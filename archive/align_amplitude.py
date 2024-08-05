import pennylane as qml
from jax import numpy as jnp
from jax import grad, jit, random
import numpy as np
from utils.data_preprocessing import data_preprocess
from utils.kernel_alignment import target_alignment
from utils.kernel import kernel_circuit, kernel_matrix
from config import train_config
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils.utils import random_params, uncertinity_sampling_subset, accuracy
from jax import vmap
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load your configurations
filepath = train_config['training_dataset_path']
dr = train_config['dr_technique']
dr_comp = train_config['dr_components']
num_layers = train_config['training_layers']
ansatz = train_config['ansatz']
train_size = train_config['train_size']
alignment_epochs = train_config['alignment_epochs']

print('Reading the Data file ...')
try:
    x, y = data_preprocess(path=filepath, dr_type=dr, dr_components=dr_comp, normalize=False)
except Exception as e:
    print("Error while Reading the file")

num_qubits = len(x[0])

print("Creating Quantum Kernel Circuit...")

# Define the quantum device and parameters
dev = qml.device("default.qubit", wires=num_qubits, shots=None)
wires = dev.wires.tolist()

# Function to initialize random parameters

params = random_params(num_wires=len(wires), num_layers=num_layers, ansatz=ansatz)

# Quantum kernel QNode definition using PennyLane
@qml.qnode(dev, interface="jax")
def kernel(x1, x2, params):
    return kernel_circuit(x1=x1, x2=x2, params=params, wires=wires, num_qubits=num_qubits)


def batch_kernel(x1_batch, x2_batch, params):
    return vmap(lambda x1: vmap(lambda x2: kernel(x1, x2, params)[0])(x2_batch))(x1_batch)

def compute_kernel_matrix(x1, x2):
    return batch_kernel(x1, x2, params)
# Gradient descent optimizer using JAX
opt = qml.GradientDescentOptimizer(0.2)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

kernel_matrix = lambda x1, x2: compute_kernel_matrix(x1, x2)

def kernel_alignment(K, K_target):
    return np.inner(K.flatten(), K_target.flatten()) / (np.linalg.norm(K) * np.linalg.norm(K_target))


K_target = kernel_matrix(x_train, x_train)

def prepare_alignment_state(params, x, y):
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)
    
    probs = circuit()
    return probs[1]  # Probability of measuring |1> state

def estimate_alignment(params, K_target, x, y):
    alignment = prepare_alignment_state(params, x, y)
    return alignment


def optimize_kernel_parameters(K_target, initial_params, data):
    def objective(params):
        reshaped_params = params.reshape((num_qubits, 3))
        alignments = [estimate_alignment(params, K_target, data[i], data[j]) 
                      for i in range(len(data)) for j in range(len(data))]  
        estimated_alignment = np.mean(alignments)
        return -estimated_alignment  # Minimize the negative alignment
    
    # Flatten initial_params
    flat_initial_params = initial_params.flatten()
    
    result = minimize(objective, flat_initial_params, method='COBYLA')
    # Reshape the optimized parameters back to the original shape
    optimized_params = result.x.reshape((num_qubits, 3))
    return optimized_params