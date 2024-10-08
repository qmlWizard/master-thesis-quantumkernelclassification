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

# Training loop
cost_list = []
for step in range(alignment_epochs):
    
    kmatrix = kernel_matrix(x_train, x_train)
    kmatrix_symmetric = (kmatrix + kmatrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(kmatrix_symmetric)

    # Zero out negative eigenvalues
    eigenvalues[eigenvalues < 0] = 0

    # Reconstruct the kernel matrix, now guaranteed to be positive semi-definite
    kmatrix_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    print(kmatrix.shape)
    svm_aligned = SVC(kernel='precomputed', probability=True).fit(kmatrix_psd, y_train)
    #svm_aligned = SVC(kernel=kernel, probability=True).fit(x_train, y_train)
 
    print('done training')
    subset = uncertinity_sampling_subset(
                X=kmatrix,
                svm_trained=svm_aligned,
                subSize=6,
                ranking=False
            )

    print(subset)
    # Define cost function with JAX-compatible gradient
    cost = lambda _params: -qml.kernels.target_alignment(
            x_train[subset],
            y_train[subset],
            lambda x1, x2: kernel(x1, x2, _params)[0],
            assume_normalized_kernel=True,
        )

    # Update parameters using JAX's gradient descent optimizer
    #params = params - 0.2 * cost_grad(params)
    cost_list.append(cost(params))
    params = opt.step(cost, params)

    """                                          
    if (step + 1) % 10 == 0:
        kernel_values = batch_kernel(x_train, x_train, params)
        alignments = []
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                alignment = qml.kernels.target_alignment(
                    x_train,
                    y_train,
                    lambda x1, x2: kernel_values[i, j],
                    assume_normalized_kernel=True,
                )
                alignments.append(alignment)
        current_alignment = sum(alignments) 
    """
    print(f"Step {step+1}") #- Alignment = {cost:.3f}")


trained_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_aligned = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

accuracy_trained = accuracy(svm_aligned, x_train, y_train)
y_test_pred = svm_aligned.predict(x_test)
testing_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {accuracy_trained:.3f}")
print(f"Testing accuracy: {testing_accuracy:.3f}")

data_dict = {
    'train_accuracy': [accuracy_trained],
    'test_accuracy': [testing_accuracy],
    'cost_list': [cost_list],
    'subset_size': [6]
}


np.save('train_greedy_6.npy', data_dict)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cost_list) + 1), cost_list, marker='o', linestyle='-', color='b', label='Cost per Epoch')

# Add titles and labels
plt.title('Cost per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()
plt.savefig('train_greedy_6.png')