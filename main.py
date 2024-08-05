import pennylane as qml
import numpy as np
from utils.data_preprocessing import data_preprocess
from utils.kernel_alignment import target_alignment
from utils.kernel import kernel_circuit, kernel_matrix
from config import train_config
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils.utils import random_params, uncertinity_sampling_subset, accuracy
import matplotlib.pyplot as plt


def get_centroids(x, y):
    classes = np.unique(y)
    
    # Initialize a dictionary to store centroids
    centroids = {}
    
    # Calculate the centroid for each class
    for cls in classes:
        class_points = x[y == cls]
        centroid = class_points.mean(axis=0)
        centroids[str(cls)] = centroid
    
    return centroids
    

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
@qml.qnode(dev)
def kernel(x1, x2, params):
    return kernel_circuit(x1=x1, x2=x2, params=params, wires=wires, num_qubits=num_qubits)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

drawer = qml.draw(kernel)
print(drawer(x1=x[0], x2=x[1], params=params))
print(f"Distance between 1st and 2nd Data Points: {kernel(x1=x[0], x2=x[1], params=params)[0]}")

c = get_centroids(x_train, y_train)

print(f"Distance between Centroid and 1st Data Points: {kernel(x1=c['1'], x2=x[0], params=params)[0]}")
print(f"Distance between Centroid and 1st Data Points: {kernel(x1=c['2'], x2=x[0], params=params)[0]}")

opt = qml.GradientDescentOptimizer(0.2)
cost_list = []
alignment_list = []

def centered_kernel_matrix(centroid, x, params):
    km = np.zeros((1, len(x)))
    for i in range(len(x)):
        km[0][i] = kernel(  x1 = centroid, 
                            x2 = x[i],  
                            params = params
                         )[0]
    
    return km

selected_centroid = 0
for i in range(1):  #train_config['alignment_epochs']):

    if selected_centroid:
        km = centered_kernel_matrix(c['2'], x_train, params)
        selected_centroid = 0
    else:
        km = centered_kernel_matrix(c['1'], x_train, params)
        selected_centroid = 1

    print(km)
    print(km.shape)
    

