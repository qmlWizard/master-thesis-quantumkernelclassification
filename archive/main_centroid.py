import pennylane as qml
import pennylane.numpy as np
from utils.data_preprocessing import data_preprocess
from utils.kernel import kernel_circuit
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

def centered_kernel_matrix(centroid_idx, centroid, x, y, kernel, params):
    
    km = np.zeros((1, len(x)))
    ideal_km = np.zeros((1, len(x)))
 
    
    for i in range(len(x)):
        print("C: ", centroid)
        print("X: ", x[i])
        print(np.asarray(kernel(    x1 = centroid, 
                                    x2 = x[i],  
                                    params = params
                                ))[0])
        km[0][i] = np.asarray(kernel(  x1 = centroid, 
                            x2 = x[i],  
                            params = params
                         ))[0]
        if centroid_idx == 0:
            if y[i] == 1:
                ideal_km[0][i] = 1
            else:
                ideal_km[0][i] = -1
        else:        
            if y[i] == 1:
                ideal_km[0][i] = -1
            else:
                ideal_km[0][i] = 1
  
    return km, ideal_km
    

def target_alignment(selected_centroid, centroid, x_train, y_train, kernel, params):

    km, ideal_km = centered_kernel_matrix(selected_centroid, centroid[str(selected_centroid + 1)], x_train, y_train, kernel, params)

    T = np.outer(ideal_km, ideal_km)
    inner_product = np.sum(km * T)
    norm = np.sqrt(np.sum(km * km) * np.sum(T * T))
    inner_product = inner_product / norm
    inner_product = np.asarray(inner_product, requires_grad = True)

    return inner_product

def plot_data_and_centroids(x, y, centroids):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green']
    print(np.unique(y))
    for cls in np.unique(y):
        class_points = x[y == cls]
        print(cls - 1)
        plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[cls - 1], label=f'Class {cls}')
    
    for i, (key, centroid) in enumerate(centroids.items()):
        plt.scatter(centroid[0], centroid[1], c='red' if i == 0 else 'orange', marker='x', s=100, label=f'Centroid {key}')
    
    plt.title('Classes with Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

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
#params = np.asarray(params, requires_grad = True)

# Quantum kernel QNode definition using PennyLane
@qml.qnode(dev)
def kernel(x1, x2, params):
    return kernel_circuit(x1=x1, x2=x2, params=params, wires=wires, num_qubits=num_qubits)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

drawer = qml.draw(kernel)
print(drawer(x1=x[0], x2=x[1], params=params))
print(f"Distance between 1st and 2nd Data Points: {kernel(x1=x[0], x2=x[1], params=params)[0]}")

c = get_centroids(x_train, y_train)


opt = qml.GradientDescentOptimizer(0.2)
cost_list = []
alignment_list = []

selected_centroid = 0



for i in range(100):  #train_config['alignment_epochs']):

    cost = lambda _params: - target_alignment(selected_centroid, c, x_train, y_train, kernel, _params)

    #cost_list.append(cost(params))

    params = opt.step(cost, params)
    
    if selected_centroid:
        selected_centroid = 0
    else:
        selected_centroid = 1

    """
    print(i)

    km, ideal_km = centered_kernel_matrix(selected_centroid, c[str(selected_centroid + 1)], x_train, y_train, kernel,  params)

    weighted_sum = np.sum(km[i] @ x.T, axis=1)
    sum_weights = np.sum(km[i])
    c[str(i + 1)] = weighted_sum / sum_weights if sum_weights != 0 else c[str(i + 1)]
    """
    


plt.plot(cost_list)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost vs Epochs")
plt.show()
