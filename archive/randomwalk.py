import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define the number of qubits and the device
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits, shots=1000)

# Define the quantum kernel
def quantum_kernel(params, x, y):
    @qml.qnode(dev)
    def circuit():
        # Encoding the data points x and y
        qml.AngleEmbedding(x, wires=range(n_qubits))
        # Apply parameterized gates
        for i in range(n_qubits):
            qml.Rot(*params[i], wires=i)
        qml.AngleEmbedding(y, wires=range(n_qubits))
        # Measure the overlap
        return qml.probs(wires=0)
    
    probs = circuit()
    return probs[0]  # Probability of measuring |0> state

# Define the kernel alignment function
def kernel_alignment(K, K_target):
    return np.inner(K.flatten(), K_target.flatten()) / (np.linalg.norm(K) * np.linalg.norm(K_target))

# Define the QAE function
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

# Optimization routine
def optimize_kernel_parameters(K_target, initial_params, data):
    def objective(params):
        reshaped_params = params.reshape((n_qubits, 3))
        alignments = [estimate_alignment(reshaped_params, K_target, data[i], data[j]) 
                      for i in range(len(data)) for j in range(len(data))]  
        estimated_alignment = np.mean(alignments)
        return -estimated_alignment  # Minimize the negative alignment
    
    # Flatten initial_params
    flat_initial_params = initial_params.flatten()
    
    result = minimize(objective, flat_initial_params, method='COBYLA')
    # Reshape the optimized parameters back to the original shape
    optimized_params = result.x.reshape((n_qubits, 3))
    return optimized_params

# Define the RBF kernel function
def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

# Define the target kernel matrix using the RBF kernel
def compute_target_kernel(data, sigma=1.0):
    n_samples = len(data)
    K_target = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K_target[i, j] = rbf_kernel(data[i], data[j], sigma)
    return K_target

# Generate a binary classification dataset
np.random.seed(0)  # For reproducibility
data = np.random.rand(50, 4)  # 50 data points with 4 features each
labels = np.random.randint(2, size=50)  # Binary labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Compute the target kernel matrix using the training data
K_target = compute_target_kernel(X_train)

# Initial parameters for your quantum kernel
initial_params = np.random.rand(n_qubits, 3)

# Optimize the parameters
optimized_params = optimize_kernel_parameters(K_target, initial_params, X_train)

# Compute the quantum kernel matrix for the training and testing data
def compute_quantum_kernel_matrix(params, X1, X2):
    n_samples1 = len(X1)
    n_samples2 = len(X2)
    K = np.zeros((n_samples1, n_samples2))
    for i in range(n_samples1):
        for j in range(n_samples2):
            K[i, j] = quantum_kernel(params, X1[i], X2[j])
    return K

K_train = compute_quantum_kernel_matrix(optimized_params, X_train, X_train)
K_test = compute_quantum_kernel_matrix(optimized_params, X_test, X_train)

# Train the SVM using the quantum kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Predict using the SVM
y_pred = svm.predict(K_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the SVM with the quantum kernel:", accuracy)
