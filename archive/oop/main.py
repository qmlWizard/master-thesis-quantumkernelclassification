import pennylane as qml
from pennylane import numpy as np
from kernels.kernels import kernel
from kernel_training.kernel_alignment import target_alignment, kernel_matrix, square_kernel_matrix
from preprocessing.data_preprocessing import data_preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import train_config

filepath = train_config['training_dataset_path']
dr = train_config['dr_technique']
dr_comp = train_config['dr_components']
print('Reading the Data file ...')
try:
    print(filepath)
    x, y = data_preprocess(path=filepath, dr_type=dr, dr_components=dr_comp, normalize=False)
except Exception as e:
    print("Error while Reading the file")
           

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_config['train_size'], random_state=42)

print("Sample Data: ")

cnt = 0
for i, j in zip(x, y):
    if cnt == 5:
        break
    print(f"X: {i} --> Y: {j}")
    cnt += 1

num_qubits = len(x[0])

dev = qml.device("default.qubit", wires=num_qubits, shots=None)
wires = dev.wires.tolist()
k = kernel('angle', 'efficient_su2', num_qubits, len(wires))

params = k.random_parameters(
            num_layers=train_config['training_layers'],
            num_wires=len(wires)
        )
    
@qml.qnode(dev)
def kernel(x1, x2, params):
    return k.kernel_function(x1=x1, x2=x2, params=params, wires=wires, num_qubits=num_qubits)


def train(params, train_type='random', subset_size=4, ranking=False ):
    
    opt = qml.GradientDescentOptimizer(0.2)
    cost_list = []
    
    for i in range(train_config['alignment_epochs']):
        if train_type == 'random':
            subset = np.random.choice(list(range(len(x_train))), subset_size)
        """
        elif train_type == 'greedy':
            
            trained_kernel_matrix_greedy = lambda X1, X2: kernel_matrix(X1, X2, num_qubits, params)
            svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy, probability=True).fit(x_train, y_train)
            
            subset = uncertinity_sampling_subset(
                X=x_train,
                svm_trained=svm_aligned_greedy,
                subSize=subset_size,
                ranking=ranking
            )
        """
        cost = lambda _params: -target_alignment(
            x_train[subset],
            y_train[subset],
            num_qubits,
            params,
            assume_normalized_kernel=True
        )

        print('Cost: ', cost)
        print('Params: ', params)
        params = opt.step(cost, params)
        
        cost_list.append(cost(params))

        if (i + 1) % 10 == 0:
            current_alignment = target_alignment(
                x_train,
                y_train,
                num_qubits,
                params,
                assume_normalized_kernel=True
            )
            print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

    trained_kernel_matrix = lambda X1, X2: kernel_matrix(X1, X2, num_qubits, params)
    svm_aligned = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

    #accuracy_trained = accuracy(svm_aligned, x_train, y_train)

    #y_test_pred = svm_aligned.predict(x_test)
    #testing_accuracy = accuracy_score(y_test, y_test_pred)

    #return accuracy_trained, testing_accuracy, cost_list


train(
            params=params,
            train_type='random',
            subset_size=4,
            ranking=False
         )