import pennylane as qml
from pennylane import numpy as np
from utils.data_preprocessing import data_preprocess
from utils.kernel_alignment import target_alignment
from utils.encoding import angle_encoding
from utils.ansatz import efficient_su2
from utils.kernel import kernel_circuit
from utils.utils import random_params, uncertinity_sampling_subset, accuracy
from config import train_config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import concurrent.futures
from datetime import datetime
import time


def train(kernel, train_type='random', subset_size=4, ranking=False):
    params = random_params(
        num_wires=len(wires),
        num_layers=train_config['training_layers'],
        ansatz=train_config['ansatz']
    )
    
    opt = qml.GradientDescentOptimizer(0.2)
    cost_list = []
    alignment_list = []
    for i in range(train_config['alignment_epochs']):
        if train_type == 'random':
            subset = np.random.choice(list(range(len(x_train))), subset_size)
        elif train_type == 'greedy':
            trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
            trained_kernel_matrix_greedy = lambda X1, X2: qml.kernel_matrix(X1, X2, trained_kernel_greedy)
            svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy, probability=True).fit(x_train, y_train)
            subset = uncertinity_sampling_subset(
                X=x_train,
                svm_trained=svm_aligned_greedy,
                subSize=subset_size,
                ranking=ranking
            )
        
        cost = lambda _params: -target_alignment(
            x_train[subset],
            y_train[subset],
            lambda x1, x2: kernel(x1, x2, _params)[0],
            assume_normalized_kernel=True,
        )

        params = opt.step(cost, params)
        cost_list.append(cost(params))

        if (i + 1) % 10 == 0:
            current_alignment = target_alignment(
                x_train,
                y_train,
                lambda x1, x2: kernel(x1, x2, params)[0],
                assume_normalized_kernel=True,
            )
            print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

            alignment_list.append(current_alignment)

    trained_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
    trained_kernel_matrix = lambda X1, X2: qml.kernel_matrix(X1, X2, trained_kernel)
    svm_aligned = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

    accuracy_trained = accuracy(svm_aligned, x_train, y_train)

    y_test_pred = svm_aligned.predict(x_test)
    testing_accuracy = accuracy_score(y_test, y_test_pred)

    return accuracy_trained, testing_accuracy, cost_list, alignment_list

if __name__ == "__main__":

    filepath = train_config['training_dataset_path']
    dr = train_config['dr_technique']
    dr_comp = train_config['dr_components']
    print('Reading the Data file ...')
    try:
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
        logging.info(f"X: {i} --> Y: {j}")
        cnt += 1

    num_qubits = len(x[0])
        
    print("Creating Quantum Kernel Circuit...")
        
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    wires = dev.wires.tolist()

    params = random_params(
        num_wires=len(wires),
        num_layers=train_config['training_layers'],
        ansatz=train_config['ansatz']
    )
    
    @qml.qnode(dev)
    def kernel(x1, x2, params):
        return kernel_circuit(x1=x1, x2=x2, params=params, wires=wires, num_qubits=num_qubits)

    drawer = qml.draw(kernel)
    print(drawer(x1=x[0], x2=x[1], params=params))
    print(f"Distance between 1st and 2nd Data Points: {kernel(x1=x[0], x2=x[1], params=params)[0]}")


    random_false_training_accuracy, random_false_testing_accuracy, random_false_cost_list, random_false_alignment_list = train(kernel, 'random', 4, False)
    save_dict = {
                        'subset_method': ['random'],
                        'ranking_method': ['false'],
                        'subset_size': [4],
                        'training_accuracy': [random_false_training_accuracy],
                        'testing_accuracy': [random_false_testing_accuracy],
                        'cost_list': [random_false_cost_list],
                        'alignment_list': [random_false_alignment_list]
                    }

    np.save("random_false_4_results.npy", save_dict)

    greedy_false_training_accuracy, greedy_false_testing_accuracy, greedy_false_cost_list, greedy_false_alignment_list = train(kernel, 'greedy', 4, False)
    save_dict = {
                        'subset_method': ['random'],
                        'ranking_method': ['false'],
                        'subset_size': [4],
                        'training_accuracy': [greedy_false_training_accuracy],
                        'testing_accuracy': [greedy_false_testing_accuracy],
                        'cost_list': [greedy_false_cost_list],
                        'alignment_list ': [greedy_false_alignment_list]
                    }

    np.save("greedy_false_4_results.npy", save_dict)


    greedy_true_training_accuracy, greedy_true_testing_accuracy, greedy_true_cost_list, greedy_true_alignment_list = train(kernel, 'greedy', 4, True)
    save_dict = {
                        'subset_method': 'random',
                        'ranking_method': 'false',
                        'subset_size': 4,
                        'training_accuracy': greedy_true_training_accuracy,
                        'testing_accuracy': greedy_true_testing_accuracy,
                        'cost_list': [greedy_true_cost_list]
                    }

    np.save("greedy_true_4_results.npy", save_dict)