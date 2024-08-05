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
    
    def rbf_kernel_approximation(X, D, gamma):

        n_samples, n_features = X.shape
        omega = np.sqrt(2 * gamma) * np.random.randn(n_features, D)
        b = 2 * np.pi * np.random.rand(D)

        print(len(omega))
        print(len(b))
        
        
        Z = np.sqrt(2.0 / D) * np.cos(X @ omega + b)
        return Z