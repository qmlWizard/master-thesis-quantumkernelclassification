import pennylane as qml
from pennylane import numpy as np
from utils.data_preprocessing import data_preprocess
from utils.kernel_alignment import target_alignment
from utils.encoding import angle_encoding
from utils.ansatz import efficient_su2
from utils.kernel import kernel_circuit, square_kernel_matrix, kernel_matrix
from utils.utils import random_params, uncertinity_sampling_subset, accuracy
from config import train_config
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
import threading
import logging
import concurrent.futures

import sys
from datetime import datetime
import time

alignment_epochs = 10
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

def train(kernel, train_type = 'random', subset_size = 4, ranking = False):
	
	
	params = random_params(
				num_wires = len(wires), 
				num_layers = train_config['training_layers'],
				ansatz = train_config['ansatz']
				)
				
	
	params = params
	opt = qml.GradientDescentOptimizer(0.2)

	cost_list = []
	for i in range(train_config['alignment_epochs']):
		
		if train_type == 'random':
			subset = np.random.choice(list(range(len(x_train))), subset_size)
		
		elif train_type == 'greedy':
			trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
			trained_kernel_matrix_greedy = lambda X1, X2: kernel_matrix(X1, X2, trained_kernel_greedy)
			svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy, probability=True).fit(x_train, y_train)

			subset = uncertinity_sampling_subset(
								X = x_train, 
								svm_trained=svm_aligned_greedy, 
								subSize=subset_size,
								ranking=ranking
							    )
		
		# Define the cost function for optimization
		cost = lambda _params: -target_alignment(
			x_train[subset],
			y_train[subset],
			lambda x1, x2: kernel(x1, x2, _params)[0],
			assume_normalized_kernel=True,
		)

		# Optimization step
		params = opt.step(cost, params)
		cost_list.append(cost)

		# Report the alignment on the full dataset every 10 steps.
		if (i + 1) % 10 == 0:
			current_alignment = target_alignment(
								x_train,
								y_train,
								lambda x1, x2: kernel(x1, x2, params)[0],
								assume_normalized_kernel=True,
							   )
			print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

	trained_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
	trained_kernel_matrix = lambda X1, X2: kernel_matrix(X1, X2, trained_kernel)
	svm_aligned = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

	accuracy_trained = accuracy(svm_aligned, x_train, y_train)
	logging.info(f"Accuracy with {train_type} sampling with Ranking = {ranking} and subset Size = {subset_size} = {accuracy_trained}")

	y_test_pred = svm_aligned.predict(x_test)
	testing_accuracy = accuracy_score(y_test, y_test_pred)

	return accuracy_trained, testing_accuracy, cost_list

if __name__ == "__main__":

	with open(train_config['file_name'], "w") as file:
		print("Experiment 1")
		logging.info(f"Experiment 1")

		print("----------------------------------------------------------------------------------------------------")
		now = datetime.now()
		current_hour = now.hour

		greeting = ""
		if current_hour < 12:
			greeting = "Good morning"
		elif current_hour < 18:
			greeting = "Good afternoon"
		else:
			greeting = "Good evening"

		print(f"{greeting}!")
		formatted_datetime = now.strftime("%A, %d %B %Y at %H:%M:%S")
		print(f"Today's date and time: {formatted_datetime}")

		print("----------------------------------------------------------------------------------------------------")

		filepath = train_config['training_dataset_path']
		dr = train_config['dr_technique']
		dr_comp = train_config['dr_components']
		print('Reading the Data file ...')
		try:
			x, y = data_preprocess(path=filepath, dr_type=dr, dr_components=dr_comp, normalize=False)
		except Exception as e:
			print("Error while Reading the file")
			print(e)
			logging.info(f"Error while Reading the file")
			logging.info(e)

		x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= train_config['train_size'], random_state=42)

		print("----------------------------------------------------------------------------------------------------")
		time.sleep(2)
		print("Sample Data: ")
		logging.info(f"Sample Data: ")
		

		cnt = 0
		for i, j in zip(x, y):
			if cnt == 5:
				break
			print("X: {} --> Y: {}".format(i, j))
			logging.info("X: {} --> Y: {}".format(i, j))
			cnt += 1

		num_qubits = len(x[0])
		
		print("----------------------------------------------------------------------------------------------------")
		time.sleep(2)
		print("Creating Quantum Kernel Circuit...")
		logging.info("Creating Quantum Kernel Circuit...")

		print("----------------------------------------------------------------------------------------------------")

		dev = qml.device("default.qubit", wires = num_qubits, shots = None)
		wires = dev.wires.tolist()

		params = random_params(
							num_wires = len(wires), 
							num_layers = train_config['training_layers'],
							ansatz = train_config['ansatz']
							) 
	
		@qml.qnode(dev)
		def kernel(x1, x2, params):
			return kernel_circuit(  x1 = x1, 
						x2 = x2, 
						params = params, 
						wires = wires, 
						num_qubits = num_qubits
					     )
		
		print("----------------------------------------------------------------------------------------------------")
		drawer = qml.draw(kernel)
		print(drawer(x1 = x[0], x2 = x[1], params = params))
		print("----------------------------------------------------------------------------------------------------")
		print('Distance between 1st and 2nd Data Points', kernel(x1 = x[0], x2 = x[1], params = params)[0])
		print("----------------------------------------------------------------------------------------------------")

		if train_config['train_classical_svm']:
			print("Training Classical Support Vector Classifier... ")
			for k in train_config['classical_kernels']:
				print("Training SVM with {0} kernel".format(k.upper()))
				classical_svm = SVC(kernel=k).fit(x_train, y_train)
				y_train_pred = classical_svm.predict(x_train)
				training_accuracy = accuracy_score(y_train, y_train_pred)

				print("Training Complete.")
				print(f"Classical Training Accuracy: {training_accuracy * 100:.2f}%")
				logging.info(f"Classical Training Accuracy ({k.upper()} Kernel): {training_accuracy * 100:.2f}%")
				
				
				print("Testing trained Classical Support Vector Classifier... ")
				y_test_pred = classical_svm.predict(x_test)
				testing_accuracy = accuracy_score(y_test, y_test_pred)

				print(f"Classical Testing Accuracy: {testing_accuracy * 100:.2f}%")
				logging.info(f"Classical Testing Accuracy ({k.upper()} Kernel): {testing_accuracy * 100:.2f}%")
				
				
				print("----------------------------------------------------------------------------------------------------")
				print("----------------------------------------------------------------------------------------------------")

		threads = []

		if train_config['train_without_alignment']:

			print("Training Quantum Support Vector Classifier... ")

			without_align_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
			without_align_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, without_align_kernel) 

			print('-------------------------------------------------------------------')
			k = square_kernel_matrix(x_train, without_align_kernel, assume_normalized_kernel=True)
			with np.printoptions(precision=3, suppress=True):
				print(k)

			without_align_svm = SVC(kernel = without_align_kernel_matrix).fit(x_train, y_train)

			y_train_pred = without_align_svm.predict(x_train)
			training_accuracy = accuracy_score(y_train, y_train_pred)

			print("Training Complete.")
			print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
			logging.info(f"Training Accuracy (without kernel alignment): {training_accuracy * 100:.2f}%")

			print("----------------------------------------------------------------------------------------------------")
			
			if train_config['test_accuracy']:
				print("Testing trained Support Vector Classifier... ")
				y_test_pred = without_align_svm.predict(x_test)
				testing_accuracy = accuracy_score(y_test, y_test_pred)

				print(f"Testing Accuracy: {testing_accuracy * 100:.2f}%")

				print("----------------------------------------------------------------------------------------------------")

		
		training_accuracy, testing_accuracy, cost_list = train(kernel, 'random', 8, False)

		"""
		for subset_method in train_config['quantum_alignments']:
			for ranking_method in train_config['ranking']:
				for subset_size in train_config['subset_sizes']:
					training_accuracy, testing_accuracy, cost_list = train(kernel, 'random', subset_size, False)
					save_dict = {	
							'subset_method': subset_method,
							'ranking_method': ranking_method,
							'subset_size': subset_size,
							'training_accuracy': training_accuracy, 
							'testing_accuracy': testing_accuracy, 
							'cost_list': [cost_list]
						    } 
					fname = subset_method + '_' + str(ranking_method) + '_' + str(subset_size) + '.csv'

					df = pd.DataFrame(save_dict)
					df.to_csv(fname)
		"""
					