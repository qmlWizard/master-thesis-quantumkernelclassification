import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_preprocessing import data_preprocess
from kernel_alignment import target_alignment
from encoding import angle_encoding
from ansatz import efficient_su2
from kernel import kernel_circuit
from utils import random_params, uncertinity_sampling_subset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
import sys
from datetime import datetime
import time

#Configs
train_without_alignment = False
train_classical_svm = True
train_with_alignment_random_sampling = True
train_with_alignment_greedy_sampling = True
train_with_alignment_prob_greedy_sampling = True

test_accuracy = False
train_size = 0.10

training_layers = 6
ansatz = 'efficient_su2'

uncertinity_sampling = False
sampling_type = 'entropy'

classical_kernels = ['rbf', 'linear', 'poly']
#subset_sizes = [4, 8, 16, 24, 48]
subset_sizes = [4]



alignment_epochs = 10

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python script.py <filepath>")
		sys.exit(1)

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

	filepath = sys.argv[1]
	dr = sys.argv[2]
	dr_comp = int(sys.argv[3])
	print('Reading the Data file ...')
	try:
		x, y = data_preprocess(path=filepath, dr_type=dr, dr_components=dr_comp, normalize=False)
	except Exception as e:
		print("Error while Reading the file")
		print(e)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= train_size, random_state=42)

	print("----------------------------------------------------------------------------------------------------")
	time.sleep(2)
	print("Sample Data: ")

	cnt = 0
	for i, j in zip(x, y):
		if cnt == 5:
			break
		print("X: {} --> Y: {}".format(i, j))
		cnt += 1

	num_qubits = len(x[0])
	
	print("----------------------------------------------------------------------------------------------------")
	time.sleep(2)
	print("Creating Quantum Kernel Circuit...")

	("----------------------------------------------------------------------------------------------------")

	dev = qml.device("default.qubit", wires = num_qubits, shots = None)
	wires = dev.wires.tolist()

	params = random_params(
						   num_wires = len(wires), 
						   num_layers = training_layers,
						   ansatz = ansatz
						  ) 
 
	@qml.qnode(dev)
	def kernel(x1, x2, params):
		return kernel_circuit(x1 = x1, 
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

	if train_classical_svm:

		print("Training Classical Support Vector Classifier... ")

		for k in classical_kernels:
			print("Training SVM with {0} kernel".format(k.upper()))
			classical_svm = SVC(kernel=k).fit(x_train, y_train)
			y_train_pred = classical_svm.predict(x_train)
			training_accuracy = accuracy_score(y_train, y_train_pred)

			print("Training Complete.")
			print(f"Classical Training Accuracy: {training_accuracy * 100:.2f}%")
			
			print("Testing trained Classical Support Vector Classifier... ")
			y_test_pred = classical_svm.predict(x_test)
			testing_accuracy = accuracy_score(y_test, y_test_pred)

			print(f"Classical Testing Accuracy: {testing_accuracy * 100:.2f}%")
			print("----------------------------------------------------------------------------------------------------")
			print("----------------------------------------------------------------------------------------------------")


	if train_without_alignment:

		print("Training Quantum Support Vector Classifier... ")
		without_align_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
		without_align_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, without_align_kernel) 

		without_align_svm = SVC(kernel = without_align_kernel_matrix).fit(x_train, y_train)

		y_train_pred = without_align_svm.predict(x_train)
		training_accuracy = accuracy_score(y_train, y_train_pred)

		print("Training Complete.")
		print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
		print("----------------------------------------------------------------------------------------------------")
		if test_accuracy:
			print("Testing trained Support Vector Classifier... ")
			y_test_pred = without_align_svm.predict(x_test)
			testing_accuracy = accuracy_score(y_test, y_test_pred)

			print(f"Testing Accuracy: {testing_accuracy * 100:.2f}%")

			print("----------------------------------------------------------------------------------------------------")

	
	if train_with_alignment_random_sampling:
		for subset_size in subset_sizes:
			print("----------------------------------------------------------------------------------------------------")
			params = random_params(
						   num_wires = len(wires), 
						   num_layers = training_layers,
						   ansatz = ansatz
						  )
			
			print("Kernel Alignment with Gradient Descent for subset size: ", subset_size)
			params = params
			opt = qml.GradientDescentOptimizer(0.2)

			for i in range(alignment_epochs):
				subset = np.random.choice(list(range(len(x_train))), subset_size)
				# Define the cost function for optimization
				cost = lambda _params: -target_alignment(
					x_train[subset],
					y_train[subset],
					lambda x1, x2: kernel(x1, x2, _params),
					assume_normalized_kernel=True,
				)
				print(cost)
				# Optimization step
				params = opt.step(cost, params)

				# Report the alignment on the full dataset every 50 steps.
				if (i + 1) % 10 == 0:
					current_alignment = target_alignment(
															x_train,
															y_train,
															lambda x1, x2: kernel(x1, x2, params),
															assume_normalized_kernel=True,
														)
					print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

			trained_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
			trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
			svm_aligned = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)
			print("----------------------------------------------------------------------------------------------------")
		

	if train_with_alignment_greedy_sampling:	
		for subset_size in subset_sizes:
			print("----------------------------------------------------------------------------------------------------")
			params = random_params(
						   num_wires = len(wires), 
						   num_layers = training_layers,
						   ansatz = ansatz
						  )
			
			print("Kernel Alignment with Greedy Sub Sampling for subset size: ", subset_size)
			opt = qml.GradientDescentOptimizer(0.2)

			for i in range(alignment_epochs):

				trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
				trained_kernel_matrix_greedy = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel_greedy)
				svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy).fit(x_train, y_train)

				subset = uncertinity_sampling_subset(x_train, svm_trained=svm_aligned_greedy, subSize=subset_size)
				# Define the cost function for optimization
				cost = lambda _params: -target_alignment(
					x_train[subset],
					y_train[subset],
					lambda x1, x2: kernel(x1, x2, _params),
					assume_normalized_kernel=True,
				)
				print(cost)
				# Optimization step
				params = opt.step(cost, params)

				# Report the alignment on the full dataset every 50 steps.
				if (i + 1) % 10 == 0:
					current_alignment = target_alignment(
															x_train,
															y_train,
															lambda x1, x2: kernel(x1, x2, params),
															assume_normalized_kernel=True,
														)
					print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

			trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
			trained_kernel_matrix_greedy = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel_greedy)
			svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy).fit(x_train, y_train)
			print("----------------------------------------------------------------------------------------------------")
	
	if train_with_alignment_prob_greedy_sampling:
		for subset_size in subset_sizes:
			print("----------------------------------------------------------------------------------------------------")
			params = random_params(
						   num_wires = len(wires), 
						   num_layers = training_layers,
						   ansatz = ansatz
						  )
			
			print("Kernel Alignment with Probabilistic Sub Sampling for subset size: ", subset_size)
			opt = qml.GradientDescentOptimizer(0.2)

			for i in range(alignment_epochs):

				trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
				trained_kernel_matrix_greedy = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel_greedy)
				svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy).fit(x_train, y_train)

				subset = uncertinity_sampling_subset(x_train, sampling='probabilistic',  svm_trained=svm_aligned_greedy, subSize=subset_size)
				# Define the cost function for optimization
				cost = lambda _params: -target_alignment(
					x_train[subset],
					y_train[subset],
					lambda x1, x2: kernel(x1, x2, _params),
					assume_normalized_kernel=True,
				)
				
				# Optimization step
				params = opt.step(cost, params)

				# Report the alignment on the full dataset every 50 steps.
				if (i + 1) % 10 == 0:
					current_alignment = target_alignment(
															x_train,
															y_train,
															lambda x1, x2: kernel(x1, x2, params),
															assume_normalized_kernel=True,
														)
					print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

			trained_kernel_greedy = lambda x1, x2: kernel(x1, x2, params)[0]
			trained_kernel_matrix_greedy = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel_greedy)
			svm_aligned_greedy = SVC(kernel=trained_kernel_matrix_greedy).fit(x_train, y_train)
			print("----------------------------------------------------------------------------------------------------")