import pennylane as qml

import pandas as pd
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
from datetime import datetime

import time

"""
#Custom functions
from kernels import kernel_xx

"""

from data_preprocessing import data_preprocess
from kernel_alignment import target_alignment
from encoding import angle_encoding
from ansatz import efficient_su2
from kernel import kernel_circuit
from utils import random_params






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

	params = random_params(len(wires), 1) 

	print(params)
 
	@qml.qnode(dev)
	def kernel_circ(x1, x2, params, wires, num_qubits):
		return kernel_circuit(x1, x2, params, wires, num_qubits)[0]

	print(kernel_circuit(x[0], x[1], params, wires, num_qubits))
	
	("----------------------------------------------------------------------------------------------------")

