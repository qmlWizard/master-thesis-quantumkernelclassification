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

"""
#Custom functions
from kernels import kernel_xx

"""

from data_preprocessing import data_preprocess
from kernel_alignment import alignment


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
	print("Sample Data: ")

	cnt = 0
	for i, j in zip(x, y):
		if cnt == 5:
			break
		print("X: {} --> Y: {}".format(i, j))
		cnt += 1

	
	print("----------------------------------------------------------------------------------------------------")





