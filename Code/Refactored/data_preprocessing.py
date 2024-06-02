import pandas as pd
import numpy as np


#Add the different read types to increase the reachability of code
def read_file(path):
	if 'csv' in path:
		df = pd.read_csv(path)
	elif 'json' in path:
		df = pd.read_json(path)
	elif 'xlsx' in path:
		df - pd.read_excel(path)
	else:
		return "Unrecognised File Path. Please provide correct right path (Supports csv, json and xlsx files)"
	
	return df

def normalize_data(x, y):
	return x, y


def dimentionality_reduction(x, y, technique = None, components = 0):

	if technique == 'pca':
		from sklearn.decomposition import PCA
		pca = PCA(n_components = components)
		x =  pca.fit_transform(x)
		return x, y.ravel()
	
	elif technique == 'lda':
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
		lda = LinearDiscriminantAnalysis(n_components = components)
		x =  lda.fit_transform(x, y)
		return x, y.ravel()
	
	else:
		return x, y.ravel()

def data_preprocess(path, dr_type = None, dr_components = 0, normalize = True):

	df = read_file(path)
	print("Columns present in file,")
	print(list(df.columns))
	cols = list(df.columns) #.remove('fault')

	x = df[cols]
	y = df[['fault']].to_numpy()		
	
	if normalize:
		x, y = normalize_data(x, y)

	x, y = dimentionality_reduction(x, y, dr_type, components = dr_components)	

	return x, y