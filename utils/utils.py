from pennylane import numpy as np
from sklearn.svm import SVC

def random_params(num_wires, num_layers, ansatz):
    if 'efficient_su2':
        return np.random.uniform(0, 2 * np.pi, (2, num_layers + 1, 2, num_wires, 4), requires_grad=True)
    
def uncertinity_sampling_subset(X, svm_trained, subSize, sampling = 'entropy', ranking = False):
	
	if sampling == 'entropy':
		if ranking:
			probabilities = svm_trained.predict_proba(X)
			entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
		
			sorted_indices = np.argsort(entropy)
			sorted_entropy_values = np.sort(entropy)

			probabilities = np.linspace(1, 0, len(entropy))
			probabilities = probabilities / probabilities.sum()

			sampled_indices = np.random.choice(sorted_indices, size=subSize, p=probabilities)

			return sampled_indices
		
		else:
			probabilities = svm_trained.predict_proba(X)
			entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
			selected_indices = np.argsort(entropy)[:subSize]

			return selected_indices
	
	return None

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)