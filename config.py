train_config = {
	'training_dataset_path': '/Users/digvijay/Developer/MasterThesis/master-thesis-quantumkernelclassification/Data/Testdata.csv',
	'dr_technique': 'pca',
	'dr_components': 3,
    'train_without_alignment': False,
    'train_classical_svm': False,
    'train_with_alignment_random_sampling': True,
    'train_with_alignment_greedy_sampling': True,
    'train_with_alignment_prob_greedy_sampling': True,
    'test_accuracy': False,
    'train_size': 0.2,
    'training_layers': 6,
    'ansatz': 'efficient_su2',
    'uncertainty_sampling': False,
    'sampling_type': 'entropy',
    'classical_kernels': ['rbf', 'linear', 'poly'],
    'quantum_alignments': ['random', 'greedy'],
    'ranking': [True, False],
    'subset_sizes': [4, 8, 16, 24, 48],
    'file_name': "Experiment_1.txt",
    'alignment_epochs': 500,
	'multithreading':True,
}
