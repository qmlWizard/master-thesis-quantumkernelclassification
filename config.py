train_config = {
	'training_dataset_path': '/Users/digvijay/Developer/MasterThesis/master-thesis-quantumkernelclassification/Data/Testdata.csv',
    'train_without_alignment': True,
    'train_classical_svm': True,
    'train_with_alignment_random_sampling': True,
    'train_with_alignment_greedy_sampling': True,
    'train_with_alignment_prob_greedy_sampling': True,
    'test_accuracy': False,
    'train_size': 0.80,
    'training_layers': 6,
    'ansatz': 'efficient_su2',
    'uncertainty_sampling': False,
    'sampling_type': 'entropy',
    'classical_kernels': ['rbf', 'linear', 'poly'],
    'subset_sizes': [4, 8, 16, 24, 48],
    'file_name': "Experiment_1.txt",
    'alignment_epochs': 10,
	'multithreading':True
}
