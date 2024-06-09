from pennylane import numpy as np

def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2 * np.pi, (2, num_layers + 1, 2, num_wires), requires_grad=True)