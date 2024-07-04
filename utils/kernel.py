import pennylane as qml
from pennylane import numpy as np
from utils.encoding import angle_encoding
from utils.ansatz import efficient_su2, basic
import concurrent.futures
from itertools import product

def variational_circuit(x, params, num_qubits, wires):

	layers = len(params) - 1
	
	for j in range(layers):
		angle_encoding(
						input_data = x, 
				  		num_qubits = num_qubits, 
				  		wires = wires, 
						gate = 'RZ', 
						input_scaling = True, 
						input_scaling_param = params[j, 0, 0, 0],  
						hadamard = True
					   )
		
		efficient_su2(num_qubits = num_qubits, params = params[j, 1:], wires = wires )

def kernel_circuit(x1, x2, num_qubits, wires, params = np.asarray([])):
	
	variational_circuit(
						x = x1,
					 	params = params[0],
						num_qubits = num_qubits,
						wires = wires
					   )
	
	adjoint_variational_circuit = qml.adjoint(variational_circuit)

	adjoint_variational_circuit(
						x = x2, 
						params = params[1],
						num_qubits = num_qubits,
						wires = wires
						)
		
	return qml.probs(wires=wires)

def kernel_without_ansatz(x1, x2, wires, num_qubits):
	angle_encoding(
						input_data = x1, 
				  		num_qubits = num_qubits, 
				  		wires = wires, 
						gate = 'RZ', 
						hadamard = True
					   )
	
	adjoint_angle_encoding = qml.adjoint(variational_circuit)

	adjoint_angle_encoding(
						x = x2, 
						num_qubits = num_qubits,
						gate = 'RZ',
						wires = wires
						)



def square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    r"""Computes the square matrix of pairwise kernel values for a given dataset.

    Args:
        X (list[datapoint]): List of datapoints
        kernel ((datapoint, datapoint) -> float): Kernel function that maps
            datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, in
            which case the diagonal of the kernel matrix is set to 1, avoiding unnecessary
            computations.

    Returns:
        array[float]: The square matrix of kernel values.
    """
    N = qml.math.shape(X)[0]
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))

    matrix = [None] * N**2

    def compute_kernel(i, j):
        return kernel(X[i], X[j])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Compute all off-diagonal kernel values, using symmetry of the kernel matrix
        futures = {executor.submit(compute_kernel, i, j): (i, j) for i in range(N) for j in range(i + 1, N)}
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            kernel_value = future.result()
            matrix[N * i + j] = kernel_value
            matrix[N * j + i] = kernel_value

    if assume_normalized_kernel:
        # Create a one-like entry that has the same interface and batching as the kernel output
        # As we excluded the case N=1 together with assume_normalized_kernel above, matrix[1] exists
        one = qml.math.ones_like(matrix[1])
        for i in range(N):
            matrix[N * i + i] = one
    else:
        # Fill the diagonal by computing the corresponding kernel values
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(compute_kernel , i, i): i for i in range(N)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                matrix[N * i + i] = future.result()

    shape = (N, N) if qml.math.ndim(matrix[0]) == 0 else (N, N, qml.math.size(matrix[0]))

    return qml.math.moveaxis(qml.math.reshape(qml.math.stack(matrix), shape), -1, 0)


def kernel_matrix(X1, X2, kernel):

    N = qml.math.shape(X1)[0]
    M = qml.math.shape(X2)[0]

    def compute_kernel(pair):
        x, y = pair
        return kernel(x, y)

    pairs = list(product(X1, X2))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_kernel, pairs))

    matrix = qml.math.stack(results)

    if qml.math.ndim(matrix[0]) == 0:
        return qml.math.reshape(matrix, (N, M))

    return qml.math.moveaxis(qml.math.reshape(matrix, (N, M, qml.math.size(matrix[0]))), -1, 0)
