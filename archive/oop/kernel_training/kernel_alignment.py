import pennylane as qml
from pennylane import numpy as np
import concurrent.futures
from itertools import product
from kernels.kernels import kernel

dev = qml.device("default.qubit", wires=3, shots=None)
wires = dev.wires.tolist()

def kernel_matrix(X1, X2, num_qubits, params):

    N = qml.math.shape(X1)[0]
    M = qml.math.shape(X2)[0]

    def compute_kernel(pair, params):
        x, y = pair
        k = kernel('angle', 'efficient_su2', num_qubits, len(wires))

        print("Kernel Matrix kernel: ", k.kernel_function(x1=x, x2=y, params=params, wires=wires, num_qubits=num_qubits)[0])
        return k.kernel_function(x1=x, x2=y, params=params, wires=wires, num_qubits=num_qubits)[0]


    pairs = list(product(X1, X2))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_kernel, pairs))

    matrix = qml.math.stack(results)

    if qml.math.ndim(matrix[0]) == 0:
        return qml.math.reshape(matrix, (N, M))

    return qml.math.moveaxis(qml.math.reshape(matrix, (N, M, qml.math.size(matrix[0]))), -1, 0)


def square_kernel_matrix(X, assume_normalized_kernel=False, num_qubits = 3, params = []):

    N = qml.math.shape(X)[0]
    
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))

    matrix = [None] * N**2

    def compute_kernel(i, j):
        k = kernel('angle', 'efficient_su2', num_qubits, len(wires))
        print*"SKM: ", k.kernel_function(x1=i, x2=j, params=params, wires=wires, num_qubits=num_qubits)[0]
        return k.kernel_function(x1=i, x2=j, params=params, wires=wires, num_qubits=num_qubits)[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_kernel, i, j): (i, j) for i in range(N) for j in range(i + 1, N)}
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            kernel_value = future.result()
            matrix[N * i + j] = kernel_value
            matrix[N * j + i] = kernel_value

    if assume_normalized_kernel:
        one = qml.math.ones_like(matrix[1])
        for i in range(N):
            matrix[N * i + i] = one
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(compute_kernel , i, i): i for i in range(N)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                matrix[N * i + i] = future.result()

    shape = (N, N) if qml.math.ndim(matrix[0]) == 0 else (N, N, qml.math.size(matrix[0]))

    return qml.math.moveaxis(qml.math.reshape(qml.math.stack(matrix), shape), -1, 0)

def target_alignment(
    X,
    Y,
    num_qubits,
    params,
    assume_normalized_kernel=False,
    rescale_class_labels=True
):

    K = square_kernel_matrix(
        X = X,
        assume_normalized_kernel=assume_normalized_kernel,
        num_qubits= num_qubits,
        params = params
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product