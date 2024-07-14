import pennylane as qml
from pennylane import numpy as np
import concurrent.futures
from itertools import product
from utils.kernel import kernel_circuit

def kernel_matrix(kernel, X1, X2):

    N = qml.math.shape(X1)[0]
    M = qml.math.shape(X2)[0]

    def compute_kernel(pair):
        x, y = pair
        return kernel(x, y)

    pairs = list(product(range(N), range(M)))

    matrix = [None] * (N * M)

    # Parallel computation of kernel values
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_kernel, (X1[i], X2[j])): (i, j) for i, j in pairs}
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            matrix[N * i + j] = future.result()

    matrix = qml.math.stack(matrix)

    if qml.math.ndim(matrix[0]) == 0:
        return qml.math.reshape(matrix, (N, M))

    return qml.math.moveaxis(qml.math.reshape(matrix, (N, M, qml.math.size(matrix[0]))), -1, 0)



def square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    N = qml.math.shape(X)[0]
    
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))

    # Initialize an empty matrix
    matrix = [None] * (N * N)

    dev = qml.device("default.qubit", wires=3, shots=None)
    wires = dev.wires.tolist()

    @qml.qnode(dev)
    def compute_kernel(i, j):
        return kernel(X[i], X[j])

    # Function to fill in the matrix for a given row i
    def fill_row(i):
        for j in range(i + 1, N):
            kernel_value = compute_kernel(i, j)
            matrix[N * i + j] = kernel_value
            matrix[N * j + i] = kernel_value
    
    # Parallel computation of the upper triangle (excluding diagonal)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fill_row, i) for i in range(N)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if assume_normalized_kernel:
        one = qml.math.ones_like(matrix[1])
        for i in range(N):
            matrix[N * i + i] = one
    else:
        # Compute the diagonal elements
        def fill_diagonal(i):
            matrix[N * i + i] = compute_kernel(i, i)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fill_diagonal, i) for i in range(N)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    shape = (N, N) if qml.math.ndim(matrix[0]) == 0 else (N, N, qml.math.size(matrix[0]))

    return qml.math.moveaxis(qml.math.reshape(qml.math.stack(matrix), shape), -1, 0)

def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
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