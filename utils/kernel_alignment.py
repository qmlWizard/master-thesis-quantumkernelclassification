import pennylane as qml
from pennylane import numpy as np
import concurrent.futures
from itertools import product
from utils.kernel import kernel_circuit
from jax import numpy as jnp
from jax import vmap

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
    N = X.shape[0]

    if assume_normalized_kernel and N == 1:
        return jnp.eye(1)

    # Create a device
    dev = qml.device("default.qubit", wires=3, shots=None)
    wires = dev.wires.tolist()

    @qml.qnode(dev, interface='jax')
    def compute_kernel(x1, x2):
        return kernel(x1, x2)[0]

    # Vectorized kernel computation
    compute_kernel_vmap = vmap(vmap(compute_kernel, in_axes=(None, 0)), in_axes=(0, None))
    matrix = compute_kernel_vmap(X, X)

    if assume_normalized_kernel:
        matrix = matrix.at[jnp.diag_indices(N)].set(1.0)
    else:
        compute_diag_kernel_vmap = vmap(compute_kernel)
        diag_elements = compute_diag_kernel_vmap(X, X)
        matrix = matrix.at[jnp.diag_indices(N)].set(diag_elements)

    return matrix



"""
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

"""
def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    K = square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = jnp.count_nonzero(jnp.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = jnp.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = jnp.array(Y)

    T = jnp.outer(_Y, _Y)
    inner_product = jnp.sum(K * T)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(T * T))
    inner_product = inner_product / norm

    return inner_product
