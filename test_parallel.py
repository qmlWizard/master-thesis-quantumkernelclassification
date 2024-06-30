import multiprocessing
import numpy as np

##Cosine SIMILARITY



def kernel(x1, x2):
    return np.dot(x1, x2) / (x1 * x2)

def worker(pairs, result_queue):
    for p in pairs:
        result = kernel(p[0], p[1])
        i, j = p[2], p[3]
        result_queue.put((i, j, result))

if __name__ == "__main__":
    multiprocessing.set_start_method('fork')

    processes = []
    result_queue = multiprocessing.Queue()

    num_processes = 2  # Number of processes to run

    matrix = np.asarray([11, 20, 300, 0.1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1])
    kernel_matrix = np.zeros((len(matrix), len(matrix)))

    print(len(matrix))

    pairs = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            pairs.append((matrix[i], matrix[j], i, j))

    chunk_size = len(pairs) // num_processes

    print(chunk_size)

    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else len(pairs)
        process = multiprocessing.Process(target=worker, args=(pairs[start:end], result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not result_queue.empty():
        i, j, result = result_queue.get()
        kernel_matrix[i][j] = result
        print(f"Matrix Index = i:{i} j:{j}, and pair result: {result}")

    with np.printoptions(precision=3, suppress=True):
        print(kernel_matrix)