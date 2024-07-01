import multiprocessing
import numpy as np
import concurrent.futures


##Cosine SIMILARITY

def kernel(x1, x2):
    return np.dot(x1, x2) / (x1 * x2)

def worker(pairs):
    result = kernel(pairs[0], pairs[1])
    i, j = pairs[2], pairs[3]
        #result_queue.put((i, j, result))
    return (result, i, j)
    
if __name__ == "__main__":
    multiprocessing.set_start_method('fork')

    processes = []
    result_queue = multiprocessing.Queue()

    num_processes = 2  # Number of processes to run

    matrix = np.random.rand(1000)
    kernel_matrix = np.zeros((len(matrix), len(matrix)))

    print(len(matrix))

    pairs = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            pairs.append((matrix[i], matrix[j], i, j))

    chunk_size = len(pairs) // num_processes

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(worker, pair) for pair in pairs}
        for future in concurrent.futures.as_completed(futures):
            kernel_value, i, j = future.result()
            kernel_matrix[i][j] = kernel_value


    with np.printoptions(precision=3, suppress=True):
        print(kernel_matrix)

    print("Shape: ", kernel_matrix.shape)

"""
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
"""