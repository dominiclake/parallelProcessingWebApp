from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import multiprocessing as mp

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# ----------- Serial Multiplication -----------
def multiply_serial(a, b):
    start_time = time.time()
    result = np.dot(a, b)
    exec_time = time.time() - start_time
    return result.tolist(), exec_time

# ----------- Parallel Multiplication -----------
def worker(args):
    a_chunk, b = args
    return np.dot(a_chunk, b)

def multiply_parallel(a, b, num_processes):
    a_np = np.array(a)
    b_np = np.array(b)

    start_time = time.time()

    # Split matrix A into chunks for workers
    chunks = np.array_split(a_np, num_processes, axis=0)
    args = [(chunk, b_np) for chunk in chunks]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker, args)

    result = np.vstack(results)
    exec_time = time.time() - start_time

    return result.tolist(), exec_time

# ----------- Routes -----------
@app.route('/multiply-serial', methods=['POST'])
def serial_endpoint():
    data = request.json
    a = data['matrixA']
    b = data['matrixB']
    result, exec_time = multiply_serial(a, b)
    return jsonify({
        'method': 'serial',
        'execution_time': exec_time,
        'result': result
    })

@app.route('/multiply-parallel', methods=['POST'])
def parallel_endpoint():
    data = request.json
    a = data['matrixA']
    b = data['matrixB']
    num_threads = int(data.get('numThreads', mp.cpu_count()))

    result, exec_time = multiply_parallel(a, b, num_threads)
    return jsonify({
        'method': 'parallel',
        'execution_time': exec_time,
        'threads_used': num_threads,
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)
