import tensorflow as tf
import time
import numpy as np

# Function to run the workload and measure time
def run_workload(device_name, matrix_size=10000, conv_size=512, iterations=10):
    print(f"\nRunning on {device_name}")
    with tf.device(device_name):
        # Large random matrices
        a = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        b = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        
        # Synthetic image for convolution (batch_size, height, width, channels)
        img = tf.random.normal([1, conv_size, conv_size, 64], dtype=tf.float32)
        kernel = tf.random.normal([3, 3, 64, 128], dtype=tf.float32)  # Conv filter
        
        # Warm-up run (to initialize GPU/CPU)
        _ = tf.matmul(a, b)
        _ = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # Start timing
        start_time = time.time()
        
        # Repeated matrix multiplications
        for _ in range(iterations):
            c = tf.matmul(a, b)
        
        # Convolution
        for _ in range(iterations):
            conv_out = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # Ensure computation completes
        c.numpy()  # Force evaluation
        conv_out.numpy()
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken on {device_name}: {elapsed_time:.3f} seconds")
    return elapsed_time

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Run on GPU if available
if tf.config.list_physical_devices('GPU'):
    gpu_time = run_workload("/GPU:0")
else:
    print("No GPU found, running on CPU only")

# Run on CPU
cpu_time = run_workload("/CPU:0")

# Performance comparison
if tf.config.list_physical_devices('GPU'):
    speedup = cpu_time / gpu_time
    print(f"\nGPU Speedup over CPU: {speedup:.2f}x")