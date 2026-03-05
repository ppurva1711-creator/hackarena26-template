import time
import torch
import numpy as np
from vit_snake_detection import load_vit_model, predict_one_image, load_index_and_meta, DEFAULT_CSV_PATH

# Configuration
TEST_IMAGE = r"D:\snakebite_project\data\images\arthropod_bite\arthropod_bite1.jpg"
ITERATIONS = 50

def benchmark():
    print(f"Starting Benchmarking on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    
    # 1. Load Resources (Model Loading Time)
    start_load = time.time()
    processor, model = load_vit_model()
    index, embeddings, meta = load_index_and_meta()
    end_load = time.time()
    load_time = end_load - start_load
    print(f"Model Load Time: {load_time:.4f} seconds")

    # Warmup
    print("Warming up...")
    predict_one_image(TEST_IMAGE, processor, model, index, meta, k=1, csv_path=DEFAULT_CSV_PATH)
    
    # 2. Inference Benchmark
    print(f"Running {ITERATIONS} iterations...")
    latencies = []
    
    for i in range(ITERATIONS):
        start_inf = time.time()
        # Note: predict_one_image includes image loading + preprocessing + inference + search
        _ = predict_one_image(TEST_IMAGE, processor, model, index, meta, k=1, csv_path=DEFAULT_CSV_PATH)
        end_inf = time.time()
        latencies.append((end_inf - start_inf) * 1000) # ms
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    std_dev = np.std(latencies)
    
    print("\n===== RESULTS =====")
    print(f"Average System Latency: {avg_latency:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency:.2f} ms")
    print(f"Throughput: {1000/avg_latency:.2f} req/sec")
    
if __name__ == "__main__":
    benchmark()
