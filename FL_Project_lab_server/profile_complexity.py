"""
Aggregator Complexity Profiler

This script performs a standalone experiment to measure and plot the
computational complexity (scalability) of the robust aggregators.

It measures the wall-clock time (in milliseconds) for each aggregator
as a function of the number of participating clients (n).

It does NOT run the full FL training loop.
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import functools
from collections import OrderedDict

# Import our project components
import config
from model import get_model
from aggregator import aegis, cw_med, multi_krum

# --- === 1. PROFILING PARAMETERS === ---

# List of client counts (n) to test
# Krum will become very slow at the high end.
N_CLIENTS_TO_TEST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500, 2000]

# --- H-Parameters for a fair test ---
# We must provide the same assumptions to Aegis and Krum
ASSUMED_FRACTION_BYZANTINE = 0.4

# Number of times to run for warmup (to get GPU caches, etc., ready)
N_WARMUP = 2
# Number of times to repeat the timing for a stable average
N_REPEATS = 5

# --- === 2. HELPER FUNCTION === ---

def create_fake_updates(n_clients, template_dict):
    """
    Creates a list of 'n' fake client updates.
    Each update is a (state_dict, n_k) tuple with random data.
    """
    print(f"  > Generating {n_clients} fake client updates...", end="", flush=True)
    fake_updates = []
    for _ in range(n_clients):
        # Create fake weights with the same shape/type as the real model
        fake_weights = OrderedDict()
        for key, tensor in template_dict.items():
            # Create random data on the correct device
            fake_weights[key] = torch.randn_like(tensor).to(config.DEVICE)
        
        # Fake data size (n_k). 100 is just a placeholder.
        n_k = 100 
        fake_updates.append((fake_weights, n_k))
    
    print(" Done.")
    return fake_updates

# --- === 3. MAIN PROFILING FUNCTION === ---

def profile():
    """
    Runs the main profiling experiment and saves the plot.
    """
    print("--- Starting Aggregator Complexity Profiling ---")
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_TYPE}, Dataset: {config.DATASET_NAME}")
    print(f"Testing n = {N_CLIENTS_TO_TEST}")
    print(f"Repeating {N_REPEATS} times (after {N_WARMUP} warmup runs)...\n")
    
    # --- Step 1: Get model template and dimension ---
    # We use the real model to get the correct shapes and dimension 'd'
    model = get_model().to(config.DEVICE)
    template_dict = model.state_dict()
    dim_d = sum(p.numel() for p in model.parameters())
    print(f"Model dimension (d): {dim_d:,} parameters")
    
    # --- Step 2: Generate all fake data *once* ---
    # We generate for the max number of clients to be efficient
    max_n = N_CLIENTS_TO_TEST[-1]
    all_fake_updates = create_fake_updates(max_n, template_dict)
    
    # --- Step 3: Define aggregators ---
    # We use functools.partial to "fix" the arguments for Krum
    # to match the setup in main.py
    krum_func = functools.partial(
        multi_krum, 
        fraction_byzantine=ASSUMED_FRACTION_BYZANTINE,
        m_selected=None, # Use internal default
        weighted=True
    )
    
    aggregators_to_test = [
        ("CWMed", cw_med),
        ("Aegis (Ours)", aegis),
        ("Multi-Krum", krum_func) 
    ]
    
    # Dictionary to store results: {"Aegis": [time1, time2, ...], ...}
    results = {name: [] for name, _ in aggregators_to_test}

    # --- Step 4: Run the experiment loop ---
    for n in N_CLIENTS_TO_TEST:
        print(f"\nProfiling for n = {n} clients...")
        
        # Get the subset of fake updates for this 'n'
        current_updates = all_fake_updates[:n]
        
        for name, agg_func in aggregators_to_test:
            print(f"  > Timing {name}...", end="", flush=True)
            times_ms = []
            
            for i in range(N_WARMUP + N_REPEATS):
                # Synchronize CUDA device before starting timer for accurate GPU timing
                if config.DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                
                t_start = time.perf_counter()
                
                # --- RUN THE AGGREGATOR ---
                _ = agg_func(current_updates)
                
                if config.DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                
                t_end = time.perf_counter()
                
                # Only save results *after* warmup runs
                if i >= N_WARMUP:
                    times_ms.append((t_end - t_start) * 1000) # Convert to milliseconds
            
            # Store the median time for this 'n'
            median_time = np.median(times_ms)
            results[name].append(median_time)
            print(f" Median time: {median_time:.2f} ms")

    print("\n--- Profiling Complete. Generating plot... ---")

    # --- Step 5: Plot the results ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, times in results.items():
        ax.plot(N_CLIENTS_TO_TEST, times, label=f"{name} (Scales ~{ 'n*log(n)' if 'Krum' not in name else 'n^2' })", marker='o', linewidth=2)
    
    ax.set_xlabel("Number of Clients (n) in one round", fontsize=12)
    ax.set_ylabel("Aggregation Time (milliseconds)", fontsize=12)
    ax.set_yscale('log') # <-- Use log scale to see all lines clearly
    ax.set_title(f"Aggregator Scalability vs. Number of Clients (n)\n(Model: {config.MODEL_TYPE}, Dim d: {dim_d:,})", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- Step 6: Save and Show ---
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, "aggregator_complexity_comparison.png")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    
    print(f"Plot saved to '{save_path}'")
    plt.show()

# --- === 4. RUN MAIN FUNCTION === ---

if __name__ == "__main__":
    # --- THIS IS THE FIX ---
    profile() # Was incorrectly set to main()