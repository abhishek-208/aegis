"""
Main entry point for the Federated Learning simulation.

This script acts as an experiment runner:
1. Defines a list of experiments (e.g., FedAvg vs. Aegis).
2. Runs each simulation.
3. Collects the results.
4. Passes the results to the plotter.
"""

import time
import copy
import torch
import os   # <-- Import the os module
import functools

# Import all our project modules
import config
from model import get_model
from data_utils import load_data, partition_data, get_test_dataloader
from client import Client
from server import Server
from aggregator import fed_avg, aegis, multi_krum, cw_med
import plotter

# --- === 1. DEFINE EXPERIMENTS === ---
# Define all experiments to run in a list of dictionaries.

# Set 'run': True to run an experiment, or 'run': False to skip it.
# This implements the {0, 0, 1} toggle system you requested.
#

EXPERIMENT_CONFIGS = [
    {
        'run': True,  # Set to False to skip this one
        'label': f"FedAvg (With no Attack)",
        'aggregator': fed_avg,
        'attack_type': 'none',
        'fraction_byzantine': 0.0,
        'color': 'g', # Green
        'marker': 'o'
    },
    {
        'run': True,  # <-- Set to False to skip the sign_flip test
        'label': f"FedAvg (With {config.ATTACK_TYPE} Attack)",
        'aggregator': fed_avg,
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE,   # Use a strong 40% attack
        'color': 'r', # Red
        'marker': 'x'
    },
    {
        'run': True,  # <-- Set to False to skip the sign_flip test
        'label': f"Aegis (With {config.ATTACK_TYPE} Attack)",
        'aggregator': aegis,
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': 'b', # Blue
        'marker': 's' # Square
    },
    
    {
        'run': False,  
        'label': f"FedAvg (With {config.ATTACK_TYPE} Attack)",
        'aggregator': fed_avg,
        'attack_type': config.ATTACK_TYPE, 
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': 'orange', 
        'marker': 'v'      
    },

    {
        'run': False,  # <-- Run this   experiment
        'label': f"FedAvg (With {config.ATTACK_TYPE} Attack)",
        'aggregator': aegis,
        'attack_type': config.ATTACK_TYPE, 
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': 'purple', 
        'marker': 'P'       
    },
    
    {
        'run': True,  
        'label': f"CWMed (With {config.ATTACK_TYPE} Attack)",
        'aggregator': cw_med,  # <-- Use the   function
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': 'magenta', #   color
        'marker': 'p'       #   marker (star)
    },
    {
        'run': True,  
        'label': f"Krum (With {config.ATTACK_TYPE} Attack)",
        # We use functools.partial to "pre-fill" the fraction_byzantine
        # argument that multi_krum needs.
        'aggregator': functools.partial(
            multi_krum, 
            fraction_byzantine=config.FRACTION_BYZANTINE, # for calculating 'f'
            m_selected=None,        # Use internal default (n - f - 2)
            weighted=True           # Use weighted avg for fair comparison
        ),
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE, # This is for the server to *create* attackers
        'color': 'black',   #   color
        'marker': 'D'       #   marker (diamond)
    }

]

# --- === 2. SIMULATION RUNNER === ---

def run_simulation(exp_config):
    """
    Runs a single, complete FL simulation based on an experiment config.
    
    Args:
        exp_config (dict): A dictionary from the EXPERIMENT_CONFIGS list.
        
    Returns:
        dict: A result dictionary containing the label, color, marker,
              accuracy_history, and loss_history.
    """
    
    print(f"\n\n------------------------------- Starting Experiment: {exp_config['label']} -------------------------------")
    exp_start_time = time.time()
    
    # --- Profiling: Initialize Timers ---
    timing_summary = {
        "data_setup": 0.0,
        "client_training": 0.0,
        "server_aggregation": 0.0,
        "evaluation": 0.0
    }
    
    # --- Step 1: Load Data & Create Clients (Timed) ---
    t_start_data = time.time()
    
    train_dataset, test_dataset = load_data()
    client_dataloaders = partition_data(train_dataset)
    test_loader = get_test_dataloader(test_dataset)
    all_clients = [Client(cid, loader) for cid, loader in enumerate(client_dataloaders)]
    
    t_end_data = time.time()
    timing_summary["data_setup"] = t_end_data - t_start_data
    
    print(f"\n[Data] Successfully created {len(all_clients)} clients. (Time: {timing_summary['data_setup']:.2f}s)")
    
    # --- Step 2: Initialize Server ---
    server = Server(
        aggregator_func=exp_config['aggregator'],
        test_loader=test_loader
    )
    
    # --- Step 3: Run Training Rounds ---
    accuracy_history = []
    loss_history = []
    
    for round_num in range(config.NUM_ROUNDS):
        print(f"\n    ------------------------------- Round {round_num + 1}/{config.NUM_ROUNDS} -------------------------------")
        
        # Run one round and get timings
        round_timings = server.run_round(
            all_clients=all_clients,
            attack_type=exp_config['attack_type'],
            fraction_byzantine=exp_config['fraction_byzantine']
        )
        
        # Add round timings to summary
        timing_summary["client_training"] += round_timings["train_time"]
        timing_summary["server_aggregation"] += round_timings["agg_time"]
        
        # --- Step 4: Evaluate (Timed) ---
        if (round_num + 1) % config.EVALUATE_EVERY_N_ROUNDS == 0:
            t_start_eval = time.time()
            loss, accuracy = server.evaluate()
            
            # Synchronize for accurate GPU timing
            if config.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_end_eval = time.time()
            
            timing_summary["evaluation"] += (t_end_eval - t_start_eval)
            
            accuracy_history.append(accuracy)
            loss_history.append(loss)
            
            print(f"\n    > EVALUATION: Global Model Loss: {loss:.4f}, Accuracy: {accuracy:.2f}% (Time: {(t_end_eval - t_start_eval):.2f}s)")
        
    exp_end_time = time.time()
    total_duration = exp_end_time - exp_start_time
    
    # --- Step 5: Print Profiling Summary ---
    print(f"\n\n------------------------------- Experiment Finished. -------------------------------")
    print(f"  > Total Duration: {total_duration:.2f}s")
    print(f"\n  > --- Profiling Summary ---")
    
    # Calculate "Other" time
    profiling_total = sum(timing_summary.values())
    other_time = total_duration - profiling_total
    timing_summary["other (loops, print, etc)"] = other_time
    
    # Print formatted summary table
    print(f"    {'Stage':<28} | {'Time (s)':<10} | {'Percentage':<10}")
    print(f"    {'-'*28:<28} | {'-'*10:<10} | {'-'*10:<10}")
    for stage, stage_time in timing_summary.items():
        percentage = (stage_time / total_duration) * 100
        print(f"    {stage:<28} | {stage_time:<10.2f} | {percentage:<10.1f}%")
    print(f"    {'-'*28:<28} | {'-'*10:<10} | {'-'*10:<10}")
    print(f"    {'Total':<28} | {total_duration:<10.2f} | 100.0%")
    
    # Return the results
    return {
        "label": exp_config['label'],
        "color": exp_config['color'],
        "marker": exp_config['marker'],
        "history": accuracy_history,
        "loss_history": loss_history,
        "duration": total_duration
    }

# --- === 3. MAIN EXECUTION === ---

def main():
    """
    Main execution function.
    """
    
    print(f"\n\nImporting modules....\n")
    print("\n\n------------------------------- Starting FL Simulation Runner -------------------------------")
    print(f"Config: {config.NUM_ROUNDS} rounds, {config.NUM_CLIENTS} total clients")
    print(f"Device: {config.DEVICE}")

    # Run all enabled experiments
    all_results = []
    for exp_config in EXPERIMENT_CONFIGS:
        if exp_config.get('run', True):
            result = run_simulation(exp_config)
            all_results.append(result)
        else:
            print(f"\n\n------------------------------- Skipping Experiment: {exp_config['label']} -------------------------------")

    # --- Step 4: Plot Results ---
    if not all_results:
        print("\n\n------------------------------- No experiments were run. Exiting. -------------------------------")
        return

    print("\n\n------------------------------- All simulations complete. Generating plot... -------------------------------")
    
    # 1. Generate line plots for training dynamics
    save_path_acc, save_path_loss = plotter.plot_results(all_results, config)
    
    # 2. Generate bar charts for final summary
    save_path_acc_bar, save_path_loss_bar = plotter.plot_final_summary_bars(all_results, config)
    
    print("\n--- Plotting Complete ---")
    print(f"Line plots saved to: '{save_path_acc}' and '{save_path_loss}'")
    print(f"Bar charts saved to: '{save_path_acc_bar}' and '{save_path_loss_bar}'")

if __name__ == "__main__":
    # Fix for multiprocessing on Windows/macOS
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    main()