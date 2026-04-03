"""
Main entry point for the Federated Learning simulation.

This script acts as an experiment runner:
1. Defines a list of experiments (e.g., FedAvg vs. Aegis).
2. Runs each simulation.
3. Collects the results.
4. Passes the results to the plotter.
"""

import sys
import time
import copy
import torch
import os
import functools
from datetime import datetime

# Import all our project modules
import config
from model import get_model
from data_utils import load_data, partition_data, get_test_dataloader
from client import Client
from server import Server
from aggregator import fed_avg, aegis, multi_krum, cw_med, fools_gold, bulyan, reset_foolsgold_history, reset_aegis_reputation
import plotter

# --- === 0. LOGGING HELPER === ---
class Tee(object):
    """
    A simple Tee to replicate stdout to a file.
    All print() calls will go to both terminal and the log file.
    """
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Mandatory for sys.stdout redirection
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# --- === 1. DEFINE EXPERIMENTS === ---
# Define all experiments to run in a list of dictionaries.

# Set 'run': True to run an experiment, or 'run': False to skip it.
# This implements the {0, 0, 1} toggle system you requested.
#

EXPERIMENT_CONFIGS = [
    {
        'run': False,  # Set to False to skip this one
        'label': f"FedAvg (With no Attack)",
        'aggregator': fed_avg,
        'attack_type': 'none',
        'fraction_byzantine': 0.0,
        'color': 'g', # Green
        'marker': 'o' 
    },
    {
        'run': False,  # <-- Set to False to skip the sign_flip test
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
        'label': f"CWMed (With {config.ATTACK_TYPE} Attack)",
        'aggregator': cw_med,  # <-- Use the   function
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': 'magenta', #   color
        'marker': 'p'       #   marker (star)
    },
    {
        'run': False,  
        'label': f"Krum (With {config.ATTACK_TYPE} Attack)",
        # We use functools.partial to "pre-fill" the fraction_byzantine
        # argument that multi_krum needs.
        'aggregator': functools.partial(
            multi_krum, 
            fraction_byzantine=config.FRACTION_BYZANTINE, # for calculating 'f'
            m_selected=None,        # Use internal default (n - f - 2)
            weighted=False           # Match paper exactly (arithmetic mean)
        ),
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE, # This is for the server to *create* attackers
        'color': 'black',   #   color
        'marker': 'D'       #   marker (diamond)
    },
    {
        'run': False,  # <-- Set to True to run FoolsGold
        'label': f"FoolsGold (With {config.ATTACK_TYPE} Attack)",
        'aggregator': fools_gold,
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': config.FRACTION_BYZANTINE,
        'color': '#e67e22',   # Orange
        'marker': '^'         # Triangle up
    },
    {
        # NOTE: Bulyan requires n >= 4f + 3.
        # With fraction_byzantine=0.2 and ~15 clients/round: f=3, need n>=15. OK.
        # If you use config.FRACTION_BYZANTINE=0.33 with small rounds it will ASSERT-fail.
        # Override fraction_byzantine here (and below) independently of config.py.
        'run': False,  # <-- Set to True to run Bulyan
        'label': f"Bulyan (With {config.ATTACK_TYPE} Attack)",
        'aggregator': functools.partial(
            bulyan,
            fraction_byzantine=0.2,   # Must satisfy: n_selected >= 4*floor(n*0.2) + 3
        ),
        'attack_type': config.ATTACK_TYPE,
        'fraction_byzantine': 0.2,    # Must match the value above (server creates this many attackers)
        'color': '#00bcd4',   # Cyan
        'marker': 'v'         # Triangle down
    }
]

# --- === OVERRIDE FOR Self COMPARISON MODE === ---
if config.COMPARE_AEGIS_SCENARIOS:
    print(f"\n>>> [Config Override] Running Automated Aegis Comparison Protocol <<<")

    # --- Master Toggles: Switch entire data-split groups ON/OFF ---
    RUN_BALANCED_IID   = False
    RUN_UNBALANCED_IID = False
    RUN_NON_IID        = True

    EXPERIMENT_CONFIGS = []

    # ======================== BALANCED IID ========================
    if RUN_BALANCED_IID:
        EXPERIMENT_CONFIGS += [
            {
                'run': True,
                'label': "Aegis (Balanced IID - No Attack)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'none',
                'fraction_byzantine': 0.0,
                'color': '#2ecc71'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Sign Flip)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'sign_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e74c3c'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Label Flip)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'label_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#9b59b6'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Additive Noise)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#3498db'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - IPM)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'ipm',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e67e22'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Volume Spam)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'volume_spam',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#1abc9c'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Catastrophic Noise)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'catastrophic_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e056fd'
            },

            {
                'run': True,
                'label': "Aegis (Balanced IID - ALIE)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'alie',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#c0392b'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Pure Additive Noise)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'pure_additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#7f8c8d'
            },
            {
                'run': True,
                'label': "Aegis (Balanced IID - Sybil)",
                'aggregator': aegis,
                'data_split': 'BALANCED_IID',
                'attack_type': 'sybil',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#2c3e50'
            },
        ]

    # ====================== UNBALANCED IID =======================
    if RUN_UNBALANCED_IID:
        EXPERIMENT_CONFIGS += [
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - No Attack)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'none',
                'fraction_byzantine': 0.0,
                'color': '#27ae60'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Sign Flip)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'sign_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#c0392b'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Label Flip)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'label_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#8e44ad'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Additive Noise)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#2980b9'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - IPM)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'ipm',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#d35400'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Volume Spam)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'volume_spam',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#16a085'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Catastrophic Noise)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'catastrophic_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#be2edd'
            },

            {
                'run': True,
                'label': "Aegis (Unbalanced IID - ALIE)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'alie',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#c0392b'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Pure Additive Noise)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'pure_additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#95a5a6'
            },
            {
                'run': True,
                'label': "Aegis (Unbalanced IID - Sybil)",
                'aggregator': aegis,
                'data_split': 'UNBALANCED_IID',
                'attack_type': 'sybil',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#34495e'
            },
        ]

    # =========================== NON-IID ===========================
    if RUN_NON_IID:
        EXPERIMENT_CONFIGS += [
            {
                'run': True,
                'label': "Aegis (Non-IID - No Attack)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'none',
                'fraction_byzantine': 0.0,
                'color': '#2ecc71'
            },
            {
                'run': 0,
                'label': "Aegis (Non-IID - Sign Flip)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'sign_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e74c3c'
            },
            {
                'run': 0,
                'label': "Aegis (Non-IID - Label Flip)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'label_flip',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#9b59b6'
            },
            {
                'run': 0,
                'label': "Aegis (Non-IID - IPM)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'ipm',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e67e22'
            },            
            {
                'run': False,
                'label': "Aegis (Non-IID - Volume Spam)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'volume_spam',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#1abc9c'
            },        
            {
                'run': False,
                'label': "Aegis (Non-IID - ALIE)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'alie',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#c0392b'
            },

            {
                'run': False,
                'label': "Aegis (Non-IID - Sybil)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'sybil',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#2c3e50'
            },
            {
                'run': False,
                'label': "Aegis (Non-IID - Pure Additive Noise)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'pure_additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#bdc3c7'
            },
            {
                'run': False,
                'label': "Aegis (Non-IID - Additive Noise)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'additive_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#3498db'
            },
            {
                'run': False,
                'label': "Aegis (Non-IID - Catastrophic Noise)",
                'aggregator': aegis,
                'data_split': 'NON_IID',
                'attack_type': 'catastrophic_noise',
                'fraction_byzantine': config.FRACTION_BYZANTINE,
                'color': '#e056fd'
            },
        ]

    # =========================== ABLATION STUDY ===========================
elif config.RUN_ABLATION_STUDY:
    print(f"\n>>> [Config Override] Running Aegis Ablation Study Protocol <<<")

    EXPERIMENT_CONFIGS = [
        {
            'run': True,
            'label': f"Full Aegis ({config.ATTACK_TYPE})",
            'aggregator': aegis,
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#2ecc71',
            'marker': 'o'
        },
        {
            'run': True,
            'label': f"Aegis [No Volume Clip] ({config.ATTACK_TYPE})",
            'aggregator': functools.partial(aegis, ablate_volume_clipping=True),
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#e74c3c',
            'marker': 'x'
        },
        {
            'run': True,
            'label': f"Aegis [No Euclidean Filter] ({config.ATTACK_TYPE})",
            'aggregator': functools.partial(aegis, ablate_euclidean_filter=True),
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#ff1493', # Deep Pink
            'marker': 'd' # thin diamond
        },
        {
            'run': True,
            'label': f"Aegis [No Directional Filter] ({config.ATTACK_TYPE})",
            'aggregator': functools.partial(aegis, ablate_directional=True),
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#9b59b6',
            'marker': 's'
        },
        {
            'run': True,
            'label': f"Aegis [No Cosine Penalty] ({config.ATTACK_TYPE})",
            'aggregator': functools.partial(aegis, ablate_cosine_penalty=True),
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#e67e22',
            'marker': '^'
        },
        {
            'run': True,
            'label': f"Aegis [No Adaptive Thresholding] ({config.ATTACK_TYPE})",
            'aggregator': functools.partial(aegis, ablate_adaptive=True),
            'attack_type': config.ATTACK_TYPE,
            'fraction_byzantine': config.FRACTION_BYZANTINE,
            'color': '#3498db',
            'marker': 'v'
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
    
    # --- Fix Random Seed for Reproducibility ---
    # Ensures every experiment comparing the same data split gets EXACTLY the same client shards and Byzantines
    import numpy as np
    import random
    seed = config.RANDOM_SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Reset stateful aggregator histories to prevent cross-experiment contamination
    reset_foolsgold_history()
    reset_aegis_reputation()
    
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
    
    # Check if experiment config overrides the global data split
    split_override = exp_config.get('data_split', None)
    
    # Pass the override (or None) to partition_data
    client_dataloaders = partition_data(train_dataset, split_type=split_override)
    
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
    attack_intensity_history = []
    participant_counts = []
    adaptive_k_history = [] # Track k over time in case of adaptive k
    
    # --- Defense Quality Metrics (for Aegis) ---
    filter_accuracy_history = []
    detection_rate_history = []
    precision_history = []

    # --- Raw Diagnostic Counts (for diagnostic dashboard) ---
    tp_history = []
    fp_history = []
    fn_history = []
    approval_rate_history = []
    
    # Best Model Tracking & Early Stopping Initialization
    best_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0

    for round_num in range(config.NUM_ROUNDS):
        print(f"\n    ------------------------------- Round {round_num + 1}/{config.NUM_ROUNDS} -------------------------------")
        
        # Run one round and get timings
        round_timings = server.run_round(
            all_clients=all_clients,
            attack_type=exp_config['attack_type'],
            fraction_byzantine=exp_config['fraction_byzantine'],
            current_round=round_num
        )
        
        # Add round timings to summary
        timing_summary["client_training"] += round_timings["train_time"]
        timing_summary["server_aggregation"] += round_timings["agg_time"]
        
        # Track attack intensity for visualization
        num_byz = round_timings.get("num_byzantine", 0)
        num_sel = round_timings.get("num_selected", 1)
        attack_intensity_history.append(num_byz / num_sel if num_sel > 0 else 0.0)
        participant_counts.append(num_sel)
        
        # Track adaptive k
        adaptive_k_history.append(round_timings.get("adaptive_k", None))
        
        # Track defense metrics
        if round_timings.get("accuracy") is not None:
             filter_accuracy_history.append(round_timings["accuracy"])
        if round_timings.get("detection_rate") is not None:
             detection_rate_history.append(round_timings["detection_rate"])
        if round_timings.get("precision") is not None:
             precision_history.append(round_timings["precision"])

        # Track raw counts for diagnostic dashboard
        tp_history.append(round_timings.get("tp", 0))
        fp_history.append(round_timings.get("fp", 0))
        fn_history.append(round_timings.get("fn", 0))

        n_approved = round_timings.get("num_approved", 0)
        n_rejected = round_timings.get("num_rejected", 0)
        total_evaluated = n_approved + n_rejected
        approval_rate = (n_approved / total_evaluated * 100) if total_evaluated > 0 else 100.0
        approval_rate_history.append(approval_rate)
        
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
            
            # --- Visualization Check ---
            should_visualize = (
                config.VISUALIZE_GRADIENTS and 
                (round_num + 1) % config.VISUALIZE_EVERY_N_ROUNDS == 0
            )
            
            if should_visualize and "viz_data" in round_timings and round_timings["viz_data"]:
                 # We only save the plot, we don't block
                 save_path_viz = plotter.plot_gradient_scatter(
                     round_timings["viz_data"], 
                     round_num + 1, 
                     config, 
                     exp_config['label']
                 )
                 print(f"      > [Visualization] Gradient Scatter Plot saved: {save_path_viz}")

            # --- Best Model Tracking & Early Stopping Check ---
            if loss < best_loss - config.MIN_DELTA:
                best_loss = loss
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                
            if config.EARLY_STOPPING_ENABLED:
                if patience_counter > 0:
                    print(f"      [Early Stopping] No improvement. Patience: {patience_counter}/{config.PATIENCE}")
                    
                    if patience_counter >= config.PATIENCE:
                        print(f"\n      [Early Stopping] Patience limit reached at round {round_num + 1}. Stopping training.")
                        break
        
    exp_end_time = time.time()
    total_duration = exp_end_time - exp_start_time
    
    # --- Step 5: Print Profiling Summary ---
    print(f"\n\n------------------------------- Experiment Finished. -------------------------------")
    print(f"  > Total Duration: {total_duration:.2f}s")
    
    if accuracy_history and loss_history:
        print(f"  > Best Model Tracker —  Accuracy: {best_accuracy:.2f}%  |  Loss: {best_loss:.4f}")
    
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
    
    # --- Compute Average Defense Metrics ---
    import numpy as np
    avg_filter_acc = np.mean(filter_accuracy_history) if filter_accuracy_history else None
    avg_detection_rate = np.mean(detection_rate_history) if detection_rate_history else None
    avg_precision = np.mean(precision_history) if precision_history else None
    
    if avg_filter_acc is not None:
        print(f"\n    [Defense Stats] Avg Filter Accuracy: {avg_filter_acc:.1f}% | Detection Rate: {avg_detection_rate:.1f}% | Precision: {avg_precision:.1f}%")

    # Derive a clean aggregator name for the summary table
    agg_func = exp_config['aggregator']
    if hasattr(agg_func, 'func'):          # functools.partial wraps the real function
        agg_name = agg_func.func.__name__
    else:
        agg_name = agg_func.__name__

    # Return the results
    return {
        "label": exp_config['label'],
        "color": exp_config['color'],
        "marker": exp_config.get('marker', 'o'),
        "history": accuracy_history,
        "loss_history": loss_history,
        "attack_intensity": attack_intensity_history,
        "participant_counts": participant_counts,
        "adaptive_k_history": adaptive_k_history,
        "avg_filter_acc": avg_filter_acc,
        "avg_detection_rate": avg_detection_rate,
        "avg_precision": avg_precision,
        "duration": total_duration,
        # Raw diagnostic histories
        "tp_history": tp_history,
        "fp_history": fp_history,
        "fn_history": fn_history,
        "approval_rate_history": approval_rate_history,
        "detection_rate_history": detection_rate_history,
        "precision_history": precision_history,
        "best_accuracy": best_accuracy,
        "best_loss": best_loss,
        # For the final summary table
        "aggregator_name": agg_name,
        "attack_type": exp_config.get('attack_type', 'unknown'),
    }

# --- === 3. MAIN EXECUTION === ---

def main():
    """
    Main execution function.
    """
    
    # 1. Initialize Logging
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a descriptive prefix based on the active run mode
    if config.RUN_ABLATION_STUDY:
        run_mode = "ablation"
    elif config.COMPARE_AEGIS_SCENARIOS:
        run_mode = "comparison"
    else:
        run_mode = f"manual_{config.ATTACK_TYPE}"

    log_prefix = f"{run_mode}_{config.DATASET_NAME}_{config.DATA_SPLIT_TYPE}"
    log_filename = os.path.join(config.RESULTS_DIR, f"log_{log_prefix}_{timestamp}.txt")
    
    original_stdout = sys.stdout
    sys.stdout = Tee(log_filename)
    
    try:
        print(f"\n\nImporting modules....\n")
        print("\n\n------------------------------- Starting FL Simulation Runner -------------------------------")
        print(f"Config: {config.NUM_ROUNDS} rounds, {config.NUM_CLIENTS} total clients")
        print(f"Device: {config.DEVICE}")
        print(f"Log File: {log_filename}")

        # Run all enabled experiments, saving plots progressively after each one
        all_results = []
        for exp_config in EXPERIMENT_CONFIGS:
            if exp_config.get('run', True):
                result = run_simulation(exp_config)
                all_results.append(result)
            else:
                print(f"\n\n------------------------------- Skipping Experiment: {exp_config['label']} -------------------------------")
                continue

            # --- Save plots after every completed scenario (progressive overwrite) ---
            # This ensures partial results are never lost if the run is interrupted.
            print(f"\n[Plotter] Saving progressive plots with {len(all_results)} scenario(s) so far...")
            plotter.plot_results(all_results, config)
            plotter.plot_final_summary_bars(all_results, config)

            # Generate per-experiment diagnostic dashboard
            plotter.plot_aegis_diagnostics(result, config)

        # --- Final confirmation ---
        if not all_results:
            print("\n\n------------------------------- No experiments were run. Exiting. -------------------------------")
            return

    finally:
        # --- Final Experiment Summary Table ---
        if all_results:
            col_label = 36
            col_agg   = 14
            col_atk   = 22
            col_acc   = 12
            col_loss  = 10
            sep = f"    {'-'*col_label} | {'-'*col_agg} | {'-'*col_atk} | {'-'*col_acc} | {'-'*col_loss}"
            print(f"\n\n{'='*90}")
            print(f"  EXPERIMENT SUMMARY ({len(all_results)} run(s) completed)")
            print(f"{'='*90}")
            print(f"    {'Experiment':<{col_label}} | {'Aggregator':<{col_agg}} | {'Attack':<{col_atk}} | {'Final Acc (%)':<{col_acc}} | {'Final Loss':<{col_loss}}")
            print(sep)
            for r in all_results:
                final_acc  = f"{r['history'][-1]:.2f}"  if r['history']  else 'N/A'
                final_loss = f"{r['loss_history'][-1]:.4f}" if r['loss_history'] else 'N/A'
                print(f"    {r['label']:<{col_label}} | {r.get('aggregator_name','?'):<{col_agg}} | {r.get('attack_type','?'):<{col_atk}} | {final_acc:<{col_acc}} | {final_loss:<{col_loss}}")
            print(sep)
            print(f"{'='*90}")

        # 2. Cleanup and Restore stdout
        print(f"\n\n[System] All experiments completed. Closing log file.")
        sys.stdout.close()
        sys.stdout = original_stdout

    print("\n\n------------------------------- All simulations complete. Final plots saved. -------------------------------")

if __name__ == "__main__":
    # Fix for multiprocessing on Windows/macOS
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    main()
    
    # --- Auto-Shutdown Logic ---
    if getattr(config, 'AUTO_SHUTDOWN', False):
        import platform
        print("\n[System] AUTO_SHUTDOWN is enabled. Shutting down in 60 seconds...")
        print("         To cancel, open a terminal and run 'shutdown -a'")
        
        system_platform = platform.system()
        if system_platform == 'Windows':
            os.system('shutdown /s /t 60')
        elif system_platform == 'Linux' or system_platform == 'Darwin':
            os.system('sudo shutdown -h +1')
        else:
            print(f"         [Warning] Auto-shutdown not supported on OS: {system_platform}")