"""
Utility for plotting the results of the FL experiments.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os 
import config as config # Import config to use its values
from matplotlib.patches import Patch



def _smooth_exponential(data, weight=0.9):
    """
    Computes Exponential Moving Average (EMA) for smoothing.
    formula: y[t] = weight * y[t-1] + (1 - weight) * x[t]
    """
    if len(data) == 0:
        return data
    smoothed = []
    last = data[0]
    for val in data:
        new_val = last * weight + (1 - weight) * val
        smoothed.append(new_val)
        last = new_val
    return smoothed


def _add_participant_bars(ax, all_results):
    """
    Adds semi-transparent bars on a twin y-axis showing number of 
    participants per round. Uses the FIRST result with participant data.
    """
    return # Temporarily disabled for cleaner plots
    for result in all_results:
        counts = result.get('participant_counts', [])
        if not counts:
            continue
        
        ax2 = ax.twinx()
        rounds = np.arange(1, len(counts) + 1)
        
        ax2.bar(
            rounds, counts,
            color='gray', alpha=0.6, width=1.0,
            label='Participants/Round', zorder=0
        )
        ax2.set_ylabel('Participants per Round', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=9)
        ax2.set_ylim(0, max(counts) * 2.5)  # Keep bars short (bottom portion)
        
        # Add to legend
        handles, labels = ax.get_legend_handles_labels()
        bar_patch = Patch(facecolor='gray', alpha=0.6, label='Participants/Round')
        handles.append(bar_patch)
        ax.legend(handles=handles, loc='lower right', fontsize=10)
        
        break  # Only use first result

def plot_results(all_results, config_module):
    """
    Generates and saves TWO SEPARATE line plot files:
    1. ..._accuracy_line.png
    2. ..._loss_line.png
    
    (Now uses simple, solid lines and includes duration in legend.)
    
    Args:
        all_results (list): A list of result dictionaries from run_simulation.
        config_module (module): The config module, for parameter info.
    """
    print(f"\n[Plotter] Generating 2 result line plots (Accuracy and Loss)...")
    
    # Box 1: System Setup
    split_str = f" ({getattr(config_module, 'SHARDS_PER_CLIENT', 'N/A')} shards/client)" if config_module.DATA_SPLIT_TYPE == 'NON_IID' else ""
    sys_param_text = (
        r"$\mathbf{System\ Setup}$" + "\n"
        f"{'Dataset':<13}: {config_module.DATASET_NAME}\n"
        f"{'Split':<13}: {config_module.DATA_SPLIT_TYPE}{split_str}\n"
        f"{'Rounds':<13}: {config_module.NUM_ROUNDS}\n"
        f"{'Clients/Rd':<13}: {config_module.MIN_CLIENTS_PER_ROUND}-{config_module.MAX_CLIENTS_PER_ROUND} (of {config_module.NUM_CLIENTS})\n"
        f"{'Local Epochs':<13}: {config_module.LOCAL_EPOCHS}\n"
        f"{'Batch Size':<13}: {config_module.BATCH_SIZE}\n"
        f"{'LR':<13}: {config_module.LEARNING_RATE}"
    )

    # Box 2: Attack Profile
    # Dynamically read labels to see which attacks are actually plotted
    plot_labels = " ".join([r.get('label', '').lower() for r in all_results])
    
    threat_lines = [f"{'Byzantine (f)':<14}: {getattr(config_module, 'FRACTION_BYZANTINE', 0.0) * 100:.0f}%"]
    
    if "noise" in plot_labels:
        threat_lines.append(f"{'Attacker (σ)':<14}: {getattr(config_module, 'ATTACK_NOISE_STD', 'N/A')}")
    if "sybil" in plot_labels:
        threat_lines.append(f"{'Sybil (k)':<14}: {getattr(config_module, 'NUM_SYBILS_PER_ATTACKER', 'N/A')}")
    if "alie" in plot_labels:
        threat_lines.append(f"{'ALIE (Z)':<14}: {getattr(config_module, 'ALIE_Z', 'N/A')}")
    if "ipm" in plot_labels:
        threat_lines.append(f"{'IPM (ε)':<14}: {getattr(config_module, 'IPM_EPSILON', 'N/A')}")

    # Pad with empty lines so the box height is exactly 8 lines (matching System Setup)
    while len(threat_lines) < 7:
        threat_lines.append(" ")

    attack_param_text = r"$\mathbf{Threat\ Model}$" + "\n" + "\n".join(threat_lines)

    # Box 3: Aegis Defense
    rep_enabled = getattr(config_module, 'REPUTATION_ENABLED', False)
    defense_lines = [
        f"{'Mode':<14}: {'Adaptive' if getattr(config_module, 'ADAPTIVE_THRESHOLD_ENABLED', True) else 'Fixed'}",
        f"{'k Range':<14}: {getattr(config_module, 'K_MAX', 'N/A')} -> {getattr(config_module, 'K_MIN', 'N/A')}  (warmup={getattr(config_module, 'WARMUP_ROUNDS', 'N/A')})",
        f"{'k Floor':<14}: {getattr(config_module, 'K_SAFE_FLOOR', 'N/A')}",
        f"{'Var Sens':<14}: {getattr(config_module, 'VARIANCE_SENSITIVITY', 'N/A')}",
        f"{'Cos Threshold':<14}: {getattr(config_module, 'PASS1_COS_THRESHOLD', 'N/A')}",
        f"{'Cos Penalty α':<14}: {getattr(config_module, 'COSINE_PENALTY_WEIGHT', 'N/A')}",
        f"{'Reputation':<14}: {'ON' if rep_enabled else 'OFF'}  (λ={getattr(config_module, 'REPUTATION_WEIGHT', 'N/A')}, γ={getattr(config_module, 'REPUTATION_DECAY', 'N/A')})",
    ]
    defense_param_text = r"$\mathbf{Aegis\ Defense}$" + "\n" + "\n".join(defense_lines)

    props = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da')

    # Subtitle line with key dataset info
    param_subtitle = (
        f"{config.DATASET_NAME} - {config.DATA_SPLIT_TYPE}"
    )

    # --- === 1. ACCURACY PLOT === ---
    
    # Determine plot mode label
    if getattr(config_module, 'RUN_ABLATION_STUDY', False):
        mode_label = "Aegis Ablation Study"
    elif getattr(config_module, 'COMPARE_AEGIS_SCENARIOS', False):
        mode_label = "Aegis Scenario Comparison"
    else:
        mode_label = "Byzantine-Resilient FL Comparison"

    fig_acc, ax_acc = plt.subplots(figsize=(14, 8)) # Wider for external legend
    
    # Title and subtitle removed as requested
    
    ax_acc.set_ylabel('Global Model Accuracy (%)', fontsize=12)
    ax_acc.set_xlabel('Communication Round', fontsize=12)
    ax_acc.grid(True, linestyle='--', alpha=0.6)

    for result in all_results:
        plot_label = f"{result['label']}"
        
        # Determine x-axis based on actual length of history
        # Because Early Stopping might have stopped it early
        num_points = len(result['history'])
        actual_rounds = np.arange(
            config_module.EVALUATE_EVERY_N_ROUNDS, 
            (num_points * config_module.EVALUATE_EVERY_N_ROUNDS) + 1, 
            config_module.EVALUATE_EVERY_N_ROUNDS
        )

        # Smoothing parameters
        # Alpha is the weight of previous points. 0.9 = very smooth.
        smoothing_weight = getattr(config_module, 'PLOT_SMOOTHING_WEIGHT', 0.85)

        # Plot the smoothed data
        smoothed_acc = _smooth_exponential(result['history'], weight=smoothing_weight)
        ax_acc.plot(
            actual_rounds,
            smoothed_acc,
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle=result.get('linestyle', 'solid'),
            linewidth=2.5,
        )
    
    
    ax_acc.legend(loc='lower right', fontsize=14, ncol=1, framealpha=0.9, borderpad=1.2, labelspacing=1.2)
    _add_participant_bars(ax_acc, all_results)
    
    # --- Adaptive K Plot (Twin Axis) ---
    # We only plot this if at least one result has valid adaptive k history
    # has_adaptive_k = any(r.get('adaptive_k_history') and any(k is not None for k in r['adaptive_k_history']) for r in all_results)
    has_adaptive_k = False # Temporarily disabled for cleaner plots
    
    if has_adaptive_k:
        ax_k = ax_acc.twinx()
        ax_k.set_ylabel('Adaptive Threshold (k)', fontsize=12, color='black')
        ax_k.spines['right'].set_position(('outward', 60)) # Move slightly right to not overlap with participants
        
        for result in all_results:
            k_hist = result.get('adaptive_k_history', [])
            if not k_hist or all(k is None for k in k_hist):
                continue
                
            # Filter None values (plot continuous line for available segments)
            # Simplest approach: Replace None with np.nan and plot
            k_values = [k if k is not None else np.nan for k in k_hist]
            
            # Determine x-axis (same as accuracy)
            num_points_k = len(k_values)
            actual_rounds_k = np.arange(1, num_points_k + 1)
            
            # Since accuracy is evaluated every N rounds, but K is recorded every round?
            # Wait, main.py records k EVERY ROUND. Accuracy is EVERY N.
            # So actual_rounds_k should be 1..Total.
            
            # Plot K as a thin black line
            ax_k.plot(actual_rounds_k, k_values, color='black', linewidth=1.0, linestyle='-', alpha=0.6, label='Adaptive k')

        # Add to legend manually or trust standard legend?
        # Standard legend on ax_acc won't show ax_k lines easily without manual work.
        # Let's add a dummy patch to ax_acc legend
        k_line_proxy = plt.Line2D([0], [0], color='black', linewidth=1.0, label='Adaptive Threshold (k)')
        handles, labels = ax_acc.get_legend_handles_labels()
        handles.append(k_line_proxy)
        ax_acc.legend(handles=handles, loc='lower right', fontsize=14, ncol=1, framealpha=0.9, borderpad=1.2, labelspacing=1.2)


    # Parameter text boxes removed as requested

    
    # --- === 2. LOSS PLOT === ---
    
    fig_loss, ax_loss = plt.subplots(figsize=(14, 8)) # Wider for external legend
    
    # Title and subtitle removed as requested

    ax_loss.set_ylabel('Global Model Loss', fontsize=12)
    ax_loss.set_xlabel('Communication Round', fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.6)

    for result in all_results:
        plot_label = f"{result['label']}"

        # Determine x-axis based on actual length of history
        num_points = len(result['loss_history'])
        actual_rounds = np.arange(
            config_module.EVALUATE_EVERY_N_ROUNDS, 
            (num_points * config_module.EVALUATE_EVERY_N_ROUNDS) + 1, 
            config_module.EVALUATE_EVERY_N_ROUNDS
        )

        # Plot the smoothed data
        smoothed_loss = _smooth_exponential(result['loss_history'], weight=smoothing_weight)
        ax_loss.plot(
            actual_rounds,
            smoothed_loss,
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle=result.get('linestyle', 'solid'),
            linewidth=2.5,
        )
    
    ax_loss.legend(loc='upper right', fontsize=14, ncol=1, framealpha=0.9, borderpad=1.2, labelspacing=1.2)
    _add_participant_bars(ax_loss, all_results)
    # Parameter text boxes removed as requested

    # --- === 3. SAVE AND SHOW === ---
    
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)

    # Build descriptive filename: aggregators_attack_mode
    agg_names = list(dict.fromkeys(
        r.get('aggregator_name', 'unknown') for r in all_results
    ))
    attacks = list(dict.fromkeys(
        r.get('attack_type', 'none') for r in all_results
    ))
    agg_part    = "+".join(agg_names)  if agg_names else "unknown"
    attack_part = "+".join(attacks)    if attacks   else "none"
    if getattr(config_module, 'RUN_ABLATION_STUDY', False):
        mode_part = "ablation"
    elif getattr(config_module, 'COMPARE_AEGIS_SCENARIOS', False):
        mode_part = "comparison"
    else:
        mode_part = "manual"
    base_filename = f"{agg_part}_{attack_part}_{mode_part}"

    save_path_acc  = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_accuracy_line.png")
    save_path_loss = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_loss_line.png")
    
    fig_acc.tight_layout(pad=3.0)
    fig_acc.savefig(save_path_acc, dpi=300, bbox_inches='tight')
    
    fig_loss.tight_layout(pad=3.0)
    fig_loss.savefig(save_path_loss, dpi=300, bbox_inches='tight')
    
    print(f"  > Line plot (Accuracy) saved to {save_path_acc}")
    print(f"  > Line plot (Loss) saved to {save_path_loss}")
    
    # --- Write Text Summary ---
    txt_sys = sys_param_text.replace(r"$\mathbf{System\ Setup}$", "=== System Setup ===")
    txt_attack = attack_param_text.replace(r"$\mathbf{Threat\ Model}$", "=== Threat Model ===")
    txt_defense = defense_param_text.replace(r"$\mathbf{Aegis\ Defense}$", "=== Aegis Defense ===")
    
    summary_path = os.path.join(config_module.RESULTS_DIR, "Experiment_Summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(txt_sys + "\n\n")
        f.write(txt_attack + "\n\n")
        f.write(txt_defense + "\n\n")
        
        f.write("=== Results ===\n")
        for res in all_results:
            final_acc = res.get('best_accuracy', res['history'][-1] if res.get('history') else 0)
            final_loss = res.get('best_loss', res['loss_history'][-1] if res.get('loss_history') else 0)
            dur = res.get('duration', 0)
            f.write(f"\nExperiment: {res['label']}\n")
            f.write(f"  Best Accuracy: {final_acc:.2f}%\n")
            f.write(f"  Best Loss: {final_loss:.4f}\n")
            f.write(f"  Duration: {dur:.1f}s\n")
            if res.get('avg_filter_acc') is not None:
                dr_val = res.get('avg_detection_rate')
                pr_val = res.get('avg_precision')
                dr_str = f"{dr_val:.1f}%" if dr_val is not None else "N/A"
                pr_str = f"{pr_val:.1f}%" if pr_val is not None else "N/A"
                f.write(f"  Avg Aegis Filter Accuracy: {res['avg_filter_acc']:.1f}%\n")
                f.write(f"  Avg Aegis Detection: {dr_str}\n")
                f.write(f"  Avg Aegis Precision: {pr_str}\n")
                
    print(f"  > Summary text saved to {summary_path}")
    
    plt.show() # Show both line plots
    plt.close('all') # Prevent memory leaks
    
    return save_path_acc, save_path_loss

# --- === 4. BAR CHART FUNCTION === ---

def plot_final_summary_bars(all_results, config_module):
    """
    Generates and saves TWO SEPARATE bar chart files for the
    FINAL results of all experiments.
    """
    print(f"\n[Plotter] Generating 2 final summary bar charts...")
    
    # Determine plot mode label
    if getattr(config_module, 'RUN_ABLATION_STUDY', False):
        mode_label = "Aegis Ablation Study"
    elif getattr(config_module, 'COMPARE_AEGIS_SCENARIOS', False):
        mode_label = "Aegis Scenario Comparison"
    else:
        mode_label = "Byzantine-Resilient FL Comparison"
    
    # Extract data for plotting
    labels = [f"{r['label']}" for r in all_results]
    colors = [r['color'] for r in all_results]
    
    # Bar chart uses best values achieved before early stopping degradation
    final_accuracies = [r.get('best_accuracy', r['history'][-1] if r.get('history') else 0) for r in all_results]
    final_losses     = [r.get('best_loss', r['loss_history'][-1] if r.get('loss_history') else 0) for r in all_results]
    
    # --- === 1. FINAL ACCURACY BAR CHART === ---
    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    
    x_ticks = np.arange(len(labels))
    ax_acc.bar(x_ticks, final_accuracies, color=colors, width=0.4)
    ax_acc.set_xticks(x_ticks)
    
    # Title removed as requested
    
    ax_acc.set_ylabel('Best Accuracy (%)', fontsize=14)
    ax_acc.set_ylim(bottom=0, top=max(final_accuracies) * 1.15)
    
    ax_acc.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
    
    for i, acc in enumerate(final_accuracies):
        ax_acc.text(i, acc + 0.5, f"{acc:.2f}%", ha='center', fontsize=14, fontweight='bold')
        
    # --- === 2. FINAL LOSS BAR CHART === ---
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    
    ax_loss.bar(x_ticks, final_losses, color=colors, width=0.4)
    ax_loss.set_xticks(x_ticks)
    
    # Title removed as requested
    
    ax_loss.set_ylabel('Best Loss', fontsize=14)
    
    # Linear scale limits
    if final_losses:
        ax_loss.set_ylim(bottom=0, top=max(final_losses) * 1.15)
    
    ax_loss.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
    
    for i, loss in enumerate(final_losses):
        # Place text inside the bar, near the top
        ax_loss.text(i, loss * 0.85, f"{loss:.4f}", ha='center', va='top', fontsize=14, fontweight='bold')

    # --- === 3. SAVE AND SHOW === ---
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)

    # Build descriptive filename: aggregators_attack_mode
    agg_names = list(dict.fromkeys(
        r.get('aggregator_name', 'unknown') for r in all_results
    ))
    attacks = list(dict.fromkeys(
        r.get('attack_type', 'none') for r in all_results
    ))
    agg_part    = "+".join(agg_names)  if agg_names else "unknown"
    attack_part = "+".join(attacks)    if attacks   else "none"
    if getattr(config_module, 'RUN_ABLATION_STUDY', False):
        mode_part = "ablation"
    elif getattr(config_module, 'COMPARE_AEGIS_SCENARIOS', False):
        mode_part = "comparison"
    else:
        mode_part = "manual"
    base_filename = f"{agg_part}_{attack_part}_{mode_part}"

    save_path_acc_bar  = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_accuracy_bar.png")
    save_path_loss_bar = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_loss_bar.png")
    
    fig_acc.tight_layout(pad=2.0)
    fig_acc.savefig(save_path_acc_bar, dpi=300)
    
    fig_loss.tight_layout(pad=2.0)
    fig_loss.savefig(save_path_loss_bar, dpi=300)
    
    print(f"  > Bar chart (Accuracy) saved to {save_path_acc_bar}")
    print(f"  > Bar chart (Loss) saved to {save_path_loss_bar}")
    
    plt.show() # Show both bar charts
    plt.close('all') # Prevent memory leaks
    
    return save_path_acc_bar, save_path_loss_bar

# --- === 5. GRADIENT SCATTER PLOT === ---

def plot_gradient_scatter(viz_data, round_num, config_module, exp_label):
    """
    Plots the PCA projection of client gradients.
    Colors: Blue (Honest), Red (Attacker)
    Markers: Circle (Accepted), X (Rejected)
    """
    import numpy as np
    import os

    coords = viz_data["coords"]         # (N, 2)
    approved_indices = set(viz_data["approved_indices"])
    original_indices = viz_data["original_indices"] # List of client IDs
    byzantine_set = set(viz_data["byzantine_set"])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Iterate through each point to plot with correct style
    for i, client_id in enumerate(original_indices):
        x, y = coords[i]
        
        # Color: Red if Byzantine (Ground Truth), Blue if Honest
        color = 'red' if client_id in byzantine_set else 'blue'
        
        # Marker: Circle (o) if Approved, Cross (x) if Rejected
        # approved_indices in stats refers to the INDEX in the updates list, not client_id
        marker = 'o' if i in approved_indices else 'x'
        
        # Edge color/transparency
        alpha = 0.8
        
        if marker == 'x':
            # 'x' is an unfilled marker, edgecolors is ignored (causes warning)
            ax.scatter(x, y, c=color, marker=marker, s=100, alpha=alpha)
        else:
            # 'o' is filled, we want a black outline
            ax.scatter(x, y, c=color, marker=marker, s=100, alpha=alpha, edgecolors='black')
        
        # Add client ID text
        ax.text(x, y, str(client_id), fontsize=8, ha='right')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Honest (True)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Attacker (True)', markersize=10),
        Line2D([0], [0], marker='o', color='gray', label='Accepted (Aggregator)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='x', color='gray', label='Rejected (Aggregator)', markersize=10, linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, borderpad=1.0, labelspacing=1.0)

    ax.set_title(f"Gradient Projection (PCA) - Round {round_num}\nExperiment: {exp_label}")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save
    scatter_dir = os.path.join(config_module.RESULTS_DIR, "ScatterPlots")
    os.makedirs(scatter_dir, exist_ok=True)
    
    # Sanitize label for filename
    safe_label = "".join(c for c in exp_label if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    filename = f"Gradients_R{round_num}_{safe_label}.png"
    save_path = os.path.join(scatter_dir, filename)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig) # Close to modify memory usage
    
    return save_path


def _add_defense_metrics_box(ax, all_results):
    """
    Adds a summary text box to the plot showing average defense metrics
    (Accuracy, Detection Rate, Precision) for Aegis scenarios only.
    """
    metrics_lines = []

    for res in all_results:
        if res.get('avg_filter_acc') is not None:
            label = res['label']
            short_label = (label[:28] + '..') if len(label) > 28 else label
            line = (f"{short_label}:\n"
                    f"  Filter Accuracy: {res['avg_filter_acc']:.1f}%\n"
                    f"  Detection Rate:  {res['avg_detection_rate']:.1f}%\n"
                    f"  Precision:       {res['avg_precision']:.1f}%")
            metrics_lines.append(line)

    if not metrics_lines:
        return

    full_text = "Aegis Defense Quality (Avg)\n" + "\n".join(metrics_lines)
    props = dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.85, edgecolor='#7f8c8d')
    # Placed at center (x=0.35), same row as Parameters box (y=-0.12)
    ax.text(0.35, -0.12, full_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')


# --- === 5. AEGIS DIAGNOSTIC DASHBOARD === ---

def plot_aegis_diagnostics(result, config_module):
    """
    Generates a 6-panel diagnostic dashboard for a single AEGIS experiment.
    
    Panels:
      1. Model Accuracy (%) over rounds
      2. Adaptive K value with K_SAFE_FLOOR reference line
      3. False Positives per round (honest clients wrongly rejected)
      4. False Negatives per round (attackers that slipped through)
      5. Approval Rate (% of clients approved each round)
      6. Detection Rate and Precision (%)
    
    This is a diagnostic tool for parameter tuning, not a publication figure.
    
    Args:
        result (dict): A single experiment's result dictionary from run_simulation.
        config_module (module): The config module for parameter values.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    label = result.get('label', 'Unknown')
    print(f"\n[Plotter] Generating AEGIS diagnostic dashboard for: {label}")

    # --- Extract data ---
    accuracy = result.get('history', [])
    k_history = result.get('adaptive_k_history', [])
    tp_history = result.get('tp_history', [])
    fp_history = result.get('fp_history', [])
    fn_history = result.get('fn_history', [])
    approval_rate = result.get('approval_rate_history', [])
    detection_rate = result.get('detection_rate_history', [])  # Already in % (0-100 scale)
    precision = result.get('precision_history', [])
    participant_counts = result.get('participant_counts', [])

    # Determine number of rounds from the longest available series
    num_rounds = max(len(accuracy), len(k_history), len(tp_history), 1)
    
    # Build x-axes (accuracy may be sampled every N rounds, others are per-round)
    eval_interval = getattr(config_module, 'EVALUATE_EVERY_N_ROUNDS', 1)
    rounds_acc = np.arange(eval_interval, len(accuracy) * eval_interval + 1, eval_interval)

    # Trim per-round series to match (they should all be the same length)
    n = min(len(k_history), len(tp_history), len(fp_history), len(fn_history), len(approval_rate))
    if n == 0:
        print("  > [Diagnostics] No diagnostic data available. Skipping.")
        return None
    
    rounds_diag = np.arange(1, n + 1)
    k_vals = k_history[:n]
    tp_vals = tp_history[:n]
    fp_vals = fp_history[:n]
    fn_vals = fn_history[:n]
    ar_vals = approval_rate[:n]
    n_vals = participant_counts[:n] # Total selection per round

    # Calculate FP Rate (%) = FP / (Total Selected - Total Byzantine)
    # Calculate FN Rate (%) = FN / (Total Byzantine)
    fp_rate_vals = []
    fn_rate_vals = []
    for i in range(n):
        num_byzantine = tp_vals[i] + fn_vals[i]
        num_honest = max(n_vals[i] - num_byzantine, 1) # Prevent div by zero
        fp_rate_vals.append((fp_vals[i] / num_honest) * 100)
        fn_rate_vals.append((fn_vals[i] / max(num_byzantine, 1)) * 100)
    
    # Detection rate and precision may have different lengths
    dr_vals = detection_rate[:n] if len(detection_rate) >= n else detection_rate
    pr_vals = precision[:n] if len(precision) >= n else precision
    rounds_dr = np.arange(1, len(dr_vals) + 1)

    # --- Smoothing helper (EMA) ---
    smoothing = getattr(config_module, 'PLOT_SMOOTHING_WEIGHT', 0.85)
    
    def ema_smooth(data, weight=0.8):
        """Exponential moving average for visual smoothing."""
        if weight <= 0 or len(data) == 0:
            return np.array(data)
        smoothed = []
        last = data[0] if data[0] is not None else 0
        for val in data:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                smoothed.append(last)
                continue
            last = weight * last + (1 - weight) * val
            smoothed.append(last)
        return np.array(smoothed)

    # --- Create 6-panel figure ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False)
    fig.suptitle(f'AEGIS Diagnostic Dashboard\n{label}', fontsize=16, fontweight='bold')

    # Color scheme
    c_accent = '#2ecc71'   # Green
    c_warn   = '#e74c3c'   # Red
    c_info   = '#3498db'   # Blue  
    c_orange = '#e67e22'   # Orange
    c_purple = '#9b59b6'   # Purple
    c_gray   = '#95a5a6'   # Gray

    # ------ Panel 1: Model Accuracy ------
    ax1 = axes[0, 0]
    if len(accuracy) > 0:
        ax1.plot(rounds_acc, ema_smooth(accuracy, smoothing), color=c_accent, linewidth=2.0, label='Accuracy (EMA)')
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Model Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ------ Panel 2: Adaptive K Value ------
    ax2 = axes[0, 1]
    k_clean = [v if v is not None else np.nan for v in k_vals]
    ax2.plot(rounds_diag, ema_smooth(k_clean, 0.9), color=c_info, linewidth=2.5, label='K (EMA)')
    
    # Reference lines for config thresholds
    k_floor = getattr(config_module, 'K_SAFE_FLOOR', None)
    k_max = getattr(config_module, 'K_MAX', None)
    k_min = getattr(config_module, 'K_MIN', None)
    if k_floor is not None:
        ax2.axhline(y=k_floor, color=c_warn, linestyle='--', linewidth=1.5, label=f'K_SAFE_FLOOR={k_floor}')
    if k_max is not None:
        ax2.axhline(y=k_max, color=c_gray, linestyle=':', linewidth=1.0, alpha=0.5, label=f'K_MAX={k_max}')
    if k_min is not None:
        ax2.axhline(y=k_min, color=c_gray, linestyle=':', linewidth=1.0, alpha=0.5, label=f'K_MIN={k_min}')
    
    ax2.set_ylabel('K Value', fontsize=11)
    ax2.set_title('Adaptive Threshold (K)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    ax2.grid(True, alpha=0.3)

    # ------ Panel 3: False Positive Rate ------
    ax3 = axes[1, 0]
    ax3.plot(rounds_diag, ema_smooth(fp_rate_vals, 0.9), color=c_warn, linewidth=2.0, label='FP Rate (EMA)')
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.set_title('False Positive Rate (Honest Clients Rejected)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105) # Rate is a percentage

    # ------ Panel 4: False Negative Rate ------
    ax4 = axes[1, 1]
    has_attackers = any((tp_vals[i] + fn_vals[i]) > 0 for i in range(n))
    if has_attackers:
        ax4.plot(rounds_diag, ema_smooth(fn_rate_vals, 0.9), color=c_orange, linewidth=2.0, label='FN Rate (EMA)')
        ax4.legend(loc='upper right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    else:
        ax4.text(0.5, 0.5, 'N/A — No Attackers', transform=ax4.transAxes,
                 ha='center', va='center', fontsize=14, color='grey', fontstyle='italic')
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title('False Negative Rate (Attackers Missed)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)

    # ------ Panel 5: Approval Rate ------
    ax5 = axes[2, 0]
    ax5.plot(rounds_diag, ema_smooth(ar_vals, 0.9), color=c_purple, linewidth=2.0, label='Approval Rate (EMA)')
    ax5.axhline(y=70, color=c_warn, linestyle='--', linewidth=1.0, alpha=0.5, label='70% threshold')
    ax5.set_ylabel('Approval Rate (%)', fontsize=11)
    ax5.set_xlabel('Communication Round', fontsize=11)
    ax5.set_title('Client Approval Rate', fontsize=12)
    ax5.legend(loc='lower right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 105)

    # ------ Panel 6: Detection Rate & Precision ------
    ax6 = axes[2, 1]
    if len(dr_vals) > 0 and has_attackers:
        ax6.plot(rounds_dr, ema_smooth(dr_vals, 0.9), color=c_accent, linewidth=2.0, label='Detection Rate (EMA)')
    if len(pr_vals) > 0 and has_attackers:
        ax6.plot(rounds_dr[:len(pr_vals)], ema_smooth(pr_vals, 0.9), color=c_info, linewidth=2.0, label='Precision (EMA)')
    if not has_attackers:
        ax6.text(0.5, 0.5, 'N/A — No Attackers', transform=ax6.transAxes,
                 ha='center', va='center', fontsize=14, color='grey', fontstyle='italic')
    else:
        ax6.legend(loc='lower right', fontsize=12, borderpad=1.0, labelspacing=1.0)
    ax6.set_ylabel('Percentage (%)', fontsize=11)
    ax6.set_xlabel('Communication Round', fontsize=11)
    ax6.set_title('Detection Rate & Precision', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 105)

    # --- Add parameter annotation box ---
    param_text = (
        f"K_MAX={getattr(config_module, 'K_MAX', '?')}  "
        f"K_MIN={getattr(config_module, 'K_MIN', '?')}  "
        f"K_FLOOR={getattr(config_module, 'K_SAFE_FLOOR', '?')}  "
        f"Warmup={getattr(config_module, 'WARMUP_ROUNDS', '?')}  "
        f"Var_Sens={getattr(config_module, 'VARIANCE_SENSITIVITY', '?')}  "
        f"Byz={getattr(config_module, 'FRACTION_BYZANTINE', '?')}"
    )
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=9, 
             style='italic', color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # --- Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for suptitle and params
    
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)
    safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    filename = f"DIAGNOSTIC_{safe_label}.png"
    save_path = os.path.join(config_module.RESULTS_DIR, filename)
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  > Diagnostic dashboard saved to: {save_path}")
    
    # --- Print summary statistics ---
    avg_fp = np.mean(fp_rate_vals) if len(fp_rate_vals) > 0 else 0
    avg_fn = np.mean(fn_rate_vals) if len(fn_rate_vals) > 0 else 0
    avg_ar = np.mean(ar_vals) if len(ar_vals) > 0 else 0
    avg_k = np.nanmean([v for v in k_clean if not np.isnan(v)]) if len(k_clean) > 0 else 0
    fn_str = f"{avg_fn:.1f}%" if has_attackers else "N/A"
    
    print(f"  > [Summary] Avg FP Rate: {avg_fp:.1f}% | Avg FN Rate: {fn_str} | "
          f"Avg Approval Rate: {avg_ar:.1f}% | Avg K: {avg_k:.2f}")
    
    return save_path

def plot_complexity_verification(result, config_module):
    import matplotlib.pyplot as plt
    import numpy as np
    
    participants = result.get('participant_counts', [])
    agg_times = result.get('agg_time_history', [])
    
    if not participants or not agg_times:
        print("No timing data available.")
        return
    
    participants = np.array(participants)
    agg_times = np.array(agg_times) * 1000  # Convert to milliseconds
    
    # Bin by participant count and compute mean time per bin
    unique_k = np.unique(participants)
    mean_times = []
    for k in unique_k:
        mask = participants == k
        mean_times.append(np.mean(agg_times[mask]))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(unique_k, mean_times, color='#3498db', s=60, zorder=3)
    
    # Linear fit
    coeffs = np.polyfit(unique_k, mean_times, 1)
    fit_line = np.poly1d(coeffs)
    
    # Calculate R-squared to measure goodness of fit
    if len(unique_k) > 1:
        y_pred = fit_line(unique_k)
        ss_res = np.sum((mean_times - y_pred)**2)
        ss_tot = np.sum((mean_times - np.mean(mean_times))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    else:
        r_squared = 1.0 # Cannot compute R^2 with only one point, assume perfect fit for a single point
        
    x_fit = np.linspace(min(unique_k), max(unique_k), 100)
    ax.plot(x_fit, fit_line(x_fit), '--', color='#e74c3c', 
            label=f'Linear fit: {coeffs[0]:.2f}k + {coeffs[1]:.2f} (R² = {r_squared:.4f})')
    
    ax.set_xlabel('Number of Clients (k)', fontsize=12)
    ax.set_ylabel('Aggregation Time (ms)', fontsize=12)
    ax.set_title('AEGIS Aggregation Time vs Client Count', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)
    label = result.get('label', 'Unknown')
    safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    save_path = os.path.join(config_module.RESULTS_DIR, f'complexity_verification_{safe_label}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  > Complexity verification plot saved to: {save_path}")