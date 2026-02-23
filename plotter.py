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

def _add_attack_shading(ax, all_results, config_module):
    """
    Adds vertical red shading bands to the plot based on attack intensity.
    Darker red = more attackers that round. Only shades if attack_intensity > 0.
    Uses the FIRST result that has a non-zero attack to determine shading.
    """
    MAX_ALPHA = 0.9  # Maximum opacity for the shading 
    
    for result in all_results:
        intensities = result.get('attack_intensity', [])
        if not intensities or max(intensities) == 0:
            continue
        
        # We found a result with attacks — use it for shading
        for round_idx, intensity in enumerate(intensities):
            if intensity > 0:
                round_x = round_idx + 1  # Rounds are 1-indexed
                alpha = intensity * MAX_ALPHA  # Scale alpha by attack fraction
                ax.axvspan(
                    round_x - 0.5, round_x + 0.5,
                    color='red', alpha=alpha, linewidth=0, zorder=2
                )
        
        # Add legend entry for the shading
        shading_patch = Patch(
            facecolor='red', alpha=MAX_ALPHA * 0.5,
            label=f'Attack Intensity (max {max(intensities)*100:.0f}% Byzantine)'
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(shading_patch)
        ax.legend(handles=handles, loc='lower right', fontsize=10)
        break  # Only shade once (all experiments share the same attack pattern)

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
    
    # Get the x-axis (evaluation rounds)
    eval_rounds = np.arange(
        config_module.EVALUATE_EVERY_N_ROUNDS, 
        config_module.NUM_ROUNDS + 1, 
        config_module.EVALUATE_EVERY_N_ROUNDS
    )
    
    # parameter text box (used on both plots) ---
    # Batch Size, Byzantine %, Removed Data Split ---
    param_text = (
        f"--- Parameters ---\n"
        f"Total Rounds: {config_module.NUM_ROUNDS}\n"
        f"Total Clients: {config_module.NUM_CLIENTS}\n"
        f"Clients/Round: {config_module.MIN_CLIENTS_PER_ROUND}-{config_module.MAX_CLIENTS_PER_ROUND}\n"
        f"Local Epochs: {config_module.LOCAL_EPOCHS}\n"
        f"Learning Rate: {config_module.LEARNING_RATE}\n"
        f"Batch Size: {config_module.BATCH_SIZE}\n"
        f"Byzantine %: {config_module.FRACTION_BYZANTINE * 100:.0f}%"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)

    # --- === 1. ACCURACY PLOT === ---
    
    fig_acc, ax_acc = plt.subplots(figsize=(12, 8)) # Single plot
    
    plot_title_acc = (
        f"Byzantine-Resilient FL Comparison (Accuracy over Time)\n"
        f"({config.DATASET_NAME} - {config.DATA_SPLIT_TYPE})"
    )
    ax_acc.set_title(plot_title_acc, fontsize=16, pad=20)
    
    ax_acc.set_ylabel('Global Model Accuracy (%)', fontsize=12)
    ax_acc.set_xlabel('Communication Round', fontsize=12)
    ax_acc.grid(True, linestyle='--', alpha=0.6)

    for result in all_results:
        plot_label = f"{result['label']} ({result['duration']:.1f}s)"
        
        # Determine x-axis based on actual length of history
        # Because Early Stopping might have stopped it early
        num_points = len(result['history'])
        actual_rounds = np.arange(
            config_module.EVALUATE_EVERY_N_ROUNDS, 
            (num_points * config_module.EVALUATE_EVERY_N_ROUNDS) + 1, 
            config_module.EVALUATE_EVERY_N_ROUNDS
        )

        ax_acc.plot(
            actual_rounds,
            result['history'], # 'history' is the accuracy history
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle='solid',
            linewidth=2.5,
        )
    
    ax_acc.legend(loc='lower right', fontsize=10)
    _add_attack_shading(ax_acc, all_results, config_module)
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
        ax_acc.legend(handles=handles, loc='lower right', fontsize=10)


    ax_acc.text(
        0.02, 0.98, param_text, 
        transform=ax_acc.transAxes, 
        fontsize=9,
        verticalalignment='top', 
        bbox=props
    )
    
    # --- === 2. LOSS PLOT === ---
    
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8)) # Single plot
    
    plot_title_loss = (
        f"Byzantine-Resilient FL Comparison (Loss over Time)\n"
        f"({config.DATASET_NAME} - {config.DATA_SPLIT_TYPE})"
    )
    ax_loss.set_title(plot_title_loss, fontsize=16, pad=20)

    ax_loss.set_ylabel('Global Model Loss', fontsize=12)
    ax_loss.set_xlabel('Communication Round', fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.set_yscale('log') # Use a log scale for loss

    for result in all_results:
        plot_label = f"{result['label']} ({result['duration']:.1f}s)"

        # Determine x-axis based on actual length of history
        num_points = len(result['loss_history'])
        actual_rounds = np.arange(
            config_module.EVALUATE_EVERY_N_ROUNDS, 
            (num_points * config_module.EVALUATE_EVERY_N_ROUNDS) + 1, 
            config_module.EVALUATE_EVERY_N_ROUNDS
        )

        ax_loss.plot(
            actual_rounds,
            result['loss_history'], # This is the   loss history
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle='solid',
            linewidth=2.5,
        )
    
    ax_loss.legend(loc='upper right', fontsize=10)
    _add_attack_shading(ax_loss, all_results, config_module)
    _add_participant_bars(ax_loss, all_results)
    ax_loss.text(
        0.02, 0.98, param_text, 
        transform=ax_loss.transAxes, 
        fontsize=9,
        verticalalignment='top', 
        bbox=props
    )

    # --- === 3. SAVE AND SHOW === ---
    
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)
    
    base_filename = f"{config_module.DATASET_NAME}_{config_module.DATA_SPLIT_TYPE}_R{config_module.NUM_ROUNDS}"
    save_path_acc = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_accuracy_line.png")
    save_path_loss = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_loss_line.png")
    
    fig_acc.tight_layout(pad=3.0)
    fig_acc.savefig(save_path_acc, dpi=300)
    
    fig_loss.tight_layout(pad=3.0)
    fig_loss.savefig(save_path_loss, dpi=300)
    
    print(f"  > Line plot (Accuracy) saved to {save_path_acc}")
    print(f"  > Line plot (Loss) saved to {save_path_loss}")
    
    plt.show() # Show both line plots
    
    return save_path_acc, save_path_loss

# --- === 4. BAR CHART FUNCTION === ---

def plot_final_summary_bars(all_results, config_module):
    """
    Generates and saves TWO SEPARATE bar chart files for the
    FINAL results of all experiments.
    """
    print(f"\n[Plotter] Generating 2 final summary bar charts...")
    
    # Extract data for plotting
    labels = [f"{r['label']}\n({r['duration']:.1f}s)" for r in all_results]
    colors = [r['color'] for r in all_results]
    
    final_accuracies = [r['history'][-1] for r in all_results]
    final_losses = [r['loss_history'][-1] for r in all_results]
    
    # --- === 1. FINAL ACCURACY BAR CHART === ---
    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    
    x_ticks = np.arange(len(labels))
    ax_acc.bar(x_ticks, final_accuracies, color=colors)
    ax_acc.set_xticks(x_ticks)
    
    plot_title_bar_acc = (
        f"Final Model Accuracy\n"
        f"({config.DATASET_NAME} - {config.DATA_SPLIT_TYPE}, {config.NUM_ROUNDS} Rounds)"
    )
    ax_acc.set_title(plot_title_bar_acc, fontsize=16, pad=20)
    
    ax_acc.set_ylabel('Final Accuracy (%)', fontsize=12)
    ax_acc.set_ylim(bottom=0, top=max(final_accuracies) * 1.15)
    
    ax_acc.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    
    for i, acc in enumerate(final_accuracies):
        ax_acc.text(i, acc + 0.5, f"{acc:.2f}%", ha='center', fontweight='bold')
        
    # --- === 2. FINAL LOSS BAR CHART === ---
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    
    ax_loss.bar(x_ticks, final_losses, color=colors)
    ax_loss.set_xticks(x_ticks)
    
    plot_title_bar_loss = (
        f"Final Model Loss\n"
        f"({config.DATASET_NAME} - {config.DATA_SPLIT_TYPE}, {config.NUM_ROUNDS} Rounds)"
    )
    ax_loss.set_title(plot_title_bar_loss, fontsize=16, pad=20)
    
    ax_loss.set_ylabel('Final Loss (Log Scale)', fontsize=12)
    ax_loss.set_yscale('log') # Use a log scale for loss
    
    ax_loss.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    
    for i, loss in enumerate(final_losses):
        ax_loss.text(i, loss * 1.1, f"{loss:.4f}", ha='center', fontweight='bold')

    # --- === 3. SAVE AND SHOW === ---
    os.makedirs(config_module.RESULTS_DIR, exist_ok=True)
    
    base_filename = f"{config_module.DATASET_NAME}_{config_module.DATA_SPLIT_TYPE}_R{config_module.NUM_ROUNDS}"
    save_path_acc_bar = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_final_accuracy_bar.png")
    save_path_loss_bar = os.path.join(config_module.RESULTS_DIR, f"{base_filename}_final_loss_bar.png")
    
    fig_acc.tight_layout(pad=2.0)
    fig_acc.savefig(save_path_acc_bar, dpi=300)
    
    fig_loss.tight_layout(pad=2.0)
    fig_loss.savefig(save_path_loss_bar, dpi=300)
    
    print(f"  > Bar chart (Accuracy) saved to {save_path_acc_bar}")
    print(f"  > Bar chart (Loss) saved to {save_path_loss_bar}")
    
    plt.show() # Show both bar charts
    
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
    ax.legend(handles=legend_elements, loc='upper right')

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