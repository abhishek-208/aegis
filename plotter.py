"""
Utility for plotting the results of the FL experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import os 
import config as config # Import config to use its values

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
        
        ax_acc.plot(
            eval_rounds,
            result['history'], # 'history' is the accuracy history
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle='solid',
            linewidth=2.5,
        )
    
    ax_acc.legend(loc='lower right', fontsize=10)
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

        ax_loss.plot(
            eval_rounds,
            result['loss_history'], # This is the   loss history
            label=plot_label,
            color=result['color'],
            marker=None,
            linestyle='solid',
            linewidth=2.5,
        )
    
    ax_loss.legend(loc='upper right', fontsize=10)
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