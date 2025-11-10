import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions_vs_targets(
    predictions,
    targets,
    output_path,
    output_filename="preds_vs_targets.pdf",
):
    """
    Plot predictions vs. targets as a scatter plot with marginal distributions,
    add a y = x line, and save the figure.

    Args:
        predictions: Predictions array or list
        targets: True target values array or list
        output_path: Directory to save the plot
        output_filename: Filename for the saved plot
    """

    # Convert to numpy arrays if needed
    predictions = np.array(predictions)
    targets = np.array(targets)

    if len(predictions) == 0 or len(targets) == 0:
        print("Warning: Empty predictions or targets arrays")
        return None

    # Create figure with custom layout for marginal distributions
    fig = plt.figure(figsize=(8, 8))
    
    # Define the layout: main plot (center), small marginal histograms (top and right)
    # Make the scatter plot dominant with thin marginal strips
    left, width = 0.12, 0.7
    bottom, height = 0.12, 0.7
    spacing = 0.01
    margin_size = 0.12  # Much smaller margins for distributions
    
    # Define rectangles for the plots
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, margin_size]
    rect_histy = [left + width + spacing, bottom, margin_size, height]
    
    # Create the axes
    ax_scatter = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx)
    ax_histy = fig.add_axes(rect_histy)
    
    # Main scatter plot (this is now the dominant feature)
    ax_scatter.scatter(targets, predictions, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    # Get min and max values for the identity line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    
    # Add identity line
    ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.8, label="Perfect prediction")
    
    # Calculate and display R² on the plot using proper coefficient of determination
    # R² = 1 - (SS_res / SS_tot) where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - y_mean)²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    ax_scatter.text(0.95, 0.05, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes, 
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    horizontalalignment='right', verticalalignment='bottom')
    
    # Labels and title for main plot
    ax_scatter.set_xlabel("True Values", fontsize=12)
    ax_scatter.set_ylabel("Predictions", fontsize=12)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(fontsize=9)
    
    # Small marginal histogram for targets (top) - much more subtle
    ax_histx.hist(targets, bins=30, alpha=0.6, color='steelblue', edgecolor='none')
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histx.tick_params(axis='x', labelbottom=False, labelsize=8)
    ax_histx.tick_params(axis='y', labelsize=8)
    ax_histx.set_ylabel('Count', fontsize=9)
    
    # Small marginal histogram for predictions (right) - much more subtle
    ax_histy.hist(predictions, bins=30, orientation='horizontal', alpha=0.6, 
                    color='lightcoral', edgecolor='none')
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.tick_params(axis='y', labelleft=False, labelsize=8)
    ax_histy.tick_params(axis='x', labelsize=8)
    ax_histy.set_xlabel('Count', fontsize=9)

    # Add subtle mean lines to the marginal plots
    target_mean = np.mean(targets)
    pred_mean = np.mean(predictions)
    ax_histx.axvline(target_mean, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_histy.axhline(pred_mean, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)

    os.makedirs(output_path, exist_ok=True)
    save_file = os.path.join(output_path, output_filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved predictions vs targets with distributions to {save_file}")
    plt.close()
    return save_file
