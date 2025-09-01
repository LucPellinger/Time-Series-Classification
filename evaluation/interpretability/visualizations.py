# visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import captum

# ANSI color codes
BLACK = '\033[0;30m'	
RED = '\033[0;31m'	
GREEN = '\033[0;32m'	
YELLOW  = '\033[0;33m'	
BLUE  = '\033[0;34m'	
PURPLE  = '\033[0;35m'	
CYAN  = '\033[0;36m'	
WHITE = '\033[0;37m'
RESET = '\033[0;0m'

BLACK_b = '\033[1;30m'	
RED_b = '\033[1;31m'	
GREEN_b = '\033[1;32m'	
YELLOW_b  = '\033[1;33m'	
BLUE_b  = '\033[1;34m'	
PURPLE_b  = '\033[1;35m'	
CYAN_b  = '\033[1;36m'	
WHITE_b = '\033[1;37m'
RESET_b = '\033[0;0m'

BLACK_ = '\e[4;30m'	
RED_ = '\e[4;31m'	
GREEN_ = '\e[4;32m'	
YELLOW_  = '\e[4;33m'	
BLUE_  = '\e[4;34m'	
PURPLE_  = '\e[4;35m'	
CYAN_  = '\e[4;36m'	
WHITE_ = '\e[4;37m'
RESET_ = '\e[0;0m'

def visualize_attributions(attr, inputs, targets, save_path=None, num_samples=1, channels_last=True, method_used="IntegratedGradients", test_feature_mapping=None):
    """
    Visualize time series attributions using Captum's visualize_timeseries_attr.

    Parameters:
        attr (numpy.ndarray): Attributions array, shape (batch_size, seq_length, input_dim).
        inputs (numpy.ndarray): Original input data, shape (batch_size, seq_length, input_dim).
        targets (list or numpy.ndarray): Target labels for each input in the batch.
        save_path (str, optional): Path to save the visualization. If None, visualization is shown interactively.
    """
    batch_size, seq_length, input_dim = inputs.shape
    for i in range(batch_size):
        # Create visualization for each instance in the batch
        if i != 1:# >= num_samples:
            pass
        else:
            instance_attr = attr[i]
            instance_input = inputs[i]
            target_label = targets[i]
            
            # # Map feature indices to names using the feature_mapping dictionary
            # channel_labels = [test_feature_mapping.get(j, f"Feature {j} ") for j in range(input_dim)]
    
    
            #selected_features = [0, 4, 5, 9]
            selected_features = None
            if selected_features is not None:
                # Subset the features
                instance_attr = instance_attr[:, selected_features]
                instance_input = instance_input[:, selected_features]
                # Adjust input_dim
                input_dim = len(selected_features)
                # Adjust channel_labels
                channel_labels = [
                    test_feature_mapping.get(j, f"Feature {j}") for j in selected_features
                ]
            else:
                # Map feature indices to names using the feature_mapping dictionary
                channel_labels = [
                    test_feature_mapping.get(j, f"Feature {j}") for j in range(input_dim)
                ]
    
            viz_method = "overlay_individual" # overlay_individual, overlay_combined, colored_graph
            viz_figsize = (8, 2 * input_dim) 
    
            if viz_method =="overlay_combined":
                viz_figsize = (15, 15) 
    
            fig, _ = captum.attr.visualization.visualize_timeseries_attr(
                attr=instance_attr,
                data=instance_input,
                method = viz_method, # overlay_individual, overlay_combined, colored_graph
                channels_last=channels_last,
                sign="all",
                show_colorbar=True,
                channel_labels=channel_labels,
                fig_size = viz_figsize,
                title = f"{method_used} Attributions for sample {i} with target label {target_label}",
                alpha_overlay = 0.7
            )
    
            if save_path:
                fig.savefig(f"{save_path}_sample_{i}_{method_used}.png")
                plt.close(fig)  # Close the figure to prevent memory leaks
                print(f"{GREEN}[+]{RESET} saved created {method_used} visual to file @ '{BLUE_b}{save_path}_sample_{i}.png{RESET_b}'.")
    
            else:
                plt.show()

def visualize_attention_weights(
    attention_weights: np.ndarray,
    data: np.ndarray,
    x_values: np.ndarray = None,
    method: str = "colored_graph", # overlay_individual 
    channel_labels: list = None,
    channels_last: bool = True,
    alpha_overlay: float = 0.7,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    title: str = None,
    fig_size: tuple = (12, 8),
    **plot_kwargs
):
    """
    Visualizes attention weights for time-series data.

    Args:
        attention_weights (np.ndarray): Attention weights array of shape (N, C) with
                                         channels last unless `channels_last` is False.
        data (np.ndarray): Original time-series data of shape (N, C) with
                           channels last unless `channels_last` is False.
        x_values (np.ndarray, optional): X-axis values of shape (N,). Defaults to range(N).
        method (str, optional): Visualization method. One of "overlay_individual",
                                "overlay_combined", or "colored_graph". Defaults to "overlay_individual".
        channel_labels (list, optional): Labels for the channels. Defaults to None.
        channels_last (bool, optional): If True, assumes channels are the last dimension. Defaults to True.
        alpha_overlay (float, optional): Opacity for overlay. Defaults to 0.7.
        cmap (str, optional): Colormap for heatmap. Defaults to "viridis".
        show_colorbar (bool, optional): Whether to show colorbar. Defaults to True.
        title (str, optional): Title for the plot. Defaults to None.
        fig_size (tuple, optional): Size of the figure. Defaults to (12, 8).
        plot_kwargs (dict, optional): Additional kwargs for `plt.plot`.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if channels_last:
        attention_weights = np.transpose(attention_weights)  # (C, N)
        data = np.transpose(data)  # (C, N)

    num_channels, timeseries_length = data.shape

    # Default x-axis values
    if x_values is None:
        x_values = np.arange(timeseries_length)

    # Normalize attention weights
    norm = Normalize(vmin=attention_weights.min(), vmax=attention_weights.max())
    cmap = get_cmap(cmap)

    # Set up the plot
    num_subplots = 1 if method == "overlay_combined" else num_channels
    fig, axes = plt.subplots(
        nrows=num_subplots, figsize=fig_size, sharex=True, squeeze=False
    )
    axes = axes.flatten()

    # Visualization methods
    if method == "overlay_individual":
        for chan in range(num_channels):
            ax = axes[chan]
            ax.plot(x_values, data[chan], label=f"Channel {chan + 1}", **plot_kwargs)
            ax.fill_between(
                x_values,
                data[chan],
                data[chan].min(),
                where=(attention_weights[chan] > 0),
                color=cmap(norm(attention_weights[chan])),
                alpha=alpha_overlay,
            )
            if channel_labels:
                ax.set_ylabel(channel_labels[chan])
            ax.legend()

    elif method == "overlay_combined":
        combined_weights = np.mean(attention_weights, axis=0)
        ax = axes[0]
        for chan in range(num_channels):
            ax.plot(x_values, data[chan], label=f"Channel {chan + 1}", **plot_kwargs)
        ax.fill_between(
            x_values,
            data.mean(axis=0),
            data.min(),
            where=(combined_weights > 0),
            color=cmap(norm(combined_weights)),
            alpha=alpha_overlay,
        )
        if channel_labels:
            ax.legend(channel_labels)

    elif method == "colored_graph":
        for chan in range(num_channels):
            ax = axes[chan]
            points = np.array([x_values, data[chan]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                array=attention_weights[chan],
                linewidth=2,
            )
            ax.add_collection(lc)
            ax.set_xlim(x_values.min(), x_values.max())
            ax.set_ylim(data[chan].min(), data[chan].max())
            if channel_labels:
                ax.set_ylabel(channel_labels[chan])

    # Title and colorbar
    if title:
        fig.suptitle(title, fontsize=16)
    if show_colorbar:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal")

    plt.tight_layout()
    plt.show()
    return fig, axes
