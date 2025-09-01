import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

def compute_global_feature_importance(attr_list, method="mean"):
    """
    Compute global feature importance from a list of attributions.
    
    Parameters:
    - attr_list (list): List of tuples (inputs, attributions, targets) for each batch.
    - method (str): Aggregation method - "mean" or "sum".
    
    Returns:
    - global_importance (np.ndarray): Aggregated global feature importance.
    """
    # Initialize a list to store aggregated importances per batch
    batch_importances = []
    
    for inputs, attributions, _ in attr_list:
        # Aggregate attributions across all time steps
        # Shape of attributions: (batch_size, time_steps, num_features)
        if method == "mean":
            batch_importance = np.mean(np.abs(attributions), axis=1)  # Mean across time steps
        elif method == "sum":
            batch_importance = np.sum(np.abs(attributions), axis=1)   # Sum across time steps
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Sum importance over all samples in the batch
        batch_importances.append(np.sum(batch_importance, axis=0))  # Sum across batch
    
    # Combine importance scores from all batches
    global_importance = np.sum(batch_importances, axis=0)
    
    return global_importance


def visualize_feature_importance(ig_global_importance, method="IntegratedGradients", aggregation="", test_feature_mapping=None, save_path=None):
    # Map feature indices to names using the feature_mapping dictionary
    if test_feature_mapping:
        channel_labels = [test_feature_mapping.get(j, f"Feature {j}") for j in range(len(ig_global_importance))]
    else:
        channel_labels = [f"Feature {j}" for j in range(len(ig_global_importance))]

    # Create a DataFrame with the importance data
    importance_data = {
        "Feature Name": channel_labels,
        "Feature Index": list(range(len(ig_global_importance))),
        "Importance Score": ig_global_importance
    }
    df = pd.DataFrame(importance_data)

    # Sort the DataFrame by 'Importance Score' in descending order to get the ranking
    df = df.sort_values(by='Importance Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1  # Add a 'Rank' column starting from 1

    # Save the DataFrame if save_path is provided
    file_name = f'{method}_feature_importance.parquet'
    if save_path:
        tmp__save_path = os.path.join(save_path, file_name)
        df.to_parquet(tmp__save_path, index=False)
        print(f"Feature importance values saved to {file_name}")

    # Plotting
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Feature Name'], df['Importance Score'], tick_label=df['Feature Name'])

    # Add annotations on top of each bar with importance score and rank
    for bar, rank in zip(bars, df['Rank']):
        height = bar.get_height()
        offset = max(0.005 * height, 0.002)  # Adjust the offset as needed
        plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                 f'{height:.4f}\n(Rank {rank})',
                 ha='center', va='bottom', fontsize=9)

    plt.title(f"Global Feature Importance ({method}) using {aggregation} over all attributions")
    plt.xlabel("Feature Name")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=90, ha='center')  # Rotate labels for better readability
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Show or save the plot
    if save_path:
        plot_file = os.path.join(save_path, f"{method}_feature_importance.png")
        plt.savefig(plot_file)
        print(f"Feature importance plot saved to {plot_file}")
    
    plt.close()
    plt.show()



# def compute_entropy(attributions, method="IntegratedGradients", aggregation="mean"):
#     """
#     Compute entropy for attribution distributions.
    
#     Parameters:
#     - attributions (np.ndarray): Attributions of shape (batch_size, time_steps, num_features).
    
#     Returns:
#     - entropies (list): Entropy scores for each sample in the batch.
#     """
#     # Aggregate attributions across time steps (if applicable)
#     aggregated_attributions = np.sum(np.abs(attributions), axis=1)  # Sum over time steps
    
#     # Normalize attributions to form a probability distribution
#     probabilities = aggregated_attributions / np.sum(aggregated_attributions, axis=1, keepdims=True)
    
#     # Compute entropy
#     entropies = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)  # Add epsilon to avoid log(0)
    
#     return entropies

def compute_entropy(attr_list, method="IntegratedGradients", aggregation="mean"):
    """
    Compute entropy for attribution distributions across all samples.
    
    Parameters:
    - attr_list (list): List of tuples (inputs, attributions, targets).
    
    Returns:
    - entropies (list): Average entropy scores for the entire dataset.
    """
    all_entropies = []  # Store entropy values for all samples

    for inputs, attributions, target in attr_list:
        # Debug: Check the shape of attributions
        print("Shape of attributions for a sample:", attributions.shape)

        # Aggregate attributions across time steps (if applicable)
        if len(attributions.shape) == 3:  # If attributions have time_steps and features
            aggregated_attributions = np.sum(np.abs(attributions), axis=1)  # Sum over time steps
        elif len(attributions.shape) == 2:  # Already aggregated
            aggregated_attributions = attributions
        else:
            raise ValueError(f"Unexpected shape of attributions: {attributions.shape}")
        
        print("Shape of aggregated attributions for a sample:", aggregated_attributions.shape)

        # Normalize attributions to form a probability distribution
        total_attributions = np.sum(aggregated_attributions, axis=1, keepdims=True)
        probabilities = aggregated_attributions / (total_attributions + 1e-8)  # Add epsilon to avoid division by zero

        # Compute entropy for each sample
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)  # Add epsilon to avoid log(0)
        all_entropies.extend(entropy)  # Add to the global list
    
    return all_entropies

def compute_sparsity(attr_list, method="IntegratedGradients", threshold=0.01):
    """
    Compute sparsity for attribution distributions across all samples.
    
    Parameters:
    - attr_list (list): List of tuples (inputs, attributions, targets).
    - method (str): Attribution method name (for debugging/logging purposes).
    - threshold (float): Threshold to consider attributions as non-zero.
    
    Returns:
    - sparsities (list): Sparsity scores for the entire dataset.
    """
    all_sparsities = []  # Store sparsity values for all samples

    for inputs, attributions, target in attr_list:
        # Debug: Check the shape of attributions
        print(f"[{method}] Shape of attributions for a sample:", attributions.shape)

        # Aggregate attributions across time steps (if applicable)
        if len(attributions.shape) == 3:  # If attributions have time_steps and features
            aggregated_attributions = np.sum(np.abs(attributions), axis=1)  # Sum over time steps
        elif len(attributions.shape) == 2:  # Already aggregated
            aggregated_attributions = attributions
        else:
            raise ValueError(f"Unexpected shape of attributions: {attributions.shape}")
        
        print(f"[{method}] Shape of aggregated attributions for a sample:", aggregated_attributions.shape)

        # Compute sparsity
        # Count non-zero attributions based on the threshold
        non_zero_counts = np.sum(aggregated_attributions > threshold, axis=1)
        total_features = aggregated_attributions.shape[1]  # Number of features
        
        # Sparsity is the proportion of zero-valued features
        sparsity = 1 - (non_zero_counts / total_features)
        all_sparsities.extend(sparsity)  # Add to the global list

    return all_sparsities



# def compute_feature_interaction(attr_list, method="IntegratedGradients"):
#     """
#     Compute pairwise correlation of attributions across features for each sample.
    
#     Parameters:
#     - attr_list (list): List of tuples (inputs, attributions, targets).
#     - method (str): Attribution method name (for debugging/logging purposes).
    
#     Returns:
#     - interaction_matrices (list): List of pairwise correlation matrices for all samples.
#     - avg_interaction_matrix (np.ndarray): Average pairwise correlation matrix across all samples.
#     """
#     interaction_matrices = []  # Store interaction matrices for all samples

#     for inputs, attributions, target in attr_list:
#         # Debug: Check the shape of attributions
#         print(f"[{method}] Shape of attributions for a sample:", attributions.shape)

#         # Aggregate attributions across time steps (if applicable)
#         if len(attributions.shape) == 3:  # If attributions have time_steps and features
#             aggregated_attributions = np.sum(np.abs(attributions), axis=1)  # Sum over time steps
#         elif len(attributions.shape) == 2:  # Already aggregated
#             aggregated_attributions = attributions
#         else:
#             raise ValueError(f"Unexpected shape of attributions: {attributions.shape}")
        
#         print(f"[{method}] Shape of aggregated attributions for a sample:", aggregated_attributions.shape)

#         # Compute pairwise correlation matrix for features
#         # Shape of aggregated_attributions: (batch_size, num_features)
#         for sample in aggregated_attributions:  # Iterate over batch
#             # Compute pairwise correlations
#             corr_matrix = np.corrcoef(sample.T)  # Transpose to correlate features
#             interaction_matrices.append(corr_matrix)

#     # Compute the average interaction matrix across all samples
#     avg_interaction_matrix = np.mean(interaction_matrices, axis=0)

#     return interaction_matrices, avg_interaction_matrix

# def visualize_feature_interaction(interaction_matrix, method="IntegratedGradients", title_suffix="", save_path=None):
#     """
#     Visualize feature interaction matrices using a heatmap.
    
#     Parameters:
#     - interaction_matrix (np.ndarray): Pairwise feature interaction matrix (e.g., average or single sample).
#     - method (str): Attribution method name (e.g., "IntegratedGradients" or "DeepLift").
#     - title_suffix (str): Additional text for the plot title.
#     - save_path (str or None): If provided, saves the plot to this path.
    
#     Returns:
#     - None
#     """
#     plt.figure(figsize=(10, 8))
#     plt.imshow(interaction_matrix, cmap="coolwarm", interpolation="nearest")
#     plt.colorbar()
#     plt.title(f"Feature Interaction Matrix ({method}) {title_suffix}")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Feature Index")
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Feature interaction matrix saved to {save_path}")
#     else:
#         plt.show()

def compute_perturb_std(dataset):
    """
    Compute the optimal perturbation standard deviation (perturb_std) based on the dataset.
    Args:
        dataset (Dataset): The dataset to analyze.
    Returns:
        perturb_std (torch.Tensor): Standard deviation of each feature scaled by a fraction.
    """
    # Compute feature-wise mean and std
    n_samples = 0
    feature_sum = None
    feature_sq_sum = None

    for features, _ in dataset:
        features = features.permute(1, 0).float()  # [seq_length, num_features] -> [num_features, seq_length]
        n_samples += features.numel()

        if feature_sum is None:
            feature_sum = features.sum(dim=1)
            feature_sq_sum = (features ** 2).sum(dim=1)
        else:
            feature_sum += features.sum(dim=1)
            feature_sq_sum += (features ** 2).sum(dim=1)

    # Compute mean and std
    mean = feature_sum / n_samples
    variance = (feature_sq_sum / n_samples) - (mean ** 2)
    std = torch.sqrt(variance)

    # Scale standard deviation to compute perturb_std
    perturb_std = 0.1 * std  # Use 10% of the standard deviation as perturbation scale
    return perturb_std