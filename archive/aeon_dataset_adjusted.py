# aeon_dataset_adjusted.py

from aeon.datasets import load_classification
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import numpy as np

# ANSI color codes
BLACK = '\e[0;30m'	
RED = '\e[0;31m'	
GREEN = '\e[0;32m'	
YELLOW  = '\e[0;33m'	
BLUE  = '\e[0;34m'	
PURPLE  = '\e[0;35m'	
CYAN  = '\e[0;36m'	
WHITE = '\e[0;37m'
RESET = '\e[0;0m'

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

class AeonDataset(Dataset):
    def __init__(self, features, labels, seq_len, num_classes):
        super().__init__()
        
        # Ensure labels are integer-encoded
        if not isinstance(labels, (list, np.ndarray, torch.Tensor)):
            raise ValueError("Labels must be a list, numpy array, or torch Tensor.")
        labels = torch.tensor(labels)
        if labels.ndim > 1:
            raise ValueError("Labels must be 1D (integer-encoded) before one-hot encoding.")


        self.features = features
        self.labels = F.one_hot(torch.tensor(labels), num_classes=num_classes).float()
        
        # extracting input parameters for the models timemil and todynet
        self.seq_len = max(seq_len, features[0].shape[1])
        self.num_classes = num_classes
        self.feat_in = features[0].shape[0]

        print(f"\t\t{YELLOW_b}[->]{RESET_b} Creating AeonDataset with feature set shape of {BLUE_b}{features[0].shape}{RESET_b} and label set shape of {BLUE_b}{self.labels.shape}{RESET_b} .")
        print(f"\t\tFeatures shape: {self.features[0].shape}, Number of features (channels): {self.feat_in}")


        # Compute class counts and weights using integer labels
        self.class_counts = np.bincount(labels, minlength=self.num_classes)
        self.class_weights = 1.0 / self.class_counts
        self.class_weights = self.class_weights / np.sum(self.class_weights) * self.num_classes  # Normalize weights
        print(f"\t\t{YELLOW_b}[->]{RESET_b} Retrieved class weigths: {BLUE_b}{self.class_weights}{RESET_b}.")


    def __getitem__(self, idx):
        feats = torch.tensor(self.features[idx]).permute(1, 0).float()  # L x d -> d x L
        feats = F.pad(feats, pad=(0, 0, max(0, self.seq_len - feats.shape[0]), 0))  # Pad to seq_len
        label = self.labels[idx]
        # print("Shape of features at the end of __get__:", feats.shape)
        return feats, label

    def __len__(self):
        return len(self.labels)


def get_aeon_datasets(dataset_name, val_size=0.2, random_state=42):
    """
    Load the dataset and split into train, validation, and test datasets.
    Args:
        dataset_name: Name of the dataset to load from Aeon.
        val_size: Fraction of the training data to be used as validation set.
        random_state: Random seed for reproducibility.
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load train and test splits from aeon library wrapper using load_classification
    print(f"{YELLOW_b}[->]{RESET_b} loading train data from {BLUE_b}{dataset_name}{RESET_b}.")
    X_train, y_train, meta = load_classification(name=dataset_name, split='train')
    print(f"{YELLOW_b}[->]{RESET_b} loading test data from {BLUE_b}{dataset_name}{RESET_b}.")
    X_test, y_test, _ = load_classification(name=dataset_name, split='test')

    # Determine sequence length and number of classes
    print(f"{YELLOW_b}[->]{RESET_b} permuting X_test from format {BLUE_b}(N, L, d){RESET_b} -> into format (N, d, L) to retrieve correct seq_len.")
    X_test_tensor = torch.from_numpy(X_test).permute(0, 2, 1).float()  # (N, L, d) -> (N, d, L)    # Ensure minimum sequence length is met
    print(f"{YELLOW_b}[->]{RESET_b} permuted X_test from shape {BLUE_b}{torch.from_numpy(X_test).shape}{RESET_b} to new shape {BLUE_b}{X_test_tensor.shape}{RESET_b} to retrieve seq_len parameter.")

    seq_len = max(21, X_test_tensor.shape[-1])  # Use a minimum length of 21
    if seq_len == 21:
        print(f"\t\t\t{X_test_tensor.shape[-1]} is less than 21, using minimum sequence length of 21.")
    else:
        print(f"\t\t\t{X_test_tensor.shape[-1]} is greater than 21, using sequence length of {seq_len}.")

    num_classes = len(meta['class_values'])
    print(f"{YELLOW_b}[->]{RESET_b} Extracted num_classes {BLUE_b}{num_classes}{RESET_b}.")


    # Encode labels
    print(f"{YELLOW_b}[->]{RESET_b} Encoding labels to one-hot encoded format.")
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(meta['class_values'])}
    y_train_encoded = [class_to_idx[label] for label in y_train]
    y_test_encoded = [class_to_idx[label] for label in y_test]

    # Check for class consistency
    if len(set(y_train_encoded)) != num_classes:
        raise ValueError("Mismatch in number of classes in training data.")
    if len(set(y_test_encoded)) != num_classes:
        raise ValueError("Mismatch in number of classes in test data.")

    # Perform train-validation split
    print(f"{YELLOW_b}[->]{RESET_b} Performing train-val split with validation size of {BLUE_b}{val_size}{RESET_b} on training set.")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_encoded, test_size=val_size, stratify=y_train_encoded
    ) # random_state=random_state, 

    # Validate splits
    if len(set(y_train_split)) != num_classes or len(set(y_val)) != num_classes:
        raise ValueError("Mismatch in number of classes between train and validation sets.")

    # Create datasets
    print(f"{YELLOW_b}[->]{RESET_b} Creating AeonDataset {BLUE_b}{val_size}{RESET_b} for train-, validation- and testset.")
    print(f"\t{YELLOW_b}[->]{RESET_b} Creating AeonDataset {BLUE_b}trainset{RESET_b}.")
    train_dataset = AeonDataset(X_train_split, y_train_split, seq_len, num_classes)
    print(f"\t{YELLOW_b}[->]{RESET_b} Creating AeonDataset {BLUE_b}validationset{RESET_b}.")
    val_dataset = AeonDataset(X_val, y_val, seq_len, num_classes)
    print(f"\t{YELLOW_b}[->]{RESET_b} Creating AeonDataset {BLUE_b}testset{RESET_b}.")    
    test_dataset = AeonDataset(X_test, y_test_encoded, seq_len, num_classes)

    return train_dataset, val_dataset, test_dataset
