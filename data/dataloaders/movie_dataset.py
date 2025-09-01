# movie_dataset_v2.py
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F


class MovieTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, split, mean=None, std=None, max_seq_len=None):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.data, self.feature_names = self.load_data()
        self.labels = [label for _, label in self.data]

        # Compute feature properties
        self.num_classes = len(set(self.labels))
        self.feat_in = self.data[0][0].shape[1]
        # Compute class counts and weights
        self.class_counts = np.bincount(self.labels, minlength=self.num_classes)
        self.class_weights = 1.0 / self.class_counts
        self.class_weights = self.class_weights / np.sum(self.class_weights) * self.num_classes  # Normalize weights

        # Store mean and std for standardization
        self.mean = mean
        self.std = std

    def load_data(self):
        file_path = os.path.join(self.data_dir, f'{self.split}.parquet')
        df = pd.read_parquet(file_path)

        features_to_exclude = ['Success_Label', 'from_imdb_imdb_id', 'daily_box_office_date_revenue_recorded']

        if "imdb" in file_path:
            features_to_exclude = ['Success_Label', 'from_imdb_imdb_id', 'daily_box_office_date_revenue_recorded', 
                              # potentially include again
                              'daily_box_office_%± YD gross change daily',
                              'daily_box_office_%± LW gross change weekly',
                              #"from_imdb_daily_number_of_ratings",
                              'days_since_first_rating',
                              #'daily_box_office_per theaters avg gross',
                              # until here
                              'daily_box_office_TotalBoxOffice_lifespan', 'days_after_release',  
                              'days_between_first_and_last_rating', 
                              'rolling_mean_trend_7_days_from_imdb_daily_mean_rating', 
                              'slope_of_from_imdb_daily_mean_rating', 
                              "mean_diff_from_imdb_daily_mean_rating", 'lagged_corr_1_from_imdb_daily_mean_rating',
                              'autocorr_lag_1_from_imdb_daily_mean_rating', 'time_to_peak_from_imdb_daily_mean_rating',
                              'time_to_trough_from_imdb_daily_mean_rating', 'longest_decrease_streak_from_imdb_daily_mean_rating',
                              'longest_increase_streak_from_imdb_daily_mean_rating'
                              ]
        elif "rotten" in file_path:
            features_to_exclude = ['Success_Label', 'from_imdb_imdb_id', 'daily_box_office_date_revenue_recorded', 
            # potentially include again
            # "rotten_review_score_encoded_max",
            # "rotten_review_score_encoded_min",
            'daily_box_office_%± YD gross change daily',
            'daily_box_office_%± LW gross change weekly',
            # "from_imdb_daily_number_of_ratings",
            'days_since_first_rating',
            # 'daily_box_office_per theaters avg gross',
            # until here
            # sentiment here
            # 'rotten_rating_sentiment_polarity_mean',
            # 'rotten_rating_sentiment_polarity_var',
            # 'rotten_rating_sentiment_polarity_min',
            # 'rotten_rating_sentiment_polarity_max',
            # 'rotten_review_word_count_mean'
            # 'rotten_review_score_encoded_mean',
            # 'rotten_review_score_encoded_var',
            # 'rotten_review_score_encoded_min',
            # 'rotten_review_score_encoded_max',
            # until here
            #'daily_box_office_TotalBoxOffice_lifespan',
            'days_between_first_and_last_rating',
            'rolling_mean_trend_7_days_rotten_rating_sentiment_polarity_mean', #'days_after_release', 
            'slope_of_rotten_rating_sentiment_polarity_mean', 
            'mean_diff_rotten_rating_sentiment_polarity_mean', 
            'lagged_corr_1_rotten_rating_sentiment_polarity_mean', 
            'autocorr_lag_1_rotten_rating_sentiment_polarity_mean', 
            'time_to_peak_rotten_rating_sentiment_polarity_mean',
            'time_to_trough_rotten_rating_sentiment_polarity_mean',
            'longest_increase_streak_rotten_rating_sentiment_polarity_mean',
            'longest_decrease_streak_rotten_rating_sentiment_polarity_mean']
        elif "twitter" in file_path:
            features_to_exclude = ['Success_Label', 'from_imdb_imdb_id', 'daily_box_office_date_revenue_recorded', 
            # potentially include again
            'daily_box_office_%± YD gross change daily',
            'daily_box_office_%± LW gross change weekly',
            # 'daily_box_office_per theaters avg gross',
            # "from_imdb_daily_number_of_ratings",
            'days_since_first_rating', 
            # include till here
            'days_after_release', 
            'days_between_first_and_last_rating', 'daily_box_office_TotalBoxOffice_lifespan', 
            'rolling_mean_trend_7_days_twitter_rating_mean', 
            'slope_of_twitter_rating_mean',  
            'mean_diff_twitter_rating_mean', 
            'lagged_corr_1_twitter_rating_mean', 
            'autocorr_lag_1_twitter_rating_mean', 
            'time_to_peak_twitter_rating_mean', 
            'time_to_trough_twitter_rating_mean', 
            'longest_increase_streak_twitter_rating_mean', 
            'longest_decrease_streak_twitter_rating_mean']

        print(f"\t\t\tNumber of unique movies in {self.split}-set: \t{len(df['from_imdb_imdb_id'].unique())}")
        print(f"\t\t\tNumber of samples in {self.split}-set: \t{len(df['from_imdb_imdb_id'])}")


        # Extract feature names
        feature_names = df.drop(
            features_to_exclude,# ['Success_Label', 'from_imdb_imdb_id', 'daily_box_office_date_revenue_recorded'], 
            axis=1
        ).columns.tolist()



        grouped = df.groupby('from_imdb_imdb_id')
        data = []
        for movie_id, group in grouped:
            group = group.sort_values('daily_box_office_date_revenue_recorded')
            features = group.drop(features_to_exclude, axis=1).values
            label = group['Success_Label'].iloc[0]
            data.append((features, label))
        return data, feature_names

    @property
    def feature_mapping(self):
        """
        Returns a dictionary mapping feature indices to feature names.
        """
        # rename_mapping = {
        #     "rotten_review_score_encoded_mean": 'mean_rating',
        #     "rotten_review_score_encoded_var": 'var_rating',
        #     "rotten_review_score_encoded_min": 'min_rating',
        #     "rotten_review_score_encoded_max": 'max_rating',
        #     "rotten_review_word_count_mean": 'mean_num_words',
        #     "rotten_rating_sentiment_polarity_mean": 'mean_sentiment',
        #     "rotten_rating_sentiment_polarity_var": 'var_sentiment',
        #     "rotten_rating_sentiment_polarity_min": 'min_sentiment',
        #     "rotten_rating_sentiment_polarity_max": 'max_sentiment',
        #     "daily_box_office_DailyBoxOfficeRevenue": 'daily_box_office',
        #     "daily_box_office_per theaters avg gross": 'avg_theater_gross',
        #     "daily_box_office_Theaters_cleaned": 'num_theaters'
        # }
        # # Rename feature names if they are in the mapping
        # renamed_features = [
        #     rename_mapping.get(name, name) for name in self.feature_names
        # ]
        
        # self.feature_names
        return {i: name for i, name in enumerate(self.feature_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        features = torch.tensor(features, dtype=torch.float)

        # Apply truncation or padding if max_seq_len is set
        if self.max_seq_len:
            seq_len = features.shape[0]
            if seq_len > self.max_seq_len:
                features = features[:self.max_seq_len]
            elif seq_len < self.max_seq_len:
                pad_size = self.max_seq_len - seq_len
                features = F.pad(features, (0, 0, 0, pad_size))  # Pad to max_seq_len

        # Apply standardization if mean and std are provided
        if self.mean is not None and self.std is not None:
            # Avoid division by zero
            std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
            features = (features - self.mean) / std

        # Permute features to [num_features, seq_length]
        #features = features.permute(1, 0).float()  # L x d -> d x L
        label = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).float()

        # print("Shape of features at the end of __get__:", features.shape)

        return features, label


def get_max_sequence_length(data_dir):
    """
    Determines the maximum sequence length across train, validation, and test datasets.
    Args:
        data_dir (str): Directory containing train, validation, and test datasets.
    Returns:
        int: Maximum sequence length.
    """
    max_length = 0
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.parquet')
        df = pd.read_parquet(file_path)

        grouped = df.groupby('from_imdb_imdb_id')
        for _, group in grouped:
            seq_len = len(group)
            max_length = max(max_length, seq_len)
    return max_length


def get_movie_datasets(data_dir, max_seq_len=None):
    """
    Prepares train, validation, and test datasets with padding/truncation to max_seq_len.
    Args:
        data_dir (str): Directory containing train, validation, and test datasets.
        max_seq_len (int, optional): Maximum sequence length for padding/truncation.
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Determine max_seq_len dynamically if not provided
    if max_seq_len is None:
        max_seq_len = get_max_sequence_length(data_dir)
    print(f"Using max_seq_len={max_seq_len}")

    # Create datasets with determined max_seq_len
    train_dataset = MovieTimeSeriesDataset(data_dir, 'train', max_seq_len=max_seq_len)
    val_dataset = MovieTimeSeriesDataset(data_dir, 'validation', max_seq_len=max_seq_len)
    test_dataset = MovieTimeSeriesDataset(data_dir, 'test', max_seq_len=max_seq_len)

    # Compute mean and std on training data
    mean, std = compute_mean_std(train_dataset)
    mean = torch.tensor(mean, dtype=torch.float)
    std = torch.tensor(std, dtype=torch.float)

    # Update datasets with mean and std
    train_dataset.mean = mean
    train_dataset.std = std
    val_dataset.mean = mean
    val_dataset.std = std
    test_dataset.mean = mean
    test_dataset.std = std

    out_seq_len = max(train_dataset.max_seq_len, val_dataset.max_seq_len, test_dataset.max_seq_len)
    out_num_classes = max(train_dataset.num_classes, val_dataset.num_classes, test_dataset.num_classes)
    out_feat_in = max(train_dataset.feat_in, val_dataset.feat_in, test_dataset.feat_in)

    return train_dataset, val_dataset, test_dataset, out_seq_len, out_num_classes, out_feat_in


def compute_mean_std(train_dataset):
    """
    Computes the mean and standard deviation per feature over the entire training dataset.
    Args:
        train_dataset (Dataset): The training dataset.
    Returns:
        mean (ndarray): Mean per feature.
        std (ndarray): Standard deviation per feature.
    """
    n_samples = 0
    feature_sum = None
    feature_sq_sum = None

    for features, _ in train_dataset:
        features = features.permute(1, 0).float()  # L x d -> d x L
        n = features.numel()
        n_samples += n

        if feature_sum is None:
            feature_sum = features.sum(dim=1)
            feature_sq_sum = (features ** 2).sum(dim=1)
        else:
            feature_sum += features.sum(dim=1)
            feature_sq_sum += (features ** 2).sum(dim=1)

    mean = feature_sum / n_samples
    variance = (feature_sq_sum / n_samples) - (mean ** 2)
    std = torch.sqrt(variance)

    return mean.numpy(), std.numpy()



def collate_fn(batch):
    features, labels = zip(*batch)
    # Get sequence lengths
    lengths = torch.tensor([f.shape[1] for f in features], dtype=torch.long)
    max_length = lengths.max().item()

    # Pad features to the max_length in the batch
    #padded_features = torch.stack([
    #    F.pad(f, (0, 0, 0, max_length - f.shape[0])) for f in features
    #])
    
    # Handle one-hot encoded or integer labels
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)  # Already tensors, stack directly
    else:
        labels = torch.tensor(labels, dtype=torch.long)  # Convert integer labels to tensor

    features = torch.stack(features)  # Shape: [batch_size, seq_length, num_features]

    # Combine one-hot encoded labels into a single tensor
    #labels = torch.stack(labels)  # Shape: [batch_size, num_classes]
    lengths = torch.tensor(lengths, dtype=torch.long)
    return features, labels, lengths  
