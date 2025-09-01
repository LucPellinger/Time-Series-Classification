# time_series_classification/data/__init__.py
"""Data handling module for time series classification."""

from time_series_classification.data.dataloaders.aeon_dataset import (
    AeonDataset,
    get_aeon_datasets
)
from time_series_classification.data.dataloaders.movie_dataset import (
    MovieTimeSeriesDataset,
    get_movie_datasets,
    collate_fn
)

__all__ = [
    "AeonDataset",
    "get_aeon_datasets",
    "MovieTimeSeriesDataset",
    "get_movie_datasets",
    "collate_fn",
]