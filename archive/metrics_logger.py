# metrics_logger.py

import pytorch_lightning as pl
import csv
import os

class MetricsLogger(pl.Callback):
    def __init__(self, experiment_dir):
        self.metrics_file = os.path.join(experiment_dir, 'metrics.csv')
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        # Include all metric names in the header
        self.metric_keys = ['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metric_keys)
        self.train_metrics = {}
        self.epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.train_metrics = {
            'train_loss': metrics.get('train_loss'),
            'train_acc': metrics.get('train_acc'),
            'train_f1': metrics.get('train_f1')
            # Add other training metrics as needed
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_metrics = {
            'val_loss': metrics.get('val_loss'),
            'val_acc': metrics.get('val_acc'),
            'val_f1': metrics.get('val_f1')
            # Add other validation metrics as needed
        }

        # Combine metrics
        metric_values = [self.epoch] + [self.train_metrics.get(k) for k in self.metric_keys[1:4]] + [val_metrics.get(k) for k in self.metric_keys[4:]]

        # Write metrics to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric_values)

        self.epoch += 1
