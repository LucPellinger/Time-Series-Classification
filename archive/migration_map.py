# migration_map.py
"""
File migration mapping for reorganizing the project structure.
Run this script to move files to their new locations.
"""

import shutil
import os
from pathlib import Path

# Define the migration mapping
FILE_MIGRATIONS = {
    # Data modules
    'aeon_dataset_adjusted.py': 'data/dataloaders/aeon_dataset.py',
    'movie_dataset_v2.py': 'data/dataloaders/movie_dataset.py',
    
    # Model modules
    'models.py': 'models/base_model.py',
    'lookahead.py': 'models/optimizers/lookahead.py',
    # Note: Individual model architectures (lstm, timemil, todynet) should be 
    # placed in models/architectures/ if you have them
    
    # Training modules
    'train.py': 'training/trainer.py',
    'train_optimization_v2.py': 'training/optimizer.py',
    'metrics_logger.py': 'training/callbacks/metrics_logger.py',
    
    # Evaluation modules
    'visualize.py': 'evaluation/interpretability/visualizations.py',
    'utils.py': 'evaluation/metrics/metric_utils.py',
    
    # Utils
    'logger.py': 'utils/logger.py',
}

def migrate_files(source_dir: str, target_dir: str):
    """
    Migrate files from source directory to new project structure.
    
    Args:
        source_dir: Path to current flat directory with all files
        target_dir: Path to new organized project root
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    for old_file, new_location in FILE_MIGRATIONS.items():
        old_path = source_path / old_file
        new_path = target_path / new_location
        
        if old_path.exists():
            # Create parent directories
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to new location
            shutil.copy2(old_path, new_path)
            print(f"✓ Migrated: {old_file} -> {new_location}")
        else:
            print(f"✗ Not found: {old_file}")
    
    print("\nMigration complete!")

if __name__ == "__main__":
    # Update these paths according to your setup
    SOURCE_DIR = os.getcwd()
    TARGET_DIR = os.getcwd()
    
    migrate_files(SOURCE_DIR, TARGET_DIR)