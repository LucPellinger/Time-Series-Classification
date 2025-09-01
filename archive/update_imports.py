# update_imports.py
"""
Script to update import statements after file reorganization.
"""

import re
from pathlib import Path
import os
from typing import Dict, List

# Define import mappings
IMPORT_MAPPINGS = {
    # Old import -> New import
    'from data.dataloaders.aeon_dataset import': 'from data.dataloaders.aeon_dataset import',
    'from data.dataloaders.movie_dataset import': 'from data.dataloaders.movie_dataset import',
    'from models.base_model import': 'from models.base_model import',
    'from models.optimizers.lookahead import': 'from models.optimizers.lookahead import',
    'from training.trainer import': 'from training.trainer import',
    'from training.optimizer import': 'from training.optimizer import',
    'from training.callbacks.metrics_logger import': 'from training.callbacks.metrics_logger import',
    'from evaluation.interpretability.visualizations import': 'from evaluation.interpretability.visualizations import',
    'from evaluation.metrics.metric_utils import': 'from evaluation.metrics.metric_utils import',
    'import evaluation.metrics.metric_utils as metric_utils': 'import evaluation.metrics.metric_utils as metric_utils',
}

def update_imports_in_file(file_path: Path):
    """Update import statements in a single file."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update each import mapping
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(re.escape(old_import), new_import, content)
    
    # Only write if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in: {file_path}")
        return True
    return False

def update_all_imports(project_root: str):
    """Update imports in all Python files in the project."""
    
    root_path = Path(project_root)
    updated_count = 0
    
    # Find all Python files
    for py_file in root_path.rglob("*.py"):
        if update_imports_in_file(py_file):
            updated_count += 1
    
    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    PROJECT_ROOT = os.getcwd()
    update_all_imports(PROJECT_ROOT)