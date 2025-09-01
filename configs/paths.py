# time_series_classification/configs/paths.py
"""Path configurations for the project."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Export paths as strings for compatibility
PROJECT_ROOT_STR = str(PROJECT_ROOT)
DATA_DIR_STR = str(DATA_DIR)
EXPERIMENTS_DIR_STR = str(EXPERIMENTS_DIR)
LOGS_DIR_STR = str(LOGS_DIR)

print(f"Project Root: {PROJECT_ROOT_STR}")