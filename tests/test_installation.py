# test_installation.py
"""Test script to verify the installation and imports work correctly."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test that all core imports work."""
    
    print("=" * 50)
    print("Testing Core Imports")
    print("=" * 50)
    
    success_count = 0
    failure_count = 0
    
    # Test data imports
    try:
        from time_series_classification.data import AeonDataset, MovieTimeSeriesDataset
        print("✓ Data module imports successful")
        success_count += 1
    except ImportError as e:
        print(f"✗ Data module import failed: {e}")
        failure_count += 1
    
    # Test model imports
    try:
        from time_series_classification.models import BaseModel, Lookahead
        print("✓ Models module imports successful")
        success_count += 1
    except ImportError as e:
        print(f"✗ Models module import failed: {e}")
        failure_count += 1
    
    # Test training imports
    try:
        from time_series_classification.training import train_experiment, MetricsLogger
        print("✓ Training module imports successful")
        success_count += 1
    except ImportError as e:
        print(f"✗ Training module import failed: {e}")
        failure_count += 1
    
    # Test evaluation imports
    try:
        from time_series_classification.evaluation import visualize_attributions
        print("✓ Evaluation module imports successful")
        success_count += 1
    except ImportError as e:
        print(f"✗ Evaluation module import failed: {e}")
        failure_count += 1
    
    # Test utils imports
    try:
        from time_series_classification.utils import setup_logger, LoggerMixin
        print("✓ Utils module imports successful")
        success_count += 1
    except ImportError as e:
        print(f"✗ Utils module import failed: {e}")
        failure_count += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {success_count} passed, {failure_count} failed")
    print("=" * 50)
    
    return failure_count == 0

def test_logger_functionality():
    """Test that the logger works correctly."""
    
    print("\n" + "=" * 50)
    print("Testing Logger Functionality")
    print("=" * 50)
    
    try:
        from time_series_classification.utils import setup_logger
        
        logger = setup_logger("TestLogger", log_to_file=False)
        logger.info("Info message test")
        logger.debug("Debug message test")
        logger.warning("Warning message test")
        logger.error("Error message test")
        
        print("✓ Logger functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False

if __name__ == "__main__":
    all_passed = True
    
    if not test_core_imports():
        all_passed = False
    
    if not test_logger_functionality():
        all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Installation successful.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)