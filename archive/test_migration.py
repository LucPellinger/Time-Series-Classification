# test_migration.py
"""Test script to verify the migration was successful."""

def test_imports():
    """Test that all major imports work."""
    
    print("Testing imports...")
    
    try:
        # Test data imports
        from data.dataloaders.aeon_dataset import AeonDataset
        print("✓ AeonDataset import successful")
        
        from data.dataloaders.movie_dataset import MovieTimeSeriesDataset
        print("✓ MovieTimeSeriesDataset import successful")
        
        # Test model imports
        from models.base_model import BaseModel
        print("✓ BaseModel import successful")
        
        from models.optimizers.lookahead import Lookahead
        print("✓ Lookahead import successful")
        
        # Test training imports
        from training.callbacks.metrics_logger import MetricsLogger
        print("✓ MetricsLogger import successful")
        
        # Test utils
        from utils.logger import setup_logger
        print("✓ Logger import successful")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()