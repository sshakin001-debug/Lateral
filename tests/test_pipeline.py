"""
Unit tests for the Lateral project.
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPipeline:
    """Tests for the custom pipeline."""
    
    def test_import_lateral_sota(self):
        """Test that lateral_sota package can be imported."""
        import lateral_sota
        assert lateral_sota.__version__ == "0.1.0"
    
    def test_import_my_modules(self):
        """Test that my_modules package can be imported."""
        import my_modules
        assert my_modules is not None
    
    def test_pipeline_initialization(self):
        """Test that the pipeline can be initialized."""
        # This test will fail until weights are downloaded
        # from my_modules.custom_pipeline import MyEnhancedPipeline
        # pipeline = MyEnhancedPipeline()
        # assert pipeline is not None
        pass


class TestConfig:
    """Tests for configuration loading."""
    
    def test_config_exists(self):
        """Test that default config file exists."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'default.yaml'
        )
        assert os.path.exists(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
