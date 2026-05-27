import unittest
from pathlib import Path
from sst_prediction_model import SSTPredictionModel
import torch

class TestSSTPredictionModel(unittest.TestCase):
    def setUp(self):
        # Use small datasets for testing purposes
        self.data_path = "test-data/GLSEA3_NETCDF"
        self.mask_path = "test-data/lakemask.nc"
        self.bathymetry_path = "test-data/interpolated_bathymetry.nc"
        self.output_dir = "test-results/"
        
        self.model = SSTPredictionModel(data_path=self.data_path,
                                        mask_path=self.mask_path,
                                        bathymetry_path=self.bathymetry_path,
                                        use_gpu=False)  # Use CPU for testing

    def test_basic_workflow(self):
        # Basic test that runs through the workflow using a small dataset
        try:
            # Load the data (use a small date range for testing)
            self.model.load_data(start_year=2007, end_year=2008, use_anomalies=False)
            
            # Train the model on a minimal amount of data
            losses, val_rmses = self.model.train(output_dir=self.output_dir,
                                                 epochs=1,  # Only 1 epoch for test
                                                 learning_rate=1e-5,
                                                 batch_size=1,
                                                 train_start='2007-01-01',
                                                 train_end='2007-12-31',
                                                 val_start='2008-01-01',
                                                 val_end='2008-12-31',
                                                 val_freq=10)  # Decrease frequency for speed
                                                 
            # Simple assertions to ensure that training completes
            self.assertTrue(len(losses) > 0, "Losses list should not be empty")
            self.assertTrue(len(val_rmses) > 0, "Validation RMSEs list should not be empty")
            self.assertTrue(all(isinstance(val, float) for val in val_rmses), "All validation RMSEs should be floats")
            
        except Exception as e:
            self.fail(f"Exception occurred during basic workflow test: {e}")

if __name__ == '__main__':
    unittest.main()