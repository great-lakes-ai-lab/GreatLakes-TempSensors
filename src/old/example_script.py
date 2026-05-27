# Import the class from the module where it's defined
from sst_prediction_model import SSTPredictionModel

# Define paths to data, mask, and bathymetry files
data_path = "/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/GLSEA3_NETCDF"
mask_path = "/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/masks/lakemask.nc"
bathymetry_path = "/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/bathymetry/interpolated_bathymetry.nc"

# Define the directory to save model and results
output_dir = "/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/results/"

# Initialize the SSTPredictionModel
model = SSTPredictionModel(data_path=data_path,
                           mask_path=mask_path,
                           bathymetry_path=bathymetry_path,
                           use_gpu=None)  # Auto-detect GPU

# Load the data
model.load_data(start_year=2007, end_year=2022)

# Train the model
losses, val_rmses = model.train(output_dir=output_dir,
                                epochs=25,
                                learning_rate=2e-5,
                                batch_size=1 if torch.cuda.is_available() else 1,
                                train_start='2007-01-01',
                                train_end='2014-12-31',
                                val_start='2015-01-01',
                                val_end='2016-12-31',
                                val_freq=5)

# Print final training and validation metrics
print(f"Training Losses: {losses}")
print(f"Validation RMSEs: {val_rmses}")