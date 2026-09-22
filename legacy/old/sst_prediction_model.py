import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
from tqdm import tqdm, notebook
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch

class SSTPredictionModel:
    def __init__(self, 
                 data_path: str, 
                 mask_path: str, 
                 bathymetry_path: str,
                 use_gpu: Optional[bool] = None):
        """
        Initialize the SST Prediction Model with necessary file paths.
        
        Args:
            data_path: Base path for GLSEA3 data files
            mask_path: Path to lake mask file
            bathymetry_path: Path to bathymetry data file
            use_gpu: Whether to use GPU. If None, will auto-detect.
        
        Example:
            model = SSTPredictionModel(
                data_path="/path/to/data",
                mask_path="/path/to/mask.nc",
                bathymetry_path="/path/to/bathymetry.nc",
                use_gpu=True
            )
        """
        self.data_path = data_path
        self.mask_path = mask_path
        self.bathymetry_path = bathymetry_path
        self.data_processor = None
        self.task_loader = None
        self.model = None
        self.device = self._setup_device(use_gpu)
        
    def _setup_device(self, use_gpu: Optional[bool] = None) -> torch.device:
        """
        Setup the computation device (CPU/GPU).
        
        Args:
            use_gpu: If True, force GPU. If False, force CPU. If None, auto-detect.
        
        Returns:
            torch.device: The device to use for computation
        
        Example:
            device = self._setup_device(use_gpu=True)
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            
        if use_gpu and torch.cuda.is_available():
            set_gpu_default_device()
            device = torch.device('cuda')
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            print("Using CPU")
            
        return device
        
    def _calculate_anomalies(self, mdat: xr.Dataset, anomaly_start: str, anomaly_end: str) -> xr.Dataset:
        """Calculate climatology and anomalies."""
        climatology = mdat.groupby('time.dayofyear').mean('time')
        anomalies = mdat.groupby('time.dayofyear') - climatology
        return anomalies.sel(time=slice(anomaly_start, anomaly_end))

    def _load_climatology(self, climatology_path: str) -> xr.Dataset:
        """Load pre-existing climatology from a file."""
        climatology = xr.open_dataset(climatology_path)
        return climatology
    
    def load_data(self, start_year: int = 2007, end_year: int = 2022, 
                file_template: str = 'GLSEA3_{date}.nc', 
                anomaly_start: str = '2007-01-01', anomaly_end: str = '2016-12-31', 
                use_anomalies: bool = True, climatology_path: Optional[str] = None) -> None:
        """
        Load and preprocess all necessary data files.
        
        Args:
            start_year: Start year for data loading
            end_year: End year for data loading
            file_template: Template for data file naming
            anomaly_start: Start date for anomaly calculation (if applicable)
            anomaly_end: End date for anomaly calculation (if applicable)
            use_anomalies: If True, calculate or load anomalies; if False, use full values
            climatology_path: Path to pre-existing climatology file (if applicable)
            
        Example:
            model.load_data(start_year=2008, end_year=2015, file_template='GLSEA3_{date}.nc', use_anomalies=True)
        """
        fpath = [f'{self.data_path}/{file_template.format(date=year)}' for year in range(start_year, end_year)]
        
        try:
            mask = xr.open_dataset(self.mask_path)
            bath = xr.open_dataset(self.bathymetry_path)
            
            print("Loading SST data...")
            dat = xr.open_mfdataset(fpath, concat_dim='time', combine='nested')
            
            # Check if the dataset has sufficient data points
            if dat.sst.isnull().all() or len(dat.time) < 1:
                raise ValueError("Insufficient data points in the dataset")
            
            # Convert invalid data values to NaN
            dat = dat.where(dat.sst != '', np.nan)
            
            mdat = dat.where(np.isnan(dat.sst) == False, -0.009)

            if use_anomalies:
                if climatology_path:
                    print("Loading pre-existing climatology...")
                    climatology = self._load_climatology(climatology_path)
                    anomalies = mdat.groupby('time.dayofyear') - climatology.sst
                else:
                    print("Calculating climatology and anomalies...")
                    anomalies = self._calculate_anomalies(mdat, anomaly_start, anomaly_end)
                self.data_to_use = anomalies
            else:
                self.data_to_use = mdat

            self.data_processor = DataProcessor(x1_name="lat", x2_name="lon")
            mask_ds = self.data_processor(mask)
            sst_ds = self.data_processor(self.data_to_use)
            bath_ds = self.data_processor(bath)
            
            self.task_loader = TaskLoader(
                context=[sst_ds, mask_ds, bath_ds],
                target=sst_ds
            )
            print("Data loading completed successfully")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def generate_tasks(self, dates: pd.DatetimeIndex, N_c_range: Union[int, Tuple[int, int]] = (0, 500), progress: bool = True) -> List:
        """
        Generate tasks for model training/validation.
        
        Args:
            dates: DatetimeIndex of dates to generate tasks for
            N_c_range: Range of context points (min, max) or an integer for fixed number of points
            progress: Whether to show progress bar
        
        Returns:
            List of tasks
        
        Example:
            tasks = model.generate_tasks(pd.date_range('2021-01-01', '2021-12-31'), N_c_range=(10, 300))
        """
        if isinstance(N_c_range, int):
            N_c_min, N_c_max = N_c_range, N_c_range
        else:
            N_c_min, N_c_max = N_c_range
        
        tasks = []
        for date in notebook.tqdm(dates, disable=not progress):
            N_c = np.random.randint(N_c_min, N_c_max)
            task = self.task_loader(date, context_sampling=[N_c, "all", "all"], target_sampling="all")
            tasks.append(task)
        return tasks
    
    def compute_validation_rmse(self, model: ConvNP, val_tasks: List) -> float:
        """
        Compute RMSE for validation tasks.

        Args:
            model: Trained ConvNP model
            val_tasks: List of validation tasks

        Returns:
            Root Mean Square Error (RMSE)
        
        Example:
            val_rmse = model.compute_validation_rmse(convnp_model, validation_tasks)
        """
        errors = []
        target_var_ID = self.task_loader.target_var_IDs[0][0]
        
        for task in val_tasks:
            with torch.no_grad():
                mean = self.data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
                true = self.data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
                errors.extend(np.abs(mean - true))
            
        return np.sqrt(np.mean(np.concatenate(errors) ** 2))
    
    def train(self, 
             output_dir: str, 
             epochs: int = 25, 
             learning_rate: float = 2e-5,
             batch_size: int = 1,
             train_start: str = '2007-01-01',
             train_end: str = '2014-12-31',
             val_start: str = '2015-01-01',
             val_end: str = '2016-12-31',
             val_freq: int = 5) -> Tuple[List, List]:
        """
        Train the model and save results.
        
        Args:
            output_dir: Directory to save model and plots
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training (adjust based on available memory)
            train_start: Start date for training range
            train_end: End date for training range
            val_start: Start date for validation range
            val_end: End date for validation range
            val_freq: Frequency of validation tasks sampling
            
        Returns:
            Tuple of training losses and validation RMSEs
        
        Example:
            losses, val_rmses = model.train(
                output_dir="/save/path",
                epochs=30,
                learning_rate=1e-4,
                batch_size=2,
                train_start='2007-01-01',
                train_end='2014-12-31',
                val_start='2015-01-01',
                val_end='2016-12-31',
                val_freq=10
            )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = ConvNP(self.data_processor, self.task_loader)
        self.model.to(self.device)
        trainer = Trainer(self.model, lr=learning_rate)
        
        train_range = pd.date_range(train_start, train_end)
        val_dates = pd.date_range(val_start, val_end, freq=f'{val_freq}D')
        val_tasks = self.generate_tasks(val_dates)
        
        losses, val_rmses = [], []
        val_rmse_best = np.inf
        
        try:
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                train_tasks = self.generate_tasks(train_range[::val_freq], progress=False)
                
                batch_losses = []
                for i in range(0, len(train_tasks), batch_size):
                    batch = train_tasks[i:i + batch_size]
                    batch_loss = trainer(batch)
                    batch_losses.extend(batch_loss)
                
                epoch_loss = np.mean(batch_losses)
                losses.append(epoch_loss)
                
                val_rmse = self.compute_validation_rmse(self.model, val_tasks)
                val_rmses.append(val_rmse)
                
                print(f"Loss: {epoch_loss:.4f}, Val RMSE: {val_rmse:.4f}")
                
                if val_rmse < val_rmse_best:
                    val_rmse_best = val_rmse
                    self.model.save(output_dir)
                    print(f"New best model saved (RMSE: {val_rmse:.4f})")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current results...")
        finally:
            self._plot_training_results(losses, val_rmses, output_dir)
            
        return losses, val_rmses
    
    def _plot_training_results(self, losses: List, val_rmses: List, output_dir: Path) -> None:
        """
        Plot and save training results.
        
        Args:
            losses: List of training losses
            val_rmses: List of validation RMSEs
            output_dir: Directory to save the plots
        
        Example:
            self._plot_training_results(training_losses, validation_rmses, Path("/save/path"))
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(losses)
        axes[1].plot(val_rmses)
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[0].set_title("Training Cost")
        axes[1].set_title("Validation RMSE")
        plt.savefig(output_dir / 'training.png')
        plt.close()

# Example usage:
if __name__ == "__main__":
    model = SSTPredictionModel(
        data_path="/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/GLSEA3_NETCDF",
        mask_path="/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/lakemask2.nc",
        bathymetry_path="/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/data/interpolated_bathymetry.nc",
        use_gpu=None  # Auto-detect GPU
    )
    
    model.load_data(start_year=2007, end_year=2022, use_anomalies=True)
    losses, val_rmses = model.train(
        output_dir="/Users/dannes/GreatLakes-TempSensors/GreatLakes-TempSensors/results/",
        epochs=25,
        learning_rate=2e-5,
        batch_size=1 if torch.cuda.is_available() else 1,
        train_start='2007-01-01',
        train_end='2014-12-31',
        val_start='2015-01-01',
        val_end='2016-12-31',
        val_freq=5
    )