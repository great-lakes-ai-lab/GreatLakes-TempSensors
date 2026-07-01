# Originally from: deepsensor_greatlakes
"""Seasonal cycle computation and anomaly processing."""

import os
import json
import uuid

import numpy as np
import pandas as pd
import xarray as xr


class SeasonalCycleProcessor:
    """
    A class to manage seasonal cycle calculation, saving, and loading.
    """

    def __init__(self, seasonal_cycle=None, metadata=None):
        """
        Initialize a SeasonalCycleProcessor.

        Parameters:
        -----------
        seasonal_cycle : xarray.Dataset, optional
            Pre-existing seasonal cycle
        metadata : dict, optional
            Metadata about the seasonal cycle
        """
        self.seasonal_cycle = seasonal_cycle
        self.metadata = metadata or {}

        # Generate a unique identifier if not provided
        if 'id' not in self.metadata:
            self.metadata['id'] = str(uuid.uuid4())

    def calculate(self, ds, dim='time'):
        """
        Calculate the seasonal cycle (climatology) of a dataset.

        Parameters:
        -----------
        ds : xarray.Dataset
            Input dataset with a time dimension
        dim : str, optional
            Dimension along which to calculate seasonal cycle (default: 'time')

        Returns:
        --------
        SeasonalCycleProcessor
            Self, with seasonal cycle calculated
        """
        # Group by month and calculate mean
        self.seasonal_cycle = ds.groupby(f'{dim}.month').mean(dim=dim)

        # Update metadata
        self.metadata.update({
            'calculation_date': str(pd.Timestamp.now()),
            'dataset_vars': list(ds.data_vars),
            'dataset_dims': dict(ds.sizes)
        })

        return self

    def save(self, base_dir='seasonal_cycles'):
        """
        Save the seasonal cycle to disk.

        Parameters:
        -----------
        base_dir : str, optional
            Base directory to save seasonal cycle files

        Returns:
        --------
        dict
            Paths to saved files
        """
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Paths for saving
        cycle_path = os.path.join(base_dir, f'{self.metadata["id"]}_seasonal_cycle.nc')
        metadata_path = os.path.join(base_dir, f'{self.metadata["id"]}_metadata.json')

        # Save seasonal cycle as NetCDF
        self.seasonal_cycle.to_netcdf(cycle_path)

        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        return {
            'seasonal_cycle_path': cycle_path,
            'metadata_path': metadata_path
        }

    @classmethod
    def load(cls, cycle_path, metadata_path=None):
        """
        Load a previously saved seasonal cycle.

        Parameters:
        -----------
        cycle_path : str
            Path to the seasonal cycle NetCDF file
        metadata_path : str, optional
            Path to the metadata JSON file

        Returns:
        --------
        SeasonalCycleProcessor
            Loaded seasonal cycle processor
        """
        # Load seasonal cycle
        seasonal_cycle = xr.load_dataset(cycle_path)

        # Load metadata if path provided
        metadata = None
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return cls(seasonal_cycle=seasonal_cycle, metadata=metadata)

    def compute_anomalies(self, ds):
        """
        Compute anomalies by subtracting seasonal cycle.

        Parameters:
        -----------
        ds : xarray.Dataset
            Input dataset

        Returns:
        --------
        xarray.Dataset
            Dataset with anomalies
        """
        if self.seasonal_cycle is None:
            raise ValueError("No seasonal cycle calculated. Call .calculate() first.")

        # Compute anomalies by subtracting seasonal cycle
        anomalies = ds.groupby('time.month') - self.seasonal_cycle

        return anomalies
