import os
import logging
import numpy as np
import h5py
from netCDF4 import Dataset

def preprocess_era5_netcdf_to_hdf5(netcdf_path, output_filename="era5_preprocessed.h5"):
    """
    Convert ERA5 NetCDF data to HDF5 format.

    Args:
        netcdf_path (str): Path to the ERA5 NetCDF file.
        output_filename (str): Name of the output HDF5 file.
    """
    try:
        # Open the NetCDF file
        nc = Dataset(netcdf_path, mode='r')
        
        # Initialize the HDF5 file
        with h5py.File(output_filename, "w") as f:
            # Create groups for different variables
            for var_name in nc.variables.keys():
                if var_name == 'time':  # Handle time separately
                    times = nc[var_name][:]
                    group = f.create_group(var_name)
                    subgroup = group.create_group('values')
                    subgroup.create_dataset('time', data=times)
                else:
                    variable_data = nc[var_name][:]  # Assuming variable is time-series
                    if len(variable_data.shape) == 1:
                        group = f.create_group(var_name)
                        group.create_dataset(var_name, data=variable_data)
                    else:
                        # Handle multi-dimensional variables (e.g., latitude, longitude)
                        lat = nc['lat'][:]
                        lon = nc['lon'][:]
                        time_steps = nc['time'][:]

                        # Create a group for each spatial point
                        for i in range(len(lat)):
                            for j in range(len(lon)):
                                point_group = f.create_group(f"point_{i}_{j}")
                                point_group.create_dataset("latitude", data=lat[i])
                                point_group.create_dataset("longitude", data=lon[j])
                                point_group.create_dataset(var_name, data=variable_data[:, i, j])

        logging.info(f"Successfully saved ERA5 preprocessed data to {output_filename}")

    except Exception as e:
        logging.error(f"Failed to preprocess ERA5 NetCDF file: {str(e)}")

def main_era5():
    """
    Main function for preprocessing ERA5 NetCDF files.
    """
    # Path to the ERA5 NetCDF directory
    ERA5_ARCHIVE_PATH = "ERA5_data"
    
    # Process all NetCDF files in the directory
    for filename in os.listdir(ERA5_ARCHIVE_PATH):
        if filename.endswith(".nc"):
            netcdf_path = os.path.join(ERA5_ARCHIVE_PATH, filename)
            preprocess_era5_netcdf_to_hdf5(netcdf_path)

if __name__ == "__main__":
    main_era5()
