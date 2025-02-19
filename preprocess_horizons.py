import os
import logging
import numpy as np
import h5py
from astropy.io import ascii

def preprocess_jpl_horizons_to_hdf5(text_file_path, output_filename="jpl_preprocessed.h5"):
    """
    Convert JPL Horizons text data to HDF5 format.

    Args:
        text_file_path (str): Path to the JPL Horizons text file.
        output_filename (str): Name of the output HDF5 file.
    """
    try:
        # Read the JPL Horizons text file
        data = ascii.read(text_file_path)
        
        # Initialize the HDF5 file
        with h5py.File(output_filename, "w") as f:
            # Create a group for each celestial object
            for obj_id in np.unique(data['object_id']):
                group = f.create_group(f"object_{obj_id}")
                
                # Extract time series data for the object
                mask = data['object_id'] == obj_id
                times = data[mask]['time']
                positions = data[mask]['position']
                velocities = data[mask]['velocity']
                
                # Create subgroups for different types of data
                pos_group = group.create_group('positions')
                vel_group = group.create_group('velocities')
                
                # Save time and corresponding values
                pos_group.create_dataset('time', data=times)
                pos_group.create_dataset('x', data=[p[0] for p in positions])
                pos_group.create_dataset('y', data=[p[1] for p in positions])
                pos_group.create_dataset('z', data=[p[2] for p in positions])
                
                vel_group.create_dataset('time', data=times)
                vel_group.create_dataset('x', data=[v[0] for v in velocities])
                vel_group.create_dataset('y', data=[v[1] for v in velocities])
                vel_group.create_dataset('z', data=[v[2] for v in velocities])

        logging.info(f"Successfully saved JPL Horizons preprocessed data to {output_filename}")

    except Exception as e:
        logging.error(f"Failed to preprocess JPL Horizons text file: {str(e)}")

def main_jpl():
    """
    Main function for preprocessing JPL Horizons text files.
    """
    # Path to the JPL Horizons directory
    JPL_ARCHIVE_PATH = "JPL_data"
    
    # Process all text files in the directory
    for filename in os.listdir(JPL_ARCHIVE_PATH):
        if filename.endswith(".txt"):
            text_file_path = os.path.join(JPL_ARCHIVE_PATH, filename)
            preprocess_jpl_horizons_to_hdf5(text_file_path)

if __name__ == "__main__":
    main_jpl()
