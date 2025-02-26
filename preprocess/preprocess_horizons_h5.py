import os
import logging
import numpy as np
import h5py
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

def read_spk_file(spk_path: str) -> dict:
    """
    Reads an SPK file and extracts timeseries data.

    Args:
        spk_path (str): Path to the SPK binary file.

    Returns:
        dict: Dictionary containing positions and velocities for each object.
            Format:
                {
                    'object_ids': list of unique object IDs,
                    'times': list of datetime objects,
                    'positions': numpy array of 3D positions,
                    'velocities': numpy array of 3D velocities
                }
    """
    try:
        # Initialize data containers
        object_ids = []
        times = []
        positions = []
        velocities = []

        with open(spk_path, "rb") as f:
            # Read the first record to get number of records and body ID
            data = f.read(16)
            num_records = int.from_bytes(data[:4], byteorder='d')
            body_id = int.from_bytes(data[4:8], byteorder='d')

            for _ in range(num_records):
                # Read the record header (contains time, position, velocity)
                header_size = 24
                header_data = f.read(header_size)
                
                if len(header_data) < header_size:
                    break  # End of file reached

                # Extract fields from the header
                t0 = int.from_bytes(header_data[0:4], byteorder='d')
                t1 = int.from_bytes(header_data[4:8], byteorder='d')
                pos_size = int.from_bytes(header_data[8:12], byteorder='d')
                vel_size = int.from_bytes(header_data[12:16], byteorder='d')

                # Read position and velocity data
                pos_data = f.read(pos_size)
                vel_data = f.read(vel_size)

                # Convert time from TDB to datetime (assuming TDB is similar to UTC for simplicity)
                t = datetime.utcfromtimestamp(t0) + timedelta(seconds=t1)

                # Extract position components
                pos_x = float(int.from_bytes(pos_data[0:4], byteorder='d'))
                pos_y = float(int.from_bytes(pos_data[4:8], byteorder='d'))
                pos_z = float(int.from_bytes(pos_data[8:12], byteorder='d'))

                # Extract velocity components
                vel_x = float(int.from_bytes(vel_data[0:4], byteorder='d'))
                vel_y = float(int.from_bytes(vel_data[4:8], byteorder='d'))
                vel_z = float(int.from_bytes(vel_data[8:12], byteorder='d'))

                # Append data
                object_ids.append(body_id)
                times.append(t)
                positions.extend([pos_x, pos_y, pos_z])
                velocities.extend([vel_x, vel_y, vel_z])

        return {
            'object_ids': np.array(object_ids),
            'times': np.array(times, dtype='datetime64[ns]'),
            'positions': np.array(positions).reshape(-1, 3),
            'velocities': np.array(velocities).reshape(-1, 3)
        }

    except Exception as e:
        logging.error(f"Error reading SPK file {spk_path}: {str(e)}")
        return None

def preprocess_jpl_horizons_to_hdf5(
    spk_directory: str,
    output_filename: str = "horizons_preprocessed.h5",
    log_file: Optional[str] = None
) -> bool:
    """
    Converts JPL Horizons SPK files to HDF5 format.

    Args:
        spk_directory (str): Directory containing SPK binary files.
        output_filename (str): Name of the output HDF5 file.
        log_file (Optional[str]): Path to a log file for logging progress.

    Returns:
        bool: True if preprocessing was successful, False otherwise.
    """
    try:
        # Initialize logging
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

        # Initialize the HDF5 file
        with h5py.File(output_filename, "w") as f:
            # Create main groups
            raw_data_group = f.create_group("raw_data")
            metadata_group = f.create_group("metadata")

            # Track all object IDs across files
            all_object_ids = set()

            total_files = 0
            total_records = 0

            for filename in os.listdir(spk_directory):
                if filename.endswith(".bsp"):
                    spk_path = os.path.join(spk_directory, filename)
                    total_files += 1

                    logging.info(f"Processing SPK file: {filename}")
                    
                    data = read_spk_file(spk_path)
                    if not data:
                        continue

                    # Update object IDs
                    all_object_ids.update(data['object_ids'].tolist())

                    # Add timeseries data for this file
                    total_records += len(data['times'])
                    
                    # Create subgroup for this file's data
                    file_group = raw_data_group.create_group(filename)
                    
                    # Save positions and velocities
                    file_group.create_dataset(
                        "positions",
                        data=data['positions'],
                        dtype=np.float64,
                        compression="gzip"
                    )
                    
                    file_group.create_dataset(
                        "velocities",
                        data=data['velocities'],
                        dtype=np.float64,
                        compression="gzip"
                    )

            # Save global metadata
            metadata_group["object_ids"] = np.array(list(all_object_ids), dtype='int64')
            metadata_group["total_files"] = total_files
            metadata_group["total_records"] = total_records

        logging.info(f"Successfully saved Horizons preprocessed data to {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Failed to preprocess JPL Horizons SPK files: {str(e)}")
        return False

def main_jpl():
    """
    Main function for preprocessing JPL Horizons SPK files.
    """
    try:
        # Path to the JPL Horizons directory
        JPL_ARCHIVE_PATH = "JPL_data"

        # Validate input directory exists
        if not os.path.exists(JPL_ARCHIVE_PATH):
            raise FileNotFoundError(f"Directory {JPL_ARCHIVE_PATH} not found")

        # Preprocess all SPK files in the directory
        success = preprocess_jpl_horizons_to_hdf5(
            spk_directory=JPL_ARCHIVE_PATH,
            output_filename="horizons_preprocessed.h5",
            log_file="preprocessing.log"
        )

        if success:
            logging.info("Preprocessing completed successfully")
        else:
            logging.error("Preprocessing failed")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main_jpl()
