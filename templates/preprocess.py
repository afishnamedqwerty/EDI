import numpy as np
import spiceypy as sp
import multiprocessing as mp
import os
import h5py
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from multiprocessing import Queue, Process

# Define paths
PDS_ARCHIVE_PATH = "PDS_archives/bepicolombo"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

def discover_kernels(kernel_type, archive_path):
    """
    Discover kernel files of a specified type in the PDS archive.

    Args:
        kernel_type (str): Type of kernel to discover ('SPK', 'CK', 'PCK', 'LSK', 'FK').
        archive_path (str): Path to the PDS archive directory.

    Returns:
        list: List of kernel file paths matching the type.
    """
    patterns = {
        "SPK": ["*.bsp"],
        "CK": ["*.bc"],
        "PCK": ["*.bpc", "*.tpc"],
        "LSK": ["*.tls"],
        "FK": ["*.tf"]
    }

    kernel_paths = []
    for ext in patterns[kernel_type]:
        pattern = f"*.{ext}"
        path = os.path.join(archive_path, kernel_type.lower(), pattern)
        kernel_paths.extend(list(Path(path).glob()))

    return kernel_paths

def load_meta_kernel(meta_kernel_path):
    """
    Load a meta-kernel file.

    Args:
        meta_kernel_path (str): Path to the meta-kernel file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        sp.furnsh(meta_kernel_path)
        logging.info(f"Successfully loaded meta-kernel: {meta_kernel_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to load meta-kernel {meta_kernel_path}: {str(e)}")
        return False


def process_kernel(queue, kernel_info):
    """
    Process a single kernel file and add results to a queue.

    Args:
        queue (Queue): Queue to store results.
        kernel_info (tuple): Tuple containing kernel type, path, and metadata.
    """
    try:
        kernel_type, path, start_time, stop_time = kernel_info

        if kernel_type == "SPK":
            times, positions, velocities = load_spk_data(path, start_time, stop_time)
            queue.put((kernel_type, "success", {
                "times": times,
                "positions": positions,
                "velocities": velocities
            }))

        elif kernel_type == "CK":
            times, quaternions = load_ck_data(path, start_time, stop_time)
            queue.put((kernel_type, "success", {
                "times": times,
                "quaternions": quaternions
            }))

        elif kernel_type == "PCK":
            if path.endswith(".bpc"):
                constants = load_bpc_data(path)
            else:
                constants = load_tpc_data(path)
            queue.put((kernel_type, "success", {
                "constants": constants
            }))

        elif kernel_type == "LSK":
            leaps = load_lsk_data(path)
            queue.put((kernel_type, "success", {
                "leaps": leaps
            }))

        elif kernel_type == "FK":
            frames = load_fk_data(path)
            queue.put((kernel_type, "success", {
                "frames": frames
            }))

    except Exception as e:
        queue.put((kernel_type, "failed", str(e)))

def load_spk_data(spk_path, start_time, stop_time):
    """
    Load position and velocity data from an SPK file.

    Args:
        spk_path (str): Path to the SPK file.
        start_time (int): Start time in seconds past J2000.
        stop_time (int): Stop time in seconds past J2000.

    Returns:
        tuple: (times, positions, velocities)
    """
    try:
        sp.furnsh(spk_path)

        times = []
        positions = []
        velocities = []

        current_time = start_time

        while current_time <= stop_time:
            try:
                pos, vel = sp.spkopa(current_time)

                times.append(current_time)
                positions.append(pos)
                velocities.append(vel)

            except Exception as e:
                logging.warning(f"Error at time {current_time}: {str(e)}")

            current_time += 86400  # Increment by one day

        return np.array(times), np.array(positions), np.array(velocities)

    except Exception as e:
        logging.error(f"Failed to load SPK file {spk_path}: {str(e)}")
        return None, None, None

def load_ck_data(ck_path, start_time, stop_time):
    """
    Load attitude data from a CK file.

    Args:
        ck_path (str): Path to the CK file.
        start_time (int): Start time in seconds past J2000.
        stop_time (int): Stop time in seconds past J2000.

    Returns:
        tuple: (times, quaternions)
    """
    try:
        sp.furnsh(ck_path)

        times = []
        quaternions = []

        current_time = start_time

        while current_time <= stop_time:
            try:
                q = sp.spkopa(current_time)  # Example CK data loading
                times.append(current_time)
                quaternions.append(q)

            except Exception as e:
                logging.warning(f"Error at time {current_time}: {str(e)}")

            current_time += 86400

        return np.array(times), np.array(quaternions)

    except Exception as e:
        logging.error(f"Failed to load CK file {ck_path}: {str(e)}")
        return None, None

def load_bpc_data(bpc_path):
    """
    Load PCK data from a BPC file.

    Args:
        bpc_path (str): Path to the BPC file.

    Returns:
        dict: Dictionary of PCK constants.
    """
    try:
        sp.furnsh(bpc_path)
        return sp.bpc2c()
    except Exception as e:
        logging.error(f"Failed to load BPC file {bpc_path}: {str(e)}")
        return None

def load_tpc_data(tpc_path):
    """
    Load PCK data from a TPC file.

    Args:
        tpc_path (str): Path to the TPC file.

    Returns:
        dict: Dictionary of PCK constants.
    """
    try:
        sp.furnsh(tpc_path)
        return sp.tpc2c()
    except Exception as e:
        logging.error(f"Failed to load TPC file {tpc_path}: {str(e)}")
        return None

def load_lsk_data(lsk_path):
    """
    Load LSK data from an LSK file.

    Args:
        lsk_path (str): Path to the LSK file.

    Returns:
        tuple: (times, positions)
    """
    try:
        sp.furnsh(lsk_path)
        times = []
        positions = []

        current_time = 0
        while True:
            try:
                pos = sp.lskpos(current_time)
                times.append(current_time)
                positions.append(pos)

            except Exception as e:
                logging.warning(f"Error at time {current_time}: {str(e)}")

            if current_time > 1e9:  # Arbitrary stop condition
                break

            current_time += 86400

        return np.array(times), np.array(positions)

    except Exception as e:
        logging.error(f"Failed to load LSK file {lsk_path}: {str(e)}")
        return None, None

def load_fk_data(fk_path):
    """
    Load FK data from an FK file.

    Args:
        fk_path (str): Path to the FK file.

    Returns:
        dict: Dictionary of FK constants.
    """
    try:
        sp.furnsh(fk_path)
        return sp.fk2c()
    except Exception as e:
        logging.error(f"Failed to load FK file {fk_path}: {str(e)}")
        return None

def align_data(data_dict, reference_times):
    """
    Align data from different kernels by time.

    Args:
        data_dict (dict): Dictionary of kernel type to data.
        reference_times (np.ndarray): Times to align to.

    Returns:
        dict: Aligned data dictionary.
    """
    aligned_data = {kernel_type: [] for kernel_type in data_dict.keys()}

    for kernel_type, data in data_dict.items():
        times = data["times"]
        values = data["values"]

        # Find matching times
        matches = np.searchsorted(times, reference_times)
        valid_indices = np.where(matches < len(times))[0]

        aligned_values = []
        for t in reference_times:
            idx = np.searchsorted(times, t)
            if idx < len(times):
                aligned_values.append(values[idx])
            else:
                aligned_values.append(None)

        aligned_data[kernel_type] = np.array(aligned_values)

    return aligned_data

def normalize_data(data, method="zscore"):
    """
    Normalize data using specified method.

    Args:
        data (np.ndarray): Data to normalize.
        method (str): Normalization method ('zscore', 'minmax').

    Returns:
        np.ndarray: Normalized data.
    """
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    else:
        raise ValueError("Unsupported normalization method")

def create_windows(data, window_size=10, step=5):
    """
    Create sliding windows of data.

    Args:
        data (np.ndarray): Input data.
        window_size (int): Size of the window.
        step (int): Step between windows.

    Returns:
        list: List of windows.
    """
    num_windows = len(data) // step
    windows = []

    for i in range(num_windows):
        start = i * step
        end = start + window_size
        if end > len(data):
            break
        windows.append(data[start:end])

    return windows

def save_preprocessed_data(data, filename="preprocessed.h5"):
    """
    Save preprocessed data to an HDF5 file.

    Args:
        data (dict): Dictionary of kernel type to data.
        filename (str): Name of the output file.
    """
    try:
        with h5py.File(filename, "w") as f:
            for kernel_type, values in data.items():
                group = f.create_group(kernel_type)
                for key, value in values.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value)
                    else:
                        group.create_dataset(key, data=np.array([value]))

        logging.info(f"Successfully saved preprocessed data to {filename}")
    except Exception as e:
        logging.error(f"Failed to save preprocessed data: {str(e)}")

def main():
    """
    Main function for preprocessing SPICE kernels.
    """
    # Discover meta-kernels and load them
    meta_kernel_paths = list(Path(PDS_ARCHIVE_PATH).glob("meta_kernels/*.tm"))
    for path in meta_kernel_paths:
        if load_meta_kernel(str(path)):
            logging.info(f"Loaded meta-kernel: {path}")

    # Prepare kernel discovery and loading
    kernels_to_load = {
        "SPK": [],
        "CK": [],
        "PCK": [],
        "LSK": [],
        "FK": []
    }

    # Initialize queues
    result_queue = Queue()
    error_queue = Queue()

    # Prepare worker processes
    workers = []
    for kernel_type, paths in kernels_to_load.items():
        for path in paths:
            start_time = 0
            stop_time = 1e9

            process = Process(
                target=process_kernel,
                args=(result_queue, (kernel_type, path, start_time, stop_time))
            )

            workers.append(process)
            process.start()

    # Collect results and errors
    for _ in range(len(workers)):
        result = result_queue.get()
        if result[1] == "success":
            logging.info(f"Successfully processed {result[0]} kernel: {result[2]}")
        else:
            logging.error(f"Failed to process {result[0]} kernel: {result[2]}")

    # Align data by time
    reference_times = np.arange(0, 1e9, 86400)
    aligned_data = align_data(data_dict, reference_times)

    # Normalize data
    normalized_data = {}
    for kernel_type, values in aligned_data.items():
        normalized_values = normalize_data(values, method="zscore")
        normalized_data[kernel_type] = normalized_values

    # Create windows
    window_size = 10
    step = 5
    windowed_data = create_windows(normalized_data, window_size=window_size, step=step)

    # Save preprocessed data
    save_preprocessed_data(windowed_data, filename="preprocessed.h5")

if __name__ == "__main__":
    main()
