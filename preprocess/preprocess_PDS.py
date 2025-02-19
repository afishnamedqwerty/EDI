import os
import logging
import numpy as np
from multiprocessing import Process, Queue
import h5py
import pandas as pd
import spiceypy as spice
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_metakernel_data(meta_kernel_path):
    """
    Load kernel metadata from a meta-kernel file.

    Args:
        meta_kernel_path (str): Path to the meta-kernel file.

    Returns:
        dict: Dictionary mapping kernel types to their file paths.
    """
    try:
        kernels = {
            "SPK": [],
            "CK": [],
            "PCK": [],
            "LSK": [],
            "FK": []
        }

        with open(meta_kernel_path, 'r') as f:
            content = f.read().split('\n')

        for line in content:
            if line.strip().startswith("'"):
                path = line.strip()[1:-2].strip()  # Remove quotes and commas
                _, ext = os.path.splitext(path)
                
                if ext == '.bsp':
                    kernels["SPK"].append(path)
                elif ext == '.bc':
                    kernels["CK"].append(path)
                elif ext == '.tls':
                    kernels["LSK"].append(path)
                elif ext == '.tpc':
                    kernels["PCK"].append(path)
                elif ext == '.tf':
                    kernels["FK"].append(path)

        return kernels

    except Exception as e:
        logging.error(f"Failed to load meta-kernel {meta_kernel_path}: {str(e)}")
        return None

def process_spk(spk_file, start_time, stop_time):
    """
    Process an SPK file and extract positions and velocities.

    Args:
        spk_file (str): Path to the SPK file.
        start_time (float): Start time in ET.
        stop_time (float): Stop time in ET.

    Returns:
        tuple: Positions and velocities arrays.
    """
    try:
        et_start = spice.str2et(start_time)
        et_stop = spice.str2et(stop_time)
        
        times = np.linspace(et_start, et_stop, num=1000)
        positions, _ = spice.spkpos(spk_file, times, 'J2000', 'NONE', 'SATURN BARYCENTER')
        velocities = spice.spkvel(spk_file, times, 'J2000', 'NONE', 'SATURN BARYCENTER')

        return {
            "times": times,
            "positions": positions,
            "velocities": velocities
        }
    except Exception as e:
        logging.error(f"Failed to process SPK file {spk_file}: {str(e)}")
        return None

def process_ck(ck_file, start_time, stop_time):
    """
    Process a CK file and extract quaternions.

    Args:
        ck_file (str): Path to the CK file.
        start_time (float): Start time in ET.
        stop_time (float): Stop time in ET.

    Returns:
        dict: Quaternions array.
    """
    try:
        et_start = spice.str2et(start_time)
        et_stop = spice.str2et(stop_time)
        
        times = np.linspace(et_start, et_stop, num=1000)
        quaternions = []
        
        for et in times:
            q = spice.mqckqup(ck_file, et)
            quaternions.append(q)
        
        return {
            "times": times,
            "quaternions": np.array(quaternions)
        }
    except Exception as e:
        logging.error(f"Failed to process CK file {ck_file}: {str(e)}")
        return None

def process_pck(pck_file):
    """
    Process a PCK file and extract constants.

    Args:
        pck_file (str): Path to the PCK file.

    Returns:
        dict: Dictionary of extracted constants.
    """
    try:
        constants = {}
        with spice.BPCD(pck_file) as f:
            for record in f:
                if record.type == 'B':
                    constants[record.center] = {
                        "rotation_period": record.rotation,
                        "radius": record.radius
                    }
        
        return {
            "constants": constants
        }
    except Exception as e:
        logging.error(f"Failed to process PCK file {pck_file}: {str(e)}")
        return None

def process_lsk(lsk_file):
    """
    Process an LSK file and extract leap seconds.

    Args:
        lsk_file (str): Path to the LSK file.

    Returns:
        dict: Array of leap seconds.
    """
    try:
        leaps = []
        with spice.TLSK(lsk_file) as f:
            for record in f:
                leaps.append(record.leap)
        
        return {
            "leaps": np.array(leaps)
        }
    except Exception as e:
        logging.error(f"Failed to process LSK file {lsk_file}: {str(e)}")
        return None

def process_fk(fk_file):
    """
    Process an FK file and extract frame definitions.

    Args:
        fk_file (str): Path to the FK file.

    Returns:
        dict: Dictionary of frame definitions.
    """
    try:
        frames = {}
        with spice.FK(fk_file) as f:
            for frame in f.frames():
                frames[frame.name] = {
                    "type": frame.type,
                    "parent": frame.parent
                }
        
        return {
            "frames": frames
        }
    except Exception as e:
        logging.error(f"Failed to process FK file {fk_file}: {str(e)}")
        return None

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
            result = process_spk(path, start_time, stop_time)
        elif kernel_type == "CK":
            result = process_ck(path, start_time, stop_time)
        elif kernel_type == "PCK":
            result = process_pck(path)
        elif kernel_type == "LSK":
            result = process_lsk(path)
        elif kernel_type == "FK":
            result = process_fk(path)
        else:
            logging.error(f"Unknown kernel type {kernel_type}")
            return

        if result:
            queue.put((kernel_type, "success", result))
        else:
            queue.put((kernel_type, "failed", f"{kernel_type} processing failed"))

    except Exception as e:
        queue.put((kernel_type, "failed", str(e)))

def align_data(data_dict, reference_times):
    """
    Align data from different kernels to a common time grid.

    Args:
        data_dict (dict): Dictionary of kernel type to data.
        reference_times (np.ndarray): Common time grid.

    Returns:
        dict: Aligned data dictionary.
    """
    aligned_data = {}
    
    for kernel_type, data in data_dict.items():
        if kernel_type == "SPK":
            # Align positions and velocities
            times = data["times"]
            positions = np.interp(reference_times, times, data["positions"])
            velocities = np.interp(reference_times, times, data["velocities"])
            aligned_data[kernel_type] = {
                "reference_times": reference_times,
                "positions": positions,
                "velocities": velocities
            }
        elif kernel_type == "CK":
            # Align quaternions
            times = data["times"]
            quats = np.interp(reference_times, times, data["quaternions"])
            aligned_data[kernel_type] = {
                "reference_times": reference_times,
                "quaternions": quats
            }
        elif kernel_type == "PCK":
            # Constants are static; no time alignment needed
            aligned_data[kernel_type] = data
        elif kernel_type == "LSK":
            # Leap seconds are static; no time alignment needed
            aligned_data[kernel_type] = data
        elif kernel_type == "FK":
            # Frame definitions are static; no time alignment needed
            aligned_data[kernel_type] = data

    return aligned_data

def normalize_data(data, method="zscore"):
    """
    Normalize data using specified method.

    Args:
        data (np.ndarray): Data to normalize.
        method (str): Normalization method ('minmax' or 'zscore').

    Returns:
        np.ndarray: Normalized data.
    """
    if method == "minmax":
        return (data - data.min()) / (data.max() - data.min())
    elif method == "zscore":
        return (data - data.mean()) / data.std()
    else:
        raise ValueError("Unsupported normalization method")

def create_windows(data_dict, window_size=10, step=5):
    """
    Create sliding windows from normalized data.

    Args:
        data_dict (dict): Dictionary of kernel type to data.
        window_size (int): Size of each window.
        step (int): Step between consecutive windows.

    Returns:
        dict: Dictionary of kernel type to list of windows.
    """
    windowed_data = {}

    for kernel_type, values in data_dict.items():
        if isinstance(values, np.ndarray):
            num_windows = int((len(values) - window_size) / step) + 1
            windows = []
            
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                if end > len(values):
                    break
                windows.append(values[start:end])
            
            windowed_data[kernel_type] = windows
        else:
            # Handle static data (PCK, LSK, FK)
            windowed_data[kernel_type] = [values]

    return windowed_data

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
                
                if isinstance(values, list):
                    for i, window in enumerate(values):
                        subgroup = group.create_group(f"window_{i}")
                        for key, arr in window.items():
                            subgroup.create_dataset(key, data=arr)
                else:
                    for key, arr in values.items():
                        group.create_dataset(key, data=arr)

        logging.info(f"Successfully saved preprocessed data to {filename}")
    except Exception as e:
        logging.error(f"Failed to save preprocessed data: {str(e)}")

def main():
    """
    Main function for preprocessing SPICE kernels.
    """
    # Discover meta-kernels and load them
    PDS_ARCHIVE_PATH = "PDS_archives/bepicolombo"
    meta_kernel_paths = list(Path(PDS_ARCHIVE_PATH).glob("meta_kernels/*.tm"))
    
    all_kernels = {}
    
    for path in meta_kernel_paths:
        kernels = load_metakernel_data(str(path))
        if kernels:
            for kernel_type, paths in kernels.items():
                if kernel_type not in all_kernels:
                    all_kernels[kernel_type] = []
                all_kernels[kernel_type].extend(paths)

    # Remove duplicates while preserving order
    seen = set()
    unique_kernels = {}
    
    for kernel_type, paths in all_kernels.items():
        unique_paths = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        unique_kernels[kernel_type] = unique_paths

    # Initialize queues
    result_queue = Queue()
    error_queue = Queue()

    # Prepare worker processes
    workers = []
    
    for kernel_type, paths in unique_kernels.items():
        for path in paths:
            start_time = "2000-01-01"  # Replace with actual start time
            stop_time = "2024-12-31"   # Replace with actual stop time
            
            kernel_info = (kernel_type, str(path), start_time, stop_time)
            process = Process(
                target=process_kernel,
                args=(result_queue, kernel_info)
            )
            
            workers.append(process)
            process.start()

    # Collect results and errors
    data_dict = {}
    
    for _ in range(len(workers)):
        result = result_queue.get()
        
        if result[1] == "success":
            kernel_type, _, data = result
            data_dict[kernel_type] = data
            logging.info(f"Successfully processed {kernel_type} kernel: {result[2]}")
        else:
            logging.error(f"Failed to process {result[0]} kernel: {result[2]}")

    # Align data by time
    reference_times = np.arange(
        spice.str2et("2000-01-01"),
        spice.str2et("2024-12-31"),
        86400.0  # Daily sampling
    )
    
    aligned_data = align_data(data_dict, reference_times)

    # Normalize data
    normalized_data = {}
    
    for kernel_type, values in aligned_data.items():
        if isinstance(values, dict):
            # Static data (PCK, LSK, FK)
            normalized_values = {k: normalize_data(v) for k, v in values.items()}
        else:
            # Time-series data (SPK, CK)
            normalized_values = normalize_data(values)
        
        normalized_data[kernel_type] = normalized_values

    # Create windows
    window_size = 10
    step = 5
    windowed_data = create_windows(normalized_data, window_size=window_size, step=step)

    # Save preprocessed data
    save_preprocessed_data(windowed_data, filename="preprocessed.h5")

if __name__ == "__main__":
    main()
