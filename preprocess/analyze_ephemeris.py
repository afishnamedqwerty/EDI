"""
This module contains functions to read, parse, and visualize ephemeris data stored in Parquet format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

def jdtdb_to_datetime(jdtdb: float) -> datetime:
    """
    Converts a Julian Date in TDB (Julian Date in Tokyo Dynamical Bureau) to a Python datetime object.
    
    Args:
        jdtdb (float): The Julian Date in TDB
        
    Returns:
        datetime: A timezone-aware UTC datetime object
    """
    try:
        # Convert JDTDB to a Time object with UTC timezone
        t = Time(jdtdb, format='jd', scale='tdb').datetime
        return t.replace(tzinfo=timezone.utc)
        
    except Exception as e:
        raise ValueError(f"Error converting JDTDB {jdtdb} to datetime: {str(e)}")

def analyze_parquet_file(parquet_path: str) -> None:
    """
    Reads, parses, and visualizes the contents of an ephemeris Parquet file.
    
    Args:
        parquet_path (str): Path to the Parquet file to be analyzed
        
    Returns:
        None
    """
    try:
        # Step 1: Verify file exists
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
            
        # Step 2: Read Parquet file into DataFrame
        df = pd.read_parquet(parquet_path)
        print(f"\n Dataframe: {df}")
        
        if df.empty:
            print(f"Warning: Empty DataFrame loaded from {parquet_path}")
            return
            
        # Step 3: Convert JDTDB to datetime and sort by time
        try:
            df['datetime'] = df['JDTDB'].apply(jdtdb_to_datetime)
            df.sort_values('datetime', inplace=True, ascending=True)
            
        except KeyError:
            raise ValueError(f"Required column 'JDTDB' not found in {parquet_path}")
            
        except Exception as e:
            print(f"Error converting JDTDB to datetime: {str(e)}")
            return
            
        # Step 4: Extract object information
        try:
            obj_name = df['object_name'].unique()[0]
            start_time = df['datetime'].min()
            end_time = df['datetime'].max()
            
        except KeyError:
            raise ValueError(f"Required metadata columns not found in {parquet_path}")
            
        # Step 5: Identify key metrics to visualize
        metrics = {
            'X': ('X Position', 'km'),
            'Y': ('Y Position', 'km'),
            'Z': ('Z Position', 'km'),
            'VX': ('X Velocity', 'km/s'),
            'VY': ('Y Velocity', 'km/s'),
            'VZ': ('Z Velocity', 'km/s'),
            'RG': ('Range from Earth', 'km')
        }
        
        # Step 6: Create visualization
        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Ephemeris Data for {obj_name}", fontsize=14, y=1.03)
        
        num_plots = len(metrics)
        current_plot = 1
        
        for metric, (metric_label, unit) in metrics.items():
            if metric not in df.columns:
                continue
                
            plt.subplot(num_plots, 1, current_plot)
            plt.plot(df['datetime'], df[metric], label=f"{metric_label} ({unit})", 
                    color='blue', linestyle='-', marker='', linewidth=0.8)
            
            plt.xlabel('Time (UTC)')
            plt.ylabel(f"{metric_label} [{unit}]")
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend()
            
            current_plot += 1
            
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing Parquet file {parquet_path}: {str(e)}")
        raise

def main():
    """
    Main function to analyze all Parquet files in the specified directory.
    """
    try:
        # Set default directory if not specified
        dir_name = "Horizons_archive"
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} not found")
            
        # Get all Parquet files in directory
        parquet_files = [f for f in os.listdir(dir_name) if f.endswith('.parquet')]
        
        if not parquet_files:
            print("No Parquet files found in the directory")
            return
            
        print(f"Analyzing {len(parquet_files)} Parquet files in {dir_name}")
        
        for file in parquet_files:
            file_path = os.path.join(dir_name, file)
            print(f"\nProcessing: {file}")
            analyze_parquet_file(file_path)
            
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
