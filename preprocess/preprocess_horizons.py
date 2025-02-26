import pandas as pd
import glob
from pathlib import Path
import pyarrow.parquet as pq

# Constants
ARCHIVE_DIR = Path(__file__).parent / "Horizons_archive"
OUTPUT_PREFIX = "ephemeris_vec_"
DEFAULT_START_TIME = "2023-01-01"  # Replace with actual start time
DEFAULT_STOP_TIME = "2024-12-31"   # Replace with actual stop time

def merge_eph_vec_parquet_files():
    """
    Merges all parquet files with the specified prefix into a single parquet file.
    """
    try:
        # Get all parquet files matching the prefix
        pattern = f"{OUTPUT_PREFIX}*.parquet"
        files = list(ARCHIVE_DIR.glob(pattern))
        
        if not files:
            print(f"No files found with prefix {OUTPUT_PREFIX}")
            return
            
        # Read each parquet file and merge into a single dataframe
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            # Add a column to track the original filename
            #df['source_file'] = file.name
            dfs.append(df)
            
        merged_df = pd.concat(dfs, axis=0, keys=[file.name for file in files])
        
        # Create output directory if it doesn't exist
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{OUTPUT_PREFIX}mb_{DEFAULT_START_TIME}_{DEFAULT_STOP_TIME}.parquet"
        output_path = ARCHIVE_DIR / output_filename
        
        # Save merged dataframe to parquet
        merged_df.to_parquet(
            output_path,
            index=False,
            compression='gzip',
            engine='pyarrow'
        )
        
        print(f"Successfully merged and saved to {output_path}")
        print(f"Merged dataframe contains {len(merged_df)} rows with columns: {merged_df.columns.tolist()}")

        return output_path
        
    except Exception as e:
        print(f"Error merging parquet files: {str(e)}")
        raise

if __name__ == "__main__":
    output_path = merge_eph_vec_parquet_files()
    dataf = pd.read_parquet(output_path)
    print(f"\n{dataf}")
