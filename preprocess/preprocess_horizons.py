import pandas as pd
import glob
from pathlib import Path
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler, StandardScaler
#from sklearn import CanonicalCorrelationAnalysis

# Constants
ARCHIVE_DIR = Path(__file__).parent / "Horizons_archive"
OUTPUT_PREFIX = "ephemeris_vec_"
DEFAULT_START_TIME = "2023-01-01"  # Replace with actual start time
DEFAULT_STOP_TIME = "2024-12-31"   # Replace with actual stop time

# Normalize each feature independently
# 
def normalize_features(df):
    """
    Perform z-score standardization or robust scaling on numerical features.
    """
    # Select only numerical columns (features)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = RobustScaler()
    # Split into features and labels (if any)
    X = df[numeric_cols].values
    normalized_X = scaler.fit_transform(X)
    # Create a new DataFrame with normalized values
    df_normalized = df.copy()
    df_normalized[numeric_cols] = normalized_X

    return df_normalized

def apply_cca(df, features1, features2):
    """
    Apply Canonical Correlation Analysis to two sets of features.
    """
    cca = CanonicalCorrelationAnalysis(n_components=5)  # Adjust n_components as needed
    X = df[features1].values
    Y = df[features2].values
    
    x_scores = cca.fit_transform(X, Y)
    y_scores = cca.transform(Y)
    
    # Create new features from CCA components
    df_cca = pd.DataFrame({
        f'CCA_{i+1}': x_scores[:, i] for i in range(cca.n_components_)
    })
    
    return df_cca

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
            # Ensure the 'JDTDB' column exists in all dataframes
            if 'JDTDB' not in df.columns:
                raise ValueError(f"File {file.name} does not contain the required 'JDTDB' column")
            
            required_columns = ['JDTDB', 'datetime', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"File {file.name} is missing columns: {missing_cols}")
            # Add a column to track the original filename
            #df['source_file'] = file.name
            missing_values = df.isna().sum()
            #print(missing_values)
            df_normalized = normalize_features(df)
            # Apply CCA on selected feature groups
            features1 = ['X', 'Y', 'Z']
            features2 = ['VX', 'VY', 'VZ']
            #cca_df = apply_cca(df_normalized, features1, features2)

            # Merge normalized and CCA-transformed data
            #df_processed = pd.concat([df_normalized, cca_df], axis=1)
            dfs.append(df) #df_processed
            
        merged_df = pd.concat(dfs, axis=0, keys=[file.name for file in files])

        # Reset index to ensure proper ordering
        merged_df.reset_index(drop=True, inplace=True)
        
        # Sort the dataframe by the 'JDTDB' column values
        if 'JDTDB' in merged_df.columns:
            merged_df.sort_values(by='JDTDB', ascending=True, inplace=True)
        else:
            raise ValueError("The merged dataframe does not contain the required 'JDTDB' column")
        
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
    print(f"\n{dataf}") #dataf.head(2000)
