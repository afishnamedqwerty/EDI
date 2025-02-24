from astropy.coordinates import CartesianDifferential
from astropy.time import Time
from astropy.units import Quantity
import sys
import json
import csv
import base64
import requests
import pandas as pd
import numpy as np
import timezone
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Constants
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
FORMAT = "json"
DEFAULT_START_TIME = "2023-01-01"
DEFAULT_STOP_TIME = "2024-01-01"
step_size = "1d"

class HorizonsApiClient:
    """A client class to interact with the Horizons API for SPK data retrieval."""
    
    def __init__(self, output_dir: str = "spk_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ephemeris(
        self,
        object_id: str,
        start_time: str = DEFAULT_START_TIME,
        stop_time: str = DEFAULT_STOP_TIME
    ) -> Dict:
        """
        Fetches ephemeris data for a given object over a specified time span.
    
        Args:
            object_id (str): The Horizons ID of the target object.
            start_time (str): Start date in 'YYYY-MM-DD' format.
            stop_time (str): End date in 'YYYY-MM-DD' format.
            step_size (str): Time interval between observations, e.g., '1d'.
        
        Returns:
            dict: Parsed JSON response from the Horizons API.
        """
        # Build the command string with necessary flags
        cmd = f"'{object_id}'"
    
        # Append NOFRAG and CAP flags for small bodies (e.g., asteroids, comets)
        if "P" in object_id or "-" in object_id:
            cmd += ";NOFRAG;CAP"
        
        params = {
            "format": FORMAT,
            "COMMAND": cmd,
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",  # Vector ephemeris
            "CENTER": "500@399",      # Earth-centered
            "START_TIME": start_time,
            "STOP_TIME": stop_time,
            "STEP_SIZE": step_size,
            #"CSV_FORMAT": "YES"
            #"QUANTITIES": "1,2,3,4"   # RA, DEC, delta, deldot
        }
        
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                #data = json.loads(response.text)
                output_path = self.preprocess_ephemeris(response.text, object_id)
                
                # Check for specific errors in the API response
                '''if "error" in data:
                    raise ValueError(f"API Error: {data['error']}")
                    
                return data'''
                print(f"{object_id} output: {response.text}")
                return output_path
                
            else:
                self.handle_error(response)
                return None
            
        except Exception as e:
            print(f"Error fetching ephemeris data for object {object_id}: {str(e)}")
            return None

    # Still unsure about parquet vs HDF5 tbh, Sundial formats Timebench to parquets so I'm now leaning that direction
    def preprocess_ephemeris(self, data: str, object_id: str) -> Optional[Path]:
        """
        Processes raw ephemeris data from JPL Horizons API and saves it as a Parquet file.
        
        Args:
            data (str): Raw response string from the Horizons API
            object_id (str): ID of the target object
            
        Returns:
            Path: Path to the saved Parquet file if successful. None otherwise.
        """
        try:
            # Find the positions of $$SOE and $$EOE markers
            soe_start = data.find("$$SOE")
            eoe_end = data.find("$$EOE") + 5  # Add 5 to include the marker in the slice
            
            if soe_start == -1 or eoe_end == -1:
                print(f"No SOE or EOE markers found for object {object_id}")
                return None
                
            # Extract the data section between $$SOE and $$EOE
            data_lines = data[soe_start:eoe_end].strip()
            
            if not data_lines:
                print(f"Empty data section for object {object_id}")
                return None
            
            # Identify header columns based on the first valid row
            headers = []
            first_valid_row = None
            
            for line in data_lines:
                parts = [x.strip() for x in line.split()]
                
                # Skip empty lines and metadata lines that contain non-numeric values
                if not parts or any(not x.replace('.', '', 1).isdigit() for x in parts):
                    continue
                    
                first_valid_row = parts
                break
                
            if first_valid_row is None:
                print(f"No valid data rows found for object {object_id}")
                return None
                
            # Define column names based on the structure of the ephemeris data
            headers = [
                'JDTDB',      # Julian Date in TDB
                'X',          # X position (km)
                'Y',          # Y position (km)
                'Z',          # Z position (km)
                'VX',         # X velocity (km/s)
                'VY',         # Y velocity (km/s)
                'VZ',         # Z velocity (km/s)
                'LT',         # Light travel time (seconds)
                'RG',         # Range (km)
                'RR'          # Range rate (km/s)
            ]
            
            # Parse data rows
            # is it useful to map position and velocity in the parquet beforehand or keep each coordinate idk questions
            data_rows = []
            for line in data_lines:
                parts = [x.strip() for x in line.split()]
                
                if not parts or any(not x.replace('.', '', 1).isdigit() for x in parts):
                    continue
                    
                # Convert string values to appropriate types
                jdtdb = float(parts[0])
                position = list(map(float, parts[1:4]))
                velocity = list(map(float, parts[4:7]))
                #x = float(parts[1])
                #y = float(parts[2])
                #z = float(parts[3])
                #vx = float(parts[4])
                #vy = float(parts[5])
                #vz = float(parts[6])
                lt = float(parts[7])
                rg = float(parts[8])
                rr = float(parts[9])

                # Calculate additional columns if needed
                ra, dec = self.cartesian_to_equatorial(position)
                #uncertainty_ra = np.nan  # Placeholder for uncertainty in RA (to be determined)
                #uncertainty_dec = np.nan  # Placeholder for uncertainty in Dec (to be determined)
                azimuth = np.nan           # For observer-based ephemeris
                elevation = np.nan         # For observer-based ephemeris
                distance = rg              # Use range as distance
                
                data_rows.append({
                    'time': self.jdtdb_to_datetime(jdtdb),
                    'object_id': object_id,
                    'ra': ra,
                    'dec': dec,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'vx': velocity[0],
                    'vy': velocity[1],
                    'vz': velocity[2],
                    #'uncertainty_ra': uncertainty_ra,
                    #'uncertainty_dec': uncertainty_dec,
                    'source': 'JPL Horizons API'
                })
                
            # Create DataFrame with the desired schema
            df = pd.DataFrame(data_rows)
            print(f"ayo df: {df}")
            df['JDTDB'] = pd.to_numeric(df['JDTDB'], downcast='float')
            
            # Add global metadata
            metadata = {
                #'object_name': self.get_object_name(object_id),
                'object_name': object_id,
                'reference_frame': 'J2000',
                'observer_location': 'Earth (399)',
                'start_time': DEFAULT_START_TIME,
                'end_time': DEFAULT_STOP_TIME
            }
            
            # Set data types for Parquet compatibility
            df = self.set_parquet_data_types(df)
            
            # Save to Parquet file with the specified naming convention
            filename = f"horizons_ephemeris_{object_id}_{DEFAULT_START_TIME.replace('-', '')}_{DEFAULT_STOP_TIME.replace('-', '')}.parquet"
            output_path = Path(__file__).parent / "processed_data" / filename
            
            df.to_parquet(
                output_path,
                index=False,
                compression='gzip',
                metadata=metadata
            )
            
            print(f"Successfully saved ephemeris data for {object_id} to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error processing ephemeris data for {object_id}: {str(e)}")
            return None
        
    def jdtdb_to_datetime(self, jdtdb: float) -> datetime:
        """
        Converts a Julian Date in TDB to a Python datetime object.
        """
        
        t = Time(jdtdb, format='jd', scale='tai')
        dt = t.to_datetime64()
        
        return datetime.fromtimestamp(dt.view('i8'), timezone.utc)
    
    def cartesian_to_equatorial(self, cart: list) -> tuple:
        """
        Converts Cartesian coordinates to Right Ascension and Declination.
        """
        
        x, y, z = cart
        ra = np.arctan2(y, x)
        dec = np.arcsin(z / np.hypot(x, y))
        
        return ra * 180 / np.pi, dec * 180 / np.pi
    
    def set_parquet_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sets data types for the Parquet file based on column types.
        """
        df = df.copy()
        
        # Set datetime type
        df['time'] = pd.to_datetime(df['time'])
        
        # Set floating-point types
        float_cols = [
            'ra', 'dec', 'azimuth', 'elevation',
            'distance', 'x', 'y', 'z', 'vx', 'vy', 'vz',
            'uncertainty_ra', 'uncertainty_dec'
        ]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Set categorical types
        df['object_id'] = df['object_id'].astype('category')
        df['source'] = df['source'].astype('category')
        
        return df
    
    def get_object_name(self, object_id: str) -> str:
        """
        Retrieves the name of the object from a predefined mapping.
        """
        # This would typically query an external database or API
        object_names = {
            '499': 'Mars',
            '502': 'Europa',
            '301': 'Asteroid Belt'
        }
        
        return object_names.get(object_id, 'Unknown Object')
    
    '''def preprocess_ephemeris_csv(self, data: str, object_id: str) -> Optional[Path]:
        """
        Processes the raw ephemeris data (in various formats) into a structured DataFrame and saves it to a CSV file.
        
        Args:
            data (str): The raw response string from the Horizons API.
            object_id (str): The ID of the target object for which the data was fetched.
                
        Returns:
            Path: The path to the saved CSV file if successful. None otherwise.
        """

        # Use the first valid row to determine column names
        columns = [
            'JDTDB',      # Julian Date in TDB
            'X',          # X position (km)
            'Y',          # Y position (km)
            'Z',          # Z position (km)
            'VX',         # X velocity (km/s)
            'VY',         # Y velocity (km/s)
            'VZ',         # Z velocity (km/s)
            'LT',         # Light travel time (s)
            'RG',         # Range (km)
            'RR'          # Range rate (km/s)
            ]
        try:
            # Find the positions of $$SOE and $$EOE markers
            soe_start = data.find("$$SOE")
            eoe_end = data.find("$$EOE") + 5  # Add 5 to include the marker in the slice
            
            if soe_start == -1 or eoe_end == -1:
                print(f"No SOE or EOE markers found for object {object_id}")
                return None
                
            # Extract the data section between $$SOE and $$EOE
            data_section = data[soe_start:eoe_end].strip()
            
            if not data_section:
                print(f"Empty data section for object {object_id}")
                return None
            
            # Parse the CSV-formatted data section
            df = pd.read_csv(StringIO(data_section), delimiter=',', header=0)
            print(f"preprocess df: {df}")
            
            # Handle cases where columns might be missing or malformed
            if not df.empty:
                # Convert time strings to datetime objects
                try:
                    df['Time'] = pd.to_datetime(df['Time'])
                except KeyError:
                    print(f"Time column not found in data for object {object_id}")
                    return None
                
                # Save the DataFrame to a CSV file (one per object)
                output_path = self.output_dir / f"{object_id}_ephemeris.csv"
                df.to_csv(output_path, index=False)
                
                print(f"Successfully saved ephemeris data for {object_id} to {output_path}")
                return output_path
            
            print(f"No valid data found for object {object_id}")
            return None
            
        except Exception as e:
            print(f"Error preprocessing ephemeris data for {object_id}: {str(e)}")
            return None'''

    def fetch_spk(
        self,
        spkid: str,
        start_time: str = DEFAULT_START_TIME,
        stop_time: str = DEFAULT_STOP_TIME
    ) -> Optional[Path]:
        """Fetches an SPK file for a given object from the Horizons API."""
        try:
            # Build the command string with necessary flags
            cmd = f"'{spkid}'"
            
            # Append NOFRAG and CAP flags if dealing with small bodies
            if "P" in spkid or "-" in spkid:
                cmd += ";NOFRAG;CAP"
                
            url = (
                f"{API_URL}"
                f"?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
                f"&COMMAND={cmd}&START_TIME={start_time}&STOP_TIME={stop_time}"
            )
            
            # Submit the API request
            response = requests.get(url)
            if response.status_code != 200:
                self.handle_error(response)
                return None
                
            data = json.loads(response.text)
            
            # Check if SPK file was generated
            if "spk" not in data:
                print("ERROR: SPK file not generated for object:", spkid)
                if "result" in data:
                    print(data["result"])
                else:
                    print(response.text)
                return None
                
            # Use suggested filename if available
            spk_filename = data.get("spk_file_id", spkid) + ".bsp"
            
            # Save the SPK file
            file_path = self.output_dir / spk_filename
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data["spk"]))
                
            print(f"Successfully wrote SPK content to {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error fetching data for object {spkid}: {str(e)}")
            return None
    
    def handle_error(self, response: requests.Response) -> None:
        """Handles errors returned by the Horizons API."""
        if response.status_code == 400:
            try:
                data = json.loads(response.text)
                if "message" in data:
                    print("MESSAGE:", data["message"])
                else:
                    print(json.dumps(data, indent=2))
            except ValueError:
                print("Unable to decode JSON error response")
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")

def main():
    # List of objects to fetch (example list)
    objects_to_fetch = [
        #"399",     # Earth
        "502",     # Europa
        #"401",     # Moon
        #"2000001", # Ceres
        #"2000002", # Pallas
        #"2000003", # Vesta
        #"141P",    # Comet Machholz 2
        #"DELD98A"  # Example asteroid designation
    ]
    
    # Initialize the Horizons client
    client = HorizonsApiClient(output_dir="Horizons_archive/spk_files")
    
    for obj in objects_to_fetch:
        print(f"\nFetching spk_data for object: {obj}")
        result = client.fetch_spk(spkid=obj)
        if not result:
            continue

    for obj in objects_to_fetch:
        print(f"\nFetching ephemeris_data for {obj}: ")
        result = client.fetch_ephemeris(object_id=obj)
        print(f"\n Payload: {result}")
        if result is None:
            print("Failed to fetch ephemeris data for object:", obj)
            continue
            
        '''df = client.preprocess_ephemeris(result, object_id=obj)
        if df is not None and not df.empty:
            print(f"\nDataframe Shape: {df.shape}")
            print("\nFirst few rows of the dataframe:")
            print(df.head())
        else:
            print("No valid data was processed for object:", obj)'''


        
if __name__ == "__main__":
    main()
