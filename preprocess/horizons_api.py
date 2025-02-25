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
#import timezone
import traceback
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pytz
import re
#import plt

# Constants
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
FORMAT = "json"
DEFAULT_START_TIME = "2023-01-01"
DEFAULT_STOP_TIME = "2024-01-01"
step_size = "1d" #for actual training we'll want granularity (1min or 10min step size)

class HorizonsApiClient:
    """A client class to interact with the Horizons API for Ephemeris and SPK data retrieval."""
    
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
        
        # Example QUANTITIES for OBSERVER Type
        #    The QUANTITIES parameter can be set to a comma-separated list of integers representing the desired observable quantities. For example:

        #    1: RA (Right Ascension) in hours.
        #    2: Dec (Declination) in degrees.
        #    3: Azimuth in degrees.
        #    4: Elevation in degrees.
        #    5: Range in kilometers.
        #    9: Range rate in km/s.
        #    20: Uncertainty in RA (arcseconds).
        #    23: Uncertainty in Dec (arcseconds).
        #    24: Uncertainty in Azimuth (arcminutes).
        #    29: Uncertainty in Elevation (arcminutes).

        obs_params = {
            "format": FORMAT,
            "COMMAND": cmd,
            "OBJECT_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",  # Vector ephemeris
            "CENTER": "500@399",      # Earth-centered
            "START_TIME": start_time,
            "STOP_TIME": stop_time,
            "STEP_SIZE": step_size,
            "ANG_FORMAT": "DEG",
            "EXTRA_PREC": "YES",
            #"CSV_FORMAT": "YES"
            "QUANTITIES": "1,2,3,4,5,9,20,23,24,29"   # RA, DEC, delta, deldot
        }

        vec_params = {
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

        # Example QUANTITIES for ELEMENTS Type
        # The QUANTITIES parameter can be set to a comma-separated list of integers representing the desired orbital elements. For example:

        # 1: RA of ascending node (Ω) in degrees.
        # 2: Inclination (i) in degrees.
        # 3: Eccentricity (e).
        # 4: Argument of perihelion (ω) in degrees.
        # 5: Mean anomaly (M0) in degrees.
        # 6: Semi-major axis (a) in AU.
        # 7: Longitude of ascending node (Ω) rate in arcseconds/year.
        # 8: Inclination rate in arcseconds/year.
        # 9: Eccentricity rate in dimensionless units.
        # 10: Argument of perihelion rate in arcseconds/year.
        # 11: Mean anomaly rate in degrees/year.
        
        elem_params = {
            "format": FORMAT,
            "COMMAND": cmd,
            "OBJECT_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "ELEMENTS",  # Vector ephemeris
            "CENTER": "0@399",      # Earth-centered
            "START_TIME": start_time,
            "STOP_TIME": stop_time,
            "STEP_SIZE": step_size,
            "ANG_FORMAT": "DEG",
            "EXTRA_PREC": "YES",
            #"CSV_FORMAT": "YES"
            "QUANTITIES": "1,2,3,4,5,6,7,8,9,10,11"   # RA, DEC, delta, deldot
        }
        
        try:
            response = requests.get(API_URL, params=vec_params)
            if response.status_code == 200:
                #data = json.loads(response.text)
                output_path = self.preprocess_ephemeris(response.text, object_id)
                
                # Check for specific errors in the API response
                '''if "error" in data:
                    raise ValueError(f"API Error: {data['error']}")
                    
                return data'''
                #print(f"{object_id} output: {response.text}")
                return output_path
                
            else:
                self.handle_error(response)
                return None
            
        except Exception as e:
            print(f"Error fetching ephemeris data for object {object_id}: {str(e)}")
            return None
        
    # Preprocesses vec_params ephemeris data into dataframe parquet file
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
            eoe_end = data.find("$$EOE")
            
            if soe_start == -1 or eoe_end == -1:
                print(f"No SOE or EOE markers found for object {object_id}")
                return None
                
            # Extract the data section between $$SOE and $$EOE
            raw_data = data[soe_start+5:eoe_end].strip()
            
            if not raw_data:
                print(f"Empty data section for object {object_id}")
                return None
            
            # Print raw data information for debugging
            print(f"Raw data length: {len(raw_data)}")
            print(f"First 200 chars: {raw_data[:200]}")
            
            # Approach: Use regex to find each data point directly
            
            # Find all JDTDB entries
            jdtdb_pattern = r"(\d+\.\d+)\s*=\s*A\.D\."
            jdtdb_matches = re.finditer(jdtdb_pattern, raw_data)
            
            # Store positions of each match to determine data blocks
            jdtdb_positions = [(m.start(), m.group(1)) for m in jdtdb_matches]
            
            print(f"Found {len(jdtdb_positions)} JDTDB entries")
            
            if len(jdtdb_positions) == 0:
                print("No JDTDB values found in data")
                return None
            
            # Initialize list to store parsed data
            data_rows = []
            
            # Process each data block
            for i, (pos, jdtdb_str) in enumerate(jdtdb_positions):
                try:
                    # Determine the end position of this block (start of next block or end of data)
                    end_pos = jdtdb_positions[i+1][0] if i < len(jdtdb_positions)-1 else len(raw_data)
                    
                    # Extract the entire data block for this time point
                    block = raw_data[pos:end_pos]
                    
                    # Parse JDTDB (Julian Date)
                    jdtdb = float(jdtdb_str)
                    
                    # Extract X, Y, Z using regex
                    x_match = re.search(r'X\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    y_match = re.search(r'Y\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    z_match = re.search(r'Z\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    
                    if not all([x_match, y_match, z_match]):
                        print(f"Could not find X, Y, Z in block {i}")
                        continue
                    
                    x = float(x_match.group(1))
                    y = float(y_match.group(1))
                    z = float(z_match.group(1))
                    
                    # Extract VX, VY, VZ using regex
                    vx_match = re.search(r'VX\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    vy_match = re.search(r'VY\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    vz_match = re.search(r'VZ\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    
                    if not all([vx_match, vy_match, vz_match]):
                        print(f"Could not find VX, VY, VZ in block {i}")
                        continue
                    
                    vx = float(vx_match.group(1))
                    vy = float(vy_match.group(1))
                    vz = float(vz_match.group(1))
                    
                    # Extract LT, RG, RR using regex
                    lt_match = re.search(r'LT\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    rg_match = re.search(r'RG\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    rr_match = re.search(r'RR\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    
                    if not all([lt_match, rg_match, rr_match]):
                        print(f"Could not find LT, RG, RR in block {i}")
                        continue
                    
                    lt = float(lt_match.group(1))
                    rg = float(rg_match.group(1))
                    rr = float(rr_match.group(1))
                    
                    # Convert JDTDB to datetime
                    dt = self.jdtdb_to_datetime(jdtdb)
                    
                    # Add row to data
                    data_rows.append({
                        'JDTDB': jdtdb,
                        'datetime': dt,
                        'X': x,
                        'Y': y,
                        'Z': z,
                        'VX': vx,
                        'VY': vy,
                        'VZ': vz,
                        'LT': lt,
                        'RG': rg,
                        'RR': rr
                    })
                    
                    # Print progress
                    if i % 50 == 0:
                        print(f"Processed {i} entries")
                    
                except Exception as e:
                    print(f"Error parsing block {i}: {str(e)}")
                    # Continue to next block
            
            # Create DataFrame
            if not data_rows:
                print(f"No data rows were parsed for object {object_id}")
                return None
            
            df = pd.DataFrame(data_rows)
            print(f"\nSuccessfully parsed {len(data_rows)} data points for {object_id}")
            
            # Get object name
            obj_name = self.get_object_name(object_id)
            print(f"\nObject ID {object_id} is {obj_name}\n")
            
            # Add metadata columns
            df['object_id'] = object_id
            df['object_name'] = obj_name
            df['reference_frame'] = 'Ecliptic of J2000.0'

            print(f"\nObject ID {object_id} df contents: {df}\n")
            
            # Save to Parquet file
            filename = f"ephemeris_{obj_name}_{DEFAULT_START_TIME.replace('-', '')}_{DEFAULT_STOP_TIME.replace('-', '')}.parquet"
            output_path = Path(__file__).parent / "Horizons_archive" / filename
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame to Parquet
            df.to_parquet(
                output_path,
                index=False,
                compression='gzip',
                engine='pyarrow'
            )
            
            print(f"Successfully saved ephemeris data for {object_id} to {output_path}")
            print(f"DataFrame contains {len(df)} rows with columns: {df.columns.tolist()}")
            
            return output_path
            
        except Exception as e:
            print(f"Error processing ephemeris data for {object_id}: {str(e)}")
            traceback.print_exc()
            return None

    def preprocess_ephemeris_test(self, data: str, object_id: str) -> Optional[Path]:
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
            raw_data = data[soe_start:eoe_end].strip()
            
            if not raw_data:
                print(f"Empty data section for object {object_id}")
                return None
                
            # Split the raw data into individual lines/entries
            #lines = raw_data.split('\n')
            
            # Initialize lists to store parsed data
            headers = []
            data_rows = []
            
            # Process each line to extract key-value pairs
            for line in raw_data.split('\n'):
                # Split the line into components (e.g., "X = value")
                parts = [x.strip() for x in line.split()]
                print(f"\nLine: {line} and parts {parts}")
                
                # Skip lines that don't contain valid numeric values
                if not parts or any(not x.replace('.', '', 1).isdigit() for x in parts):
                    continue
                data_rows.append(parts)
                    
                # Define column names based on the structure of the ephemeris data
            headers = [
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

            # Parse data rows
            data_rows = []
            for line in raw_data:
                parts = [x.strip() for x in line.split()]
                
                if not parts or any(not x.replace('.', '', 1).isdigit() for x in parts):
                    continue
                    
                jdtdb = float(parts[0])
                position = list(map(float, parts[1:4]))
                velocity = list(map(float, parts[4:7]))
                lt = float(parts[7])
                rg = float(parts[8])
                rr = float(parts[9])

                # Calculate additional columns if needed
                ra, dec = self.cartesian_to_equatorial(position)
                #uncertainty_ra = np.nan  # Placeholder for uncertainty in RA (to be determined)
                #uncertainty_dec = np.nan  # Placeholder for uncertainty in Dec (to be determined)
                #azimuth = np.nan           # For observer-based ephemeris
                #elevation = np.nan         # For observer-based ephemeris
                #distance = rg              # Use range as distance
                
                data_rows.append({
                    'time': self.jdtdb_to_datetime(jdtdb),
                    #'object_id': object_id,
                    #'ra': ra,
                    #'dec': dec,
                    #'azimuth': azimuth,
                    #'elevation': elevation,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'vx': velocity[0],
                    'vy': velocity[1],
                    'vz': velocity[2],
                    'lt': lt,
                    'rg': rg,
                    'rr': rr
                    #'uncertainty_ra': uncertainty_ra,
                    #'uncertainty_dec': uncertainty_dec,
                    #'source': 'JPL Horizons API'
                })
                
            # Create DataFrame with the desired schema
            #df = pd.DataFrame(data_rows)


                    
            # Create DataFrame from parsed data
            print(f"\nPre-dataframe data_rows: {data_rows}")
            df = pd.DataFrame(data_rows, columns=headers)
            obj_name = self.get_object_name(object_id)
            print(f"\nObject ID {object_id} is {obj_name}\n")
            print(f"\nPre-parquet dataframe for {obj_name}: {df}")
            
            # Set data types for Parquet compatibility
            #df = self.set_parquet_data_types(df)
                        # Add global metadata
            '''metadata = {
                #'object_name': self.get_object_name(object_id),
                'object_name': object_id,
                'reference_frame': 'J2000',
                'observer_location': 'Earth (399)',
                'start_time': DEFAULT_START_TIME,
                'end_time': DEFAULT_STOP_TIME
            }'''
            
            # Save to Parquet file with the specified naming convention
            filename = f"ephemeris_{obj_name}_{DEFAULT_START_TIME.replace('-', '')}_{DEFAULT_STOP_TIME.replace('-', '')}.parquet"
            output_path = Path(__file__).parent / "Horizons_archive" / filename
            
            df.to_parquet(
                output_path,
                index=False,
                compression='gzip',
                engine='pyarrow'
            )
            
            print(f"Successfully saved ephemeris data for {object_id} to {output_path}")
            dataf = pd.read_parquet(output_path)
            print(f"\n Dataframe: {dataf}")
            return output_path
            
        except Exception as e:
            print(f"Error processing ephemeris data for {object_id}: {str(e)}")
            return None

        
    def jdtdb_to_datetime(self, jdtdb: float) -> datetime:
        """
        Converts a Julian Date in TDB to a Python datetime object.
        """
        t = Time(jdtdb, format='jd', scale='tdb')
        return t.datetime.replace(tzinfo=pytz.UTC)
    
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
        # X99 suffix denotes a Planet
        # 5XX or 5XXXX prefix denotes a Jovian Satellite
        # 6XX or 6XXXX prefix denotes a Saturnian Satellite
        # 7XX or 7XXXX prefix denotes a Uranian Satellite
        # 8XX or 8XXXX prefix denotes a Neptunian Satellite

        object_names = {
            '10': 'Sun',
            '199': 'Mercury',
            '299': 'Venus',
            '301': 'Moon(Luna)',
            '399': 'Earth',
            '401': 'Phobos(MI)',
            '402': 'Deimos(MII)',
            '499': 'Mars',
            '501': 'Io(JI)',
            '502': 'Europa(JII)',
            '503': 'Ganymede(JIII)',
            '504': 'Callisto(JIV)',
            '505': 'Amalthea(JV)',
            '506': 'Himalia',
            '507': 'Elara(JVII)',
            '508': 'Pasiphae(JVIII)',
            '509': 'Sinope(JIX)',
            '510': 'Lysithea(JX)',
            '511': 'Carme(JXI)',
            '512': 'Ananke(JXII)',
            '513': 'Leda(JXIII)',
            '514': 'Thebe(JXIV)',
            '515': 'Adrastea(JXV)',
            '516': 'Metis(JXVI)',
            '517': 'Callirrhoe(JXVII=199J1)',
            '518': 'Themisto(JXVIII=1975J1)',
            '519': 'Megaclite(JXIX=2000J8)',
            '520': 'Taygete(JXX=2000J9)',
            '521': 'Chaldene(JXXI=2000J10)',
            '522': 'Harpalyke(JXXII=2000J5)',
            '523': 'Kalyke(JXXIII=2000J2)',
            '524': 'Iocaste(JXXIV=2000J3)',
            '525': 'Erinome(JXXV=2000J4)',
            '526': 'Isonoe(JXXVI=2000J6)',
            '527': 'Praxidike(JXXVII=2000J7)',
            '528': 'Autonoe(JXXVIII=2001J1)',
            '529': 'Thyone(JXXIX=2001J2)',
            '530': 'Hermippe(JXXX=2001J3)',
            '531': 'Aitne(JXXXI=2001J11)',
            '532': 'Eurydome(JXXXII=2001J4)',
            '533': 'Euanthe(JXXXIII=2001J7)',
            '534': 'Euporie(JXXXIV=2001J10)',
            '535': 'Orthosie(JXXXV=2001J9)',
            '536': 'Sponde(JXXXVI=2001J5)',
            '537': 'Kale(JXXXVII=2001J8)',
            '538': 'Pasithee(JXXXVIII=2001J6)',
            '539': 'Hegemone(JXXXIX=2003J8)',
            '540': 'Mneme(JXL=2003J21)',
            '541': 'Aoede(JXLI=2003J7)',
            '542': 'Thelxinoe(JXLII=2003J22)',
            '543': 'Arche(JXLIII=2002J1)',
            '544': 'Kallichore(JXLIV=2003J11)',
            '545': 'Helike(JXLV=2003J6)',
            '546': 'Carpo(JXLVI=2003J20)',
            '547': 'Eukelade(JXLVII=2003J1)',
            '548': 'Cyllene(JXLVIII=2003J13)',
            '549': 'Kore(JXLIX=2003J14)',
            '550': 'Herse(JXLXII=2003J17)',
            '551': '2010J1',
            '552': '2010J2',
            '553': 'Dia(JLIII=2000J11)',
            '554': '2016J1',
            '555': '2003J18',
            '556': '2011J2',
            '557': 'Eirene(JLVII=2003J5)',
            '558': 'Philophrosyne(JLVIII=2003J15)',
            '559': '2017J1',
            '560': 'Eupheme(JLX=2003J3)',
            '561': '2003J19',
            '562': 'Valetudo(2016J2)',
            '563': '2017J2',
            '564': '2017J3',
            '565': 'Pandia(JLXV=2017J4)',
            '566': '2017J5',
            '567': '2017J6',
            '568': '2017J7',
            '569': '2017J8',
            '570': '2017J9',
            '571': 'Ersa(2018J1)',
            '572': '2011J1',
            '55501': '2003J2',
            '55502': '2003J4',
            '55503': '2003J9',
            '55504': '2003J10',
            '55505': '2003J12',
            '55506': '2003J16',
            '55507': '2003J23',
            '55508': '2003J24',
            '55509': '2011J3',
            '55510': '2018J2',
            '55511': '2018J3',
            '55512': '2021J1',
            '55513': '2021J2',
            '55514': '2021J3',
            '55515': '2021J4',
            '55516': '2021J5',
            '55517': '2021J6',
            '55518': '2016J3',
            '55519': '2016J4',
            '55520': '2018J4',
            '55521': '2022J1',
            '55522': '2022J2',
            '55523': '2022J3',
            '599': 'Jupiter',
            '601': 'Mimas(SI)',
            '602': 'Enceladus(SII)',
            '603': 'Tethys(SIII)',
            '604': 'Dione(SIV)',
            '605': 'Rhea(SV)',
            '606': 'Titan(SVI)',
            '607': 'Hyperion(SVII)',
            '608': 'Iapetus(SVIII)',
            '609': 'Phoebe(SIX)',
            '610': 'Janus(SX)',
            '611': 'Epimetheus(SXI)',
            '612': 'Helene(SXII)',
            '613': 'Telesto(SXIII)',
            '614': 'Calypso(SXIV)',
            '615': 'Atlas(SXV)',
            '616': 'Prometheus(SXVI)',
            '617': 'Pandora(SXVII)',
            '618': 'Pan(SXVIII)',
            '619': 'Ymir(SXIX=2000S1)',
            '620': 'Paaliaq(SXX=2000S2)',
            '621': 'Tarvos(SXXI=2000S4)',
            '622': 'Ijiraq(SXXII=2000S6)',
            '623': 'Suttungr(SXXIII=2000S12)',
            '624': 'Kiviuq(SXXIV=2000S5)',
            '625': 'Mundilfari(SXXV=2000S9)',
            '626': 'Albiorix(SXXVI=2000S11)',
            '627': 'Skathi(SXXVII=2000S8)',
            '628': 'Erriapus(SXXVIII=2000S10)',
            '629': 'Siarnaq(SXXIX=2000S3)',
            '630': 'Thrymr(SXXX=2000S7)',
            '631': 'Narvi(SXXXI=2003S1)',
            '632': 'Methone(SXXXII=2004S1)',
            '633': 'Pallene(SXXXIII=2004S2)',
            '634': 'Polydeuces(SXXXIV=2004S5)',
            '635': 'Daphnis(SXXXV=2005S1)',
            '636': 'Aegir(SXXXVI=2004S10)',
            '637': 'Bebhionn(SXXXVII=2004S11)',
            '638': 'Bergelmir(SXXXVIII=2004S15)',
            '639': 'Bestla(SXXXIX=2004S18)',
            '640': 'Farbauti(SXL=2004S9)',
            '641': 'Fenrir(SXLI=2004S16)',
            '642': 'Fornjot(SXLII=2004S8)',
            '643': 'Hati(SXLIII=2004S14)',
            '644': 'Hyrrokkin(SXLIV=2004S19)',
            '645': 'Kari(SXLV=2006S2)',
            '646': 'Loge(SXLVI=2006S5)',
            '647': 'Skoll(SXLVII=2006S8)',
            '648': 'Surtur(SXLVIII=2006S7)',
            '649': 'Anthe(SXLIX=2007S4)',
            '650': 'Jarnsaxa(SL=2006S6)',
            '651': 'Greip(SLI=2006S4)',
            '652': 'Tarqeq(SLII=2007S1)',
            '653': 'Aegaeon(SLIII=2008S1)',
            '654': 'Gridr(65080=2004S20)',
            '655': 'Angrboda(65073=2004S22)',
            '656': 'Skrymir(65071=2004S23)',
            '657': 'Gerd(65072=2004S25)',
            '658': '2004S26',
            '659': 'Eggther(2004S27)',
            '660': '2004S29',
            '661': 'Beli(65078=2004S30)',
            '662': 'Gunnlod(65074=2004S32)',
            '663': 'Thiazzi(65075=2004S33)',
            '664': '2004S34',
            '665': 'Alvaldi(65069=2004S35)',
            '666': 'Geirrod(65083=2004S38)',
            '65067': '2004S31',
            '65070': '2004S24',
            '65077': '2004S28',
            '65079': '2004S21',
            '65081': '2004S36',
            '65082': '2004S37',
            '65084': '2004S39',
            '65085': '2004S7',
            '65086': '2004S12',
            '65087': '2004S13',
            '65088': '2004S17',
            '65089': '2006S1',
            '65090': '2006S3',
            '65091': '2007S2',
            '65092': '2007S3',
            '65093': '2019S1',
            '65094': '2019S2',
            '65095': '2019S3',
            '65096': '2020S1',
            '65097': '2020S2',
            '65098': '2004S40',
            '65100': '2006S9',
            '65101': '2007S5',
            '65102': '2020S3',
            '65103': '2019S4',
            '65104': '2004S41',
            '65105': '2020S4',
            '65106': '2020S5',
            '65107': '2007S6',
            '65108': '2004S42',
            '65109': '2006S10',
            '65110': '2019S5',
            '65111': '2004S43',
            '65112': '2004S44',
            '65113': '2004S45',
            '65114': '2006S11',
            '65115': '2006S12',
            '65116': '2019S6',
            '65117': '2006S13',
            '65118': '2019S7',
            '65119': '2019S8',
            '65120': '2019S9',
            '65121': '2004S46',
            '65122': '2019S10',
            '65123': '2004S47',
            '65124': '2019S11',
            '65125': '2006S14',
            '65126': '2019S12',
            '65127': '2020S6',
            '65128': '2019S13',
            '65129': '2005S4',
            '65130': '2007S7',
            '65131': '2007S8',
            '65132': '2020S7',
            '65133': '2019S14',
            '65134': '2019S15',
            '65135': '2005S5',
            '65136': '2006S15',
            '65137': '2006S16',
            '65138': '2006S17',
            '65139': '2004S48',
            '65140': '2020S8',
            '65141': '2004S49',
            '65142': '2004S50',
            '65143': '2006S18',
            '65144': '2019S16',
            '65145': '2019S17',
            '65146': '2019S18',
            '65147': '2019S19',
            '65148': '2019S20',
            '65149': '2006S19',
            '65150': '2004S51',
            '65151': '2020S9',
            '65152': '2004S52',
            '65153': '2007S9',
            '65154': '2004S53',
            '65155': '2020S10',
            '65156': '2019S21',
            '65157': '2006S20',
            '699': 'Saturn',
            '701': 'Ariel(UI)',
            '702': 'Umbriel(UII)',
            '703': 'Titania(UIII)',
            '704': 'Oberon(UIV)',
            '705': 'Miranda(UV)',
            '706': 'Cordelia(UVI)',
            '707': 'Ophelia(UVII)',
            '708': 'Bianca(UVIII)',
            '709': 'Cressida(UIX)',
            '710': 'Desdemona(UX)',
            '711': 'Juliet(UXI)',
            '712': 'Portia(UXII)',
            '713': 'Rosalind(UXIII)',
            '714': 'Belinda(UXIV)',
            '715': 'Puck(UXV)',
            '716': 'Caliban(UXVI)',
            '717': 'Sycorax(UXVII)',
            '718': 'Prospero(UXVIII=1999U3)',
            '719': 'Setebos(UXIX=1999U1)',
            '720': 'Stephano(UXX=1999U2)',
            '721': 'Trinculo(UXXI=2001U1)',
            '722': 'Francisco(UXXII=2001U3)',
            '723': 'Margaret(UXXIII=2003U3)',
            '724': 'Ferdinand(UXXIV=2001U2)',
            '725': 'Perdita(UXXV=1986U10)',
            '726': 'Mab(UXXVI=2003U1)',
            '727': 'Cupid(UXXVII=2003U2)',
            '75051': '2023U1',
            '799': 'Uranus', # non-sexual
            '801': 'Triton(NI)',
            '802': 'Nereid(NII)',
            '803': 'Naiad(NIII)',
            '804': 'Thalassa(NIV)',
            '805': 'Despina(NV)',
            '806': 'Galatea(NVI)',
            '807': 'Larissa(NVII)',
            '808': 'Proteus(NVIII)',
            '809': 'Halimede(2002N1)',
            '810': 'Psamathe(2003N1)',
            '811': 'Sao(2002N2)',
            '812': 'Laomedeia(2002N)',
            '813': 'Neso(2002N4)',
            '814': 'Hippocamp(2004N1)',
            '85051': '2002N5',
            '85052': '2021N1',
            '899': 'Neptune',
            '901': 'Charon(PI)',
            '902': 'Nix(PII)',
            '903': 'Hydra(PIII)',
            '904': 'Kerberos(2011P1)',
            '905': 'Styx(2012P1)',
            #'301': 'Asteroid Belt'
        }
        
        return object_names.get(object_id, 'Unknown Object')
    

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
        "301",      # Moon
        #"399",     # Earth
        #"401",     # Phobos
        "502",     # Europa
        #"503"      # Ganymede
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
        #print(f"\n Payload: {result}")
        if result is None:
            #print(f"\n Analyzing data for object: {obj}")
            #client.analyze_parquet_file(result)
            #print("Failed to fetch ephemeris data for object:", obj)
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
