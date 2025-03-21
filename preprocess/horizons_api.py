from astropy.coordinates import CartesianDifferential
from astropy.time import Time
from astropy.units import Quantity
import sys
import json
import csv
import base64
import requests
import pandas as pd
import math
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

    def fetch_ephemeris_obs(
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
            cmd += ";NOFRAG;" #CAP;"
        
        # Specific Quantities (https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan)
        # 1. Astrometric RA & DEC
        # 2. Apparent RA & DEC
        # 3. Rates: RA & DEC
        # 4. Apparent azimuth & elevation (AZ-EL)
        # 5. Rates: azimuth and elevation (AZ-EL)
        # 6. X & Y satellite offset & position angle
        # 7. Local apparent sidereal time
        # 8. Airmass & visual magnitude extinction
        # 9. Visual magnitude & surface brightness
        # 10. Illuminated fraction
        # 11. Defect of illumination
        # 12. Angular separation/visibility
        # 13. Target angular diameter
        # 14. Observer sub-longitude & sub-latitude
        # 15. Solar sub-longitude & sub-latitude
        # 16. Sub-solar position angle & distance from disc center
        # 17. North pole position angle & distance from disc center
        # ...
        # 48. Sky brightness and target visual signal-to-noise ratio (SNR)


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

        quantities = [1,2,4,5,9]
        n = "1"

        obs_params = {
            "format": FORMAT,
            "COMMAND": cmd,
            #"OBJECT_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",  # Vector ephemeris
            "CENTER": "500@399",      # Earth-centered
            "START_TIME": start_time,
            "STOP_TIME": stop_time,
            "STEP_SIZE": step_size,
            "ANG_FORMAT": "DEG",
            "EXTRA_PREC": "YES",
            #"CSV_FORMAT": "YES"
            "QUANTITIES": "1" #1   # RA, DEC, delta, deldot
        }

        try:
            response = requests.get(API_URL, params=obs_params)
            if response.status_code == 200:
                #data = json.loads(response.text)
                output_path = self.preprocess_ephemeris_obs(response.text, object_id)
                
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
        
    def fetch_ephemeris_vec(
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

        vec_params = {
            "format": FORMAT,
            "COMMAND": cmd,
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",  # Vector ephemeris
            "CENTER": "@10",      # Earth-centered 500@399, @10 5500@0 (Solar-system barycenter) however it could be beneficial to re-generate all vector parquets from different centers
            "START_TIME": start_time,
            "STOP_TIME": stop_time,
            "STEP_SIZE": step_size,
            #"CSV_FORMAT": "YES"
            #"QUANTITIES": "EC"   # RA, DEC, delta, deldot
        }
        
        try:
            response = requests.get(API_URL, params=vec_params)
            if response.status_code == 200:
                #data = json.loads(response.text)
                output_path = self.preprocess_ephemeris_vec(response.text, object_id)
                
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
        
    def preprocess_phys_data(self, data:str):


        # Define regex patterns for each parameter
        # Total solar system mass is like 98% sun and 1.8% jupiter, potential value in prioritizing those centers and weighting them?
        # still need to append daf on the end of the obs_dataframe
        # Then merge the obs_dataframe with the vec_dataframe using datetime as leading standardized parameter
        # 199 (Mercury) GM uncertainty incorrectly captured don't forget
        # don't forget density g/cm^-3 conversion to g/cm^3 (including density uncertainties)
        # Mean Radius (+ Uncertainty), GM (still failing empty GM 1-sigma values), Eccentricity complete
        # Inclination still an issue (only available for small bodies through Horizons API)



        gm_uncertainty_patterns = [
            r'GM 1\-sigma\s*,?\s*\s*\(km\^3/s\^2\)\s*\s*=\s*\s*\+-\s*(\d+\.\d+)', # Variation 1
            r'GM 1\-sigma\,\s*?\s*\s*km\^3/s\^2\s*\s*=\s*\s*\+-\s*(\d+\.\d+)',
            r'GM 1\-sigma\s*\(\s*km^3/s^2\)\s*=\s*\+-\s*(\d+\.\d+)', #(?= ) # Variation 2
            r'GM 1\-sigma\s*\(\s*km^3/s^2\)\s*=\s*\s*\+-(\d+\.\d+)(?= )', # Variation 3
            r'GM 1\-sigma\s*,?\s*\(\s*km^3/s^2\)\s*=\s*(\d+\.\d+)', # Variation 4
            #r'GM 1\-sigma\s*,?\s*km^3/s^2\s*=\s*\+-(\d+\.\d+)',
            #r'GM 1\-sigma,\s*?km\^3/s\^2\s*=\s*\+-\s*(\d+\.\d+)',  # With uncertainty
            #r'(?i)GM 1-sigma\,\s*km^3/s^2\s*=\s*(?:\+-?\d+\.\d+)',
            #r'(?i)([GM,|GM])\s*km^3/s^2\s*=\s*(\d+\.\d+)(?:\+-?\s*)(\d+\.\d+)?', # Variation\
            r'(?i)\bGM\s+1\-sigma\b\s*$\w+$\s*=\s*(?:\+\-)?\s*(\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma.*?\=\s*(?:\+\-)?\s*([0-9.]+)(?=\s*(?:\n|\S+\s+=|$))',
            #r'(?i)GM\s+1\-sigma.*?\=\s*(\d+(?:\.\d+)?)'
            #r'(?i)GM\s+1\-sigma.*?\=\s*([+-]?\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma\s*=\s*([+-]?\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma.*?\=\s*(?:\+\-)?\s*(\d+(?:\.\d+)?)', # Variation main
            #r'(?i)GM\s+1\-sigma.*?\=\s*[+-]?\s*(\d+(?:\.\d+)?)\b'
            #r'(?i)\bGM\s+1\-sigma\b\s*$(?:km^3/s^2)$\s*=\s*(\d+(?:\.\d+)?)',

        ]

        orb_period_patterns = [
            r'(?i)orbital period\s*=\s*(\d+\.\d+) d',
            #r'orbital period.*?\s+(\d+\.\d+)',
            r'(?i)Sidereal orbital period\s*=\s*(\d+\.\d+) d',
            r'(?i)\bsidereal\s+orb\.\s+per\,\s+y\s*=\s*(\d+\.\d+)',
            r'(?i)\bsidereal\s+orb\.\s+per\.=\s*(\d+\.\d+)\s*y',
            r'(?i)\bmean\s+sidereal\s+orb\s+per\s*=\s*(\d+\.\d+(?:[eE]\d+)?)\s*y',
            r'(?i)\bSidereal orb\. per\.\s*=\s*(\d+\.\d+)\s*y?\b',
            r'(?i)\bsidereal orb\. per\.\,\s+y =\s*(\d+\.\d+)?\b',
            r'(?i)\bSidereal orbit period\s*=\s*(\d+\.\d+)\s*yr?\b',
            r'(?i)\bSidereal orbit period\s*=\s*(\d+\.\d+)\s*y?\b',
            r'(?i)orbital period,\s+~\s(\d+\.\d+)'
        ]

        aph_peri_list = [
            #r'Perihelion.*?\s+(\d+\.\d+)\s+(\d+\.\d+)'
            r'.*Solar Constant \(W/m\^2\)\s+(\d+)\s+(\d+)'

        ]

        density_negative_patterns = [
            r'Density \(g cm\^\-3\)\s*\s*=\s*(\d+.\d+)',
        ]
        
        patterns = {
            'Mass (kg)': [
                r'(?i)^.*x10\^(\d+).*=(\d+\.\d+)',
                r'Mass x10\^(\d+) \(kg\)\s*=\s*(\d+.\d+)',
                r'Mass, x10\^(\d+) kg\s*=\s*(\d+.\d+)',
                r'Mass x 10\^(\d+) \(kg\)\s*\s*=\s*(\d+.\d+)',
                #r'(?i)^.*x10\^(\d+).*=\s*(\d+\.?\d*([eE][+-]?\d+)?)'
            ],
            'Mean Radius (km)': [
                r'Vol\. mean radius, km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 1
                r'Mean radius \(km\)\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)',  # Variation 2
                r'(?i)vol\. mean radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 3
                r'(?i)Vol\. Mean Radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.?\d*)\s*\+-?\s*(\d+\.?\d*)',  # Variation 4
                r'(?i)Vol\. Mean Radius \(km\)\s*=\s*(\d+)\s*\+\-\s*(\d+)',  # Variation 5
                r'(?i)vol\. mean radius\s*,\s*km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'  # Variation 6
                #r'(?i)mean radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'  # Variation 7
            ],
            'Density (g/cm³)': [
                r'density \(g\/cm\^3\)\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 1
                r'density \(g\/cm\^3\)\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)',  # Variation 2
                #r'(?i)density\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 3
                #r'(?i)density\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.?\d*)\s*\+-?\s*(\d+\.?\d*)',  # Variation 4
                r'Density \(g\/cm\^3\)\s*\s*=\s*(\d+.\d+)',  # Variation 5
                r'Density \(R=1195 km\)\s*\s*=\s*(\d+.\d+)',
                r'(?i)density\s*,\s*km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'
            ],
            'Semi-Major Axis': [
                r'(?i)Semi-major axis,\s+a\s*=\s*(\d+\.\d+)',
                r'(?i)Semi-major axis,\s+a\s+~\s(\d+\.\d+)',
                r'(?i)\bSemi-major axis,\s+a\s*=\s*(\d{1,3},?\d{3}(\.\d+)?)',
                r'(?i)\bSemi-major axis,\s+a\s*=\s*(\d+\.\d+$\d+\^\d+$)',
                r'(?i)\bSemi-major axis,\s+a\s*[=~]\s*(\d+\,\d+(\.\d+)?)'
            ],
        }

        params = {}
        mass_v = float('inf')
        m_radius = float('inf')
        m_radius_u = float('inf')
        density_v = float('inf')
        sm_axis = float('inf')
        for param, pattern_list in patterns.items():
            if param == 'Mass (kg)':
                # Try each pattern until a match is found
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        exp = int(match.group(1).strip())
                        value = float(match.group(2).strip())

                        scaling_factor = 10 ** (exp - 24)

                        # Standardize the mass
                        std_mass = value * scaling_factor
                        #std_mass_rounded = round(std_mass, 2)
                        params[param] = std_mass
                        mass_v = std_mass
                        #if uncertainty is not None:
                        #    params[f'{param} Uncertainty'] = uncertainty
                        print(f"Mass Match - exp: {exp} value: {value} standardized 10^24: {std_mass}")
                        #print(f" ± {uncertainty}")
                        #f uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
            elif param == 'Mean Radius (km)':
                # Try each pattern until a match is found
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        value = match.group(1).strip()
                        uncertainty = match.group(2).strip() if match.group(2) else None
                        params[param] = value
                        m_radius = value
                        if uncertainty is not None:
                            params[f'{param} Uncertainty'] = uncertainty
                            m_radius_u = uncertainty
                        print(f"Radius Match: {param} {value}")
                        print(f" ± {uncertainty}")
                        #f uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
            elif param == 'Density (g/cm³)':
                # Try each pattern until a match is found
                value = None
                #uncertainty = None
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        value = float(match.group(1).strip())
                        #uncertainty = float(match.group(2).strip()) if match.group(2) else None
                        params[param] = value
                        density_v = value
                        #if uncertainty is not None:
                        #    params[f'{param} Uncertainty'] = uncertainty
                        print(f"Density Match: {param} {value}")
                        #print(f" ± {uncertainty}")
                        #if uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found

                if value is None:
                    for pattern in density_negative_patterns:
                        match = re.search(pattern, data, flags=re.IGNORECASE)
                        #print(f"Density pattern: {pattern}")
                        if match:
                            value = float(match.group(1).strip())
                            scaling_factor = 10 ** (6)

                            # Standardize the mass
                            std_density = value * scaling_factor
                            params[param] = value
                            density_v = value
                            print(f"Density Match: {value}")
                            break
            elif param == 'Semi-Major Axis':
                # Try each pattern until a match is found
                value = None
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"Semi major pat: {pattern}")
                    if match:
                        #print(f"Semi match: {match}")
                        value = float(match.group(1).strip())
                        params[param] = value
                        sm_axis = value
                        print(f"Match: {param} {value}")
                        break  # Exit the loop once a match is found
                
                # If no direct semi-major axis match, try orbital period patterns
                if value is None:
                    for pattern in orb_period_patterns:
                        match = re.search(pattern, data, flags=re.IGNORECASE)
                        if match:
                            print(f"Orb period pattern: {pattern}")
                            orbital_period = float(match.group(1).strip())
                            print(f"Semi Axis Match group 0 check: {match.group(0)}")
                            if 'y' in match.group(0):
                                print(f"Orb period value = {orbital_period} years")
                                orbital_period_years = orbital_period
                            else:
                                # Convert orbital period (days) to years
                                print(f"Orb period value = {orbital_period} days")
                                orbital_period_years = orbital_period / 365.25
                            value = float(self.calculate_semi_major_axis(orbital_period_years))
                            params[param] = value
                            sm_axis = value
                            print(f"Match: {param} {value} (calculated from orbital period)")
                            break

            else:
                match = re.search(pattern_list, data)
                if match:
                    
                    value = match.group(1).strip()
                    params[param] = value
                    print(f"Else Param: {param}")
                    print(f"Match: {value}")
        


        # Define the list of required headers for the DataFrame
        headers = [
            'Mass (kg)',
            'Mean Radius (km)',
            'Mean Radius (km) Uncertainty',
            'Density (g/cm³)',
            'Semi-Major Axis'
        ]

        # Initialize a dictionary to hold the standardized data
        std_data = {}

        for header in headers:
            if f"{header}" in params:
                std_data[header] = params[f"{header}"]
            else:
                continue

        # Create the standardized DataFrame with each header as a column
        daf = pd.DataFrame([std_data], columns=headers)

        # Create DataFrame with extracted parameters
        '''daf = pd.DataFrame({
            'Parameter': list(params.keys()),
            'Value': list(params.values())
        })'''

        # Print the DataFrame
        #print(f"\nMajor body info:\n{daf}")

        return float(mass_v), float(m_radius), float(m_radius_u), float(density_v), float(sm_axis)


    # Preprocesses obs_params ephemeris data into dataframe parquet file (obs might actually not be necessary)       
    def preprocess_ephemeris_obs(self, data: str, object_id: str) -> Optional[Path]:
        """
        Processes raw ephemeris data from JPL Horizons API and saves it as a Parquet file.
        
        Args:
            data (str): Raw response string from the Horizons API
            object_id (str): ID of the target object
            
        Returns:
            Path: Path to the saved Parquet file if successful. None otherwise.
        """
        
        # Define regex patterns for each parameter
        # Total solar system mass is like 98% sun and 1.8% jupiter, potential value in prioritizing those centers and weighting them?
        # still need to append daf on the end of the obs_dataframe
        # Then merge the obs_dataframe with the vec_dataframe using datetime as leading standardized parameter
        # 199 (Mercury) GM uncertainty incorrectly captured don't forget
        # don't forget density g/cm^-3 conversion to g/cm^3 (including density uncertainties)
        # Mean Radius (+ Uncertainty), GM (still failing empty GM 1-sigma values), Eccentricity complete
        # Inclination still an issue (only available for small bodies through Horizons API)



        gm_uncertainty_patterns = [
            r'GM 1\-sigma\s*,?\s*\s*\(km\^3/s\^2\)\s*\s*=\s*\s*\+-\s*(\d+\.\d+)', # Variation 1
            r'GM 1\-sigma\,\s*?\s*\s*km\^3/s\^2\s*\s*=\s*\s*\+-\s*(\d+\.\d+)',
            r'GM 1\-sigma\s*\(\s*km^3/s^2\)\s*=\s*\+-\s*(\d+\.\d+)', #(?= ) # Variation 2
            r'GM 1\-sigma\s*\(\s*km^3/s^2\)\s*=\s*\s*\+-(\d+\.\d+)(?= )', # Variation 3
            r'GM 1\-sigma\s*,?\s*\(\s*km^3/s^2\)\s*=\s*(\d+\.\d+)', # Variation 4
            #r'GM 1\-sigma\s*,?\s*km^3/s^2\s*=\s*\+-(\d+\.\d+)',
            #r'GM 1\-sigma,\s*?km\^3/s\^2\s*=\s*\+-\s*(\d+\.\d+)',  # With uncertainty
            #r'(?i)GM 1-sigma\,\s*km^3/s^2\s*=\s*(?:\+-?\d+\.\d+)',
            #r'(?i)([GM,|GM])\s*km^3/s^2\s*=\s*(\d+\.\d+)(?:\+-?\s*)(\d+\.\d+)?', # Variation\
            r'(?i)\bGM\s+1\-sigma\b\s*$\w+$\s*=\s*(?:\+\-)?\s*(\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma.*?\=\s*(?:\+\-)?\s*([0-9.]+)(?=\s*(?:\n|\S+\s+=|$))',
            #r'(?i)GM\s+1\-sigma.*?\=\s*(\d+(?:\.\d+)?)'
            #r'(?i)GM\s+1\-sigma.*?\=\s*([+-]?\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma\s*=\s*([+-]?\d+(?:\.\d+)?)',
            r'(?i)GM\s+1\-sigma.*?\=\s*(?:\+\-)?\s*(\d+(?:\.\d+)?)', # Variation main
            #r'(?i)GM\s+1\-sigma.*?\=\s*[+-]?\s*(\d+(?:\.\d+)?)\b'
            #r'(?i)\bGM\s+1\-sigma\b\s*$(?:km^3/s^2)$\s*=\s*(\d+(?:\.\d+)?)',

        ]

        orb_period_patterns = [
            r'(?i)orbital period\s*=\s*(\d+\.\d+) d',
            #r'orbital period.*?\s+(\d+\.\d+)',
            r'(?i)Sidereal orbital period\s*=\s*(\d+\.\d+) d',
            r'(?i)\bsidereal\s+orb\.\s+per\,\s+y\s*=\s*(\d+\.\d+)',
            r'(?i)\bsidereal\s+orb\.\s+per\.=\s*(\d+\.\d+)\s*y',
            r'(?i)\bmean\s+sidereal\s+orb\s+per\s*=\s*(\d+\.\d+(?:[eE]\d+)?)\s*y',
            r'(?i)\bSidereal orb\. per\.\s*=\s*(\d+\.\d+)\s*y?\b',
            r'(?i)\bsidereal orb\. per\.\,\s+y =\s*(\d+\.\d+)?\b',
            r'(?i)\bSidereal orbit period\s*=\s*(\d+\.\d+)\s*yr?\b',
            r'(?i)\bSidereal orbit period\s*=\s*(\d+\.\d+)\s*y?\b',
            r'(?i)orbital period,\s+~\s(\d+\.\d+)'
        ]

        aph_peri_list = [
            #r'Perihelion.*?\s+(\d+\.\d+)\s+(\d+\.\d+)'
            r'.*Solar Constant \(W/m\^2\)\s+(\d+)\s+(\d+)'

        ]

        density_negative_patterns = [
            r'Density \(g cm\^\-3\)\s*\s*=\s*(\d+.\d+)',
        ]
        
        patterns = {
            'Mass (kg)': [
                r'(?i)^.*x10\^(\d+).*=(\d+\.\d+)',
                r'Mass x10\^(\d+) \(kg\)\s*=\s*(\d+.\d+)',
                r'Mass, x10\^(\d+) kg\s*=\s*(\d+.\d+)',
                r'Mass x 10\^(\d+) \(kg\)\s*\s*=\s*(\d+.\d+)',
                #r'(?i)^.*x10\^(\d+).*=\s*(\d+\.?\d*([eE][+-]?\d+)?)'
            ],
            'Mean Radius (km)': [
                r'Vol\. mean radius, km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 1
                r'Mean radius \(km\)\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)',  # Variation 2
                r'(?i)vol\. mean radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 3
                r'(?i)Vol\. Mean Radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.?\d*)\s*\+-?\s*(\d+\.?\d*)',  # Variation 4
                r'(?i)Vol\. Mean Radius \(km\)\s*=\s*(\d+)\s*\+\-\s*(\d+)',  # Variation 5
                r'(?i)vol\. mean radius\s*,\s*km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'  # Variation 6
                #r'(?i)mean radius\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'  # Variation 7
            ],
            'GM (km³/s²)': [
                r'GM\s*,?\s*\s*\(km\^3/s\^2\)\s*\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)', # Variation 1
                r'GM\s*\(\s*km^3/s^2\)\s*=\s*(\d+\.\d+[\+-].*?)', #(?= ) # Variation 2
                r'GM\s*$\s*km^3/s^2$\s*=\s*(\d+\.\d+[\+-].*?)(?= )', # Variation 3
                r'GM\s*\(km\^3/s\^2\)\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)',  # With uncertainty
                r'GM\s*\(km\^3/s\^2\)\s*=\s*(\d+(?:\.\d+)?)\s*(?:\+-\s*(\d+(?:\.\d+)?))?', # Optional uncertainty
                r'GM\s*,?\s*\(\s*km^3/s^2\)\s*=\s*(\d+\.\d+)',
                r'GM\,\s*km\^3/s\^2\s*=\s*((\d+\.\d+)(?: \+- (\d+\.\d+))?)',
                r'GM\,\s*km^3/s^2\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',
                r'(?i)(?:gm)\s*(?:\(|\s)km^3/s^2(?:\)|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',
                r'(?i)GM\,\s*?km\^3/s\^2?\s*=\s*(\d+\.\d+(?:\+|-)\d+\.\d+)',
                r'(?i)GM\,\s*km^3/s^2\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',
                r'(?i)GM\,\s*km^3/s^2\s*=\s*((?:\d+\.\d+)(?:\+-?\d+\.\d+)?)',
                r'(?i)([GM,|GM])\s*km^3/s^2\s*=\s*(\d+\.\d+)(?:\+-?\s*)(\d+\.\d+)?'


            ],
            'Density (g/cm³)': [
                r'density \(g\/cm\^3\)\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 1
                r'density \(g\/cm\^3\)\s*=\s*(\d+\.\d+)\s*\+-\s*(\d+\.\d+)',  # Variation 2
                #r'(?i)density\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)',  # Variation 3
                #r'(?i)density\s*(?:,?\s*)?(?:$|\s)km(?:$|\s)?\s*=\s*(\d+\.?\d*)\s*\+-?\s*(\d+\.?\d*)',  # Variation 4
                r'Density \(g\/cm\^3\)\s*\s*=\s*(\d+.\d+)',  # Variation 5
                r'Density \(R=1195 km\)\s*\s*=\s*(\d+.\d+)',
                r'(?i)density\s*,\s*km\s*=\s*(\d+\.\d+)\s*\+-?\s*(\d+\.\d+)'
            ],
            'Semi-Major Axis': [
                r'(?i)Semi-major axis,\s+a\s*=\s*(\d+\.\d+)',
                r'(?i)Semi-major axis,\s+a\s+~\s(\d+\.\d+)',
                r'(?i)\bSemi-major axis,\s+a\s*=\s*(\d{1,3},?\d{3}(\.\d+)?)',
                r'(?i)\bSemi-major axis,\s+a\s*=\s*(\d+\.\d+$\d+\^\d+$)',
                r'(?i)\bSemi-major axis,\s+a\s*[=~]\s*(\d+\,\d+(\.\d+)?)'
            ],
        }

        params = {}
        for param, pattern_list in patterns.items():
            if param == 'Mass (kg)':
                # Try each pattern until a match is found
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        exp = int(match.group(1).strip())
                        value = float(match.group(2).strip())

                        scaling_factor = 10 ** (exp - 24)

                        # Standardize the mass
                        std_mass = value * scaling_factor
                        #std_mass_rounded = round(std_mass, 2)
                        params[param] = std_mass
                        #if uncertainty is not None:
                        #    params[f'{param} Uncertainty'] = uncertainty
                        print(f"Mass Match - exp: {exp} value: {value} standardized 10^24: {std_mass}")
                        #print(f" ± {uncertainty}")
                        #f uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
            elif param == 'Mean Radius (km)':
                # Try each pattern until a match is found
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        value = match.group(1).strip()
                        uncertainty = match.group(2).strip() if match.group(2) else None
                        params[param] = value
                        if uncertainty is not None:
                            params[f'{param} Uncertainty'] = uncertainty
                        print(f"Radius Match: {param} {value}")
                        print(f" ± {uncertainty}")
                        #f uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
            elif param == 'GM (km³/s²)':
                # Try each pattern until a match is found
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        #print(f"GM match: {match}")
                        value = match.group(1).strip()
                        uncertainty = match.group(2).strip() if match.group(2) else None
                        params[param] = value
                        if uncertainty is not None and value != uncertainty:
                            params[f'{param} Uncertainty'] = uncertainty
                        print(f"GM Match: {param} {value}")
                    elif uncertainty is None:
                        for pat in gm_uncertainty_patterns:
                            umatch = re.search(pat, data, flags=re.IGNORECASE)
                            #print(f"\nPat Param: {param}")
                            #print(f"Pattern: {pat}")
                            if umatch:
                                #print(f"umatch: {umatch}")
                                uncertainty = umatch.group(1).strip()
                                params[f'{param} Uncertainty'] = uncertainty
                                print(f" ± {uncertainty}")
                                break
                                
                        #if uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
            elif param == 'Density (g/cm³)':
                # Try each pattern until a match is found
                value = None
                #uncertainty = None
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"\nParam: {param}")
                    #print(f"Pattern: {pattern}")
                    if match:
                        value = float(match.group(1).strip())
                        #uncertainty = float(match.group(2).strip()) if match.group(2) else None
                        params[param] = value
                        #if uncertainty is not None:
                        #    params[f'{param} Uncertainty'] = uncertainty
                        print(f"Density Match: {param} {value}")
                        #print(f" ± {uncertainty}")
                        #if uncertainty:
                        #    print(f" ± {uncertainty}")
                        break  # Exit the loop once a match is found
                # If no direct semi-major axis match, try orbital period patterns
                if value is None:
                    for pattern in density_negative_patterns:
                        match = re.search(pattern, data, flags=re.IGNORECASE)
                        #print(f"Density pattern: {pattern}")
                        if match:
                            value = float(match.group(1).strip())
                            scaling_factor = 10 ** (6)

                            # Standardize the mass
                            std_density = value * scaling_factor
                            params[param] = value
                            print(f"Density Match: {value}")
                            break
            elif param == 'Semi-Major Axis':
                # Try each pattern until a match is found
                value = None
                for pattern in pattern_list:
                    match = re.search(pattern, data, flags=re.IGNORECASE)
                    #print(f"Semi major pat: {pattern}")
                    if match:
                        #print(f"Semi match: {match}")
                        value = float(match.group(1).strip())
                        params[param] = value
                        print(f"Match: {param} {value}")
                        break  # Exit the loop once a match is found
                
                # If no direct semi-major axis match, try orbital period patterns
                if value is None:
                    for pattern in orb_period_patterns:
                        match = re.search(pattern, data, flags=re.IGNORECASE)
                        if match:
                            print(f"Orb period pattern: {pattern}")
                            orbital_period = float(match.group(1).strip())
                            print(f"Semi Axis Match group 0 check: {match.group(0)}")
                            if 'y' in match.group(0):
                                print(f"Orb period value = {orbital_period} years")
                                orbital_period_years = orbital_period
                            else:
                                # Convert orbital period (days) to years
                                print(f"Orb period value = {orbital_period} days")
                                orbital_period_years = orbital_period / 365.25
                            value = float(self.calculate_semi_major_axis(orbital_period_years))
                            params[param] = value
                            print(f"Match: {param} {value} (calculated from orbital period)")
                            break

            else:
                match = re.search(pattern_list, data)
                if match:
                    
                    value = match.group(1).strip()
                    params[param] = value
                    print(f"Else Param: {param}")
                    print(f"Match: {value}")
        


        # Define the list of required headers for the DataFrame
        headers = [
            'Mass (kg)',
            'Mean Radius (km)',
            'Mean Radius (km) Uncertainty',
            'GM (km³/s²)',
            'GM (km³/s²) Uncertainty',
            'Density (g/cm³)',
            'Density (g/cm³) Uncertainty',
            'Semi-Major Axis'
        ]

        # Initialize a dictionary to hold the standardized data
        std_data = {}

        for header in headers:
            if f"{header}" in params:
                std_data[header] = params[f"{header}"]
            else:
                continue

        # Create the standardized DataFrame with each header as a column
        daf = pd.DataFrame([std_data], columns=headers)

        # Create DataFrame with extracted parameters
        '''daf = pd.DataFrame({
            'Parameter': list(params.keys()),
            'Value': list(params.values())
        })'''

        # Print the DataFrame
        print(f"\nMajor body info:\n{daf}")


        try:
            # Find the positions of $$SOE and $$EOE markers
            #print(f"\nRAW: {data}")
            soe_start = data.find("$$SOE")
            eoe_end = data.find("$$EOE") #+ 7  # Add 5 to include the marker in the slice
            
            if soe_start == -1 or eoe_end == -1:
                print(f"No SOE or EOE markers found for object {object_id}")
                return None
                
            # Extract the data section between $$SOE and $$EOE
            phys_data = data[:soe_start].strip()
            raw_data = data[soe_start+7:eoe_end].strip()
            
            if not raw_data:
                print(f"Empty data section for object {object_id}")
                return None
                
            # Print raw data information for debugging
            #print(f"Raw data length: {len(raw_data)}")
            #print(f"First 200 chars: {raw_data[:200]}")
            print(f"\n{phys_data}")

            # Approach: Use regex to find each data point directly
            
            # Define regex pattern to match the date-time, RA, and DEC values
            obs_pattern = re.compile(
                r'^\s*(\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2})'  # Date-Time
                r'\s+(-?\d+\.\d+[Ee]?[-+]?\d*)'                  # RA
                r'\s+(-?\d+\.\d+[Ee]?[-+]?\d*)'                  # DEC'
            )

            obs_pat = r"^\s*(\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2})"

            
            # Find all matches in the raw data
            #obs_matches = obs_pattern.findall(raw_data)
            obs_matches = re.finditer(obs_pat, raw_data)
            obs_positions = [(m.start(), m.group(1)) for m in obs_matches]
            
            if not obs_positions:
                print("No observation data found in the section")
                return None

            #print(f"Found {len(obs_positions)} JDTDB entries")
            
            if len(obs_positions) == 0:
                print("No JDTDB values found in data")
                return None
            
            # Split into lines and filter out empty lines
            lines = [line.strip() for line in raw_data.split('\\n') if line.strip()]
            #print(f"{lines}")
            #print(f"Found {len(lines)} data lines to process")
            
            # Initialize list to store parsed data
            data_rows = []
            success_count = 0
            
            # Process each line
            for i, line in enumerate(lines, 1):
                try:
                    # Extract datetime, RA, and DEC using regex
                    #print(f"Lineee: {line}")
                    match = re.match(
                        r'^\s*(\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',
                        line
                    )
                    
                    if not match:
                        print(f"Line {i} doesn't match expected pattern: '{line}'")
                        continue
                        
                    dt_str, ra_str, dec_str = match.groups()
                    #print(f"Line: {dt_str}, {ra_str}, {dec_str}")
                    
                    # Parse datetime
                    try:
                        dt = datetime.strptime(dt_str.strip(), '%Y-%b-%d %H:%M')
                        dt = pytz.UTC.localize(dt)
                    except ValueError as e:
                        print(f"Error parsing datetime in line {i}: '{dt_str}'")
                        print(f"Error: {str(e)}")
                        continue
                    
                    # Parse RA and DEC
                    try:
                        ra = float(ra_str)
                        dec = float(dec_str)
                    except ValueError as e:
                        print(f"Error parsing coordinates in line {i}: RA='{ra_str}', DEC='{dec_str}'")
                        print(f"Error: {str(e)}")
                        continue
                    
                    # Add data row
                    data_rows.append({
                        'datetime': dt,
                        'RA_ICRF': ra,
                        'DEC_ICRF': dec
                    })
                    success_count += 1
                    
                    # Print progress every 50 successful parses
                    if success_count % 50 == 0:
                        print(f"Successfully parsed {success_count} lines")
                        
                except Exception as e:
                    print(f"Error processing line {i}: '{line}'")
                    print(f"Exception: {str(e)}")
                    continue
            
            print(f"Successfully parsed {success_count} lines out of {len(lines)}")
            
            if not data_rows:
                print(f"No data rows were parsed for object {object_id}")
                return None
                
            # Create DataFrame from parsed data
            df = pd.DataFrame(data_rows)
            print(f"Successfully created DataFrame with {len(df)} rows")
            #print(f"{object_id} Dataframe: {df}")
            
            # Get object name
            obj_name = self.get_object_name(object_id)
            #print(f"Object ID {object_id} is {obj_name}")
            
            # Add metadata columns
            df['object_id'] = object_id
            #df['object_name'] = obj_name
            #df['reference_frame'] = 'ICRF'

            #print(f"\nObject ID {object_id} df contents: {df}\n")
            
            # Save to Parquet file
            filename = f"ephemeris_obs_{obj_name}_{DEFAULT_START_TIME.replace('-', '')}_{DEFAULT_STOP_TIME.replace('-', '')}.parquet"
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
            print(f"Error processing data for object {object_id}: {str(e)}")
            traceback.print_exc()
            return None

    # Preprocesses vec_params ephemeris data into dataframe parquet file
    def preprocess_ephemeris_vec(
        self,
        data: str,
        object_id: str,
    ) -> Optional[Path]:
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
            phys_data = data[:soe_start].strip()
            raw_data = data[soe_start+5:eoe_end].strip()
            
            if not raw_data:
                print(f"Empty data section for object {object_id}")
                return None
            
            # Print raw data information for debugging
            print(f"Raw data length: {len(raw_data)}\n")
            #print(f"First 200 chars: {raw_data[:200]}")
            #print(f"{phys_data}")
            mass = float('inf')
            m_radius = float('inf')
            m_radius_uncertainty = float('inf')
            density = float('inf')
            sm_axis = float('inf')

            mass, m_radius, m_radius_uncertainty, density, sm_axis = self.preprocess_phys_data(phys_data)
            surface_g = self.calculate_surface_gravity(mass, m_radius)

            print(f"\nMass: {mass}")
            print(f"\nMean Radius: {m_radius}")
            print(f"\nMean Radius Uncertainty: {m_radius_uncertainty}")
            print(f"\nDensity: {density}")
            print(f"\nSemi-Major Axis: {sm_axis}")
            print(f"\nSurface Gravity: {surface_g}")


            
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
            min_distance = float('inf')
            max_distance = 0.0
            
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
                    # LT = Light Travel Time from target object to the observer site (days)
                    # RG = Range, or distance between observer and target object
                    # RR = Range Rate, rate of change of distance between observer and target object, measured in AU per day
                    lt_match = re.search(r'LT\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    rg_match = re.search(r'RG\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    rr_match = re.search(r'RR\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)', block)
                    
                    if not all([lt_match, rg_match, rr_match]):
                        print(f"Could not find LT, RG, RR in block {i}")
                        continue
                    
                    lt = float(lt_match.group(1))
                    rg = float(rg_match.group(1))
                    rr = float(rr_match.group(1))

                    # Update minimum and maximum distances with corresponding times
                    if rg < min_distance:
                        min_distance = rg
                    if rg > max_distance:
                        max_distance = rg

                    r = np.array([x,y,z])
                    v = np.array([vx,vy,vz])
                    inc = (90-self.calculate_inclination(r,v))
                    
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
                        'RR': rr,
                        'Inclination': inc
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
            #print(f"{object_id} Dataframe: {df}")
            print(f"\nMin distance (perihelion): {min_distance}\nMax distance (aphelion): {max_distance}")
            
            # Get object name
            obj_name = self.get_object_name(object_id)
            #print(f"\nObject ID {object_id} is {obj_name}\n")
            
            # Add metadata columns
            df['Mass'] = mass # (kg)
            df['Mean Volume Radius'] = m_radius
            df['Mean Volume Radius Uncertainty'] = m_radius_uncertainty
            df['Eccentricity'] = self.calculate_eccentricity(min_distance, max_distance)
            # Inclination (defines shape and orientation of orbit, affecting position and velocity vectors over time)
            df['Semi-Major Axis'] = sm_axis # Semi-Major axis (determines scale of the orbit)
            df['Density'] = density # Density k/m^3 (influences gravitational interactions with other bodies)
            df['Surface Gravity'] = self.calculate_surface_gravity(mass, m_radius) # Surface Gravity (affects the trajectory of objects near the planet or moon) calculated with g = GM/R^2 (G gravitational constant, M mass, R radius)
            df['object_id'] = object_id
            df['object_name'] = obj_name
            df['reference_frame'] = 'Ecliptic of J2000.0'

            print(f"\nObject ID {object_id} df contents: {df}\n")
            
            # Save to Parquet file
            filename = f"ephemeris_vec_{obj_name}_{DEFAULT_START_TIME.replace('-', '')}_{DEFAULT_STOP_TIME.replace('-', '')}.parquet"
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
            #print(f"DataFrame contains {len(df)} rows with columns: {df.columns.tolist()}")
            
            return output_path
            
        except Exception as e:
            print(f"Error processing ephemeris data for {object_id}: {str(e)}")
            traceback.print_exc()
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
    
    def calculate_inclination(self, r: float, v: float):
        """
        Calculate the orbital inclination using the position and velocity vectors.

        Parameters:
            r (numpy array): Position vector in meters.
            v (numpy array): Velocity vector in meters per second.

        Returns:
            float: Inclination angle in degrees.
        """
        # Compute specific angular momentum
        h = np.cross(r, v)
        
        # Calculate the magnitude of h
        h_magnitude = np.linalg.norm(h)
        
        if h_magnitude == 0:
            return 0.0  # Avoid division by zero
        
        # Calculate sine of inclination
        sin_i = h[2] / h_magnitude

        # Ensure sin_i is within the valid range for arcsin
        sin_i_clipped = np.clip(sin_i, -1.0, 1.0)
        
        # Use arcsin to find the inclination in radians
        i_rad = np.arcsin(sin_i_clipped)
        
        # Convert to degrees and ensure it's within [0, 180]
        i_deg = np.degrees(i_rad)
        
        if i_deg < 0:
            i_deg += 360
        
        return min(i_deg, 180.0)

    
    def calculate_surface_gravity(self, mass: float, radius: float):
        """
        Calculate the surface gravity of a body.
        
        Args:
            mass (float): Mass of the body in 10^24 kg.
            radius (float): Radius of the body in km (converted to meters).
            
        Returns:
            float: Surface gravity in m/s^2.
        """
        G = 6.67430e-11
        M_body = mass * (10**24)
        R_body = radius * 1000
        g = (G * M_body) / (R_body**2)

        return g
    
    def calculate_semi_major_axis(self, orbital_period_years):
        """
        Calculate the semi-major axis of a planet's orbit using Kepler's Third Law.
        
        Args:
            orbital_period_years (float): Orbital period in years.
            
        Returns:
            float: Semi-major axis in kilometers.
        """
        # Gravitational constant [m^3 kg^-1 s^-2]
        G = 6.67430e-11 #If data discrepancy likely unit constistency change to 6.67430e-20
        # Mass of the Sun [kg]
        M_sun = 1.9885e30
        # Convert orbital period from years to seconds
        seconds_per_year = 365.25 * 24 * 3600
        T = orbital_period_years * seconds_per_year
        
        # Calculate a^3 using Kepler's Third Law
        a_cubed = (G * M_sun * T**2) / (4 * math.pi**2)
        
        # Take the cube root to find a in meters, then convert to kilometers
        a_meters = a_cubed ** (1/3)
        a_kilometers = a_meters / 1000
        
        return a_kilometers
    
    def calculate_eccentricity(self, perihelion, aphelion):
        """
        Calculate the orbital eccentricity given the perihelion and aphelion distances.
        
        Parameters:
            perihelion (float): The distance of perihelion.
            aphelion (float): The distance of aphelion.
            
        Returns:
            float: The orbital eccentricity.
            
        Raises:
            ValueError: If either perihelion or aphelion is non-positive.
        """
        if perihelion <= 0 or aphelion <= 0:
            raise ValueError("Perihelion and Aphelion distances must be positive numbers.")
        
        e = (aphelion - perihelion) / (aphelion + perihelion)
        return e

    
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
            '1': 'Mercury Barycenter',
            '2': 'Venus Barycenter',
            '3': 'Earth Barycenter',
            '4': 'Mars Barycenter',
            '5': 'Jupiter Barycenter',
            '6': 'Saturn Barycenter',
            '7': 'Uranus Barycenter',
            '8': 'Neptune Barycenter',
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
            '999': 'Pluto'
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
        #"1",
        #"2",
        #"3",
        #"4",
        #"5",
        #"6",
        #"7",
        #"8",
        #"9",
        "199",
        "299",
        #"301",      # Moon
        #"399",     # Earth
        #"401",     # Phobos
        "499",
        #"502",     # Europa
        #"503",     # Ganymede
        "599",
        "699",
        "799",
        "899",
        "999"
    ]

    center_points = [
        "@10", # might not need so many earth observer sites, focus on orbits? eg. @10 sun for all planets, 500@399 earth for moon, 500@599 jupiter for its satellites
        "009@399",
        "016@399",
        "026@399",
        "043@399",
        "057@399",
        "066@399",
        "069@399",
        "098@399",
        "132@399",
        "187@399",
        "197@399",
        "200@399",
        "209@399",
        "217@399",
        "271@399",
        "300@399",
        "348@399",
        "433@399",
        "500@399",
        "506@399",
        "536@399",
        "548@399",
        "627@399",
        "758@399",
        "828@399",
        "861@399",
        "918@399"
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
        #result = client.fetch_ephemeris_obs(object_id=obj)
        result = client.fetch_ephemeris_vec(object_id=obj)
        #print(f"\n Payload: {result}")
        if result is None:
            #print(f"\n Analyzing data for object: {obj}")
            #client.analyze_parquet_file(result)
            #print("Failed to fetch ephemeris data for object:", obj)
            continue


        
if __name__ == "__main__":
    main()
