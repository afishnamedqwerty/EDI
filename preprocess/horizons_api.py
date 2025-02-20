import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Constants
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
DEFAULT_START_TIME = "2030-01-01"
DEFAULT_STOP_TIME = "2031-01-01"

class HorizonsApiClient:
    """A client class to interact with the Horizons API for SPK data retrieval."""
    
    def __init__(self, output_dir: str = "spk_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    # This can be expanded or modified based on your needs
    objects_to_fetch = [
        "399",     # Earth
        "502",     # Europa
        "401",     # Moon
        "2000001", # Ceres
        "2000002", # Pallas
        "2000003", # Vesta
        "141P",    # Comet Machholz 2
        "DELD98A"  # Example asteroid designation
    ]
    
    # Initialize the Horizons client
    client = HorizonsApiClient(output_dir="spk_files")
    
    for obj in objects_to_fetch:
        print(f"\nFetching data for object: {obj}")
        result = client.fetch_spk(spkid=obj)
        if not result:
            continue
        
if __name__ == "__main__":
    main()
