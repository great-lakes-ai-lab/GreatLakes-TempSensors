import requests
import os
from datetime import datetime

# Base ERDDAP griddap URL for the dataset
base_url = "https://apps.glerl.noaa.gov/erddap/griddap/GL_Ice_Concentration_GCS.nc"

# Output directory
output_dir = "/Users/jagraha/dev/deepsensor_projects/data/temp_gridded_data/ice_conc_requests_lib/"
os.makedirs(output_dir, exist_ok=True)

# Define your spatial extent (adjust to match your area of interest)
# Full Great Lakes extent (approximate) — modify as needed
lat_min = 38.8749871947297
lat_max = 50.6059751976539
lon_min = -92.4199507342304
lon_max = -75.8816402880531

# Download each year
current_year = datetime.now().year

for year in range(1995, current_year + 1):
    start_date = f"{year}-01-01T00:00:00Z"
    end_date = f"{year}-12-31T23:59:59Z"

    # ERDDAP constraint expression
    # Format: variable[(time_start):(time_end)][(lat_min):(lat_max)][(lon_min):(lon_max)]
    constraint = (
        f"?ice_concentration"
        f"[({start_date}):1:({end_date})]"
        f"[({lat_min}):1:({lat_max})]"
        f"[({lon_min}):1:({lon_max})]"
    )

    url = base_url + constraint
    output_file = os.path.join(output_dir, f"ice_concentration_{year}.nc")

    if os.path.exists(output_file):
        print(f"Already exists, skipping: {output_file}")
        continue

    print(f"Downloading {year}...")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            f.write(response.content)

        print(f"  Saved: {output_file} ({len(response.content) / 1e6:.1f} MB)")
    except requests.exceptions.HTTPError as e:
        print(f"  ERROR for {year}: {e}")
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT for {year} — file may be too large, try smaller chunks")

print("Done!")