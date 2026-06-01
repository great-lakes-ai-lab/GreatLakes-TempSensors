# src/lakes.py
"""Lake bounding boxes for spatial subsetting."""

LAKE_BOUNDS = {
    "erie": {
        "lat": [41.3, 42.9],
        "lon": [-83.5, -78.8],
    },
    "ontario": {
        "lat": [43.1, 44.3],
        "lon": [-80.0, -75.7],
    },
    "huron": {
        "lat": [43.0, 46.4],
        "lon": [-84.8, -79.6],
    },
    "michigan": {
        "lat": [41.6, 46.1],
        "lon": [-88.0, -84.7],
    },
    "superior": {
        "lat": [46.4, 49.0],
        "lon": [-92.2, -84.3],
    },
    "all": {
        "lat": [41.3, 49.0],
        "lon": [-92.2, -75.7],
    },
}