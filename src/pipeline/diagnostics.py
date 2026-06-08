import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_alignment(datasets: dict, lake_name: str = ""):
    """
    Plot the valid-data boundary from each dataset on one figure.
    Expects datasets dict with {name: xr.Dataset} in raw/standardized form (lat/lon coords).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for (name, ds), color in zip(datasets.items(), colors):
        # Get first data variable
        var = list(ds.data_vars)[0]

        # For temporal datasets, take one time slice
        if "time" in ds.dims:
            data = ds[var].isel(time=0)
        else:
            data = ds[var]

        # Plot the valid/invalid boundary (coastline proxy)
        valid_mask = data.notnull().astype(float)

        ax.contour(
            valid_mask.lon if "lon" in valid_mask.coords else valid_mask.x2,
            valid_mask.lat if "lat" in valid_mask.coords else valid_mask.x1,
            valid_mask.values,
            levels=[0.5],
            colors=[color],
            linewidths=1.5,
        )
        ax.plot([], [], color=color, label=f"{name} ({dict(data.sizes)})")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial alignment check: {lake_name}")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def check_point_values(datasets: dict, test_points: list):
    """
    Check values at specific lat/lon points across all datasets.

    test_points: list of (lat, lon) tuples
    """
    print(f"{'Dataset':<20} {'Variable':<15} {'Point':<20} {'Value'}")
    print("-" * 70)

    for lat, lon in test_points:
        for name, ds in datasets.items():
            for var in ds.data_vars:
                da = ds[var]
                if "time" in da.dims:
                    da = da.isel(time=0)

                val = da.sel(lat=lat, lon=lon, method="nearest").values
                print(f"{name:<20} {var:<15} ({lat:.2f}, {lon:.2f})    {val:.4f}")
        print()


def print_grid_summary(datasets: dict):
    """Print resolution and extent for each dataset."""
    print(f"{'Dataset':<20} {'Lat pts':<10} {'Lon pts':<10} "
          f"{'Lat res':<12} {'Lon res':<12} "
          f"{'Lat range':<25} {'Lon range':<25}")
    print("-" * 115)

    for name, ds in datasets.items():
        lat_name = "lat" if "lat" in ds.coords else "x1"
        lon_name = "lon" if "lon" in ds.coords else "x2"

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        lat_res = np.mean(np.diff(lats))
        lon_res = np.mean(np.diff(lons))

        print(f"{name:<20} {len(lats):<10} {len(lons):<10} "
              f"{lat_res:<12.6f} {lon_res:<12.6f} "
              f"[{lats.min():.4f}, {lats.max():.4f}]   "
              f"[{lons.min():.4f}, {lons.max():.4f}]")


def run_diagnostics(config, bundle):
    """Run all spatial alignment checks on a processed bundle."""
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS: {config.run.name} ({config.lake})")
    print(f"{'='*60}\n")

    # Collect datasets that have lat/lon or x1/x2 coords
    datasets = {}
    for name, obj in bundle.items():
        if hasattr(obj, "data_vars") and len(obj.data_vars) > 0:
            # Skip non-spatial things like mask_time_ds with only time-varying vars
            has_spatial = any(
                d in obj.dims for d in ["lat", "lon", "x1", "x2"]
            )
            if has_spatial:
                datasets[name] = obj

    # 1. Grid summary
    print("--- Grid Summary ---")
    print_grid_summary(datasets)

    # 2. Point checks
    print("\n--- Point Value Check ---")
    test_points = _get_test_points(config.lake)
    check_point_values(datasets, test_points)

    # 3. Visual overlay
    print("\n--- Spatial Alignment Plot ---")
    plot_spatial_alignment(datasets, lake_name=config.lake)

    # Save plot
    plots_dir = Path(config.paths.run_dir) / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.savefig(plots_dir / "spatial_alignment.png", dpi=150, bbox_inches="tight")
    print(f"Saved to: {plots_dir / 'spatial_alignment.png'}")


def _get_test_points(lake: str) -> list:
    """Return test points (lat, lon) for a given lake: center, near-shore, land."""
    points = {
        "erie": [
            (42.0, -81.0),    # center
            (42.5, -79.5),    # near shore
            (43.0, -79.0),    # land
        ],
        "ontario": [
            (43.6, -77.5),
            (43.3, -79.5),
            (44.0, -76.0),
        ],
        "huron": [
            (44.8, -82.0),
            (43.5, -82.0),
            (44.0, -80.0),
        ],
        "michigan": [
            (43.5, -87.0),
            (42.0, -86.5),
            (44.0, -85.0),
        ],
        "superior": [
            (47.5, -88.0),
            (46.8, -85.0),
            (48.5, -86.0),
        ],
        "all": [
            (42.0, -81.0),
            (44.8, -82.0),
            (47.5, -88.0),
        ],
    }
    return points.get(lake, points["all"])


