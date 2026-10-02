# src/pipeline/preview.py
"""Quick script to show task settings, and model structure."""
import deepsensor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path

from scipy.stats import kendalltau

warnings.filterwarnings("ignore", module="cartopy.*")

import cartopy.crs as ccrs
import cartopy.feature as cf

from utils.dates import dates_from_intervals
from pipeline.config import PipelineConfig
from pipeline.task_builder import make_train_val_dates, make_train_date_sampler


from deepsensor.model import ConvNP

def run_preview(config: PipelineConfig, bundle: dict, tl_config):
    print("\n" + "=" * 60)
    print("STAGE: PREVIEW")
    print("=" * 60)
    train_dates, val_dates = make_train_val_dates(config)
    tc = config.training
    id_sweeps, n_layer_sweeps, kernel_sweep = config.training.internal_density_sweep, config.training.n_unet_layer_sweep, config.training.unet_kernel_size_sweeep

    print("Validation Task Sampling:")
    print("With the current configuration there will be:")
    print(f"              {len(val_dates)} validation tasks. Striding {tc.val_date_stride} days")
    print(f"              first date: {val_dates[0].date()}, last date: {val_dates[-1].date()}")
    print(f"              Tasks dropped from these months: {tc.val_months_drop}")

    print("\n" + "-" * 60)
    print("Training Task Sampling:")
    if tc.train_date_mode == "random":
        date_sampler, _, _ = make_train_date_sampler(config)
        print(f"              first date in pool: {train_dates[0].date()}, last date in pool: {train_dates[-1].date()}")
    else:
        print(f"              {len(train_dates)} train tasks. Striding {tc.train_date_stride} days")
        print(f"              first date: {train_dates[0].date()}, last date: {train_dates[-1].date()}")
    print(f"              Tasks dropped from these months: {tc.train_months_drop}")

    print("\n" + "-" * 60)
    print("Evaluation Tasks")
    ec = config.evaluation
    eval_dates = dates_from_intervals(tc.test_range, ec.date_subsample_factor, months_to_drop=tc.test_months_drop)
    print(f"                 Evaluation will use {len(eval_dates)} dates. Striding {ec.date_subsample_factor} days ")
    print(f"                  first date: {eval_dates[0].date()}, last date: {eval_dates[-1].date()}")
    print(f"              Tasks dropped from these months: {tc.test_months_drop}")

    # TODO: Go back to chat and https://umgpt.umich.edu/conversations/4480410 and get add some caclutions for adjusting the params
    #   what is the spatial resolution of GLSEA3, how many row and cols in the data extent -> what's an appropriate internal density

    out_dir = Path(config.paths.run_dir) / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idens in id_sweeps:
        for nl in n_layer_sweeps:
            for ks in kernel_sweep:
                print(f"Internal Density: {idens}, n Layers: {nl}, "
        f"kernel size: {ks}")
                model = ConvNP(bundle["data_processor"], tl_config.task_loader, internal_density=idens, unet_channels=(64,) * nl, unet_kernels = ks)
                try:
                    rf_fig = _make_rf_plot(config=config, bundle=bundle, model=model,internal_d=idens, n_layers=nl, kernel_size=ks, out_dir=out_dir, scale="50m")
                    # rf_fig.show()
                except Exception as e:
                    print(f"An error occurred making the plot: {e}")


def _get_extent(ds, lon_name="lon", lat_name="lat"):
    """Return (lon_min, lon_max, lat_min, lat_max) for an xarray object."""
    lon = ds[lon_name]
    lat = ds[lat_name]
    return (
        float(lon.min()),
        float(lon.max()),
        float(lat.min()),
        float(lat.max()),
    )


def _get_patch(data_processor, receptive_field, extent, crs):
    x2_min, x2_max, x1_min, x1_max = extent
    x11, x12 = data_processor.config["coords"]["x1"]["map"]
    x21, x22 = data_processor.config["coords"]["x2"]["map"]

    x1_rf_raw = receptive_field * (x12 - x11)
    x2_rf_raw = receptive_field * (x22 - x21)

    x1_midpoint_raw = (x1_max + x1_min) / 2
    x2_midpoint_raw = (x2_max + x2_min) / 2

    # Compute bottom left corner of receptive field
    x1_corner = x1_midpoint_raw - x1_rf_raw / 2
    x2_corner = x2_midpoint_raw - x2_rf_raw / 2

    patch = mpatches.Rectangle(xy=[x2_corner, x1_corner],  # Cartesian fmt: x2, x1
            width=x2_rf_raw,
            height=x1_rf_raw,
            facecolor="black",
            alpha=0.15,
            edgecolor="crimson",
            linewidth=2.0,
            transform=crs,
        )

    return patch


def _make_rf_plot(config, bundle, model, internal_d, n_layers, kernel_size, out_dir, scale="10m"):
    crs = ccrs.PlateCarree()
    extent = _get_extent(bundle["lakemask_sampling"])
    patch = _get_patch(
        bundle["data_processor"], model.model.receptive_field, extent, crs
    )

    x0, y0 = patch.get_xy()
    w, h = patch.get_width(), patch.get_height()

    lon_min, lon_max, lat_min, lat_max = extent
    data_w = lon_max - lon_min
    data_h = lat_max - lat_min

    overflow_x = w > data_w
    overflow_y = h > data_h

    if overflow_x or overflow_y:
        pad = 0.05 * max(w, h)

        if overflow_x:
            lon_min = min(lon_min, x0) - pad
            lon_max = max(lon_max, x0 + w) + pad
        if overflow_y:
            lat_min = min(lat_min, y0) - pad
            lat_max = max(lat_max, y0 + h) + pad


    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(projection=crs))

    # ---- water: paint the whole axes, then draw land on top -------------
    ax.add_feature(cf.OCEAN.with_scale(scale), facecolor="#cfe4f2", zorder=0)
    ax.add_feature(cf.LAND.with_scale(scale),  facecolor="#efe9dd", zorder=1)
    ax.add_feature(cf.LAKES.with_scale(scale), facecolor="#cfe4f2",
                   edgecolor="#5a7d96", linewidth=0.5, zorder=2)

    # ---- boundaries -----------------------------------------------------
    ax.add_feature(cf.COASTLINE.with_scale(scale),
                   edgecolor="#333333", linewidth=0.8, zorder=3)
    ax.add_feature(cf.STATES.with_scale(scale),
                   edgecolor="#444444", linewidth=1.4, zorder=4)   # bold
    ax.add_feature(cf.BORDERS.with_scale(scale),
                   edgecolor="black", linewidth=0.6,
                   linestyle=(0, (4, 3)), zorder=5)                # thin dashed

    # ---- receptive-field patch on top -----------------------------------
    patch.set_zorder(10)
    ax.add_patch(patch)

    ax.set_extent((lon_min, lon_max, lat_min, lat_max), crs=crs)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="gray", alpha=0.5, linestyle=":", zorder=6)
    gl.top_labels = False
    gl.right_labels = False

    n_params = f"{deepsensor.backend.nps.num_params(model.model):,}"

    title_text = (
        f"Internal Density: {internal_d}, n Layers: {n_layers}, "
        f"kernel size: {kernel_size}, N Parameters: {n_params}"
    )
    fig.subplots_adjust(top=0.99)

    fig.text(
        0.5,
        0.94,
        title_text,
        ha="center",
        va="top",
        fontsize=14,
        color="black",
    )

    # fig.tight_layout()
    fig.savefig(out_dir / f"rf_plot_id_{internal_d}_nl_{n_layers}_ks_{kernel_size}.png")
    # _finish_plot(config, out_dir / f"rf_plot_id_{internal_d}_nl_{n_layers}_ks_{kernel_size}.png")

    return fig