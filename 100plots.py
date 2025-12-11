import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

"NOTICE: Please ensure that you CHANGE the file directories before running the code"

def load_historical_dataset(zarr_path: str) -> xr.Dataset:
    """Load (or build + cache) the historical reference dataset."""
    if os.path.exists(zarr_path):
        print("Loading historical data from Zarr store …")
        return xr.open_zarr(zarr_path, chunks={"time": 365, "lat": 192, "lon": 288})

    print("First time historical load; this may take a while …")
    hist = xr.open_mfdataset(
        "/yourdirectoryhere/b.e21.BHIST*.f09_g17.LE2-*.cam.h1.PRECT.*.nc",
        chunks={"time": 365, "lat": 192, "lon": 288},
        data_vars="minimal",
        coords="minimal",
        combine="by_coords",
        compat="override",
    )

    print("Caching historical data to Zarr …")
    hist.to_zarr(zarr_path, mode="w")
    return hist


def extract_nearest(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Select the grid cell closest to the requested lat/lon."""
    return ds.sel(lat=lat, lon=lon, method="nearest")

def extract_regional_data(ds, historical_data, lat_min, lat_max, lon_min, lon_max):
    # Extract regional data for both datasets
    regional_ds = ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    regional_historical = historical_data.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    return regional_ds, regional_historical


def calculate_annual_max(ds_point: xr.Dataset, hist_point: xr.Dataset):
    """Return (annual_max, historical_max) in precip (m) for a single grid point."""
    # future data
    daily = ds_point.PRECT.where(ds_point.PRECT != -900) * 86400  # kg m‑2 s‑1 → m day‑1
    daily = daily.chunk({"time": -1})
    rolling_7d = daily.rolling(time=2, center=True).sum()  # 7‑day totals

    weights = np.cos(np.deg2rad(ds_point.lat)) 
    annual_max = (rolling_7d.weighted(weights).mean(dim=["lat", "lon"]).groupby("time.year").max())

    # historical baseline
    hist_daily = hist_point.PRECT.where(hist_point.PRECT != -900) * 86400
    hist_daily = hist_daily.chunk({"time": -1})
    hist_roll = hist_daily.rolling(time=2, center=True).sum()
    hist_max = (hist_roll.weighted(weights).mean(dim=["lat", "lon"]).groupby("time.year").max())

    return annual_max, hist_max


def compute_threshold(historical_max: xr.DataArray) -> float:
    """Return mean + 4σ threshold for exceedance detection."""
    std = historical_max.std().compute()
    mean = historical_max.mean().compute()
    return float((mean + 4 * std).values)


def build_plot(annual_max: xr.DataArray, threshold: float, exceed: xr.DataArray, ensemble_name: str, i: int):
    """Create and return a Matplotlib Figure for a single ensemble member."""
    fig, ax = plt.subplots(figsize=(11, 6))

    years = annual_max.year
    ax.plot(years, annual_max, label="Annual Max 7‑day Precip", lw=1.0)
    ax.axhline(threshold, color="red", ls="--", label=f"Threshold (4σ = {threshold:.3f} m)")
    ax.scatter(exceed.year, exceed, color="orange", s=25, zorder=5, label=f"Exceedances ({exceed.size})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (m)")
    ax.set_title(f"Ensemble {i} – Kerr County, TX (2d rolling window maxima - historical)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def main():
    target_lat = 30.0
    target_lon = 260.6  # CESM uses 0–360° longitudes

    # # Define the region of interest (Kerala, India)
    # lat_min = 8   # Southern border of Kerala
    # lat_max = 12  # Northern border of Kerala
    # lon_min = 74  # Western border of Kerala
    # lon_max = 77  # Eastern border of Kerala

    # 1. Load / prepare historical reference once (one target)
    hist = load_historical_dataset("/yourdirectoryhere/historical_data_global.zarr")
    hist_point = extract_nearest(hist, target_lat, target_lon)

    # # 1. Load / prepare historical reference once (regional target)
    # hist = load_historical_dataset("/data01/zacharmi/historical_data_global.zarr")
    # hist_point, _ = extract_regional_data(hist, hist, lat_min, lat_max, lon_min, lon_max)

    # 2. Compute threshold from historical data
    _, historical_max = calculate_annual_max(hist_point, hist_point)
    threshold = compute_threshold(historical_max)
    print(f"Exceedance threshold (mean +4 σ): {threshold:.3f} m")

    # 3. Gather ensemble patterns
    with open("/yourdirectoryhere/textfiles/historicalensembles.txt") as f:
        patterns = [line.strip() for line in f if line.strip()]

    output_pdf = "/yourdirectoryhere/ensemble_precipitation_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        for idx, pattern in enumerate(patterns, start=1):
            full_glob = f"/yourdirectoryhere/dataset/{pattern}"
            print(f"[{idx}/{len(patterns)}] Processing {pattern} …")

            try:
                ds = xr.open_mfdataset(
                    full_glob,
                    chunks={"time": 365, "lat": 192, "lon": 288},
                    data_vars="minimal",
                    coords="minimal",
                    combine="by_coords",
                    compat="override",
                )
            except FileNotFoundError:
                print(f"Warning: No files matched {full_glob}; skipping.")
                continue
            # One point
            ds_point = extract_nearest(ds, target_lat, target_lon)
            annual_max, _ = calculate_annual_max(ds_point, hist_point)
            annual_max = annual_max.compute()
            annual_max = annual_max.where(annual_max.year <= 2014, drop=True)
            exceed = annual_max.where(annual_max > threshold, drop=True)

            # #Regional Point
            # ds_regional, _ = extract_regional_data(ds, hist_point, lat_min, lat_max, lon_min, lon_max)
            # annual_max_regional, _ = calculate_annual_max(ds_regional, hist_point)
            # annual_max_regional = annual_max_regional.compute()
            # exceed_regional = annual_max_regional.where(annual_max_regional > threshold, drop=True)

            # One Point
            fig = build_plot(annual_max, threshold, exceed, ensemble_name=pattern.split(".")[5], i=idx)
            pdf.savefig(fig)
            plt.close(fig)

            # # Regional Point
            # fig = build_plot(annual_max_regional, threshold, exceed_regional, ensemble_name=pattern.split(".")[5], i=idx)
            # pdf.savefig(fig)
            # plt.close(fig)

            # ds.close()

    print(f"  PDF written to {output_pdf}")


if __name__ == "__main__":
    main()
