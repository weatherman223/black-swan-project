import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

"NOTICE: Please Change File Directories"

def load_historical_dataset(zarr_path: str) -> xr.Dataset:
    """Load (or build + cache) the historical reference dataset."""
    if os.path.exists(zarr_path):
        print("Loading historical data from Zarr store …")
        return xr.open_zarr(zarr_path, chunks={"time": 365, "lat": 192, "lon": 288})

    print("First time historical load; this may take a while …")
    hist = xr.open_mfdataset(
        "/yourdirectoryhere/dataset/b.e21.BHIST*.f09_g17.LE2-*.cam.h1.PRECT.*.nc",
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

def compute_extreme_threshold(historical_max: xr.DataArray) -> float:
    """Return mean + 10σ threshold for extreme exceedance detection."""
    std = historical_max.std().compute()
    mean = historical_max.mean().compute()
    return float((mean + 10 * std).values)

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
    # hist = load_historical_dataset("/yourdirectoryhere/historical_data_global.zarr")
    # hist_point, _ = extract_regional_data(hist, hist, lat_min, lat_max, lon_min, lon_max)

    # 2. Compute threshold from historical data
    _, historical_max = calculate_annual_max(hist_point, hist_point)
    threshold = compute_threshold(historical_max)
    extreme_threshold = compute_extreme_threshold(historical_max)
    mean = float(historical_max.mean().compute().values)
    print(f"Exceedance threshold (mean +4 σ): {threshold:.3f} m")
    print(f"Extreme exceedance threshold (mean +10 σ): {extreme_threshold:.3f} m")

    # 3. Gather ensemble patterns
    with open("/yourdirectoryhere/textfiles/futureensembles.txt") as f:
        patterns = [line.strip() for line in f if line.strip()]

    all_years = []
    all_maxima = []
    all_exceedance_years = []
    all_exceedance_values = []
    all_extreme_exceedance_years = []
    all_extreme_exceedance_values = []

    output_pdf = "/yourdirectoryhere/ensemble_precipitation_analysis.pdf"
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
            exceed = annual_max.where(annual_max > threshold, drop=True).where(annual_max < extreme_threshold, drop=True)
            extreme_exceed = annual_max.where(annual_max > extreme_threshold, drop=True)

            # #Regional Point
            # ds_regional, _ = extract_regional_data(ds, hist_point, lat_min, lat_max, lon_min, lon_max)
            # annual_max_regional, _ = calculate_annual_max(ds_regional, hist_point)
            # annual_max_regional = annual_max_regional.compute()
            # exceed_regional = annual_max_regional.where(annual_max_regional > threshold, drop=True)

            all_years.extend(annual_max.year.values)
            all_maxima.extend(annual_max.values)
            all_exceedance_years.extend(exceed.year.values)
            all_exceedance_values.extend(exceed.values)
            all_extreme_exceedance_years.extend(extreme_exceed.year.values)
            all_extreme_exceedance_values.extend(extreme_exceed.values)

            ds.close()

        all_maxima_array = np.array(all_maxima)
    
        # Count values in different ranges
        left_of_mean = all_maxima_array[all_maxima_array < mean]
        right_of_mean_left_threshold = all_maxima_array[(all_maxima_array >= mean) & (all_maxima_array < threshold)]
        right_threshold_left_extreme = all_maxima_array[(all_maxima_array >= threshold) & (all_maxima_array < extreme_threshold)]
        right_of_extreme = all_maxima_array[all_maxima_array >= extreme_threshold]
    
        # Print the counts and percentages
        total_points = len(all_maxima)
        print(f"\n=== Distribution Analysis ===")
        print(f"Total data points: {total_points}")
        print(f"Historical mean: {mean:.3f} m")
        print(f"Threshold (4σ): {threshold:.3f} m") 
        print(f"Extreme threshold (10σ): {extreme_threshold:.3f} m")
        print(f"\n=== Counts by Range ===")
        print(f"Left of mean (< {mean:.3f} m): {len(left_of_mean)} ({len(left_of_mean)/total_points*100:.1f}%)")
        print(f"Mean to threshold ({mean:.3f} - {threshold:.3f} m): {len(right_of_mean_left_threshold)} ({len(right_of_mean_left_threshold)/total_points*100:.1f}%)")
        print(f"Threshold to extreme ({threshold:.3f} - {extreme_threshold:.3f} m): {len(right_threshold_left_extreme)} ({len(right_threshold_left_extreme)/total_points*100:.1f}%)")
        print(f"Right of extreme (> {extreme_threshold:.3f} m): {len(right_of_extreme)} ({len(right_of_extreme)/total_points*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(11, 6))

        # Create histogram of all annual maxima
        bin_width = 0.02
        max_val = max(all_maxima)
        bin_edges = np.arange(0, max_val + bin_width, bin_width)
        ax.hist(all_maxima, bins=bin_edges, alpha=0.7, color='lightblue', edgecolor='black', 
                label=f'Annual Maxima (n={len(all_maxima)})', density=True)

        # Add vertical lines for thresholds and mean
        ax.axvline(mean, color="blue", ls="--", linewidth=2, label=f"Historical Mean = {mean:.3f} m")
        ax.axvline(threshold, color="red", ls="--", linewidth=2, label=f"Threshold (4σ = {threshold:.3f} m)")
        ax.axvline(extreme_threshold, color="purple", ls="--", linewidth=2, label=f"Extreme Threshold (10σ = {extreme_threshold:.3f} m)")

        # Optionally, overlay exceedances as separate histogram
        # if all_exceedance_values:
        #     ax.hist(all_exceedance_values, bins=n_bins, alpha=0.8, color='orange', 
        #             edgecolor='red', label=f'Exceedances (n={len(all_exceedance_values)})', 
        #             density=True, histtype='step', linewidth=2)

        # if all_extreme_exceedance_values:
        #     ax.hist(all_extreme_exceedance_values, bins=n_bins, alpha=0.8, color='red', 
        #             edgecolor='darkred', label=f'Extreme Exceedances (n={len(all_extreme_exceedance_values)})', 
        #             density=True, histtype='step', linewidth=2)

        ax.set_xlabel("Precipitation (m)")
        ax.set_ylabel("Probability Density")

        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 20)
        
        ax.set_title(f"Distribution of Annual Maxima (historical) – Kerr County, TX (2d rolling window)\n{len(patterns)} Ensembles Combined")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("/yourdirectoryhere/result_histogram.pdf", dpi=300, bbox_inches='tight')
        print("Histogram saved to result_histogram.pdf")
        plt.close()

        # Print summary statistics
        total_exceedances = len(all_exceedance_values)
        total_points = len(all_maxima)
        exceedance_rate = (total_exceedances / total_points) * 100
        print(f"Summary: {total_exceedances} exceedances out of {total_points} total points ({exceedance_rate:.2f}%)")
        print(f"Extreme exceedances: {len(all_extreme_exceedance_values)} out of {total_points} total points ({(len(all_extreme_exceedance_values) / total_points) * 100:.2f}%)")

        ds.close()

        ds.close()

    print(f"  PDF written to {output_pdf}")

if __name__ == "__main__":
    main()
