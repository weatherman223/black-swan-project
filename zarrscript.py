import xarray as xr
import zarr

'NOTICE: Please ensure that you CHANGE the file directory before running the code'

def convert_historical_to_zarr():
    print("Loading historical dataset...")
    historical_data = xr.open_mfdataset(
        '/yourdirectoryhere/b.e21.BHIST*.f09_g17.LE2-*.cam.h1.PRECT.*.nc', 
        chunks={'time': 365, 'lat': 96, 'lon': 144},
        data_vars='minimal',
        coords='minimal',
        combine='by_coords',
        compat="override")
    
    print(f"Dataset loaded: {historical_data.dims}")
    print("Converting to Zarr...")
    
    # Save to Zarr
    zarr_path = '/yourdirectoryhere/historical_data_global.zarr'
    historical_data.to_zarr(zarr_path, mode='w')
    print(f"Conversion complete! Saved to {zarr_path}")

if __name__ == "__main__":
    convert_historical_to_zarr()