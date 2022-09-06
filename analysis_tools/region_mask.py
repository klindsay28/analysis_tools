import numpy as np
import xarray as xr

from .utils import lon_shift


def gen_region_mask(ds, da, region_names):
    """return region mask appropriate for da in ds"""

    rmask = xr.DataArray(
        np.zeros((len(region_names), da.shape[-2], da.shape[-1])),
        dims=("region", da.dims[-2], da.dims[-1]),
        coords={"region": region_names},
    )
    rmask.region.encoding["dtype"] = "S1"

    # construct 'surface' mask
    isel_dict = {}
    for dim in da.dims[:-2]:
        isel_dict[dim] = 0
    surf_mask = xr.where(da.isel(isel_dict).notnull(), 1.0, 0.0)

    for region_ind, region_name in enumerate(region_names):
        if region_name == "Global":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] > -91.0) & (da.cf["longitude"] > -361.0), surf_mask, 0.0
            )
        elif region_name == "Ocean":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] > -91.0) & (da.cf["longitude"] > -361.0), surf_mask, 0.0
            )
            if "LANDFRAC" in ds:
                rmask.values[region_ind, :, :] *= (1.0 - ds["LANDFRAC"][0, :])
        elif region_name == "SH":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] < 0.0) & (da.cf["longitude"] > -361.0), surf_mask, 0.0
            )
        elif region_name == "NH":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] > 0.0) & (da.cf["longitude"] > -361.0), surf_mask, 0.0
            )
        elif region_name == "NPac":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] > 40.0) & (da.cf["latitude"] < 65.0) \
                & (lon_shift(da.cf["longitude"], 0) > 115.0) \
                & (lon_shift(da.cf["longitude"], 0) < 240.0), surf_mask, 0.0
            )
            if "LANDFRAC" in ds:
                rmask.values[region_ind, :, :] *= (1.0 - ds["LANDFRAC"][0, :])
        elif region_name == "LabSea":
            rmask.values[region_ind, :, :] = xr.where(
                (da.cf["latitude"] > 52.0) & (da.cf["latitude"] < 66.0) \
                & (lon_shift(da.cf["longitude"], 0) > 295.0) \
                & (lon_shift(da.cf["longitude"], 0) < 315.0), surf_mask, 0.0
            )
            if "LANDFRAC" in ds:
                rmask.values[region_ind, :, :] *= (1.0 - ds["LANDFRAC"][0, :])
        else:
            raise ValueError(f"Unknown region name {region_name}")

    return rmask
