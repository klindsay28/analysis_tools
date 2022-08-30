from inspect import signature

import numpy as np
import xarray as xr

from .utils import time_set_mid


# TODO: add postprocess_cam, which does the following
#   1) add AREA to ds if not available
#      Q: Is there a better option for rearth than using CIME's shr value?
#   2) add cell_measures attribute with "area: AREA" to vars on lat, lon dimensions
#   3) add axis attributes to time, lat, lon, lev, ilev
#   4) call time_set_mid


# TODO: add dt, in seconds, to all postprocess functions


def open_mfdataset_kwargs(scomp):
    """return dict of arguments for open_mfdataset appropriate for scomp"""

    if scomp == "mom6":
        # the following variables can have a CF-compliant units with no
        # calendar attribute, causing problems with time conversion
        return {"drop_variables": ["average_T1", "average_T2"]}
    else:
        return {}


def postprocess(ds, scomp, **kwargs):
    """call postprocessing function appropriate for scomp on ds"""

    if scomp == "pop":
        pp_fcn = postprocess_pop
    elif scomp == "mom6":
        pp_fcn = postprocess_mom6
    else:
        return ds

    # construct args to post processing function
    kwargs_avail = {"ds": ds}
    kwargs_avail.update(kwargs)
    pp_kwargs = {
        arg: kwargs_avail[arg] for arg in signature(pp_fcn).parameters
    }

    return pp_fcn(**pp_kwargs)


def postprocess_pop(ds):
    """
    POP specific Dataset postprocessing
    add standard_name, if known
    add area to cell_measures attribute
    add axis attributes to coordinates
    add nlon, nlat coordinates
    set time to average of time:bounds
    """
    add_standard_names(ds, standard_names["pop"])
    for varname in ds.data_vars:
        if "coordinates" in ds[varname].encoding:
            var_coords = ds[varname].encoding["coordinates"]
            if ("TLONG" in var_coords) and ("TLAT" in var_coords):
                ds[varname].attrs["cell_measures"] = "area: TAREA"
            if ("ULONG" in var_coords) and ("ULAT" in var_coords):
                ds[varname].attrs["cell_measures"] = "area: UAREA"
        if varname in standard_names["pop"]:
            ds[varname].attrs["standard_name"] = standard_names["pop"][varname]
    ds["time"].attrs["axis"] = "T"
    for coordname in ds.coords:
        if "depth" in ds[coordname].attrs["long_name"]:
            ds[coordname].attrs["axis"] = "Z"
    ds["nlat"] = ("nlat", np.arange(ds.sizes["nlat"]), {"axis": "Y"})
    ds["nlon"] = ("nlon", np.arange(ds.sizes["nlon"]), {"axis": "X"})
    return time_set_mid(ds, "time")


def postprocess_mom6(ds, catalog, case):
    """
    MOM6 specific Dataset postprocessing
    add standard_name, if known
    change time:calendar to lower case
    add axis attribute to coordinates
    add data_vars from corresponding static stream to dataset
    add coordinates to data variables
    """
    add_standard_names(ds, standard_names["mom6"])
    tb_name = ds["time"].bounds
    for varname in ["time", tb_name]:
        for d in [ds[varname].attrs, ds[varname].encoding]:
            if "calendar" in d:
                d["calendar"] = d["calendar"].lower()
            for att_name in ["calendar_type", "_FillValue", "missing_value"]:
                if att_name in d:
                    del d[att_name]
    for coordname in ds.coords:
        if "cartesian_axis" in ds[coordname].attrs:
            ds[coordname].attrs["axis"] = ds[coordname].attrs["cartesian_axis"]
            del ds[coordname].attrs["cartesian_axis"]
    df = catalog.df
    df = df[df["case"] == case]
    df = df[df["scomp"] == "mom6"]
    df = df[df["stream"] == "static"]
    if len(df) == 0:
        raise ValueError(f"no static stream for {case} found in catalog")
    ds_static = xr.open_dataset(df["path"].values[0])
    for varname in ds_static.data_vars:
        ds[varname] = ds_static[varname]
    for varname in ds.data_vars:
        dims = ds[varname].dims
        if "xh" in dims:
            if "yh" in dims:
                ds[varname].attrs["coordinates"] = "geolat geolon"
            if "yq" in dims:
                ds[varname].attrs["coordinates"] = "geolat_v geolon_v"
        if "xq" in dims:
            if "yh" in dims:
                ds[varname].attrs["coordinates"] = "geolat_u geolon_u"
            if "yq" in dims:
                ds[varname].attrs["coordinates"] = "geolat_c geolon_c"
    return ds


def add_standard_names(ds, standard_names):
    """add standard_name to data_vars' attributes, if known"""
    for varname in ds.data_vars:
        if varname in standard_names:
            ds[varname].attrs["standard_name"] = standard_names[varname]


standard_names = {
    "pop": {
        "UVEL": "sea_water_x_velocity",
        "VVEL": "sea_water_y_velocity",
        "TEMP": "sea_water_potential_temperature",
        "SALT": "sea_water_salinity",
        "IAGE": "sea_water_age_since_surface_contact",
    },
    "mom6": {
        "agessc": "sea_water_age_since_surface_contact",
    },
    "MARBL": {
        "FG_CO2": "surface_downward_mole_flux_of_carbon_dioxide",
        "DpCO2": "surface_carbon_dioxide_partial_pressure_difference_between_sea_water_and_air",
        "photoC_TOT_zint": "net_primary_mole_productivity_of_biomass_expressed_as_carbon_by_phytoplankton",
        "DIC": "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",
        "ALK": "sea_water_alkalinity_expressed_as_mole_equivalent",
        "PO4": "mole_concentration_of_dissolved_inorganic_phosphorus_in_sea_water",
        "SiO3": "mole_concentration_of_dissolved_inorganic_silicon_in_sea_water",
        "NO3": "mole_concentration_of_nitrate_in_sea_water",
        "NH4": "mole_concentration_of_ammonium_in_sea_water",
        "Fe": "mole_concentration_of_dissolved_iron_in_sea_water",
        "O2": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    },
}

standard_names["pop"].update(standard_names["MARBL"])
standard_names["mom6"].update(standard_names["MARBL"])