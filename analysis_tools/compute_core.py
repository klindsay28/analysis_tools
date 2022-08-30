import cf_xarray as cfxr
import xarray as xr

from .utils import key_value_str_to_dict
from .utils_units import clean_units


def reduce(ds, da_in, reduce_op, reduce_axes):
    """reduce da along reduce_axes, using reduce_op"""

    weight = get_cell_measure(ds, da_in, "area").fillna(0)

    # TODO: introduce region dimension here to weight

    # TODO: if "T" in reduce_axes then multiply weight by dt

    # apply reduction operation
    with xr.set_options(keep_attrs=True):
        da_out = da_in.cf.weighted(weight).sum(dim=reduce_axes)
        if reduce_op == "average":
            ones_masked = xr.ones_like(da_in).where(da_in.notnull())
            da_out /= ones_masked.cf.weighted(weight).sum(dim=reduce_axes)

    # set reduction specific attributes
    da_in_units = clean_units(da_in.attrs["units"])
    if reduce_op == "integrate":
        da_out.attrs["long_name"] = "Integrated " + da_in.attrs["long_name"]
        da_out.attrs["units"] = f"({weight.attrs['units']})({da_in_units})"
    if reduce_op == "average":
        da_out.attrs["long_name"] = "Averaged " + da_in.attrs["long_name"]
        da_out.attrs["units"] = da_in_units

    # delete attributes that are no longer applicable after reduction
    # TODO: modify cell_methods, cell_measures, and coordinates appropriately for reduce_axes
    for key in ["cell_measures", "coordinates", "grid_loc"]:
        if key in da_out.attrs:
            del da_out.attrs[key]

    # propagate particular encoding values
    for name in ["dtype", "missing_value", "_FillValue"]:
        da_out.encoding[name] = da_in.encoding[name]

    return da_out


def get_cell_measure(ds, da, measure):
    """return measure for da in ds"""

    cell_measures = key_value_str_to_dict(da.attrs["cell_measures"])

    return ds[cell_measures[measure]]