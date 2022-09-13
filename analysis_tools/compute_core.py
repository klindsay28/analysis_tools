import cf_xarray as cfxr  # noqa: F401
import xarray as xr

from .utils import key_value_str_to_dict


def reduce(ds, da_in, reduce_op, reduce_axes, measure=None, mask=None):
    """reduce da along reduce_axes, using reduce_op"""

    weight = get_cell_measure(ds, da_in, measure).fillna(0)

    # note that mask may have additional dimensions, e.g., region
    if mask is not None:
        attrs = weight.attrs
        weight = mask * weight
        weight.attrs = attrs

    # TODO: if "T" in reduce_axes then multiply weight by dt

    # apply reduction operation
    with xr.set_options(keep_attrs=True):
        if reduce_op == "integrate":
            da_out = da_in.cf.weighted(weight).sum(dim=reduce_axes)
        if reduce_op == "average":
            da_out = da_in.cf.weighted(weight).mean(dim=reduce_axes)

    # set reduction specific attributes
    if reduce_op == "integrate":
        da_out.attrs["long_name"] = "Integrated " + da_in.attrs["long_name"]
        da_out.attrs["units"] = f"({weight.attrs['units']})({da_in.attrs['units']})"
    if reduce_op == "average":
        da_out.attrs["long_name"] = "Averaged " + da_in.attrs["long_name"]

    # delete attributes that are no longer applicable after reduction
    # TODO: https://github.com/klindsay28/analysis_tools/issues/1
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
