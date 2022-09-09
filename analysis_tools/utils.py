"""utility functions"""

from datetime import datetime
import inspect

import cftime
import dask
import numpy as np
import xarray as xr


def print_timestamp(msg):
    print(f"{str(datetime.now())}({inspect.stack()[1][3]}):{msg}")


def lon_shift(lon, cut):
    """convert lon to values in range cut:cut+360"""
    return (lon - cut) % 360 + cut


def repl_coord(coordname, ds1, ds2):
    """
    Return copy of d2 with coordinate coordname replaced, using coordname from ds1.
    Drop ds2.coordname.attrs['bounds'] in result, if ds2.coordname has bounds attribute.
    Add ds1.coordname.attrs['bounds'] to result, if ds1.coordname has bounds attribute.
    Except for coordname, the returned Dataset is a non-deep copy of ds2.
    """
    if "bounds" in ds2[coordname].attrs:
        tb_name = ds2[coordname].attrs["bounds"]
        ds_out = ds2.drop_vars(tb_name).assign_coords({coordname: ds1[coordname]})
    else:
        ds_out = ds2.assign_coords({coordname: ds1[coordname]})
    if "bounds" in ds1[coordname].attrs:
        tb_name = ds1[coordname].attrs["bounds"]
        ds_out = xr.merge([ds_out, ds1[tb_name]])
    return ds_out


def time_set_mid(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with midpoints of
    ds[time_name].attrs['bounds'], if bounds attribute exists.
    Except for time_name, the returned Dataset is a copy of ds2.
    The copy is deep or not depending on the argument deep.
    """

    ds_out = ds.copy(deep)

    if "bounds" not in ds[time_name].attrs:
        return ds_out

    tb_name = ds[time_name].attrs["bounds"]
    tb = ds[tb_name]
    bounds_dim = next(dim for dim in tb.dims if dim != time_name)

    # Use da = da.copy(data=...), in order to preserve attributes and encoding.

    # If tb is dask array of datetime objects then apply compute before applying mean.
    # This ensures that chunking of the time variable is preserved
    if isinstance(tb.data, dask.array.Array) and tb.dtype == np.dtype("O"):
        ds_out[time_name] = ds[time_name].copy(data=tb.compute().mean(bounds_dim))
    else:
        ds_out[time_name] = ds[time_name].copy(data=tb.mean(bounds_dim))

    return ds_out


def time_year_plus_frac(ds, time_name):
    """return time variable, as numpy array of year plus fraction of year values"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype("O"):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values,
            ds[time_name].attrs["units"],
            ds[time_name].attrs["calendar"],
        )

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(
        tvals_cftime, "days since 0000-01-01", calendar="noleap"
    )

    return tvals_days / 365.0


def key_value_str_to_dict(key_value_str):
    """convert string of space separated key: value pairs to a dict"""

    ret_val = {}
    key_value_list = key_value_str.split()
    for i in range(0, len(key_value_list), 2):
        word = key_value_list[i]
        if word[-1] != ":":
            raise ValueError(
                f"word={word} not colon terminated in {key_value_str}"
            )
        ret_val[word[:-1]] = key_value_list[i+1]

    return ret_val


def expand_list_of_dicts(l_of_d, key):
    """
    construct list of dicts from list of dicts,
    duplicating each dict d if d[key] is list or tuple,
        except d[key] is replaced with objects returned by iterating over d[key]
    [{"key1": ["foo", "bar"], "key2": "val2"}] ->
        [{"key1": "foo", "key2": "val2"}, {"key1": "bar", "key2": "val2"}]
    """

    ret_val = []
    for d in l_of_d:
        if isinstance(d, dict) and isinstance(d.get(key, None), (list, tuple)):
            for obj in d[key]:
                d_new = d.copy()
                d_new[key] = obj
                ret_val.append(d_new)
        else:
            ret_val.append(d)
    return ret_val
