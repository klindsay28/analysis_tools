"""
utility functions related to units

The arithmetic functions are rather ad-hoc.
"""

import re

import cf_units
import xarray as xr


def clean_units(units):
    """replace some troublesome unit terms with acceptable replacements"""
    replacements = {
        "kgC": "kg",
        "gC": "g",
        "gC13": "g",
        "gC14": "g",
        "gN": "g",
        "unitless": "1",
        "fraction": "1",
        "years": "common_years",
        "yr": "common_year",
        "meq": "mmol",
        "neq": "nmol",
        "psu": "1e-3",
    }
    units_split = re.split(r"( |\(|\)|\^|\*|/|-[0-9]+|[0-9]+)", units)
    units_split_repl = [
        replacements[token] if token in replacements else token for token in units_split
    ]
    return "".join(units_split_repl)


def conv_units_np(values, units_in, units_out):
    """
    return a copy of numpy array values, with units converted from units_in to units_out
    """
    units_in_cf = cf_units.Unit(clean_units(units_in))
    units_out_cf = cf_units.Unit(clean_units(units_out))
    return units_in_cf.convert(values, units_out_cf, inplace=False)


def conv_units(da, units_out):
    """
    return a copy of da, with units converted to units_out
    """
    # use apply_ufunc to preserve dask-ness of da
    func = lambda values: conv_units_np(values, da.attrs["units"], units_out)
    da_out = xr.apply_ufunc(
        func, da, keep_attrs=True, dask="parallelized", output_dtypes=[da.dtype]
    )
    da_out.attrs["units"] = units_out
    da_out.encoding = da.encoding
    return da_out
