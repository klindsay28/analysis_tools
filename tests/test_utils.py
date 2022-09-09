#! /usr/bin/env python3

import pytest
import cftime
import numpy as np
import xarray as xr

from xr_ds_ex import xr_ds_ex, gen_time_bounds_values
from analysis_tools.utils import lon_shift, time_year_plus_frac, time_set_mid
from analysis_tools.utils import repl_coord, key_value_str_to_dict
from analysis_tools.utils import expand_list_of_dicts

nyrs = 300
var_const = False


@pytest.mark.parametrize("lon", [-270.0, -135.0, -90.0, 0.0, 90.0, 135.0, 270.0])
@pytest.mark.parametrize("cut", [-180.0, -90.0, 0.0, 90.0, 180.0])
def test_lon_shift(lon, cut):
    ret_val = lon_shift(lon, cut)
    assert ret_val >= cut
    # the following assertions might not hold for all inputs, due to roundoff
    # but are expected to hold for the test values
    assert ret_val < cut + 360.0
    assert (ret_val - lon) % 360.0 == 0.0


@pytest.mark.parametrize("decode_times1", [True, False])
@pytest.mark.parametrize("decode_times2", [True, False])
@pytest.mark.parametrize("apply_chunk1", [True, False])
def test_repl_coord(decode_times1, decode_times2, apply_chunk1):
    ds1 = time_set_mid(xr_ds_ex(decode_times1, nyrs=nyrs, var_const=var_const), "time")
    if apply_chunk1:
        ds1 = ds1.chunk({"time": 12})

    # change time:bounds attribute variable rename corresponding variable
    tb_name_old = ds1["time"].attrs["bounds"]
    tb_name_new = tb_name_old + "_new"
    ds1["time"].attrs["bounds"] = tb_name_new
    ds1 = ds1.rename({tb_name_old: tb_name_new})

    # verify that repl_coord on xr_ds_ex gives same results as
    # 1) executing time_set_mid
    # 2) manually changing bounds
    ds2 = repl_coord(
        "time", ds1, xr_ds_ex(decode_times2, nyrs=nyrs, var_const=var_const)
    )
    assert ds2.identical(ds1)

    assert ds2["time"].encoding == ds1["time"].encoding
    assert ds2["time"].chunks == ds1["time"].chunks


@pytest.mark.parametrize("decode_times", [True, False])
@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("apply_chunk", [True, False])
def test_time_set_mid(decode_times, deep, apply_chunk):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const, time_mid=False)
    if apply_chunk:
        ds = ds.chunk({"time": 12})

    mid_month_values = gen_time_bounds_values(nyrs).mean(axis=1)
    if decode_times:
        time_encoding = ds["time"].encoding
        expected_values = cftime.num2date(
            mid_month_values, time_encoding["units"], time_encoding["calendar"]
        )
    else:
        expected_values = mid_month_values

    ds_out = time_set_mid(ds, "time", deep)

    assert ds_out.attrs == ds.attrs
    assert ds_out.encoding == ds.encoding
    assert ds_out.chunks == ds.chunks

    for varname in ds.variables:
        assert ds_out[varname].attrs == ds[varname].attrs
        assert ds_out[varname].encoding == ds[varname].encoding
        assert ds_out[varname].chunks == ds[varname].chunks
        if varname == "time":
            assert np.all(ds_out[varname].values == expected_values)
        else:
            assert np.all(ds_out[varname].values == ds[varname].values)
            assert (ds_out[varname].data is ds[varname].data) == (not deep)

    # verify that values are independent of ds being chunked in time
    ds_chunk = xr_ds_ex(
        decode_times, nyrs=nyrs, var_const=var_const, time_mid=False
    ).chunk({"time": 6})
    ds_chunk_out = time_set_mid(ds_chunk, "time")
    assert ds_chunk_out.identical(ds_out)


@pytest.mark.parametrize("decode_times", [True, False])
def test_time_year_plus_frac(decode_times):
    ds = xr_ds_ex(decode_times, nyrs=nyrs, var_const=var_const)

    # call time_year_plus_frac to ensure that it doesn't raise an exception
    ty = time_year_plus_frac(ds, "time")


def test_key_value_str_to_dict():
    d = {"key1": "val1", "key2": "val2", "key3": "val3"}
    key_value_str = " ".join([f"{str(key)}: {str(d[key])}" for key in d])
    assert key_value_str_to_dict(key_value_str) == d


def test_expand_list_of_dicts():
    # list example
    l_of_d = [{"key1": ["foo", "bar"], "key2": "val2"}]
    expected = [{"key1": "foo", "key2": "val2"}, {"key1": "bar", "key2": "val2"}]
    assert expand_list_of_dicts(l_of_d, "key1") == expected
    assert expand_list_of_dicts(l_of_d, "key2") == l_of_d
    assert expand_list_of_dicts(l_of_d, "key3") == l_of_d

    # tuple example
    l_of_d = [{"key1": ("foo", "bar"), "key2": "val2"}]
    expected = [{"key1": "foo", "key2": "val2"}, {"key1": "bar", "key2": "val2"}]
    assert expand_list_of_dicts(l_of_d, "key1") == expected
    assert expand_list_of_dicts(l_of_d, "key2") == l_of_d
    assert expand_list_of_dicts(l_of_d, "key3") == l_of_d
