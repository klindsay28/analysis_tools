"""
utility functions related to testing
"""


def dict_skip_keys(dict_in, skip_key_list):
    """return copy of dict_in, skipping keys in skip_key_list"""
    return {key: value for key, value in dict_in.items() if key not in skip_key_list}


def ds_identical_skip_attr_list(ds_baseline, ds_test, skip_attr_list):
    """compare ds_test to ds_baseline, omitting attrs in skip_attr_list"""
    ds_baseline_cp = ds_baseline.copy()
    ds_test_cp = ds_test.copy()
    for ds in [ds_baseline_cp, ds_test_cp]:
        ds.attrs = dict_skip_keys(ds.attrs, skip_attr_list)
    return ds_baseline_cp.identical(ds_test_cp)
