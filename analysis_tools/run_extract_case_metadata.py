#!/usr/bin/env python
"""execute extract_case_metadata based on command line arguments"""

import argparse
import sys

import yaml

from analysis_tools.catalog_utils import extract_case_metadata


def parse_args(args):
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="execute extract_case_metadata based on command line arguments",
    )
    parser.add_argument(
        "--caseroot",
        help="caseroot directory for case that metadata is being extracted from",
    )
    parser.add_argument(
        "--sname",
        help="shortname for case",
    )
    parser.add_argument(
        "--esmcol_spec_dir",
        help="directory where esmcol spec file will reside",
    )
    return parser.parse_args()


def main(args):
    """execute extract_case_metadata based on command line arguments"""

    case_metadata = extract_case_metadata(
        args.caseroot, args.sname, args.esmcol_spec_dir
    )
    print(yaml.dump([case_metadata], sort_keys=False))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
