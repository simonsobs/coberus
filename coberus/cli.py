#!python3

"""
Runs the background tasks from librarian server. Does not actually
run a web server instance.
"""

import argparse as ap
from pathlib import Path
from dask.distributed import Client
from .core import Coadder, coadd
from .fits import save_to_fits, extract_wcs
import json

def main():
    parser = ap.ArgumentParser(
        description=(
            "Runs coberus in parallel on your data using dask by spawning a new dask "
            "client and server. This script cannot connect to pre-exisitng dask clusters."
            "Your files must be in the directory you provide as the --input argument, "
            "with names map{1:N}.fits, map{1:N}_mask.fits, and cov_map{1:N}_map{1:N}.fits. "
            "Responses must be provided in a resopnses.json file."
        )
    )

    parser.add_argument("--input", required=True, type=Path, help="Path to the directory containing input data.")
    parser.add_argument("--number", required=True, type=int, help="Number of maps that you will use.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output FITS file.")

    args = parser.parse_args()

    with open(args.input / "responses.json", "r") as f:
        responses = json.load(f)
        responses = [float(r) for r in responses]

    client = Client()

    maps = [args.input / f"map{i+1}.fits" for i in range(0, args.number)]
    masks = [args.input / f"map{i+1}_mask.fits" for i in range(0, args.number)]
    covariance_maps = [
        [
            args.input
            / (
                f"cov_map{i+1}_map{j+1}.fits"
                if i < j
                else f"cov_map{j+1}_map{i+1}.fits"
            )
            for j in range(0, args.number)
        ]
        for i in range(0, args.number)
    ]

    wcs = extract_wcs(maps[0])

    coadder = Coadder(
        maps=maps, masks=masks, responses=responses, covariance_maps=covariance_maps
    )

    main_array = coadd(client, coadder)

    save_to_fits(args.output, main_array, wcs)
