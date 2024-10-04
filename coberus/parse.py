"""
Parse an input JSON file and return a WaveletMetadata and associated
Maps.
"""

from pydantic import BaseModel

from pixell.wavelets import CosineNeedlet

from .wavelets import Map, WaveletMetadata
from pathlib import Path
import numpy as np
import math


def gate(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


postprocess_funcs = {"gate": gate}


class BeamMetadata(BaseModel):
    fwhm: float

    def to_func(self):
        def beam(ells: float) -> float:
            fwhm_radians = np.deg2rad(self.fwhm / 60.0)
            square_ells = ells * ells

            prefactor = -(fwhm_radians * fwhm_radians) / (16.0 * math.log(2))

            return np.exp(prefactor * square_ells)

        return beam


class MapMetadata(BaseModel):
    tag: str
    path: Path
    mask: Path
    lmin: int | None = None
    lmax: int | None = None
    response: float

    beam: BeamMetadata
    postprocess_map: str | None = None
    postprocess_mask: str | None = None

    def to_map(self) -> Map:
        return Map(
            tag=self.tag,
            path=self.path,
            mask=self.mask,
            lmin=self.lmin,
            lmax=self.lmax,
            response=self.response,
            beam=self.beam.to_func(),
            postprocess_map=postprocess_funcs.get(self.postprocess_map, lambda x: x),
            postprocess_mask=postprocess_funcs.get(self.postprocess_mask, lambda x: x),
        )


class CoberusInput(BaseModel):
    map_metadata: list[MapMetadata]

    primary_map_tag: str

    lpeaks: list[float]

    output_beam_fwhm: float
    cov_smooth_factor: int

    io_suffix: str | None = None
    output_root: Path
    n_workers: int | None = None

    output_map: Path

    def to_wavelet_metadata(self) -> WaveletMetadata:
        primary_map = [m for m in self.map_metadata if m.tag == self.primary_map_tag][0]

        return WaveletMetadata.from_primary_map(
            filename=primary_map.path,
            basis=CosineNeedlet(self.lpeaks),
            output_beam_fwhm=self.output_beam_fwhm,
            cov_smooth_factor=self.cov_smooth_factor,
            output_root=self.output_root,
            io_suffix=self.io_suffix,
        )

    def to_maps(self) -> list[Map]:
        return [m.to_map() for m in self.map_metadata]


def main():
    from pydantic_yaml import parse_yaml_raw_as
    import argparse as ap
    from .core import coadd
    from .wavelets import wavelet_prepare, wavelet_to_map
    from dask.distributed import Client

    parser = ap.ArgumentParser(
        description=(
            "Parse an input JSON/yml file and return a WaveletMetadata and associated Maps. "
            "This will then be coadded, creating a final map."
        )
    )

    parser.add_argument(
        "input_file", type=Path, help="The input JSON/yml file to parse."
    )

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = f.read()

    input_data = parse_yaml_raw_as(CoberusInput, data)
    wavelet_metadata = input_data.to_wavelet_metadata()
    maps = input_data.to_maps()
    primary_map = [m.path for m in maps if m.tag == input_data.primary_map_tag][0]
    primary_mask = [m.mask for m in maps if m.tag == input_data.primary_map_tag][0]

    with Client(n_workers=input_data.n_workers) as client:
        coadders = wavelet_prepare(client=client, metadata=wavelet_metadata, maps=maps)

        result_maps = {
            s: coadd(client=client, coadder=coadder) for s, coadder in coadders.items()
        }

        coadded_map = wavelet_to_map(
            primary_map=primary_map,
            primary_mask=primary_mask,
            metadata=wavelet_metadata,
            result_maps=result_maps,
        )

        # Save the final map
        coadded_map.save(input_data.output_map)

    # Clean up intermediate files.
    for _, coadder in coadders.items():
        coadder.cleanup()
