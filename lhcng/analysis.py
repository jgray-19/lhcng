"""
optics.py
=========

This module contains functions for performing optics analysis for the LHC
using MAD-NG. It provides routines to:
    - Extract RDTs from a Turn-by-Turn (TBT) file.
    - Run Harpy frequency analysis on the TBT data.

The analysis is based on TFS file handling via the tfs package and the
omc3.hole_in_one entrypoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tfs
from omc3.hole_in_one import hole_in_one_entrypoint

from .config import ALL_RDTS, ANALYSIS_DIR, FREQ_OUT_DIR
from .model_compressor import ModelCompressor
from .tfs_utils import filter_out_BPM_near_IPs

logger = logging.getLogger(__name__)


def get_output_dir(tbt_name: str, output_dir: Path = None) -> Path:
    """
    Return (and create, if needed) the output directory based on the TBT filename.

    Parameters
    ----------
    tbt_name : str
        The name of the TBT file.
    output_dir : Path, optional
        Custom output directory. If None, it is created under ANALYSIS_DIR.

    Returns
    -------
    Path
        The output directory.
    """
    if output_dir is None:
        output_dir = ANALYSIS_DIR / f"{tbt_name.split('.')[0]}"
        output_dir.mkdir(exist_ok=True)
    return output_dir


def get_rdt_type(rdt: str) -> tuple[str, str]:
    """
    Determine the type of an RDT based on its naming.

    The function assumes an RDT name in the format "f####_x" or "f####_y" where the digits
    define its order. It returns a tuple:
      - First element is "skew" if the sum of the third and fourth digits is odd; otherwise "normal".
      - Second element is "octupole" if the sum of the digits equals 4, else "sextupole".

    Parameters
    ----------
    rdt : str
        RDT name.

    Returns
    -------
    tuple[str, str]
        A tuple (type, order) e.g. ("normal", "sextupole").
    """
    # Extract the digits after the initial 'f'
    rdt_numbers = [int(num) for num in rdt.split("_")[0][1:]]
    is_skew = (rdt_numbers[2] + rdt_numbers[3]) % 2 == 1
    order = sum(rdt_numbers)
    return ("skew" if is_skew else "normal", "octupole" if order == 4 else "sextupole")


def get_rdt_paths(rdts: list[str], output_dir: Path) -> dict[str, Path]:
    """
    Return a dictionary mapping each RDT to its corresponding TFS file path.

    The file paths are constructed based on the RDT type and order.

    Parameters
    ----------
    rdts : list[str]
        List of RDT names.
    output_dir : Path
        Output directory for the analysis.

    Returns
    -------
    dict[str, Path]
        Dictionary where keys are RDT names and values are file paths.
    """
    rdt_paths = {}
    for rdt in rdts:
        if rdt.lower() in ["f1001", "f1010"]:
            rdt_paths[rdt] = output_dir / f"{rdt}.tfs"
            continue
        rdt_type, order_name = get_rdt_type(rdt)
        # Example folder structure: <output_dir>/<rdt_type>_<order>/<rdt>.tfs
        rdt_paths[rdt] = output_dir / "rdt" / f"{rdt_type}_{order_name}" / f"{rdt}.tfs"
    return rdt_paths


def get_tunes(output_dir: Path) -> list[float]:
    """
    Extract the tunes from the optics analysis file.

    Assumes that the file "beta_amplitude_x.tfs" is present in the output directory
    and contains headers with keys "Q1" and "Q2".

    Parameters
    ----------
    output_dir : Path
        Directory where the optics analysis file is located.

    Returns
    -------
    list[float]
        A list containing the two tunes.
    """
    optics_file = output_dir / "f1001.tfs"
    headers = tfs.reader.read_headers(optics_file)
    return [headers["Q1"], headers["Q2"]]


def get_rdts_from_optics_analysis(
    beam: int,
    tbt_path: Path,
    model_dir: Path,
    output_dir: Path = None,
    rdts: list[str] = ALL_RDTS,
    compensation="none",
) -> dict[str, tfs.TfsDataFrame]:
    """
    Run the optics analysis to extract RDTs for the given beam using a TBT file.

    This function will:
      - Determine the output directory based on the TBT file name.
      - Construct file paths for each RDT.
      - Invoke the optics analysis via the hole_in_one entrypoint.
      - Read the generated TFS files, filter them, and return them in a dictionary.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    tbt_path : Path
        Path to the TBT file.
    output_dir : Path, optional
        Directory to store output files. If not provided, it is created based on tbt_path.

    Returns
    -------
    dict[str, tfs.TfsDataFrame]
        Dictionary mapping each RDT to its corresponding TFS DataFrame.
    """
    # Determine if only coupling analysis is needed.
    only_coupling = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
    # Define the RDT magnet order; for sextupoles use 3, adjust if needed.
    rdt_order = 3
    output_dir = get_output_dir(tbt_path.name, output_dir)

    # Build file paths for each RDT.
    rdt_paths = get_rdt_paths(rdts, output_dir)

    # Run optics analysis using omc3's hole_in_one_entrypoint.
    with ModelCompressor(model_dir):
        hole_in_one_entrypoint(
            files=[
                p.parent / p.stem
                for p in Path(FREQ_OUT_DIR).glob(f"{tbt_path.name}*.freqsx")
            ],
            outputdir=output_dir,
            year="2024",
            optics=True,
            accel="lhc",
            beam=beam,
            model_dir=model_dir,
            only_coupling=only_coupling,
            compensation="none",
            nonlinear=["rdt"],
            rdt_magnet_order=rdt_order,
        )

    tunes = get_tunes(output_dir)
    logger.info(f"Tunes for beam {beam}: {tunes}")

    # Read the generated TFS files and filter out unwanted BPM rows.
    rdts_dfs = {
        rdt: filter_out_BPM_near_IPs(tfs.read(path, index="NAME"))
        for rdt, path in rdt_paths.items()
    }
    return rdts_dfs


def run_harpy(
    tbt_files: Path | list[Path],
    beam: int,
    model_dir: Path,
    tunes: list[float] = [0.28, 0.31, 0.0],
    natdeltas: list[float] = [0.0, -0.0, 0.0],
    linfile_dir: Path = None,
    clean: bool = False,
) -> None:
    """
    Run Harpy frequency analysis on the Turn-by-Turn (TBT) file for the given beam.

    This function constructs the TBT file path and then calls the OMC3 Harpy entrypoint
    to generate frequency analysis outputs. Optionally, it can use SVD cleaning to remove
    noise from the data.

    Parameters
    ----------
    beam : int
        The beam number (1 or 2).
    linfile_dir : Path, optional
        Directory containing the linear output files. Defaults to FREQ_OUT_DIR.
    clean : bool, optional
        If True, perform SVD cleaning on the data (default is False).
    """
    from omc3.hole_in_one import hole_in_one_entrypoint

    if linfile_dir is None:
        linfile_dir = FREQ_OUT_DIR

    if isinstance(tbt_files, Path | str):
        tbt_files = [tbt_files]

    with ModelCompressor(model_dir):
        hole_in_one_entrypoint(
            harpy=True,
            files=tbt_files,
            outputdir=linfile_dir,
            to_write=["lin", "spectra"],
            opposite_direction=(beam == 2),
            tunes=tunes,
            natdeltas=natdeltas,
            clean=clean,
        )
        logger.info(f"Harpy analysis complete for beam {beam}.")
