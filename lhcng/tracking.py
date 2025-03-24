"""
tracking.py
===========

This module provides functions to run tracking simulations for the LHC using MAD‐NG.
It includes routines to:
  - Run a tracking simulation over a specified number of turns.
  - Execute Harpy frequency analysis on the tracking data.

Functions:
    run_tracking(beam: int, nturns: int, kick_amp: float = 1e-3) -> tfs.TfsDataFrame
    run_harpy(beam: int, linfile_dir: Path = None, clean: bool = False) -> None
"""

import logging
from pathlib import Path

import tfs
from pymadng import MAD

from .config import DATA_DIR, FREQ_OUT_DIR
from .model import initialise_model

logger = logging.getLogger(__name__)


def get_file_suffix(beam: int, nturns: int) -> str:
    """
    Return a file suffix based on the beam number and number of turns.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    nturns : int
        Number of turns.

    Returns
    -------
    str
        Suffix in the format "b<beam>_<nturns>t".
    """
    assert beam in [1, 2], "Beam must be 1 or 2"
    return f"b{beam}_{nturns}t"


def get_tfs_path(beam: int, nturns: int) -> Path:
    """
    Return the path for the TFS file for given beam and number of turns.

    Parameters
    ----------
    beam : int
        Beam number.
    nturns : int
        Number of turns.

    Returns
    -------
    Path
        Path to the TFS file.
    """
    return DATA_DIR / f"{get_file_suffix(beam, nturns)}.tfs.bz2"


def get_tbt_path(beam: int, nturns: int, index: int | str) -> Path:
    """
    Return the path for the TBT file for given beam, number of turns, and index.

    Parameters
    ----------
    beam : int
        Beam number.
    nturns : int
        Number of turns.
    index : int
        Index for the file (if multiple TBT files are generated).

    Returns
    -------
    Path
        Path to the TBT file.
    """
    suffix = get_file_suffix(beam, nturns) + f"_{index}"
    return DATA_DIR / f"tbt_{suffix}.sdds"


def get_tbt_name(beam: int, sdds: bool = True) -> str:
    """
    Return the TBT filename for the given beam.

    Parameters
    ----------
    beam : int
        Beam number.
    sdds : bool, optional
        If True, use "sdds" extension, otherwise "tfs.bz2".

    Returns
    -------
    str
        TBT filename.
    """
    ext = "sdds" if sdds else "tfs.bz2"
    return f"tbt_data_b{beam}.{ext}"


def run_tracking(beam: int, nturns: int, kick_amp: float = 1e-3) -> tfs.TfsDataFrame:
    """
    Run a tracking simulation using MAD-NG for a given beam over a specified number of turns.

    This function initializes the model, applies a kick amplitude to the beam,
    and tracks the beam for `nturns` turns. It returns the resulting tracking data as
    a TFS DataFrame.

    Parameters
    ----------
    beam : int
        The beam number (1 or 2).
    nturns : int
        Number of turns to simulate.
    kick_amp : float, optional
        Kick amplitude to apply to the beam (default is 1e-3).

    Returns
    -------
    tfs.TfsDataFrame
        A TFS DataFrame containing the tracking results with columns such as "name", "x",
        "y", "eidx", "turn", and "id".
    """
    with MAD() as mad:
        # Initialize the model in MAD‐NG.
        initialise_model(mad, beam)

        # Run the tracking simulation. The tracking command in MAD‐NG uses the "track" command.
        mad.send(f"""
local kick_amp = py:recv();

local tws = twiss {{sequence = MADX.lhcb{beam}}}

local betx = tws["beta11"][1];
local bety = tws["beta22"][1];

local sqrt_betx = math.sqrt(betx);
local sqrt_bety = math.sqrt(bety);

print("Found beta functions: betx =", betx, "bety =", bety, "at", tws:name_of(1));
local X0 = {{x = kick_amp * sqrt_betx, y = -kick_amp * sqrt_bety, px = 0, py = 0, t = 0, pt = 0}};
print("Running tracking for beam {beam} with X0:", X0.x, X0.y);

local t0 = os.clock();
mtbl = track {{sequence = MADX.lhcb{beam}, nturn = {nturns}, X0 = X0, info=3}};
print("Tracking runtime:", os.clock() - t0);
        """).send(kick_amp)

        # Retrieve the tracking table as a pandas DataFrame.
        df = mad.mtbl.to_df(columns=["name", "x", "y", "eidx", "turn", "id"])
        logger.info(f"Tracking complete for beam {beam} over {nturns} turns.")
    return df
