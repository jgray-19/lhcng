"""
model.py
========

This module provides functions for creating and manipulating the LHC accelerator model
using MAD-NG. It includes routines to create the model directory, generate the MAD-X sequence,
initialise the model within MAD-NG, and perform tune matching.
"""

import os
import shutil
from pathlib import Path

from cpymad.madx import Madx
from omc3.model_creator import create_instance_and_model
from pymadng import MAD

from .config import ACC_MODELS, CURRENT_DIR
from .model_compressor import ModelCompressor
from .model_constants import MODEL_COLUMNS, MODEL_HEADER, MODEL_STRENGTHS
from .tfs_utils import export_tfs_to_madx

# Define the MADX job filename (as created by the model_creator)
MADX_FILENAME = "job.create_model_nominal.madx"


def get_folder_suffix(
    beam: int,
    coupling_knob: bool | float = False,
    tunes: list[float] = [0.28, 0.31],
) -> str:
    """
    Return a file suffix based on the beam number and number of turns.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    coupling_knob : bool | float, optional
        Set the value of the cmrs.b<beam> knob for coupling (default is False, i.e. no coupling).
    tunes : list[float], optional
        Natural tunes (default is [0.28, 0.31]).

    Returns
    -------
    str
        Suffix in the format "b<beam>_<nturns>t_c<coupling>_t<tune1>_<tune2>".
    """
    assert beam in [1, 2], "Beam must be 1 or 2"
    coupling = ""
    if coupling_knob is not False:
        coupling = f"_c{coupling_knob}"

    tunes_str = f"t{tunes[0]}_{tunes[1]}"
    return f"b{beam}_{coupling}_{tunes_str}"


def get_model_dir(
    beam: int, coupling_knob: bool | float = False, tunes: list[float] = [0.28, 0.31]
) -> Path:
    """
    Return the model directory for the given beam.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    coupling_knob : bool | float, optional
        Set the value of the cmrs.b<beam> knob for coupling (default is False, i.e. no coupling).
    tunes : list[float], optional
        Natural tunes (default is [0.28, 0.31]).
    
    Returns
    -------
    Path
        Path to the model directory.
    """
    assert beam in [1, 2], "Beam must be 1 or 2"
    model_dir = CURRENT_DIR / ("model_" + get_folder_suffix(beam, coupling_knob, tunes))
    model_dir.mkdir(exist_ok=True)
    return model_dir


def create_model_dir(
    beam: int,
    *,
    coupling_knob: bool | float = False,
    # optics_type: str = "nominal",
    year: str = "2024",
    driven_excitation: str = "acd",
    energy: float = 6800.0,
    nat_tunes: list[float] = [0.28, 0.31],
    drv_tunes: list[float] | None = None,
    modifiers: Path | list[Path] = ACC_MODELS
    / "operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx",
    **kwargs,
) -> Path:
    """
    Create and initialise the accelerator model for the given beam.

    This routine generates the MAD-X sequence file, updates the model with MAD-NG,
    and generates the beam4 sequence for tracking if beam 2.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    coupling_knob : bool | float, optional
        Set the value of the cmrs.b<beam> knob for coupling (default is False, i.e. no coupling).
    optics_type : str, optional
        Optics type (default is "nominal").
    year : str, optional
        Year of the optics (default is "2024").
    driven_excitation : str, optional
        Driven excitation type (default is "acd").
    energy : float, optional
        Beam energy in GeV (default is 6800.0 GeV).
    nat_tunes : list[float], optional
        Natural tunes (default is [0.28, 0.31]).
    drv_tunes : list[float] | None, optional
        Driven tunes (default is None, i.e. use natural tunes).
    modifiers : Path | list[Path], optional
        Path to the MAD-X modifiers file or list of modifiers (default is the nominal optics file).
    **kwargs
        Additional keyword arguments to pass to the omc3.create_instance_and_model function.
    """
    model_dir = get_model_dir(beam, coupling_knob, nat_tunes)

    if isinstance(modifiers, (Path, str)):
        modifiers = [modifiers]

    if drv_tunes is None:
        drv_tunes = nat_tunes

    create_instance_and_model(
        accel="lhc",
        fetch="path",
        path = ACC_MODELS,
        # type=optics_type,
        beam=beam,
        year=year,
        driven_excitation=driven_excitation,
        energy=energy,
        nat_tunes=nat_tunes,
        drv_tunes=drv_tunes,
        modifiers=modifiers,
        outputdir=model_dir,
        **kwargs,
    )

    # Generate the MAD-X sequence and update with MAD-NG
    make_madx_seq(beam, model_dir, coupling_knob)
    update_model_with_ng(beam, model_dir, tunes=nat_tunes)

    # Generate beam4 sequence for tracking if beam 2
    if beam == 2:
        make_madx_seq(beam, model_dir, coupling_knob, beam4=True)

    ModelCompressor.compress_model_folder(model_dir)
    return model_dir

def model_to_ng(
    model_dir: Path,
    beam: int = 1,
    year = "2025",
    coupling_knob: bool | float = False,
    tunes: list[float] = [0.28, 0.31],
    beam4: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Convert the model in the given directory to MAD-NG format.

    Parameters
    ----------
    model_dir : Path
        Path to the model directory containing the MAD-X sequence file.
    beam : int, optional
        Beam number (1 or 2) for which the model is being converted (default is 1).
    coupling_knob : bool | float, optional
        Set the value of the cmrs.b<beam> knob for coupling (default is False, i.e. no coupling).
    tunes : list[float], optional
        Natural tunes (default is [0.28, 0.31]).
    beam4 : bool, optional
        If True, generate the sequence for beam 4 (default is False).
    out_dir : Path | None, optional
        Output directory for the converted model. If None, uses the model_dir (default is None).
    """

    # Copy the model directory to the output directory
    if out_dir and out_dir != model_dir:
        shutil.copytree(model_dir, out_dir, dirs_exist_ok=True,  ignore=shutil.ignore_patterns("acc-models-lhc"))
        model_dir = out_dir
        os.system(f"ln -s /afs/cern.ch/eng/acc-models/lhc/{year}/ {model_dir/'acc-models-lhc'}")
    
    # First create the MAD-X sequence file
    make_madx_seq(beam, model_dir, coupling_knob, beam4)
    update_model_with_ng(beam, model_dir, tunes=tunes)

    if beam4:
        # Generate the beam4 sequence for tracking
        make_madx_seq(beam, model_dir, coupling_knob, beam4=True)
    
    ModelCompressor.compress_model_folder(model_dir)

def make_madx_seq(
    beam: int,
    model_dir: Path,
    coupling_knob: bool | float,
    beam4: bool = False,
) -> None:
    """
    Generate the MAD-X sequence file for the given beam.

    If beam4 is True, adjust the sequence for beam 4 settings (used for tracking).
    """
    madx_file = model_dir / MADX_FILENAME
    with open(madx_file, "r") as f:
        lines = f.readlines()
    if beam4:
        assert beam == 2, "Beam 4 sequence can only be generated for beam 2"
        print("Generating beam4 sequence for tracking")

    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))
        for i, line in enumerate(lines):
            if beam4:
                if "define_nominal_beams" in line:
                    madx.input(
                        "beam, sequence=LHCB2, particle=proton, energy=450, kbunch=1, npart=1.15E11, bv=1;"
                    )
                    continue
                elif "acc-models-lhc/lhc.seq" in line:
                    line = line.replace(
                        "acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq"
                    )
            if "coupling_knob" in line:
                print(f"Setting coupling knob to {coupling_knob}")
                madx.input(line)
                if coupling_knob is not False:
                    madx.input(f"cmrs.b{beam} = {coupling_knob};")
                break  # The coupling knob is the last line to be read
            if "match_tunes" not in line:
                madx.input(line)
        madx.input(
            f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq", noexpr=false;
            """
        )


def add_strengths_to_twiss(mad: MAD, mtable_name: str) -> None:
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({mtable_name}, strength_cols)
    """).send(MODEL_STRENGTHS)


def observe_BPMs(mad: MAD, beam: int) -> None:
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:  select(observed, {{pattern="BPM"}})
    """)


def update_model_with_ng(beam: int, model_dir: Path, tunes: list[float] = [0.28, 0.31], deltap = "nil") -> None:
    """
    Update the accelerator model with MAD-NG and perform tune matching.

    This routine loads the saved sequence into MAD-NG, initialises the beam parameters,
    and then calls the tune matching routine.
    """
    with MAD() as mad:
        seq_dir = -1 if beam == 2 else 1
        start_madng(mad, beam, model_dir, tunes=tunes, sequence_direction=seq_dir)
        mad.send(f"""
-- Set the twiss table information needed for the model update
hnams = py:recv()
cols = py:recv()
str_cols = py:recv()

cols = MAD.utility.tblcat(cols, str_cols)

-- Calculate the twiss parameters with coupling and observe the BPMs
! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, deltap={deltap} }} 

-- Select everything
twiss_elements:select(nil, \ -> true)

-- Deselect the drifts
twiss_elements:deselect{{pattern="drift"}}
""")
        mad.send(MODEL_HEADER).send(MODEL_COLUMNS).send(MODEL_STRENGTHS)
        add_strengths_to_twiss(mad, "twiss_elements")
        mad.send(
            # True below is to make sure only selected rows are written
            f"""twiss_elements:write("{model_dir / "twiss_elements.dat"}", cols, hnams, true)"""
        )
        observe_BPMs(mad, beam)
        mad.send(f"""
twiss_ac   = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1, deltap={deltap}}}
twiss_data = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1, deltap={deltap}}}
        """)
        add_strengths_to_twiss(mad, "twiss_ac")
        add_strengths_to_twiss(mad, "twiss_data")
        mad.send(f"""
twiss_ac:write("{model_dir / "twiss_ac.dat"}", cols, hnams)
twiss_data:write("{model_dir / "twiss.dat"}", cols, hnams)
print("Replaced twiss data tables")
py:send("write complete")
""")
        assert mad.receive() == "write complete", "Error in writing twiss tables"

    # Read the twiss data tables and then convert all the headers to uppercase and column names to uppercase
    export_tfs_to_madx(model_dir / "twiss_ac.dat")
    export_tfs_to_madx(model_dir / "twiss_elements.dat")
    export_tfs_to_madx(model_dir / "twiss.dat")


def start_madng(
    mad: MAD,
    beam: int,
    model_dir: Path,
    *,
    tunes: list[float] = [0.28, 0.31],
    deltap = "nil",
    sequence_direction: int = 1,
) ->  None:
    """
    Initialise the accelerator model within MAD-NG.

    Loads the saved sequence and sets the beam parameters.
    """
    saved_seq = model_dir / f"lhcb{beam}_saved.seq"
    saved_mad = model_dir / f"lhcb{beam}_saved.mad"

    mad.MADX.load(f"'{saved_seq}'", f"'{saved_mad}'")
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy=450}};
MADX.lhcb{beam}.beam = lhc_beam;
MADX.lhcb{beam}.dir = {sequence_direction};
print("Initialising model with beam:", {beam}, "direction:", MADX.lhcb{beam}.dir);
    """)
    if sequence_direction == 1 and beam == 1:
        mad.send('MADX.lhcb1:cycle("MSIA.EXIT.B1")')
    match_tunes(mad, beam, tunes, deltap=deltap)


def _print_tunes(mad: MAD, beam: int, sdir: int, label: str) -> tuple[float, float]:
    mad.send(f"""
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, dir={sdir}}};
py:send({{tbl.q1, tbl.q2}}, true)
    """)
    q1, q2 = mad.recv() # type: ignore
    assert isinstance(q1, float) and isinstance(q2, float), "Received tunes are not floats"
    print(f"{label} tunes: ", q1, q2)
    return q1, q2


def match_tunes(mad: MAD, beam: int, tunes: list[float] = [0.28, 0.31], deltap = "nil") -> None:
    """
    Match the tunes of the model to the desired values using MAD-NG.

    The target tunes are hardcoded (62.28 and 60.31 in absolute value) and the routine
    uses a matching command to adjust the optics accordingly.
    """
    sdir = 1
    q1, q2 = _print_tunes(mad, beam, sdir, "Initial")
    if abs(tunes[0] - q1 % 1) < 1e-6 and abs(tunes[1] - q2 % 1) < 1e-6:
        print("Tunes already matched, skipping matching.")
        return
    mad.send(f"""
match {{
  command := twiss {{sequence=MADX.lhcb{beam}, mapdef=4, dir={sdir}, deltap={deltap}}},
  variables = {{
    rtol=1e-6,
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{
    {{ expr = \\t -> math.abs(t.q1)-(62+{tunes[0]}), name='q1' }},
    {{ expr = \\t -> math.abs(t.q2)-(60+{tunes[1]}), name='q2' }},
  }},
  objective = {{ fmin=1e-7 }},
}};
    """)
    _print_tunes(mad, beam, sdir, "Final")