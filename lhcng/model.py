"""
model.py
========

This module provides functions for creating and manipulating the LHC accelerator model
using MAD-NG. It includes routines to create the model directory, generate the MAD-X sequence,
initialise the model within MAD-NG, and perform tune matching.
"""

from pathlib import Path

from cpymad.madx import Madx
from omc3.model_creator import create_instance_and_model
from pymadng import MAD

from .config import ACC_MODELS, CURRENT_DIR
from .model_constants import MODEL_COLUMNS, MODEL_HEADER, MODEL_STRENGTHS
from .tfs_utils import export_tfs_to_madx

# Define the MADX job filename (as created by the model_creator)
MADX_FILENAME = "job.create_model_nominal.madx"


def get_model_dir(beam: int) -> Path:
    """
    Return the model directory for the given beam.

    For example, for beam 1 this returns <CURRENT_DIR>/model_b1.
    """
    assert beam in [1, 2], "Beam must be 1 or 2"
    model_dir = CURRENT_DIR / f"model_b{beam}"
    model_dir.mkdir(exist_ok=True)
    return model_dir


def create_model_dir(
    beam: int,
    *,
    coupling_knob: bool | float = False,
    optics_type: str = "nominal",
    year: str = "2024",
    driven_excitation: str = "acd",
    energy: float = 6800.0,
    nat_tunes: list[float] = [0.28, 0.31],
    drv_tunes: list[float] | None = None,
    modifiers: Path | list[Path] = ACC_MODELS
    / "operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx",
    **kwargs,
) -> None:
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
    model_dir = get_model_dir(beam)

    if isinstance(modifiers, (Path, str)):
        modifiers = [modifiers]

    if drv_tunes is None:
        drv_tunes = nat_tunes

    create_instance_and_model(
        accel="lhc",
        fetch="afs",
        type=optics_type,
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

    # Read and process the generated MADX file
    madx_file = model_dir / MADX_FILENAME
    with open(madx_file, "r") as f:
        lines = f.readlines()

    # Generate the MAD-X sequence and update with MAD-NG
    make_madx_seq(model_dir, lines, beam, coupling_knob)
    update_model_with_ng(beam)

    # Generate beam4 sequence for tracking if beam 2
    if beam == 2:
        make_madx_seq(beam, model_dir, lines, coupling_knob, beam4=True)


def make_madx_seq(
    beam: int,
    model_dir: Path,
    lines: list[str],
    coupling_knob: bool | float,
    beam4: bool = False,
) -> None:
    """
    Generate the MAD-X sequence file for the given beam.

    If beam4 is True, adjust the sequence for beam 4 settings (used for tracking).
    """
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
            # Process only the first 32 lines (as in the original workflow)
            if "coupling_knob" in line:
                madx.input(line)
                if coupling_knob is not False:
                    madx.input(f"cmrs.b{beam} = {coupling_knob};")
                break
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


def update_model_with_ng(beam: int) -> None:
    """
    Update the accelerator model with MAD-NG and perform tune matching.

    This routine loads the saved sequence into MAD-NG, initialises the beam parameters,
    and then calls the tune matching routine.
    """
    model_dir = get_model_dir(beam)
    with MAD() as mad:
        seq_dir = -1 if beam == 2 else 1
        start_madng(mad, beam, sequence_direction=seq_dir)
        mad.send(f"""
-- Set the twiss table information needed for the model update
hnams = py:recv()
cols = py:recv()
str_cols = py:recv()

cols = MAD.utility.tblcat(cols, str_cols)

-- Calculate the twiss parameters with coupling and observe the BPMs
! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}} 

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
twiss_ac   = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
twiss_data = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
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
        *, 
        sequence_direction: int = 1
    ) -> None:
    """
    Initialise the accelerator model within MAD-NG.

    Loads the saved sequence and sets the beam parameters.
    """
    model_dir = get_model_dir(beam)
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
    match_tunes(mad, beam)


def _print_tunes(mad: MAD, beam: int, sdir: int, label: str) -> None:
    mad.send(f"""
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, dir={sdir}}};
print("{label} tunes: ", tbl.q1, tbl.q2);
    """)


def match_tunes(mad: MAD, beam: int) -> None:
    """
    Match the tunes of the model to the desired values using MAD-NG.

    The target tunes are hardcoded (62.28 and 60.31 in absolute value) and the routine
    uses a matching command to adjust the optics accordingly.
    """
    sdir = 1 if beam == 1 else -1
    _print_tunes(mad, beam, sdir, "Initial")
    mad.send(f"""
match {{
  command := twiss {{sequence=MADX.lhcb{beam}, mapdef=4, dir={sdir}}},
  variables = {{
    rtol=1e-6,
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{
    {{ expr = \\t -> math.abs(t.q1)-62.28, name='q1' }},
    {{ expr = \\t -> math.abs(t.q2)-60.31, name='q2' }},
  }},
  objective = {{ fmin=1e-7 }},
}};
py:send("match complete");
    """)
    _print_tunes(mad, beam, sdir, "Final")
    response = mad.receive()
    if response != "match complete":
        raise RuntimeError("Error in matching tunes: " + str(response))
