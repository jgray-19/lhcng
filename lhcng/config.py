"""
config.py
=========

This module defines configuration settings and constants for the lhcng package,
including directory paths, accelerator model settings, and RDT definitions for
MAD-NG operations with the LHC.
"""

import os
from pathlib import Path

# BASE DIRECTORY now set to the directory of the importing file
CURRENT_DIR = Path(os.getcwd()).resolve()

# Directory definitions
ANALYSIS_DIR = CURRENT_DIR / "analysis"
FREQ_OUT_DIR = ANALYSIS_DIR / "lin_files"
DATA_DIR = CURRENT_DIR / "data"
ACC_MODELS = CURRENT_DIR / "acc-models-lhc"
PLOT_DIR = CURRENT_DIR / "plots"

# Ensure that output directories exist
ANALYSIS_DIR.mkdir(exist_ok=True)
FREQ_OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# If the accelerator models directory does not exist, create a symbolic link
if not ACC_MODELS.exists():
    os.system(f"ln -s /afs/cern.ch/eng/acc-models/lhc/2024/ {ACC_MODELS}")

# RDT Definitions for the LHC (MAD-NG)
# Normal sextupole RDTs
NORMAL_SEXTUPOLE_RDTS = (
    "f1200_x",
    "f3000_x",
    "f1002_x",
    "f1020_x",
    "f0111_y",
    "f0120_y",
    "f1011_y",
    "f1020_y",
)

# Skew sextupole RDTs
SKEW_SEXTUPOLE_RDTS = (
    "f0012_y",
    "f0030_y",
    "f1101_x",
    "f1110_x",
    "f2001_x",
    "f2010_x",
    "f0210_y",
    "f2010_y",
)

# Normal octupole RDTs (if needed)
NORMAL_OCTUPOLE_RDTS = (
    "f1300_x",
    "f4000_x",
    "f0013_y",
    "f0040_y",
    "f1102_x",
    "f1120_x",
    "f2002_x",
    "f2020_x",
    "f0211_y",
    "f0220_y",
    "f2011_y",
    "f2020_y",
)

# Skew octupole RDTs (if needed)
SKEW_OCTUPOLE_RDTS = (
    "f0112_y",
    "f0130_y",
    "f0310_y",
    "f1003_x",
    "f1012_y",
    "f1030_x",
    "f1030_y",
    "f1201_x",
    "f1210_x",
    "f3001_x",
    "f3010_x",
    "f3010_y",
)

# Additional RDT sets for coupling (if used in tests or analysis)
COUPLING_RDTS = [
    "f1001",
    "f1010",
]

# Combined RDTs (if you need a single list of all available RDTs)
ALL_RDTS = (
    NORMAL_SEXTUPOLE_RDTS
    + SKEW_SEXTUPOLE_RDTS
    + NORMAL_OCTUPOLE_RDTS
    + SKEW_OCTUPOLE_RDTS
)

# Optionally, you can also define model-specific directories or prefixes here.
