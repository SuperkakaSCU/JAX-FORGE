"""Shared paths for the 400 C isothermal series (roll2, roll3.1, roll3.2).

Every script takes the series from the AGF_SERIES environment variable or a
--series flag and resolves its inputs here, so the same code runs on all three
without edits.  Outputs go to data_400C/calibration/<series>/.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data_400C" / "info" / "raw"
INFO = ROOT / "data_400C" / "info"
IMG = ROOT / "data_400C" / "img"

SERIES = os.environ.get("AGF_SERIES", "roll2")          # roll2 | roll3.1 | roll3.2


def out_dir(series=None):
    d = ROOT / "data_400C" / "calibration" / (series or SERIES)
    d.mkdir(parents=True, exist_ok=True)
    return d


OUT = out_dir()
MCAP = RAW / f"{SERIES}_stripped.mcap"
H5 = RAW / f"{SERIES}.h5"
PT = RAW / f"{SERIES}.pt"
ACTIONS = RAW / f"{SERIES}_toolpath.jsonl"
DIE_STL = INFO / "torm_tool_lib" / "Tool2.stl"

D_STOCK_MM = 38.1
R_STOCK_MM = 19.05
L_STOCK_MM = 100.0          # exposed length; roll1 was 75
T_ISO_C = 400.0             # isothermal assumption, no temperature record exists
