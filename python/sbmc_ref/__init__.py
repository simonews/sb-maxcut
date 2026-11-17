"""P0: the Python reference of the project, in float64 with NumPy, independent of the C++ code.

model: rudy reader, couplings, total weight, Ising energy, direct cut, automatic c0 (exact
integers, c0 in float64). sb: sign, pressure schedule, one SB step, a full SB run with the
best-cut rule, and the best-cut rule applied to recorded states (used by T8).
"""

from . import model, sb
from .model import (
    Graph,
    auto_c0,
    coupling_matrix,
    cut_direct,
    ising_energy,
    read_rudy,
    total_weight,
)
from .sb import SbRun, best_from_states, pressure_schedule, run_sb, sb_step, sign

__version__ = "0.1.0"

__all__ = [
    "model",
    "sb",
    "Graph",
    "read_rudy",
    "coupling_matrix",
    "total_weight",
    "ising_energy",
    "cut_direct",
    "auto_c0",
    "SbRun",
    "sign",
    "pressure_schedule",
    "sb_step",
    "run_sb",
    "best_from_states",
]
