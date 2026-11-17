"""Simulated bifurcation (bSB and dSB) in float64, independent of the C++ code.

At step m, with the field F computed from the positions before the step,
    y <- y + dt * (-p[m] * x + c0 * F),    x <- x + dt * a0 * y,
then the inelastic walls (x > 1 or x < -1: x <- sgn(x), y <- 0), with F = J sign(x) in dSB
(discrete) and F = J x in bSB (ballistic), sign(0) = +1 and p[m] = a0 * (1 - (m + 1) / M).
Best-cut rule of a replica (F04-C1, section 5.3): the configurations s(m) = sign(x) before
step m are evaluated at every step in dSB and, in bSB, only when eval_every > 0 and
m mod eval_every == 0; s(M), after the last step, is always evaluated. A configuration becomes
the best one when it is the first evaluated or strictly better: ties keep the smallest m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import ising_energy, total_weight

__all__ = ["SbRun", "sign", "pressure_schedule", "sb_step", "run_sb", "best_from_states"]


@dataclass
class SbRun:
    """Result of run_sb: arrays of R integers, and the recorded states (record x R x n)."""

    best_energy: np.ndarray
    best_cut: np.ndarray
    best_step: np.ndarray
    final_cut: np.ndarray
    final_x: np.ndarray
    final_y: np.ndarray
    states_x: Optional[np.ndarray] = field(default=None)
    states_y: Optional[np.ndarray] = field(default=None)


def sign(x: np.ndarray) -> np.ndarray:
    """int64 array: -1 where x < 0, +1 otherwise (sign(0) = sign(-0) = +1)."""
    return np.where(np.asarray(x) < 0, -1, 1).astype(np.int64)


def pressure_schedule(a0: float, steps: int) -> np.ndarray:
    """p[m] = a0 * (1 - (m + 1) / steps), m = 0 .. steps - 1, in float64."""
    m = np.arange(steps, dtype=np.float64)
    return a0 * (1.0 - (m + 1.0) / steps)


def _rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.newaxis, :] if x.ndim == 1 else x


def sb_step(
    j: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    dt: float,
    a0: float,
    c0: float,
    discrete: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One step for every replica; x and y are R x n float64 matrices."""
    x = _rows(x)
    y = _rows(y)
    if discrete:
        f = (sign(x) @ np.asarray(j, dtype=np.int64).T).astype(np.float64)
    else:
        f = x @ np.asarray(j, dtype=np.float64).T
    y = y + dt * (-p * x + c0 * f)
    x = x + dt * a0 * y
    wall = (x > 1.0) | (x < -1.0)
    x = np.where(wall, np.where(x < 0.0, -1.0, 1.0), x)
    y = np.where(wall, 0.0, y)
    return x, y


class _Best:
    """Best configuration of every replica, with the rule of the module docstring."""

    def __init__(self, replicas: int) -> None:
        self.energy = np.zeros(replicas, dtype=np.int64)
        self.step = np.zeros(replicas, dtype=np.int64)
        self.evaluated = np.zeros(replicas, dtype=bool)

    def consider(self, energies: np.ndarray, m: int) -> None:
        better = ~self.evaluated | (energies < self.energy)
        self.energy = np.where(better, energies, self.energy)
        self.step = np.where(better, m, self.step)
        self.evaluated |= better


def _evaluated(m: int, discrete: bool, eval_every: int) -> bool:
    return discrete or (eval_every > 0 and m % eval_every == 0)


def run_sb(
    j: np.ndarray,
    x0: np.ndarray,
    steps: int,
    dt: float,
    a0: float,
    c0: float,
    discrete: bool,
    eval_every: int = 0,
    record: int = 0,
) -> SbRun:
    """Run SB from x0 (one row per replica) and y = 0; record the states after the first steps."""
    x = _rows(x0).copy()
    y = np.zeros_like(x)
    p = pressure_schedule(a0, steps)
    best = _Best(x.shape[0])
    states_x, states_y = [], []
    for m in range(steps):
        if _evaluated(m, discrete, eval_every):
            best.consider(np.atleast_1d(ising_energy(j, sign(x))), m)
        x, y = sb_step(j, x, y, p[m], dt, a0, c0, discrete)
        if m < record:
            states_x.append(x.copy())
            states_y.append(y.copy())
    final_energy = np.atleast_1d(ising_energy(j, sign(x)))
    best.consider(final_energy, steps)
    w = total_weight(j)
    return SbRun(
        best_energy=best.energy,
        best_cut=(w - best.energy) // 2,
        best_step=best.step,
        final_cut=(w - final_energy) // 2,
        final_x=x,
        final_y=y,
        states_x=np.array(states_x) if record > 0 else None,
        states_y=np.array(states_y) if record > 0 else None,
    )


def best_from_states(
    j: np.ndarray,
    x0: np.ndarray,
    states_x: np.ndarray,
    discrete: bool,
    eval_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-cut rule on s(0) = sign(x0), s(k + 1) = sign(states_x[k]), k = 0 .. M - 1.

    states_x[k] is the state after step k + 1 (M x R x n). Returns (best_energy, best_step,
    final_energy) per replica, final_energy being E(s(M)).
    """
    configurations = [_rows(x0)] + [_rows(s) for s in np.asarray(states_x, dtype=np.float64)]
    steps = len(configurations) - 1
    best = _Best(configurations[0].shape[0])
    for m in range(steps):
        if _evaluated(m, discrete, eval_every):
            best.consider(np.atleast_1d(ising_energy(j, sign(configurations[m]))), m)
    final_energy = np.atleast_1d(ising_energy(j, sign(configurations[steps])))
    best.consider(final_energy, steps)
    return best.energy, best.step, final_energy
