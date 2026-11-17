"""Ising model of Max-Cut in exact integers, independent of the C++ code.

Conventions (reference document, section 5): J_ij = -w_ij, J_ii = 0;
W = sum_{i<j} w_ij, E(s) = sum_{i<j} w_ij s_i s_j, C(s) = sum_{i<j} w_ij [s_i != s_j],
so that 2C = W - E. Weights, energies and cuts are exact integers (NumPy int64); the automatic
c0 is a float64.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

__all__ = [
    "Graph",
    "read_rudy",
    "coupling_matrix",
    "total_weight",
    "ising_energy",
    "cut_direct",
    "auto_c0",
]

_INTEGER = re.compile(r"[+-]?[0-9]+")
_BLANKS = re.compile(r"[ \t]+")
_MAX_NODES = 2**31 - 1
_MAX_ABS_WEIGHT = 2**31 - 1


@dataclass(frozen=True)
class Graph:
    """Canonical graph: n nodes and the edges (u[k], v[k], w[k]), 0-based, u < v, sorted."""

    n: int
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray


def _tokens(line: str) -> list[str]:
    """Tokens of a line separated by spaces or tabs; empty for a blank line."""
    stripped = line.strip(" \t")
    return _BLANKS.split(stripped) if stripped else []


def _integer(token: str, where: str) -> int:
    if _INTEGER.fullmatch(token) is None:
        raise ValueError(f"{where}: {token!r} is not an integer")
    return int(token)


def read_rudy(path: Union[str, Path]) -> Graph:
    """Read a graph in the rudy format of the G-set.

    Format: a header line "N M", then exactly M lines "i j w" with 1-based node indices,
    i != j, an integer w != 0 and no edge repeated in either direction. Blank lines are ignored
    and one trailing carriage return per line is removed. Every violation raises ValueError,
    with the line number when there is one.
    """
    path = Path(path)
    text = path.read_bytes().decode("ascii", errors="strict")
    header = None
    edges: dict[tuple[int, int], tuple[int, int]] = {}  # (u, v) -> (w, line)
    for number, line in enumerate(text.split("\n"), start=1):
        if line.endswith("\r"):
            line = line[:-1]
        tokens = _tokens(line)
        if not tokens:
            continue
        where = f"{path}:{number}"
        if header is None:
            if len(tokens) != 2:
                raise ValueError(f"{where}: the header must be 'N M'")
            n, m = (_integer(t, where) for t in tokens)
            if not 1 <= n <= _MAX_NODES:
                raise ValueError(f"{where}: N = {n} is out of range")
            if not 0 <= m <= n * (n - 1) // 2:
                raise ValueError(f"{where}: M = {m} is out of range")
            header = (n, m)
            continue
        n, m = header
        if len(edges) == m:
            raise ValueError(f"{where}: unexpected line after the {m} edge lines")
        if len(tokens) != 3:
            raise ValueError(f"{where}: an edge line must be 'i j w'")
        i, j, w = (_integer(t, where) for t in tokens)
        if not (1 <= i <= n and 1 <= j <= n):
            raise ValueError(f"{where}: node index out of [1, {n}]")
        if i == j:
            raise ValueError(f"{where}: self-loop on node {i}")
        if w == 0 or abs(w) > _MAX_ABS_WEIGHT:
            raise ValueError(f"{where}: weight {w} is zero or out of range")
        key = (min(i, j) - 1, max(i, j) - 1)
        if key in edges:
            raise ValueError(f"{where}: edge {i} {j} repeats the edge of line {edges[key][1]}")
        edges[key] = (w, number)
    if header is None:
        raise ValueError(f"{path}: no header line")
    if len(edges) != header[1]:
        raise ValueError(f"{path}: {header[1]} edge lines expected, {len(edges)} found")
    keys = sorted(edges)
    return Graph(
        n=header[0],
        u=np.array([k[0] for k in keys], dtype=np.int64),
        v=np.array([k[1] for k in keys], dtype=np.int64),
        w=np.array([edges[k][0] for k in keys], dtype=np.int64),
    )


def coupling_matrix(graph: Graph) -> np.ndarray:
    """n x n int64 matrix with J_ij = J_ji = -w_ij and a zero diagonal."""
    j = np.zeros((graph.n, graph.n), dtype=np.int64)
    j[graph.u, graph.v] = -graph.w
    j[graph.v, graph.u] = -graph.w
    return j


def _upper_weights(j: np.ndarray) -> np.ndarray:
    """w_ij = -J_ij on the strict upper triangle, 0 elsewhere."""
    return -np.triu(np.asarray(j, dtype=np.int64), k=1)


def _as_rows(s: np.ndarray) -> tuple[np.ndarray, bool]:
    s = np.asarray(s, dtype=np.int64)
    return (s[np.newaxis, :], True) if s.ndim == 1 else (s, False)


def total_weight(j: np.ndarray) -> int:
    """W = sum_{i<j} w_ij."""
    return int(_upper_weights(j).sum())


def ising_energy(j: np.ndarray, s: np.ndarray) -> Union[int, np.ndarray]:
    """E(s) = sum_{i<j} w_ij s_i s_j for a spin vector, or per row of an R x n matrix."""
    w = _upper_weights(j)
    rows, single = _as_rows(s)
    energies = np.einsum("ri,ij,rj->r", rows, w, rows)
    return int(energies[0]) if single else energies


def cut_direct(j: np.ndarray, s: np.ndarray) -> Union[int, np.ndarray]:
    """C(s) = sum_{i<j} w_ij [s_i != s_j], counted directly, per spin vector or row."""
    w = _upper_weights(j)
    rows, single = _as_rows(s)
    cuts = np.array(
        [int((w * (row[:, np.newaxis] != row[np.newaxis, :])).sum()) for row in rows],
        dtype=np.int64,
    )
    return int(cuts[0]) if single else cuts


def auto_c0(j: np.ndarray) -> float:
    """c0 = 1 / (2 sigma sqrt(n)), sigma = numpy.std (ddof = 0) of the off-diagonal J_ij."""
    j = np.asarray(j, dtype=np.int64)
    n = j.shape[0]
    off_diagonal = j[~np.eye(n, dtype=bool)]
    sigma = float(np.std(off_diagonal.astype(np.float64)))
    return 1.0 / (2.0 * sigma * math.sqrt(n))
