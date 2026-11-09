"""Locate a point in a tetrahedral mesh without mutating it.

A barycentric ray walk (Marcum 1995) uses face adjacency; an octree seed and
fallback walks precede a brute-force scan. Return -1 outside the domain.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .kernels import traverse_jit, brute_force_locate_jit
from .metric import NOISE_FLOOR

if TYPE_CHECKING:
    from .grid import Grid


def _arrays(grid: 'Grid'):
    """(coords float64, tets int32, adj int32) for the JIT kernels.

    coords and tets are asarray views (no copy when the dtype matches). The
    converted adjacency is cached on the grid, keyed on ``_mesh_version``, the
    identity of ``grid.adjacents`` and its length."""
    adj_src = grid.adjacents
    key = (getattr(grid, "_mesh_version", None), id(adj_src), len(adj_src))
    cache = getattr(grid, "_search_adj_cache", None)
    if cache is None or cache[0] != key:
        cache = (key, np.asarray(adj_src, dtype=np.int32))
        grid._search_adj_cache = cache
    return (np.asarray(grid.coordinates, dtype=np.float64),
            np.asarray(grid.tetrahedra, dtype=np.int32), cache[1])


def traverse(grid: 'Grid', start_tet: int,
             goal_point: List[float]) -> Tuple[bool, int]:
    """
    Barycentric ray-walk: starting from *start_tet*, cross faces
    toward *goal_point* using the most-negative barycentric coordinate
    to pick the exit face.

    Parameters
    ----------
    grid : Grid
    start_tet : int
        Index of the tetrahedron to start from.
    goal_point : list[float]
        Target coordinates [x, y, z].

    Returns
    -------
    (found, tet_idx)
        *found* is True if *goal_point* lies inside *tet_idx*.
        If False, *tet_idx* is the last visited tet.
    """
    coords, tets, adj = _arrays(grid)
    goal = np.asarray(goal_point, dtype=np.float64)
    found, tet = traverse_jit(coords, tets, adj, int(start_tet), goal, NOISE_FLOOR)
    return bool(found), int(tet)


def locate(x: float, y: float, z: float,
           grid: 'Grid',
           start_tet: Optional[int] = None) -> int:
    """
    Find the tetrahedron containing (x, y, z).

    Strategy (in order):
        1. Octree-seeded ray-walk from nearest-node tet.
        2. Walk from each tet in the octree leaf.
        3. Brute-force scan of all tets.

    Parameters
    ----------
    x, y, z : float
        Query point coordinates.
    grid : Grid
    start_tet : int, optional
        Seed tet index for the first ray-walk attempt.

    Returns
    -------
    int
        Tet index containing the point, or -1 if not found.
    """
    goal = np.array([x, y, z], dtype=np.float64)
    coords, tets, adj = _arrays(grid)

    # ── 1. Octree-seeded start ────────────────────────────────────
    if start_tet is None and grid.octree is not None:
        near = grid.octree.searchNearest(x, y, z)
        if near is not None and near >= 0:
            start_tet = grid.adjPointTets[near]

    if start_tet is None:
        start_tet = 0

    found, tet = traverse_jit(coords, tets, adj, int(start_tet), goal, NOISE_FLOOR)
    if found:
        return int(tet)

    # ── 2. Try all tets around points in leaf node ────────────────
    if grid.octree is not None:
        leaf = grid.octree._findLeafNode(x, y, z)
        candidates: List[int] = []
        if leaf and leaf.points:
            for pt in leaf.points:
                if pt in grid.ballOfNode:
                    candidates.extend(grid.ballOfNode[pt])
        else:
            candidates = list(range(len(grid.tetrahedra) // 4))

        for t in candidates:
            found2, tet2 = traverse_jit(coords, tets, adj, int(t), goal, NOISE_FLOOR)
            if found2:
                return int(tet2)

    # ── 3. Brute-force fallback ───────────────────────────────────
    return int(brute_force_locate_jit(coords, tets, goal, NOISE_FLOOR))
