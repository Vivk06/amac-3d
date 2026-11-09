"""Reject proposed local mesh edits that intersect existing tets.

Read candidate tets and optional coordinate overrides without mutating Grid.
Use bounding-box filtering, edge-face crossings, and containment tests; return
True on a genuine local intersection.
"""

from typing import List, Dict, Set, Optional, TYPE_CHECKING

import numpy as np

from .metric import signed_volume

_ignored_scratch = np.zeros(0, dtype=np.bool_)

if TYPE_CHECKING:
    from .grid import Grid


def segment_triangle_intersect(
    p0: List[float], p1: List[float],
    a: List[float], b: List[float], c: List[float],
    epsilon: float = 1e-7
) -> bool:
    """
    Check if segment (p0, p1) intersects triangle (a, b, c) using Möller-Trumbore.
    """
    ex, ey, ez = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    tx, ty, tz = p0[0] - a[0], p0[1] - a[1], p0[2] - a[2]
    
    px = ey * e2z - ez * e2y
    py = ez * e2x - ex * e2z
    pz = ex * e2y - ey * e2x
    
    D = e1x * px + e1y * py + e1z * pz
    
    if abs(D) < 1e-12:
        return False
        
    invD = 1.0 / D
    u = (tx * px + ty * py + tz * pz) * invD
    if u < epsilon or u > 1.0 - epsilon:
        return False
        
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    
    v = (ex * qx + ey * qy + ez * qz) * invD
    if v < epsilon or u + v > 1.0 - epsilon:
        return False
        
    t = (e2x * qx + e2y * qy + e2z * qz) * invD
    if t < epsilon or t > 1.0 - epsilon:
        return False
        
    return True


def point_in_tet(
    p: List[float],
    a: List[float], b: List[float], c: List[float], d: List[float],
    epsilon: float = 1e-7
) -> bool:
    """Check if point p is inside tetrahedron (a, b, c, d)."""
    v_total = signed_volume(a, b, c, d)
    if abs(v_total) < 1e-12:
        return False
    
    v0 = signed_volume(p, b, c, d)
    v1 = signed_volume(a, p, c, d)
    v2 = signed_volume(a, b, p, d)
    v3 = signed_volume(a, b, c, p)
    
    sign = 1.0 if v_total > 0 else -1.0
    for v in (v0, v1, v2, v3):
        if v * sign < epsilon:
            return False
    return True


def check_local_intersection(
    grid: 'Grid',
    tets_to_check: List[List[int]],
    coords_override: Optional[Dict[int, List[float]]] = None,
    ignored_tets: Optional[Set[int]] = None
) -> bool:
    """
    Check if any tetrahedron in `tets_to_check` intersects any existing non-ignored element in grid.

    Uses union AABB bounding box pre-filtering to achieve high performance.
    """
    from .kernels import check_local_intersection_jit

    coords = grid.coordinates
    tets = grid.tetrahedra
    n_total_tets = len(tets) // 4
    n_nodes = len(grid.dead_mask)

    # Marshal tets_to_check to int32 2D array
    ttc = np.array(tets_to_check, dtype=np.int32)

    # Handle coords_override: at most one entry (verified from all callers)
    if coords_override:
        node, xyz = next(iter(coords_override.items()))
        override_node = np.int32(node)
        override_xyz = np.asarray(xyz, dtype=np.float64)
    else:
        override_node = np.int32(-1)
        override_xyz = np.empty(3, dtype=np.float64)

    # The CSR ball is stale after a committed topology change. Refresh it from
    # tetrahedra (operators keep them consistent with ballOfNode at commit
    # boundaries); stand-ins without refresh_ball_csr rebuild from the dict.
    ball_ptr = grid.ball_ptr
    ball_idx = grid.ball_idx
    _stale_fn = getattr(grid, "csr_is_stale", None)
    if (_stale_fn() if _stale_fn is not None
            else len(ball_ptr) - 1 != n_nodes):
        _refresh = getattr(grid, "refresh_ball_csr", None)
        if _refresh is not None:
            _refresh()
            ball_ptr = grid.ball_ptr
            ball_idx = grid.ball_idx
        else:
            ball = grid.ballOfNode
            ptr = np.zeros(n_nodes + 1, dtype=np.int32)
            for v in range(n_nodes):
                b = ball.get(v)
                ptr[v + 1] = ptr[v] + (len(b) if b else 0)
            idx = np.empty(int(ptr[n_nodes]), dtype=np.int32)
            for v in range(n_nodes):
                b = ball.get(v)
                if b:
                    start = ptr[v]
                    for k, ti in enumerate(b):
                        idx[start + k] = ti
            ball_ptr = ptr
            ball_idx = idx
            _set = getattr(grid, "set_ball_csr", None)
            if _set is not None:
                _set(ptr, idx)

    # Reused scratch mask: a per-call whole-mesh allocation dominated the cost.
    # Assumes serial Python callers (the threaded coloured path passes its own mask).
    global _ignored_scratch
    if _ignored_scratch.shape[0] < n_total_tets:
        _ignored_scratch = np.zeros(
            max(n_total_tets, 2 * _ignored_scratch.shape[0]), dtype=np.bool_)
    marked = []
    try:
        if ignored_tets:
            for ti in ignored_tets:
                if 0 <= ti < n_total_tets:
                    _ignored_scratch[ti] = True
                    marked.append(ti)
        return check_local_intersection_jit(
            coords, tets, ball_ptr, ball_idx, grid.dead_mask,
            ttc, override_node, override_xyz, _ignored_scratch[:n_total_tets]
        )
    finally:
        for ti in marked:
            _ignored_scratch[ti] = False
