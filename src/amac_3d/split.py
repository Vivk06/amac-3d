"""Refine metric-long edges by subdividing their tetrahedral shells.

Place new nodes at metric midpoints with Log-Euclidean interpolation; preserve
boundary classification and projection, positive volume, and quality limits.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

from .metric import (
    log_metric, length_in_metric, length_in_node_metric,
    edge_metric_interpolate, det_metric, tet_quality,
    signed_volume, check_loseille_lohner_gate,
    SPLIT_THRESHOLD, NOISE_FLOOR,
)
from .cavity import collect_edges, shell_of_edge
from .adj import pack_face
from .kernels import tet_volume_quality_jit

if TYPE_CHECKING:
    from .grid import Grid


# ──────────────── Boundary helpers ─────────────────────────────────

def _is_bdy_edge(grid: 'Grid', p1: int, p2: int) -> bool:
    return p1 in grid.bdyPoints and p2 in grid.bdyPoints


# ──────────────── Edge Split Point Computation ─────────────────────

def split_edge(grid: 'Grid', p1: int, p2: int) -> Tuple[List[float], List[float]]:
    """
    Compute the split point and its metric for edge (p1, p2).

    Uses arc-length weighting:
        t = l1 / (l1 + l2)
    clipped to [0.05, 0.95].

    Returns
    -------
    (point, metric)
        point  : [x, y, z]
        metric : [m11, m12, m13, m22, m23, m33]
    """
    coords = grid.coordinates

    seg_start = coords[p1 * 3: p1 * 3 + 3]
    seg_end = coords[p2 * 3: p2 * 3 + 3]

    m_p1 = grid.metricMesh[p1 * 6: p1 * 6 + 6]
    m_p2 = grid.metricMesh[p2 * 6: p2 * 6 + 6]

    l1 = length_in_node_metric(m_p1, seg_start, seg_end)
    l2 = length_in_node_metric(m_p2, seg_start, seg_end)
    l12 = length_in_metric(m_p1, m_p2, seg_start, seg_end)

    # Arc-length weight for node 2
    node2_wt = 0.5
    if (l1 + l2) > NOISE_FLOOR:
        t = l1 / (l1 + l2)
        if 0.25 < t < 0.75:
            node2_wt = 1.0 - t
        else:
            if l12 > NOISE_FLOOR:
                if l1 < l2:
                    node2_wt = 1.0 - l1 / l12
                else:
                    node2_wt = l2 / l12

    node2_wt = min(0.95, max(0.05, node2_wt))

    x = (1.0 - node2_wt) * seg_start[0] + node2_wt * seg_end[0]
    y = (1.0 - node2_wt) * seg_start[1] + node2_wt * seg_end[1]
    z = (1.0 - node2_wt) * seg_start[2] + node2_wt * seg_end[2]

    new_metric = edge_metric_interpolate(grid, p1, p2, [x, y, z])
    return [x, y, z], new_metric


# ──────────────── Shell Subdivision ────────────────────────────────

def subdivide_shell(
    grid: 'Grid',
    p1: int, p2: int,
    point: List[float],
    shell: List[int],
    metric: Optional[List[float]] = None,
) -> None:
    """
    Replace each shell tetrahedron with two sub-tets by inserting
    *point* on edge (p1, p2).

    For each tet [p1, p2, a, b]:
        - Modify existing tet: replace p2 → newNode
        - Create new tet:      replace p1 → newNode

    Updates ballOfNode, metricMesh, metricLog in-place.
    """
    tets = grid.tetrahedra
    ball = grid.ballOfNode

    # Append new node coordinates
    grid.coordinates = np.append(grid.coordinates, point)
    new_node = len(grid.coordinates) // 3 - 1
    grid.dead_mask = np.append(grid.dead_mask, False)

    if metric:
        grid.metricMesh = np.append(grid.metricMesh, np.array(metric, dtype=np.float64))
        ml = log_metric(metric)
        grid.metricLog = np.append(grid.metricLog, np.array(ml, dtype=np.float64))

    ball[new_node] = []

    coords = grid.coordinates

    for tet_idx in shell:
        base = tet_idx * 4
        tet = [tets[base], tets[base + 1], tets[base + 2], tets[base + 3]]
        t_new = tet.copy()

        p1_pos = tet.index(p1)
        p2_pos = tet.index(p2)

        # Modify existing tet: replace p2 with newNode
        tets[base + p2_pos] = new_node
        m_idx = [tets[base + k] for k in range(4)]
        m_pts = [coords[v * 3 : v * 3 + 3] for v in m_idx]
        if signed_volume(m_pts[0], m_pts[1], m_pts[2], m_pts[3]) < 0:
            tets[base + 2], tets[base + 3] = tets[base + 3], tets[base + 2]

        # Create new tet: replace p1 with newNode
        t_new[p1_pos] = new_node
        n_pts = [coords[v * 3 : v * 3 + 3] for v in t_new]
        if signed_volume(n_pts[0], n_pts[1], n_pts[2], n_pts[3]) < 0:
            t_new[2], t_new[3] = t_new[3], t_new[2]
            
        grid.tetrahedra = np.append(grid.tetrahedra, np.array(t_new, dtype=np.int32))
        tets = grid.tetrahedra
        new_tet_idx = len(tets) // 4 - 1

        # Update ball-of-node
        ball[p2].remove(tet_idx)
        ball[new_node].append(tet_idx)

        for v in tets[new_tet_idx * 4: new_tet_idx * 4 + 4]:
            if v not in ball:
                ball[v] = []
            ball[v].append(new_tet_idx)
    # Topology and coordinates changed; test stand-ins may lack the counters.
    grid._topo_version = getattr(grid, "_topo_version", 0) + 1
    grid._mesh_version = getattr(grid, "_mesh_version", 0) + 1


def _new_split_tets_valid(grid: 'Grid', new_node: int, tol: float = 1e-12) -> bool:
    """
    True iff every live tet incident to *new_node* has signed_volume > tol.

    This is the post-operation signed_volume > 0 validity check for the split
    operator.  ``split_quality`` validates the sub-tets at the projected
    midpoint, but ``bdy_edge_subdivide`` re-projects the inserted boundary
    node afterwards, which can move it off that validated position (PN-triangle
    and curve projection are not idempotent) and degenerate a sub-tet.  This
    detects that so the caller can restore the validated coordinate.
    """
    tets = grid.tetrahedra
    coords = grid.coordinates
    for t in grid.ball_of(new_node):
        base = 4 * t
        if tets[base] == -1:
            continue
        if any(grid.dead_mask[tets[base + k]] for k in range(4)):
            return False
        pts = [coords[tets[base + k] * 3: tets[base + k] * 3 + 3] for k in range(4)]
        if signed_volume(pts[0], pts[1], pts[2], pts[3]) <= tol:
            return False
    return True


# ──────────────── Boundary Split Classification ───────────────────

def bdy_edge_subdivide(grid: 'Grid', p1: int, p2: int) -> None:
    """
    Classify the newly inserted boundary node, project it onto the appropriate
    geometric entity, and register it in the boundary bookkeeping sets.

    Level combinations:
        1-1: same surfid → Level 1 + project_to_surface
        1-2, 1-3: inherit Level 1 surface + project
        2-2: same ednr → Level 2 + project_to_curve
        2-3: inherit Level 2 edge + project_to_curve
        3-3: look up cornerEdgeMap → Level 2 + project_to_curve

    After projection the new node's registration is reconciled with the mesh
    topology via ``grid.reclassify_node``: ``bdy_edge_subdivide`` runs whenever
    *both* endpoints are boundary nodes, but their connecting edge can be an
    interior chord whose midpoint lies on no boundary face.  Reclassification
    demotes such a midpoint to interior so it is never registered — or exported
    — as a surface node.
    """
    _bdy_edge_subdivide_classify(grid, p1, p2)
    new_idx = len(grid.coordinates) // 3 - 1
    grid.reclassify_node(new_idx)


def _inherit_corner_surfs(grid: 'Grid', new_idx: int, p1: int, p2: int) -> None:
    """Give a split-created ridge node its cornerPointSurfMap entry.

    Ridge nodes need their adjacent-surface set for the collapse
    surf-compatibility rule and the smoothing surface filter; without it every
    surface-node → ridge-node collapse at the new node is vetoed forever.
    The shared surfaces of the parents are the ridge's pair; fall back to
    the union when one parent has no entry.
    """
    s1 = grid.cornerPointSurfMap.get(p1, set())
    s2 = grid.cornerPointSurfMap.get(p2, set())
    surfs = (s1 & s2) or (s1 | s2)
    if surfs:
        grid.cornerPointSurfMap[new_idx] = set(surfs)


def _bdy_edge_subdivide_classify(grid: 'Grid', p1: int, p2: int) -> None:
    """Project and tentatively register the new boundary node (per level combo)."""
    from .geom import project_to_surface, project_to_curve

    levels = grid.pointLevels
    new_idx = len(grid.coordinates) // 3 - 1
    x, y, z = (grid.coordinates[new_idx * 3],
               grid.coordinates[new_idx * 3 + 1],
               grid.coordinates[new_idx * 3 + 2])

    lv1 = levels[p1]
    lv2 = levels[p2]

    # 1-1
    if lv1 == 1 and lv2 == 1:
        s1 = grid.bdyFaceMap.get(p1, -1)
        s2 = grid.bdyFaceMap.get(p2, -1)
        if s1 == s2 and s1 != -1:
            proj = project_to_surface([x, y, z], grid, s1)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
            grid.bdyPoints.append(new_idx)
            grid.bdyFacePoints.append(new_idx)
            grid.bdyFaceMap[new_idx] = s1
            levels.append(1)
        else:
            # Different surfaces → node is on a ridge between them.
            # Classify as boundary on the nearest surface rather than
            # interior, so smoothing can constrain it properly.
            from .geom import classify_point_to_surface
            best_surf = classify_point_to_surface(x, y, z, grid)
            if best_surf != -1:
                proj = project_to_surface([x, y, z], grid, best_surf)
                grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
                grid.bdyPoints.append(new_idx)
                grid.bdyFacePoints.append(new_idx)
                grid.bdyFaceMap[new_idx] = best_surf
                levels.append(1)
            else:
                levels.append(0)
        return

    # 1-2 or 2-1
    if (lv1 == 1 and lv2 == 2) or (lv1 == 2 and lv2 == 1):
        surf = grid.bdyFaceMap.get(p1 if lv1 == 1 else p2, -1)
        if surf != -1:
            proj = project_to_surface([x, y, z], grid, surf)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
        grid.bdyPoints.append(new_idx)
        grid.bdyFacePoints.append(new_idx)
        if surf is not None and surf != -1:
            grid.bdyFaceMap[new_idx] = surf
        levels.append(1)
        return

    # 1-3 or 3-1
    if (lv1 == 1 and lv2 == 3) or (lv1 == 3 and lv2 == 1):
        surf = grid.bdyFaceMap.get(p1 if lv1 == 1 else p2, -1)
        if surf != -1:
            proj = project_to_surface([x, y, z], grid, surf)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
        grid.bdyPoints.append(new_idx)
        grid.bdyFacePoints.append(new_idx)
        if surf is not None and surf != -1:
            grid.bdyFaceMap[new_idx] = surf
        levels.append(1)
        return

    # 2-2
    if lv1 == 2 and lv2 == 2:
        e1 = grid.bdyEdgeMap.get(p1, -1)
        e2 = grid.bdyEdgeMap.get(p2, -1)
        if e1 == e2 and e1 != -1:
            proj = project_to_curve([x, y, z], grid, e1)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
            grid.bdyPoints.append(new_idx)
            grid.bdyEdgePoints.append((new_idx, e1))
            grid.bdyEdgeMap[new_idx] = e1
            _inherit_corner_surfs(grid, new_idx, p1, p2)
            levels.append(2)
        else:
            # Different edges → project to surface
            from .geom import classify_point_to_surface
            surf = classify_point_to_surface(x, y, z, grid)
            grid.bdyPoints.append(new_idx)
            grid.bdyFacePoints.append(new_idx)
            if surf is not None and surf != -1:
                grid.bdyFaceMap[new_idx] = surf
            levels.append(1)
        return

    # 2-3 or 3-2
    if (lv1 == 2 and lv2 == 3) or (lv1 == 3 and lv2 == 2):
        ednr = grid.bdyEdgeMap.get(p1 if lv1 == 2 else p2, -1)
        if ednr != -1:
            proj = project_to_curve([x, y, z], grid, ednr)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
        grid.bdyPoints.append(new_idx)
        grid.bdyEdgePoints.append((new_idx, ednr))
        if ednr is not None and ednr != -1:
            grid.bdyEdgeMap[new_idx] = ednr
        _inherit_corner_surfs(grid, new_idx, p1, p2)
        levels.append(2)
        return

    # 3-3
    if lv1 == 3 and lv2 == 3:
        key = (min(p1, p2), max(p1, p2))
        ednr = grid.cornerEdgeMap.get(key)
        if ednr is not None:
            proj = project_to_curve([x, y, z], grid, ednr)
            grid.coordinates[new_idx * 3: new_idx * 3 + 3] = proj
            grid.bdyPoints.append(new_idx)
            grid.bdyEdgePoints.append((new_idx, ednr))
            grid.bdyEdgeMap[new_idx] = ednr
            _inherit_corner_surfs(grid, new_idx, p1, p2)
            levels.append(2)
        else:
            from .geom import classify_point_to_surface
            surf = classify_point_to_surface(x, y, z, grid)
            grid.bdyPoints.append(new_idx)
            grid.bdyFacePoints.append(new_idx)
            if surf is not None and surf != -1:
                grid.bdyFaceMap[new_idx] = surf
            levels.append(1)
        return

    # Default: interior
    levels.append(0)


def get_projected_midpoint(grid: 'Grid', p1: int, p2: int, point: List[float]) -> List[float]:
    """
    Get the projected coordinates of the candidate midpoint of edge (p1, p2)
    without modifying the grid.
    """
    if not _is_bdy_edge(grid, p1, p2):
        return point

    from .geom import project_to_surface, project_to_curve
    levels = grid.pointLevels
    lv1 = levels[p1]
    lv2 = levels[p2]
    x, y, z = point

    # 1-1
    if lv1 == 1 and lv2 == 1:
        s1 = grid.bdyFaceMap.get(p1, -1)
        s2 = grid.bdyFaceMap.get(p2, -1)
        if s1 == s2 and s1 != -1:
            return project_to_surface([x, y, z], grid, s1)
        else:
            from .geom import classify_point_to_surface
            best_surf = classify_point_to_surface(x, y, z, grid)
            if best_surf != -1:
                return project_to_surface([x, y, z], grid, best_surf)
            return point

    # 1-2 or 2-1
    if (lv1 == 1 and lv2 == 2) or (lv1 == 2 and lv2 == 1):
        surf = grid.bdyFaceMap.get(p1 if lv1 == 1 else p2, -1)
        if surf != -1:
            return project_to_surface([x, y, z], grid, surf)
        return point

    # 1-3 or 3-1
    if (lv1 == 1 and lv2 == 3) or (lv1 == 3 and lv2 == 1):
        surf = grid.bdyFaceMap.get(p1 if lv1 == 1 else p2, -1)
        if surf != -1:
            return project_to_surface([x, y, z], grid, surf)
        return point

    # 2-2
    if lv1 == 2 and lv2 == 2:
        e1 = grid.bdyEdgeMap.get(p1, -1)
        e2 = grid.bdyEdgeMap.get(p2, -1)
        if e1 == e2 and e1 != -1:
            return project_to_curve([x, y, z], grid, e1)
        else:
            from .geom import classify_point_to_surface
            surf = classify_point_to_surface(x, y, z, grid)
            if surf != -1:
                return project_to_surface([x, y, z], grid, surf)
            return point

    # 2-3 or 3-2
    if (lv1 == 2 and lv2 == 3) or (lv1 == 3 and lv2 == 2):
        ednr = grid.bdyEdgeMap.get(p1 if lv1 == 2 else p2, -1)
        if ednr != -1:
            return project_to_curve([x, y, z], grid, ednr)
        return point

    # 3-3
    if lv1 == 3 and lv2 == 3:
        key = (min(p1, p2), max(p1, p2))
        ednr = grid.cornerEdgeMap.get(key)
        if ednr is not None:
            return project_to_curve([x, y, z], grid, ednr)
        else:
            from .geom import classify_point_to_surface
            surf = classify_point_to_surface(x, y, z, grid)
            if surf != -1:
                return project_to_surface([x, y, z], grid, surf)
            return point

    return point


# ──────────────── Split Quality Check ──────────────────────────────

def split_quality(
    grid: 'Grid',
    u: int, v: int,
    new_point: List[float],
    new_metric: List[float],
    shell: List[int],
    q_r: float = 1.0,
    q_a: float = 1e-3,
    _reject: dict = None,
 ) -> float:
    """
    Evaluate whether splitting edge (u, v) with *new_point* produces
    acceptable sub-tets.

    CRITICAL: Sub-tet vertex ordering must match what subdivide_shell
    actually produces — replace v→newNode in original for sub-tet 1,
    replace u→newNode for sub-tet 2.  This preserves the parent tet's
    winding so signed_volume gives the correct sign.

    Returns 1 if all pass, 0 otherwise. When ``_reject`` is given, the
    failing check's counter is incremented (rejection diagnostics).
    """
    coords = grid.coordinates
    tets = grid.tetrahedra
    mm = grid.metricMesh

    new_pt = np.asarray(new_point, dtype=np.float64)
    new_met = np.asarray(new_metric, dtype=np.float64)

    # Old worst quality of shell tets (for the gate)
    old_worst = float('inf')
    for tet_idx in shell:
        tet = list(tets[tet_idx * 4: tet_idx * 4 + 4])
        if any(grid.dead_mask[n] for n in tet):
            continue
        old_worst = min(old_worst, tet_volume_quality_jit(
            coords, mm, int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]),
            -2, new_pt, new_met, False)[1])

    new_worst = float('inf')

    for tet_idx in shell:
        tet = list(tets[tet_idx * 4: tet_idx * 4 + 4])

        if u not in tet or v not in tet:
            if _reject is not None:
                _reject["shell_corrupt"] += 1
            return 0

        u_pos = tet.index(u)
        v_pos = tet.index(v)

        # Sub-tet 1: original tet with v replaced by newNode
        sub1 = tet.copy()
        sub1[v_pos] = -1  # placeholder for new node

        if any(grid.dead_mask[n] for n in sub1 if n != -1):
            if _reject is not None:
                _reject["shell_corrupt"] += 1
            return 0
        sv1, q1 = tet_volume_quality_jit(
            coords, mm, int(sub1[0]), int(sub1[1]), int(sub1[2]), int(sub1[3]),
            -1, new_pt, new_met, True)
        if sv1 < 1e-12:
            if _reject is not None:
                _reject["subtet_volume"] += 1
            return 0

        # Sub-tet 2: original tet with u replaced by newNode
        sub2 = tet.copy()
        sub2[u_pos] = -1  # placeholder for new node

        if any(grid.dead_mask[n] for n in sub2 if n != -1):
            if _reject is not None:
                _reject["shell_corrupt"] += 1
            return 0
        sv2, q2 = tet_volume_quality_jit(
            coords, mm, int(sub2[0]), int(sub2[1]), int(sub2[2]), int(sub2[3]),
            -1, new_pt, new_met, True)
        if sv2 < 1e-12:
            if _reject is not None:
                _reject["subtet_volume"] += 1
            return 0

        new_worst = min(new_worst, q1, q2)

    # Refine-style relative floor (ref_split.c:617-622, split_quality_relative
    # = 0.1): in a shell whose existing worst tet is already below the
    # absolute floor, accept sub-tets down to 0.1× that worst — otherwise the
    # regions that most need refinement are the only ones that can never be
    # refined. Healthy shells (worst ≥ 10×q_a) keep the absolute floor.
    q_a_eff = min(q_a, max(0.1 * old_worst, 1e-8))
    if not check_loseille_lohner_gate(old_worst, new_worst, q_r, q_a_eff):
        if _reject is not None:
            _reject["ll_gate"] += 1
        return 0

    # Intersection prevention check
    from .intersection import check_local_intersection
    tets_to_check = []
    for tet_idx in shell:
        tet = list(tets[tet_idx * 4: tet_idx * 4 + 4])
        u_pos = tet.index(u)
        v_pos = tet.index(v)

        sub1 = tet.copy()
        sub1[v_pos] = -1

        sub2 = tet.copy()
        sub2[u_pos] = -1

        tets_to_check.append(sub1)
        tets_to_check.append(sub2)

    if check_local_intersection(
        grid, tets_to_check,
        coords_override={-1: new_point},
        ignored_tets=set(shell)
    ):
        if _reject is not None:
            _reject["intersection"] += 1
        return 0

    return 1


# ──────────────── Split Pass ───────────────────────────────────────

def _is_complete_shell(grid: 'Grid', shell: List[int], p1: int, p2: int) -> bool:
    """
    Check that the shell of tetrahedra sharing edge (p1, p2) is complete.
    Supports both closed rings (interior edges) and open chains (boundary edges).
    """
    from collections import defaultdict
    tets = grid.tetrahedra
    ring_node_count: dict = defaultdict(int)
    for t_idx in shell:
        tet = tets[4 * t_idx: 4 * t_idx + 4]
        others = [v for v in tet if v != p1 and v != p2]
        if len(others) != 2:
            return False  # Tet doesn't contain both p1 and p2
        for v in others:
            ring_node_count[v] += 1

    if not ring_node_count:
        return False

    counts = list(ring_node_count.values())
    
    # Case 1: Closed ring (all counts are 2)
    if all(c == 2 for c in counts):
        return True
        
    # Case 2: Open chain / boundary fan (exactly two counts are 1, all others are 2)
    ones = [v for v, c in ring_node_count.items() if c == 1]
    twos = [v for v, c in ring_node_count.items() if c == 2]
    if len(ones) == 2 and len(twos) == len(ring_node_count) - 2:
        # Verify both end faces (p1, p2, w) are actual boundary faces of the entire mesh
        for w in ones:
            ball_p1 = grid.ball_of(p1)
            ball_p2 = grid.ball_of(p2)
            ball_w = grid.ball_of(w)
            common = set(ball_p1) & set(ball_p2) & set(ball_w)
            live_count = sum(1 for t in common if tets[4 * t] != -1)
            if live_count != 1:
                return False
        return True
        
    return False


def _split_subedge_ratios_ok(grid: 'Grid', u: int, v: int,
                             new_point: List[float],
                             new_metric: List[float],
                             shell: List[int],
                             post_min_ratio: float,
                             post_max_ratio: float) -> bool:
    """
    NASA Refine ``ref_split_edge_ratio``: every edge the split
    creates at the new node — to u, to v, and to each shell ring vertex —
    must have metric length within [post_min_ratio, post_max_ratio].

    adapt.py passes the per-iteration ratcheting band (never stricter than
    the current mesh extremes, eased near convergence), so this rejects
    only splits that would *worsen* the extremes — the source of
    split/collapse churn: sub-edges shorter than the coarsening threshold
    get collapsed next iteration and the parent edge re-splits forever.
    """
    coords = grid.coordinates
    mm = grid.metricMesh

    neighbours = {u, v}
    for t in shell:
        for n in grid.tetrahedra[4 * t: 4 * t + 4]:
            if n != u and n != v:
                neighbours.add(n)

    for n in neighbours:
        if grid.dead_mask[n]:
            return False
        p = coords[3 * n: 3 * n + 3]
        m = mm[6 * n: 6 * n + 6]
        L = length_in_metric(new_metric, m, new_point, p)
        if L < post_min_ratio or L > post_max_ratio:
            return False
    return True


def _apply_splits(grid: 'Grid', edges, long_edges, q_r, q_a,
                   show_rejections, post_min_ratio, post_max_ratio) -> int:
    """Apply loop shared by serial and parallel paths — all guards verbatim."""
    coords = grid.coordinates
    mm = grid.metricMesh

    _reject = {
        "dead_node": 0,
        "length_reverify": 0,
        "shell_error": 0,
        "incomplete_shell": 0,
        "shell_corrupt": 0,
        "subtet_volume": 0,
        "ll_gate": 0,
        "intersection": 0,
        "subedge_ratio": 0,
    }

    n_splits = 0

    for idx, _ in long_edges:
        p1, p2 = edges[idx]

        if grid.dead_mask[p1] or grid.dead_mask[p2]:
            _reject["dead_node"] += 1
            continue

        seg_s = coords[p1 * 3: p1 * 3 + 3]
        seg_e = coords[p2 * 3: p2 * 3 + 3]

        m1 = mm[p1 * 6: p1 * 6 + 6]
        m2 = mm[p2 * 6: p2 * 6 + 6]
        L = length_in_metric(m1, m2, seg_s, seg_e)
        if L < SPLIT_THRESHOLD:
            _reject["length_reverify"] += 1
            continue

        try:
            shell = shell_of_edge(grid, p1, p2)
        except ValueError:
            _reject["shell_error"] += 1
            continue

        if not _is_complete_shell(grid, shell, p1, p2):
            _reject["incomplete_shell"] += 1
            continue

        _ring_count: dict = {}
        for _t in shell:
            for _v in grid.tetrahedra[4 * _t: 4 * _t + 4]:
                if _v != p1 and _v != p2:
                    _ring_count[_v] = _ring_count.get(_v, 0) + 1
        _parent_faces = [x for x, c in _ring_count.items() if c == 1]
        edge_in_boundary = bool(_parent_faces)

        new_point, new_metric = split_edge(grid, p1, p2)
        new_point_proj = (get_projected_midpoint(grid, p1, p2, new_point)
                          if edge_in_boundary else new_point)

        q = split_quality(grid, p1, p2, new_point_proj, new_metric, shell,
                          q_r=q_r, q_a=q_a, _reject=_reject)
        if q == 0:
            continue

        if not _split_subedge_ratios_ok(grid, p1, p2, new_point_proj,
                                        new_metric, shell,
                                        post_min_ratio, post_max_ratio):
            _reject["subedge_ratio"] += 1
            continue

        subdivide_shell(grid, p1, p2, new_point_proj, shell, new_metric)
        new_node = len(grid.coordinates) // 3 - 1
        grid.nodeAge.append(0)

        if edge_in_boundary and _is_bdy_edge(grid, p1, p2):
            bdy_edge_subdivide(grid, p1, p2)
            if not _new_split_tets_valid(grid, new_node):
                grid.coordinates[new_node * 3: new_node * 3 + 3] = list(new_point_proj)
                grid._mesh_version = getattr(grid, "_mesh_version", 0) + 1
        else:
            grid.pointLevels.append(0)

        grid.resample_node_metric(new_node)

        for x in _parent_faces:
            sid = grid.faceSurfMap.pop(pack_face(p1, p2, x), None)
            if sid is not None:
                grid.faceSurfMap[pack_face(p1, new_node, x)] = sid
                grid.faceSurfMap[pack_face(new_node, p2, x)] = sid
            elif grid.solidData:
                s1 = grid._classify_face_geo(p1, new_node, x)
                s2 = grid._classify_face_geo(new_node, p2, x)
                if s1 is not None:
                    grid.faceSurfMap[pack_face(p1, new_node, x)] = s1
                if s2 is not None:
                    grid.faceSurfMap[pack_face(new_node, p2, x)] = s2

        n_splits += 1

    logger.debug(f"  Edges split: {n_splits}")
    if show_rejections:
        logger.debug(f"  Split rejection breakdown: {_reject}")
    return n_splits


def split_pass(grid: 'Grid', q_r: float = 1.0, q_a: float = 1e-3,
                       show_rejections: bool = False,
                       post_min_ratio: float = 0.0,
                       post_max_ratio: float = float('inf')) -> int:
    """Split metric-long edges while preserving boundary and quality guards.

    Candidate edges are scanned by the kernel, ordered by length, then checked
    against the current metric and post-split edge-length band.
    """
    coords = grid.coordinates
    mm = grid.metricMesh
    edges, edges_arr, n_scan = grid.get_scan_edges()

    from .kernels import split_candidate_scan_jit
    n_points = len(coords) // 3
    le, ll, n_long = split_candidate_scan_jit(
        coords, mm, grid.dead_mask, edges_arr, n_points, SPLIT_THRESHOLD
    )
    long_edges = [(int(le[i]), float(ll[i])) for i in range(n_long)]

    if edges:
        frac_str = f"  Bigger edges fraction: {len(long_edges)/len(edges):.6f}"
        if grid._active_edge_frac < 1.0:
            frac_str += f"  (scanned {n_scan} active of {int(n_scan / max(grid._active_edge_frac, 1e-9))})"
        logger.debug(frac_str)

    long_edges.sort(key=lambda x: x[1], reverse=True)

    return _apply_splits(grid, edges, long_edges, q_r, q_a,
                         show_rejections, post_min_ratio, post_max_ratio)




# ──────────────── Hanging-Node Diagnostic ─────────────────────────

def count_hanging_nodes(grid: 'Grid', tol: float = 1e-9) -> int:
    """
    Count hanging nodes (T-junctions) on the mesh boundary skin.

    A hanging node is a node that lies strictly in the interior of a
    boundary edge of which it is not an endpoint — a non-conforming
    T-junction.  Such a node arises only if an edge is split in some of
    its incident tetrahedra but not all of them.

    Only the boundary skin is examined: ``subdivide_shell`` operates on
    the full shell returned by ``shell_of_edge`` (= ballOfNode[p1] ∩
    ballOfNode[p2]), so every tet sharing an interior edge is subdivided
    together.  Boundary edges (open fans) are the only place a partial
    split could leave a dangling node, so that is where T-junctions would
    appear if shell registration were incomplete.

    Returns the number of distinct hanging nodes (0 for a conforming mesh).
    """
    tets = grid.tetrahedra
    coords = grid.coordinates
    n_tets = len(tets) // 4

    # Boundary faces = faces incident to exactly one live tet.
    from collections import defaultdict
    face_to_tets: dict = defaultdict(int)
    for t in range(n_tets):
        base = 4 * t
        if tets[base] == -1:
            continue
        v = (tets[base], tets[base + 1], tets[base + 2], tets[base + 3])
        for fi in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
            face = tuple(sorted((v[fi[0]], v[fi[1]], v[fi[2]])))
            face_to_tets[face] += 1

    bdy_edges: set = set()
    bdy_nodes: set = set()
    for face, count in face_to_tets.items():
        if count != 1:
            continue
        a, b, c = face
        bdy_nodes.update((a, b, c))
        bdy_edges.add((a, b))
        bdy_edges.add((a, c))
        bdy_edges.add((b, c))

    if not bdy_edges:
        return 0

    pos = {n: coords[n * 3: n * 3 + 3] for n in bdy_nodes}

    # Spatial hash of boundary nodes for candidate lookup.  Cell size is
    # the mean boundary-edge length, so each edge spans only a few cells.
    total_len = 0.0
    for a, b in bdy_edges:
        pa, pb = pos[a], pos[b]
        total_len += ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                      + (pa[2] - pb[2]) ** 2) ** 0.5
    h = total_len / len(bdy_edges)
    if h <= 0.0:
        return 0

    def _cell(p):
        return (int(p[0] // h), int(p[1] // h), int(p[2] // h))

    grid_hash: dict = defaultdict(list)
    for n in bdy_nodes:
        grid_hash[_cell(pos[n])].append(n)

    hanging: set = set()
    for a, b in bdy_edges:
        pa, pb = pos[a], pos[b]
        ab = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
        if len2 <= 0.0:
            continue

        lo = (min(pa[0], pb[0]), min(pa[1], pb[1]), min(pa[2], pb[2]))
        hi = (max(pa[0], pb[0]), max(pa[1], pb[1]), max(pa[2], pb[2]))
        c_lo = (int(lo[0] // h), int(lo[1] // h), int(lo[2] // h))
        c_hi = (int(hi[0] // h), int(hi[1] // h), int(hi[2] // h))

        for cx in range(c_lo[0], c_hi[0] + 1):
            for cy in range(c_lo[1], c_hi[1] + 1):
                for cz in range(c_lo[2], c_hi[2] + 1):
                    for n in grid_hash.get((cx, cy, cz), ()):
                        if n == a or n == b:
                            continue
                        pv = pos[n]
                        # Parametric position along the segment.
                        t = ((pv[0] - pa[0]) * ab[0] + (pv[1] - pa[1]) * ab[1]
                             + (pv[2] - pa[2]) * ab[2]) / len2
                        if t <= tol or t >= 1.0 - tol:
                            continue
                        proj = (pa[0] + t * ab[0], pa[1] + t * ab[1],
                                pa[2] + t * ab[2])
                        d2 = ((pv[0] - proj[0]) ** 2 + (pv[1] - proj[1]) ** 2
                              + (pv[2] - proj[2]) ** 2)
                        if d2 < tol * tol:
                            hanging.add(n)

    return len(hanging)
