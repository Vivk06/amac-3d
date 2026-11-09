"""Move mesh nodes to improve metric quality without changing topology.

Boundary nodes stay on their surfaces or ridges; interior nodes use serial or
coloured trials at the interior-node threshold. Every move passes quality and
intersection guards. The force follows NASA refine
ref_smooth.c::ref_smooth_add_pliant_force; smoothing follows
ref_smooth.c::ref_smooth_pass.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import math
from typing import List, Optional, TYPE_CHECKING

from .metric import (
    length_in_metric, det_metric, tet_quality,
    signed_volume, NOISE_FLOOR,
)
import numpy as np

from .cavity import nodes_around_node

SMALL_MESH_EDGES = 2000

if TYPE_CHECKING:
    from .grid import Grid


# ──────────────── Constants ────────────────────────────────────────

SMOOTH_ALPHA: float = 0.2        # Default damping factor (ref_adapt->smooth_pliant_alpha)
QUALITY_FLOOR: float = 1e-3      # Default minimum quality floor
QUALITY_IMPROVEMENT_TOL: float = 1e-8  # Accept if new_q > old_q - this
BACKOFF_TRIES: int = 4            # NASA-style backoff attempts


# ──────────────── Geometric Validity ───────────────────────────────

def _clamp_to_domain(grid: 'Grid', point: List[float]) -> List[float]:
    """Clamp *point* to the domain bounds defined by solidData or EGADS model."""
    _egads_model = getattr(grid, 'egadsModel', None)
    if _egads_model is not None:
        from .egads_adapter import get_bounding_box
        bb_min, bb_max = get_bounding_box(_egads_model)
        return [max(bb_min[k], min(bb_max[k], point[k])) for k in range(3)]

    plane_list = []
    if grid.solidData and 'surfaceList' in grid.solidData:
        for surf in grid.solidData['surfaceList']:
            if surf['type'] == 'plane':
                plane_list.append((surf['origin'], surf['normal']))
    return _clamp_to_planes(point, plane_list)


# ──────────────── Node Quality Evaluation ──────────────────────────

def _evaluate_node_quality(
    grid: 'Grid', node: int,
    coords_override: Optional[List[float]] = None,
) -> float:
    """Compute the minimum tet quality across all tetrahedra in ball(node).

    If *coords_override* is provided, those [x, y, z] coordinates are
    used in place of the node's current coordinates.

    Returns -1.0 if any tet in the ball is inverted or contains dead nodes.
    """
    return _evaluate_quality_arrays(
        grid.coordinates, grid.tetrahedra, grid.metricMesh, grid.dead_mask,
        node, list(grid.ball_of(node)),
        coords_override_xyz=coords_override)


# ──────────────── Pliant Force Computation ─────────────────────────

def _compute_pliant_force(
    grid: 'Grid', center: int, neighbor: int,
) -> List[float]:
    """Compute the pliant (spring) force vector from *neighbor* toward *center*.

    Force formula follows NASA refine ref_smooth.c::ref_smooth_add_pliant_force
    (lines 38-58):
        norm   = (center - neighbor) / ratio
        l4     = ratio**4
        force  = (1 - l4) * exp(-l4)
        if ratio < 1:  force += (1 - ratio)**2
    """
    return _pliant_force_arrays(
        grid.coordinates, grid.metricMesh, grid.dead_mask, center, neighbor)


def _pliant_force_total(coords, metrics, dead_mask, center, neighbours):
    """Summed pliant force over *neighbours* in list order (one JIT call)."""
    from .kernels import pliant_force_sum_jit
    return pliant_force_sum_jit(
        coords, metrics, dead_mask, center,
        np.asarray(neighbours, dtype=np.int64), NOISE_FLOOR).tolist()


# ──────────────── Interior Node Smoothing ──────────────────────────

def smooth_tet_node(grid: 'Grid', node: int,
                    q_a: float = 1e-3) -> bool:
    """
    Smooth an interior node (Level 0) using pliant forces.

    Accumulates spring forces from all neighbours, applies a damped
    displacement with NASA-style backoff (ref_smooth.c:1477-1501),
    and clamps the result to the domain bounds.

    Returns True if the node was moved.
    """
    coords = grid.coordinates

    neighbours = nodes_around_node(grid, node)
    if not neighbours:
        return False

    # Accumulate pliant forces
    total_force = _pliant_force_total(grid.coordinates, grid.metricMesh,
                                      grid.dead_mask, node, neighbours)

    old_pos = list(coords[node * 3: node * 3 + 3])
    old_q = _evaluate_node_quality(grid, node)

    # NASA-style backoff: try full step, then half, quarter, etc.
    backoff = 1.0
    for _ in range(BACKOFF_TRIES):
        trial = [old_pos[k] + backoff * SMOOTH_ALPHA * total_force[k]
                 for k in range(3)]
        # Clamp to domain bounds — prevents protrusions
        trial = _clamp_to_domain(grid, trial)

        trial_q = _evaluate_node_quality(grid, node, coords_override=trial)

        if (trial_q >= q_a
                and trial_q > old_q - QUALITY_IMPROVEMENT_TOL):

            # Intersection prevention check
            ball = grid.ball_of(node)
            tets_to_check = []
            if len(ball) > 0:
                for tet_idx in ball:
                    base = tet_idx * 4
                    if grid.tetrahedra[base] != -1:
                        tets_to_check.append(list(grid.tetrahedra[base: base + 4]))

            from .intersection import check_local_intersection
            if not check_local_intersection(grid, tets_to_check, coords_override={node: trial}, ignored_tets=set(ball) if len(ball) > 0 else None):
                coords[node * 3] = trial[0]
                coords[node * 3 + 1] = trial[1]
                coords[node * 3 + 2] = trial[2]
                grid.resample_node_metric(node)
                return True

        backoff *= 0.5

    return False


# ──────────────── Boundary Node Smoothing ──────────────────────────

def smooth_boundary_node(grid: 'Grid', node: int,
                         q_a: float = 1e-3) -> bool:
    """
    Smooth a boundary node (Level 1 = face, Level 2 = edge).

    For face nodes:
        - Accumulates forces from neighbours on the same surface.
        - Re-projects the result onto the surface plane.
    For edge nodes:
        - Accumulates forces from neighbours on the same edge.
        - Re-projects the result onto the ridge curve.

    Returns True if the node was moved.
    """
    from .geom import project_to_surface, project_to_curve

    coords = grid.coordinates
    levels = grid.pointLevels
    level = levels[node]

    if level == 3:
        return False  # Corners are locked

    neighbours = nodes_around_node(grid, node)
    if not neighbours:
        return False

    # Filter neighbours to same geometric entity
    valid_neighbours: List[int] = []

    if level == 2:
        # Edge node — only neighbours on the same ridge
        my_ednr = grid.bdyEdgeMap.get(node, -1)
        if my_ednr == -1:
            return False
        for nb in neighbours:
            nb_level = levels[nb] if nb < len(levels) else 0
            if nb_level >= 2:
                nb_ednr = grid.bdyEdgeMap.get(nb, -1)
                if nb_ednr == my_ednr:
                    valid_neighbours.append(nb)
                # Corner nodes are also on this edge if they share the ednr
                if nb_level == 3:
                    for p, ednrs in grid.cornerPoints:
                        if p == nb and my_ednr in ednrs:
                            valid_neighbours.append(nb)
                            break

    elif level == 1:
        # Face node — only neighbours on the SAME surface
        # Ridge/corner neighbours count only if they lie on this surface;
        # neighbours from other surfaces pull the node across surfaces and
        # cause boundary protrusions.
        my_surfid = grid.bdyFaceMap.get(node, -1)
        if my_surfid == -1:
            return False
        for nb in neighbours:
            nb_level = levels[nb] if nb < len(levels) else 0
            if nb_level == 1:
                # Same-level face node — must share surfid
                nb_surfid = grid.bdyFaceMap.get(nb, -1)
                if nb_surfid == my_surfid:
                    valid_neighbours.append(nb)
            elif nb_level == 2:
                # Ridge node — include only if it shares the surface of the face node
                if my_surfid in grid.cornerPointSurfMap.get(nb, set()):
                    valid_neighbours.append(nb)
            elif nb_level == 3:
                # Corner node — include only if it shares the surface of the face node
                if my_surfid in grid.cornerPointSurfMap.get(nb, set()):
                    valid_neighbours.append(nb)

    # De-duplicate
    valid_neighbours = list(set(valid_neighbours))

    if not valid_neighbours:
        return False

    # Accumulate pliant forces from valid neighbours only
    total_force = _pliant_force_total(grid.coordinates, grid.metricMesh,
                                      grid.dead_mask, node, valid_neighbours)

    old_pos = list(coords[node * 3: node * 3 + 3])
    old_q = _evaluate_node_quality(grid, node)

    # NASA-style backoff with projection + domain clamp
    backoff = 1.0
    for _ in range(BACKOFF_TRIES):
        trial = [old_pos[k] + backoff * SMOOTH_ALPHA * total_force[k]
                 for k in range(3)]

        # Re-project onto geometric entity
        if level == 1:
            surfid = grid.bdyFaceMap.get(node, -1)
            if surfid != -1:
                trial = project_to_surface(trial, grid, surfid)
        elif level == 2:
            ednr = grid.bdyEdgeMap.get(node, -1)
            if ednr != -1:
                trial = project_to_curve(trial, grid, ednr)

        # Final safety: clamp to domain bounds
        trial = _clamp_to_domain(grid, trial)

        trial_q = _evaluate_node_quality(grid, node, coords_override=trial)

        if (trial_q >= q_a
                and trial_q > old_q - QUALITY_IMPROVEMENT_TOL):

            # Intersection prevention check
            ball = grid.ball_of(node)
            tets_to_check = []
            if len(ball) > 0:
                for tet_idx in ball:
                    base = tet_idx * 4
                    if grid.tetrahedra[base] != -1:
                        tets_to_check.append(list(grid.tetrahedra[base: base + 4]))

            from .intersection import check_local_intersection
            if not check_local_intersection(grid, tets_to_check, coords_override={node: trial}, ignored_tets=set(ball) if len(ball) > 0 else None):
                coords[node * 3] = trial[0]
                coords[node * 3 + 1] = trial[1]
                coords[node * 3 + 2] = trial[2]
                grid.resample_node_metric(node)
                return True

        backoff *= 0.5

    return False


# ──────────────── Array-based Interior Helpers ───────────────────────

def _pliant_force_arrays(coords, metrics, dead_mask, center, neighbor):
    """Pliant spring force on snapshot arrays — mirrors _compute_pliant_force."""
    if dead_mask[center] or dead_mask[neighbor]:
        return [0.0, 0.0, 0.0]

    p_c = list(coords[3 * center: 3 * center + 3])
    p_n = list(coords[3 * neighbor: 3 * neighbor + 3])
    m_c = list(metrics[6 * center: 6 * center + 6])
    m_n = list(metrics[6 * neighbor: 6 * neighbor + 6])

    if m_c[0] == -1 or m_n[0] == -1:
        return [0.0, 0.0, 0.0]

    ratio = length_in_metric(m_c, m_n, p_c, p_n)
    if ratio < NOISE_FLOOR:
        return [0.0, 0.0, 0.0]

    norm = [(p_c[k] - p_n[k]) / ratio for k in range(3)]
    l4 = ratio * ratio * ratio * ratio
    try:
        force = (1.0 - l4) * math.exp(-l4)
    except OverflowError:
        force = 0.0
    if ratio < 1.0:
        force += (1.0 - ratio) * (1.0 - ratio)

    return [force * norm[k] for k in range(3)]


def _evaluate_quality_arrays(coords, tets, metrics, dead_mask,
                             node, ball_tet_indices,
                             coords_override_xyz=None):
    """Min tet quality over ball on snapshot arrays — mirrors _evaluate_node_quality."""
    from .kernels import ball_min_quality_jit
    ball = np.asarray(ball_tet_indices, dtype=np.int64)
    if coords_override_xyz is None:
        has_override = False
        override = np.zeros(3, dtype=np.float64)
    else:
        has_override = True
        override = np.asarray(coords_override_xyz, dtype=np.float64)
    return float(ball_min_quality_jit(
        coords, tets, metrics, dead_mask, node, ball, has_override, override))


def _clamp_to_planes(point, plane_list):
    """Domain clamp on a pre-extracted plane list — mirrors _clamp_to_domain."""
    result = list(point)
    for origin, normal in plane_list:
        ox, oy, oz = origin
        nx, ny, nz = normal
        dist = nx * (result[0] - ox) + ny * (result[1] - oy) + nz * (result[2] - oz)
        if dist > 0:
            result[0] -= dist * nx
            result[1] -= dist * ny
            result[2] -= dist * nz
    return result


def _commit_if_safe(grid, node, trial_xyz, q_a, skip_intersection=False):
    """Serial commit: live-grid quality re-check, intersection, and metric resampling.

    Returns 1 on commit, 0 on rejection.
    """
    from .intersection import check_local_intersection

    old_q = _evaluate_node_quality(grid, node)
    trial_q = _evaluate_node_quality(grid, node,
                                     coords_override=list(trial_xyz))

    if trial_q < q_a or trial_q <= old_q - QUALITY_IMPROVEMENT_TOL:
        return 0

    if not skip_intersection:
        ball = grid.ball_of(node)
        tets_to_check = []
        if len(ball) > 0:
            for tet_idx in ball:
                base = tet_idx * 4
                if grid.tetrahedra[base] != -1:
                    tets_to_check.append(list(grid.tetrahedra[base: base + 4]))

        if check_local_intersection(grid, tets_to_check,
                                    coords_override={node: list(trial_xyz)},
                                    ignored_tets=set(ball) if len(ball) > 0 else None):
            return 0

    coords = grid.coordinates
    coords[node * 3] = trial_xyz[0]
    coords[node * 3 + 1] = trial_xyz[1]
    coords[node * 3 + 2] = trial_xyz[2]
    grid.resample_node_metric(node)
    return 1


# ──────────────── Smooth Pass Driver ───────────────────────────────

def _smooth_pass_serial(grid: 'Grid', iterations: int = 3,
                        verbose: bool = True,
                        q_a: float = 1e-3) -> int:
    """Smooth boundary and interior nodes serially."""
    from .grid import POINT_INTERIOR, POINT_SURFACE, POINT_RIDGE

    levels = grid.pointLevels
    coords = grid.coordinates
    n_points = len(coords) // 3

    if verbose:
        # Print the q_a actually used as the acceptance floor, not the
        # QUALITY_FLOOR default (adapt.py passes the annealed per-iteration
        # value, e.g. 0.115 — the old line always claimed 1e-3).
        logger.debug(f"  Smooth: quality_floor={q_a:.4e}, "
              f"alpha={SMOOTH_ALPHA:.3f}")

    total_moves = 0
    moved_nodes: set = set()

    for it in range(iterations):
        it_moves = 0

        # ── Pass 1: Edge nodes (Level 2) ──────────────────────────
        for node in range(n_points):
            if grid.dead_mask[node]:
                continue
            if node >= len(levels):
                continue
            if levels[node] != POINT_RIDGE:
                continue
            if smooth_boundary_node(grid, node, q_a=q_a):
                it_moves += 1
                moved_nodes.add(node)

        # ── Pass 2: Face nodes (Level 1) ──────────────────────────
        for node in range(n_points):
            if grid.dead_mask[node]:
                continue
            if node >= len(levels):
                continue
            if levels[node] != POINT_SURFACE:
                continue
            if smooth_boundary_node(grid, node, q_a=q_a):
                it_moves += 1
                moved_nodes.add(node)

        # ── Pass 3: Interior nodes (Level 0) ──────────────────────
        for node in range(n_points):
            if grid.dead_mask[node]:
                continue
            if node >= len(levels):
                continue
            if levels[node] != POINT_INTERIOR:
                continue
            if smooth_tet_node(grid, node, q_a=q_a):
                it_moves += 1
                moved_nodes.add(node)

        total_moves += it_moves

        if it_moves == 0:
            break  # Converged

    # Node age (Refine ref_node_age): one age unit per smooth_pass
    # call — reset for nodes that moved, incremented for attempted-but-stuck
    # movable nodes (Levels 0-2). Level 3+ nodes are immovable, never aged.
    ages = grid.nodeAge
    for node in range(min(n_points, len(ages), len(levels))):
        if grid.dead_mask[node]:
            continue
        if levels[node] in (POINT_RIDGE, POINT_SURFACE, POINT_INTERIOR):
            if node in moved_nodes:
                ages[node] = 0
            else:
                ages[node] += 1

    logger.debug(f"  Smooth: {total_moves} nodes moved ({iterations} max iter)")
    return total_moves


def _assert_move_clean(grid, node, xyz):
    """Debug check of the distance-2 argument: the trial move must pass the
    intersection test on the live grid."""
    from .intersection import check_local_intersection
    ball = grid.ball_of(node)
    tets = [list(grid.tetrahedra[4 * t: 4 * t + 4]) for t in ball
            if grid.tetrahedra[4 * t] != -1]
    assert not check_local_intersection(
        grid, tets, coords_override={node: list(xyz)},
        ignored_tets=set(ball) if len(ball) > 0 else None), \
        f"coloured trial for node {node} intersects on the live grid"


def _domain_arrays(grid):
    """Domain description for the trial kernel, mirroring _clamp_to_domain.

    Returns (planes_o, planes_n, has_bbox, bb_min, bb_max).
    """
    has_bbox = False
    bb_min = np.zeros(3, dtype=np.float64)
    bb_max = np.zeros(3, dtype=np.float64)
    plane_o, plane_n = [], []
    if getattr(grid, 'egadsModel', None) is not None:
        from .egads_adapter import get_bounding_box
        lo, hi = get_bounding_box(grid.egadsModel)
        has_bbox = True
        bb_min[:] = lo
        bb_max[:] = hi
    elif grid.solidData and 'surfaceList' in grid.solidData:
        for surf in grid.solidData['surfaceList']:
            if surf['type'] == 'plane':
                plane_o.append(surf['origin'])
                plane_n.append(surf['normal'])
    planes_o = np.array(plane_o, dtype=np.float64).reshape(-1, 3)
    planes_n = np.array(plane_n, dtype=np.float64).reshape(-1, 3)
    return planes_o, planes_n, has_bbox, bb_min, bb_max


def _colour_trials(grid, nodes, nbr_ptr, nbr_idx, domain, q_a):
    """Trial positions for one colour class: returns (out_xyz, out_ok)."""
    from .kernels import smooth_colour_trials_jit
    planes_o, planes_n, has_bbox, bb_min, bb_max = domain
    out_xyz = np.zeros((len(nodes), 3), dtype=np.float64)
    out_ok = np.zeros(len(nodes), dtype=np.bool_)
    smooth_colour_trials_jit(
        grid.coordinates, grid.tetrahedra, grid.metricMesh,
        grid.dead_mask, grid.ball_ptr, grid.ball_idx,
        nbr_ptr, nbr_idx, nodes, planes_o, planes_n, has_bbox,
        bb_min, bb_max, q_a, SMOOTH_ALPHA, QUALITY_IMPROVEMENT_TOL,
        BACKOFF_TRIES, NOISE_FLOOR, out_xyz, out_ok)
    return out_xyz, out_ok


def _smooth_pass_coloured(grid: 'Grid', iterations: int = 3,
                          verbose: bool = True,
                          q_a: float = 1e-3) -> int:
    """Smooth pass with coloured thread-parallel interior smoothing.

    ``smooth_pass`` selects this path for meshes at or above
    ``SMALL_MESH_EDGES`` interior nodes.

    Levels 2 and 1 run serially as in ``_smooth_pass_serial``. Level 0
    is greedy distance-2 coloured on the node adjacency; each colour class gets its trial
    positions from ``smooth_colour_trials_jit`` and is then committed serially
    in node order through ``_commit_if_safe``. The trial's in-kernel
    intersection verdict is final, so the commit skips the second check; the
    parent still re-checks quality and resamples the metric.

    Within a colour no two nodes are within distance 2, so each node's ball
    quality and intersection verdict are evaluated against exactly the state
    it commits into. Coords and metrics are updated in place by the commit,
    so the next colour sees them.
    """
    from .grid import POINT_INTERIOR, POINT_SURFACE, POINT_RIDGE
    from .kernels import node_neighbours_csr_jit, greedy_colour_jit

    levels = grid.pointLevels
    coords = grid.coordinates
    n_points = len(coords) // 3

    if verbose:
        logger.debug(f"  Smooth: quality_floor={q_a:.4e}, "
              f"alpha={SMOOTH_ALPHA:.3f} (coloured)")

    # Smoothing leaves the topology unchanged: the CSR, the neighbour CSR and
    # the colouring are built once per call.
    if grid.csr_is_stale():
        grid.refresh_ball_csr()
    nbr_ptr, nbr_idx = node_neighbours_csr_jit(
        grid.tetrahedra, grid.ball_ptr, grid.ball_idx, n_points)
    cand = np.zeros(n_points, dtype=np.bool_)
    for node in range(n_points):
        if grid.dead_mask[node] or node >= len(levels):
            continue
        if levels[node] != POINT_INTERIOR:
            continue
        if nbr_ptr[node + 1] > nbr_ptr[node]:
            cand[node] = True
    colour = greedy_colour_jit(cand, nbr_ptr, nbr_idx, n_points)
    classes = []
    if cand.any():
        for c in range(int(colour.max()) + 1):
            classes.append(np.flatnonzero(colour == c).astype(np.int64))

    domain = _domain_arrays(grid)

    total_moves = 0
    n_checked = 0
    moved_nodes: set = set()

    for it in range(iterations):
        it_moves = 0

        for node in range(n_points):
            if grid.dead_mask[node]:
                continue
            if node >= len(levels):
                continue
            if levels[node] != POINT_RIDGE:
                continue
            if smooth_boundary_node(grid, node, q_a=q_a):
                it_moves += 1
                moved_nodes.add(node)

        for node in range(n_points):
            if grid.dead_mask[node]:
                continue
            if node >= len(levels):
                continue
            if levels[node] != POINT_SURFACE:
                continue
            if smooth_boundary_node(grid, node, q_a=q_a):
                it_moves += 1
                moved_nodes.add(node)

        for nodes in classes:
            out_xyz, out_ok = _colour_trials(grid, nodes, nbr_ptr, nbr_idx,
                                             domain, q_a)
            for i in range(len(nodes)):
                if not out_ok[i]:
                    continue
                node = int(nodes[i])
                if __debug__ and n_checked % 50 == 0:
                    _assert_move_clean(grid, node, out_xyz[i])
                n_checked += 1
                if _commit_if_safe(grid, node, out_xyz[i], q_a,
                                   skip_intersection=True):
                    it_moves += 1
                    moved_nodes.add(node)

        total_moves += it_moves

        if it_moves == 0:
            break  # Converged

    ages = grid.nodeAge
    for node in range(min(n_points, len(ages), len(levels))):
        if grid.dead_mask[node]:
            continue
        if levels[node] in (POINT_RIDGE, POINT_SURFACE, POINT_INTERIOR):
            if node in moved_nodes:
                ages[node] = 0
            else:
                ages[node] += 1

    logger.debug(f"  Smooth: {total_moves} nodes moved ({iterations} max iter)")
    return total_moves


def smooth_pass(grid: 'Grid', iterations: int = 3,
                verbose: bool = True,
                q_a: float = 1e-3) -> int:
    """Constrained vertex smoothing, dispatched by mesh size.

    A mesh with at least ``SMALL_MESH_EDGES`` interior nodes runs
    ``_smooth_pass_coloured`` (thread-parallel interior colours); a smaller
    mesh runs ``_smooth_pass_serial``. The worker count plays no role.
    """
    from .grid import POINT_INTERIOR

    levels = grid.pointLevels
    coords = grid.coordinates
    n_points = len(coords) // 3

    n_interior = sum(1 for n in range(n_points)
                     if coords[n * 3] is not None
                     and n < len(levels)
                     and levels[n] == POINT_INTERIOR)

    if n_interior >= SMALL_MESH_EDGES:
        return _smooth_pass_coloured(grid, iterations, verbose, q_a)
    return _smooth_pass_serial(grid, iterations, verbose, q_a)
