"""Retile local tetrahedral cavities to improve mesh quality.

Alternative edge and face configurations must preserve manifold topology,
positive volume, cavity volume, and metric quality. Candidate thresholds
follow Freitag's dihedral and solid-angle method.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .cavity import shell_of_edge, fetch_faces, fetch_edges
from .adj import pack_face
from .kernels import (
    dihedral_angles_jit, solid_angle_jit, tet_angle_measures_jit,
    mesh_angle_extrema_jit, edge_swap_configs_jit,
    swap_conserves_volume_jit, config_min_quality_jit, tet_volume_quality_jit,
    swap_score_configs_jit,
)
from .metric import (
    signed_volume, volume_abs,
    check_loseille_lohner_gate,
)

if TYPE_CHECKING:
    from .grid import Grid


# ──────────────── Angle Computations ──────────────────────────────

def _tet_nodes(grid: 'Grid', tet_idx: int) -> Tuple[int, int, int, int]:
    """The four node ids of tet *tet_idx* as Python ints."""
    tets = grid.tetrahedra
    base = tet_idx * 4
    return (int(tets[base]), int(tets[base + 1]),
            int(tets[base + 2]), int(tets[base + 3]))


def dihedral_angle(grid: 'Grid', tet_idx: int) -> List[float]:
    """
    Compute the six dihedral angles of tetrahedron *tet_idx*.

    Returns a list of 6 angles (radians), one per edge, computed as
    the angle between normals of the two adjacent faces.
    """
    n0, n1, n2, n3 = _tet_nodes(grid, tet_idx)
    return dihedral_angles_jit(
        np.asarray(grid.coordinates, dtype=np.float64), n0, n1, n2, n3
    ).tolist()


def solid_angle(grid: 'Grid', tet_idx: int, vertex: int) -> float:
    """
    Compute the solid angle at *vertex* in tetrahedron *tet_idx*
    using the spherical excess formula.
    """
    n0, n1, n2, n3 = _tet_nodes(grid, tet_idx)
    return solid_angle_jit(
        np.asarray(grid.coordinates, dtype=np.float64), n0, n1, n2, n3,
        int(vertex))


# ──────────────── Quality Helpers ──────────────────────────────────

def _angle_measures(grid: 'Grid', tet_idx: int) -> Tuple[float, float, float]:
    """(min dihedral, max dihedral, max solid angle) of tet *tet_idx*."""
    n0, n1, n2, n3 = _tet_nodes(grid, tet_idx)
    return tet_angle_measures_jit(
        np.asarray(grid.coordinates, dtype=np.float64), n0, n1, n2, n3)


def _min_dihedral(grid: 'Grid', tet_idx: int) -> float:
    return _angle_measures(grid, tet_idx)[0]


def _max_dihedral(grid: 'Grid', tet_idx: int) -> float:
    return _angle_measures(grid, tet_idx)[1]


def _max_solid(grid: 'Grid', tet_idx: int) -> float:
    return _angle_measures(grid, tet_idx)[2]


def _evaluate_quality(grid: 'Grid', tet_idx: int,
                      criterion: str) -> float:
    """Evaluate a tetrahedron's quality under a given criterion."""
    coords = grid.coordinates
    tets = grid.tetrahedra
    base = tet_idx * 4
    tet = [tets[base + k] for k in range(4)]
    pts = [coords[v * 3: v * 3 + 3] for v in tet]

    sv = signed_volume(pts[0], pts[1], pts[2], pts[3])
    if sv <= 0:
        return -1e6

    if criterion == "min-dihedral":
        return _angle_measures(grid, tet_idx)[0]
    elif criterion == "max-dihedral":
        return -((_angle_measures(grid, tet_idx)[1] - math.pi))
    elif criterion == "max-solid":
        return -_angle_measures(grid, tet_idx)[2]
    elif criterion == "volume":
        return abs(sv)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


# ──────────────── Adaptive Thresholds ──────────────────────────────

def compute_adaptive_thresholds(grid: 'Grid') -> Optional[Dict]:
    """
    Compute adaptive quality thresholds following Freitag's methodology.

    Returns dict with keys:
        smallDihedralThreshold, largeDihedralThreshold,
        largeSolidThreshold, meshMinDihedral, meshMaxDihedral,
        meshMaxSolid
    """
    n_tets = len(grid.tetrahedra) // 4
    if n_tets == 0:
        return None

    found, mesh_min, mesh_max, mesh_max_solid = mesh_angle_extrema_jit(
        np.asarray(grid.coordinates, dtype=np.float64),
        np.asarray(grid.tetrahedra, dtype=np.int64),
        np.asarray(grid.dead_mask, dtype=np.bool_))
    if not found:
        return None

    return {
        'smallDihedralThreshold': min(math.radians(30), mesh_min + math.radians(10)),
        'largeDihedralThreshold': max(math.radians(150), mesh_max - math.radians(20)),
        'largeSolidThreshold': max(math.radians(240), mesh_max_solid - math.radians(60)),
        'meshMinDihedral': mesh_min,
        'meshMaxDihedral': mesh_max,
        'meshMaxSolid': mesh_max_solid,
    }


def _is_poor_quality(grid: 'Grid', tet_idx: int,
                     thresholds: Dict) -> bool:
    """Check if a tet has poor quality using adaptive thresholds."""
    tets = grid.tetrahedra
    coords = grid.coordinates
    base = tet_idx * 4
    tet = [tets[base + k] for k in range(4)]
    if tet[0] == -1:
        return False
    if any(grid.dead_mask[v] for v in tet):
        return False
    pts = [coords[v * 3: v * 3 + 3] for v in tet]
    vol = volume_abs(pts[0], pts[1], pts[2], pts[3])
    if vol < 1e-12:
        return True

    min_d, max_d, max_s = _angle_measures(grid, tet_idx)

    return (min_d < thresholds['smallDihedralThreshold']
            or max_d > thresholds['largeDihedralThreshold']
            or max_s > thresholds['largeSolidThreshold'])


# ──────────────── Edge Swap Patterns ──────────────────────────────

def _get_edge_swap_patterns() -> Dict[int, List]:
    """Canonical edge swap patterns for N incident tets."""
    return {
        3: [[(0, 1, 2)]],
        4: [[(0, 1, 2), (0, 2, 3)], [(0, 1, 3), (1, 2, 3)]],
        5: [
            [(0, 1, 2), (0, 2, 3), (0, 3, 4)],
            [(0, 1, 3), (0, 3, 4), (1, 2, 3)],
            [(0, 1, 4), (1, 2, 3), (1, 3, 4)],
            [(0, 1, 4), (1, 2, 4), (2, 3, 4)],
            [(0, 1, 2), (0, 4, 2), (2, 3, 4)],
        ],
        6: [
            [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)],
            [(0, 1, 3), (0, 3, 5), (1, 2, 3), (3, 4, 5)],
            [(0, 1, 4), (0, 4, 5), (1, 2, 4), (2, 3, 4)],
            [(0, 1, 5), (1, 2, 5), (2, 3, 4), (2, 4, 5)],
        ],
    }


def get_edge_swap_configs(
    grid: 'Grid', edge: Tuple[int, int], shell: Optional[List[int]] = None,
) -> List[List[List[int]]]:
    """
    Get all valid edge swap configurations for *edge*.

    *shell* is the precomputed ``shell_of_edge`` result; when ``None`` it is
    computed here.

    For each pattern, generate rotations around the equatorial ring,
    validate that all proposed tets have positive volume.
    """
    coords = grid.coordinates
    tets = grid.tetrahedra

    if shell is not None:
        neighbours = shell
    else:
        try:
            neighbours = shell_of_edge(grid, edge[0], edge[1])
        except ValueError:
            return []

    n_shell = len(neighbours)
    if n_shell < 3 or n_shell > 6:
        return []

    edge_set = set(edge)
    # Extract equatorial edges for each tet in the shell
    eq_edges: List[Tuple[int, int]] = []
    for t_idx in neighbours:
        verts = [tets[4 * t_idx + k] for k in range(4)]
        eq = [v for v in verts if v not in edge_set]
        if len(eq) == 2:
            eq_edges.append((eq[0], eq[1]))
        else:
            return []

    if len(eq_edges) != n_shell:
        return []

    # Sort the equatorial vertices into a consecutive loop
    surround = [eq_edges[0][0], eq_edges[0][1]]
    used_edges = {0}
    
    while len(surround) < n_shell:
        current = surround[-1]
        found = False
        for idx, (u, v) in enumerate(eq_edges):
            if idx in used_edges:
                continue
            if u == current:
                surround.append(v)
                used_edges.add(idx)
                found = True
                break
            elif v == current:
                surround.append(u)
                used_edges.add(idx)
                found = True
                break
        if not found:
            return []

    # Verify the loop closes
    last = surround[-1]
    first = surround[0]
    closed = False
    for idx, (u, v) in enumerate(eq_edges):
        if (u == last and v == first) or (v == last and u == first):
            closed = True
            break
    if not closed:
        return []

    patterns = _get_edge_swap_patterns()
    v_top, v_bot = edge
    pat = np.asarray(patterns[n_shell], dtype=np.int64)
    out, n_valid = edge_swap_configs_jit(
        np.asarray(coords, dtype=np.float64),
        np.asarray(grid.dead_mask, dtype=np.bool_),
        np.asarray(surround, dtype=np.int64), int(v_top), int(v_bot), pat)
    return out[:n_valid].tolist()


def _faces_of(v) -> Tuple[Tuple[int, int, int], ...]:
    """The four sorted faces of a tet given as a 4-vertex sequence."""
    return (tuple(sorted((v[1], v[2], v[3]))), tuple(sorted((v[0], v[2], v[3]))),
            tuple(sorted((v[0], v[1], v[3]))), tuple(sorted((v[0], v[1], v[2]))))


def _shell_boundary_faces(grid: 'Grid', shell: List[int]) -> set:
    """
    Boundary faces of a set of tets: faces incident to exactly one tet of
    the set.  This is the surface of the cavity the shell occupies.
    """
    from collections import defaultdict
    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for t_idx in shell:
        v = grid.tetrahedra[4 * t_idx: 4 * t_idx + 4]
        if v[0] == -1:
            continue
        for f in _faces_of(v):
            counts[f] += 1
    return {f for f, c in counts.items() if c == 1}


def _mapped_boundary_before(grid: 'Grid', old_tets: List[int]) -> set:
    """Debug guard: packed keys of the swap cavity's boundary faces that carry
    a ``faceSurfMap`` tag. Must be captured before the old tets are marked dead."""
    return {pack_face(*f) for f in _shell_boundary_faces(grid, old_tets)
            if pack_face(*f) in grid.faceSurfMap}


def _assert_boundary_preserved(mapped_before: set,
                               new_config: List[List[int]], ctx: str) -> None:
    """Debug guard: a swap must not drop a mapped boundary face — it re-tiles the
    cavity interior, so the cavity's boundary faces are invariant."""
    if not mapped_before:
        return
    from collections import defaultdict
    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for tet in new_config:
        for f in _faces_of(tet):
            counts[f] += 1
    after = {pack_face(*f) for f, c in counts.items() if c == 1}
    assert mapped_before <= after, (
        f"swap ({ctx}) dropped mapped boundary face(s): {mapped_before - after}")


def _edge_swap_preserves_manifold(
    grid: 'Grid', shell: List[int], new_config: List[List[int]],
) -> bool:
    """
    True iff replacing *shell* with *new_config* keeps every face shared by
    at most two tets and exactly re-fills the shell's cavity.

    Two failure modes are rejected:
      1. The config's surface (faces appearing once) differs from the shell
         cavity surface — leaves a gap or mismatched re-triangulation.
      2. A config-internal face (shared by two new tets) coincides with a
         face of a live tet *outside* the shell — happens when the
         equatorial ring is non-convex, and would make that face shared by
         three tets (non-manifold; the source of hanging nodes downstream).
    """
    from collections import defaultdict
    cavity_surface = _shell_boundary_faces(grid, shell)

    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for tet in new_config:
        for f in _faces_of(tet):
            counts[f] += 1

    once = set()
    internal = set()
    for f, c in counts.items():
        if c > 2:
            return False
        (once if c == 1 else internal).add(f)

    if once != cavity_surface:
        return False

    # Faces of live tets outside the shell, restricted to those touching the
    # config's vertices (cheap: only the balls of the involved nodes).
    shell_set = set(shell)
    verts = {v for tet in new_config for v in tet}
    external_faces = set()
    for v in verts:
        for t in grid.ball_of(v):
            if t in shell_set:
                continue
            tv = grid.tetrahedra[4 * t: 4 * t + 4]
            if tv[0] == -1:
                continue
            external_faces.update(_faces_of(tv))

    return internal.isdisjoint(external_faces)


def _swap_conserves_volume(
    grid: 'Grid', old_tets, new_config: List[List[int]],
    tol: float = 1e-9,
) -> bool:
    """
    True iff the reconnection tiles the same cavity without folding.

    A swap replaces *old_tets* with *new_config* over an identical cavity, so a
    geometrically valid (fold-free) reconnection conserves the cavity volume::

        sum(|vol(new)|) == sum(|vol(old)|).

    A fold makes two new tets overlap, so the new total overshoots by twice the
    overlap; a gap undershoots.  This is the geometric guard the per-tet
    signed-volume checks miss: a folded tet is individually positive (after
    winding normalisation), so it passes ``signed_volume > 0`` yet still
    double-covers part of the domain — the source of the volume
    inflation, which is produced exclusively by the swap pass.
    """
    tets = grid.tetrahedra
    old_nodes = np.array(
        [[int(tets[4 * t + k]) for k in range(4)] for t in old_tets],
        dtype=np.int64).reshape(-1, 4)
    new_nodes = np.array(new_config, dtype=np.int64).reshape(-1, 4)
    return bool(swap_conserves_volume_jit(
        np.asarray(grid.coordinates, dtype=np.float64),
        np.asarray(grid.dead_mask, dtype=np.bool_),
        old_nodes, new_nodes, float(tol)))


def perform_edge_swap(
    grid: 'Grid', edge: Tuple[int, int],
    new_config: List[List[int]],
    active_tets: set,
    face_to_tets_map: dict = None,
) -> bool:
    """
    Perform an edge swap: remove old shell tets, insert new config.

    Rebuilds ballOfNode and adjacency for the affected region.
    If face_to_tets_map is provided, incrementally updates it.
    """
    try:
        neighbours = shell_of_edge(grid, edge[0], edge[1])
    except ValueError:
        return False

    tets = grid.tetrahedra
    ball = grid.ballOfNode

    # Reject configs that do not exactly re-fill the shell's cavity.
    # Without this, a non-convex equatorial ring can yield positively
    # oriented tets that overlap mesh outside the shell, producing faces
    # shared by >2 tets (non-manifold boundary, hanging nodes downstream).
    if not _edge_swap_preserves_manifold(grid, neighbours, new_config):
        return False

    # Geometric guard: the reconnection must tile the shell
    # cavity without folding.  A non-convex equatorial ring can produce
    # positively-oriented tets that overlap mesh inside the cavity; the manifold
    # check above is purely topological and accepts those, so reject any config
    # whose total volume does not equal the cavity volume.
    if not _swap_conserves_volume(grid, neighbours, new_config):
        return False

    # Check for duplicate tetrahedra
    deleted_tets = set()
    for t_idx in neighbours:
        verts = tuple(sorted(tets[4 * t_idx : 4 * t_idx + 4]))
        deleted_tets.add(verts)

    new_sorted_tets = []
    for new_tet in new_config:
        sorted_verts = tuple(sorted(new_tet))
        if sorted_verts in active_tets and sorted_verts not in deleted_tets:
            return False
        new_sorted_tets.append((new_tet, sorted_verts))

    # Debug-only invariance guard (zero cost under python -O): the swap must
    # not drop any mapped boundary face of the cavity.
    if __debug__:
        _mapped_before = _mapped_boundary_before(grid, neighbours)

    # Remove old tet faces from face_to_tets_map
    if face_to_tets_map is not None:
        for t_idx in neighbours:
            old_verts = tets[4 * t_idx: 4 * t_idx + 4]
            if old_verts[0] == -1:
                continue
            for _face in fetch_faces(tets, t_idx):
                fkey = tuple(sorted(_face))
                if fkey in face_to_tets_map:
                    face_to_tets_map[fkey] = [
                        t for t in face_to_tets_map[fkey] if t != t_idx
                    ]

    # Mark old tets dead
    for t_idx in neighbours:
        verts = tets[4 * t_idx: 4 * t_idx + 4]
        for v in verts:
            if v in ball and ball[v] is not None and t_idx in ball[v]:
                ball[v].remove(t_idx)
        tets[4 * t_idx: 4 * t_idx + 4] = [-1, -1, -1, -1]

    # Remove deleted from active_tets
    for deleted in deleted_tets:
        active_tets.discard(deleted)

    # Bump mesh version — invalidates geom and diagnostic caches
    grid._mesh_version += 1
    grid._topo_version += 1

    # Add new tets
    for new_tet, sorted_verts in new_sorted_tets:
        # get_edge_swap_configs builds every tet with signed_volume > 0 and the
        # cavity-volume guard above has confirmed the config is fold-free, so no
        # winding flip is applied here (never mask an inversion
        # by flipping a negative tet positive).
        tets.extend(new_tet)
        new_idx = len(tets) // 4 - 1
        for v in new_tet:
            if v not in ball or ball[v] is None:
                ball[v] = []
            ball[v].append(new_idx)
        active_tets.add(sorted_verts)

        # Add new tet faces to face_to_tets_map
        if face_to_tets_map is not None:
            for _face in [(new_tet[1], new_tet[2], new_tet[3]),
                          (new_tet[0], new_tet[2], new_tet[3]),
                          (new_tet[0], new_tet[1], new_tet[3]),
                          (new_tet[0], new_tet[1], new_tet[2])]:
                fkey = tuple(sorted(_face))
                face_to_tets_map[fkey].append(new_idx)

    if __debug__:
        _assert_boundary_preserved(_mapped_before, new_config, f"edge {edge}")

    return True

# ──────────────── Metric Quality Helpers ──────────────────────────

def _metric_quality_of_config(grid: 'Grid', tet_idx: int) -> float:
    """Evaluate the anisotropic metric quality of a single tet."""
    n0, n1, n2, n3 = _tet_nodes(grid, tet_idx)
    return tet_volume_quality_jit(
        np.asarray(grid.coordinates, dtype=np.float64),
        np.asarray(grid.metricMesh, dtype=np.float64),
        n0, n1, n2, n3, -2, np.zeros(3), np.zeros(6), False)[1]


def _metric_quality_of_new_config(
    grid: 'Grid', config: List[List[int]],
) -> float:
    """Evaluate the minimum anisotropic metric quality of a config."""
    return config_min_quality_jit(
        np.asarray(grid.coordinates, dtype=np.float64),
        np.asarray(grid.metricMesh, dtype=np.float64),
        np.asarray(config, dtype=np.int64).reshape(-1, 4))


# ──────────────── Local Reconnection ──────────────────────────────

def local_reconnection(
    grid: 'Grid',
    criterion: str = "min-dihedral",
    max_passes: int = 3,
    verbose: bool = True,
    q_r: float = 1.0,
    q_a: float = 1e-3,
) -> int:
    """
    Iterative local reconnection via edge and face swaps.

    For each poor-quality tet (ranked by severity):
        1. Try edge swaps on all 6 edges, accept if worst quality
           improves (Freitag criterion).
        2. If no edge swap helped, try face swaps on all 4 faces.

    Parameters
    ----------
    grid : Grid
    criterion : str
        Quality criterion: "min-dihedral", "max-dihedral", "max-solid",
        or "volume".
    max_passes : int
        Maximum number of sweeps.
    verbose : bool
        Print progress info.
    q_r : float
        Loseille-Löhner relative decay tolerance (1.0 = strict).
    q_a : float
        Loseille-Löhner absolute quality floor.

    Returns
    -------
    int
        Total number of swaps performed.
    """
    total_swaps = 0

    # swap uses .extend() and del on tetrahedra; convert to list for the
    # duration of the function, then back to numpy on exit.
    grid.tetrahedra = list(grid.tetrahedra)

    for pass_num in range(max_passes):
        if verbose:
            logger.debug(f"  Swap pass {pass_num + 1}/{max_passes}")

        thresholds = compute_adaptive_thresholds(grid)
        if thresholds is None:
            if verbose:
                logger.debug("    No valid tetrahedra found")
            break

        if verbose:
            logger.debug(f"    Small dihedral: {math.degrees(thresholds['smallDihedralThreshold']):.1f}°")
            logger.debug(f"    Large dihedral: {math.degrees(thresholds['largeDihedralThreshold']):.1f}°")

        n_tets = len(grid.tetrahedra) // 4

        # Track active tets to prevent duplicate creation
        active_tets = set()
        for _t in range(n_tets):
            if grid.tetrahedra[4 * _t] != -1:
                active_tets.add(tuple(sorted(grid.tetrahedra[4 * _t : 4 * _t + 4])))

        # Build O(N) face-to-tet lookup for face swaps
        from collections import defaultdict
        _face_to_tets_map = defaultdict(list)
        for _t in range(n_tets):
            if grid.tetrahedra[4 * _t] == -1:
                continue
            for _face in fetch_faces(grid.tetrahedra, _t):
                _face_to_tets_map[tuple(sorted(_face))].append(_t)

        poor_tets: List[Tuple[int, float]] = []

        for t in range(n_tets):
            if grid.tetrahedra[4 * t] == -1:
                continue
            if _is_poor_quality(grid, t, thresholds):
                severity = (thresholds['smallDihedralThreshold']
                            - _min_dihedral(grid, t))
                poor_tets.append((t, severity))

        poor_tets.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            logger.debug(f"    Found {len(poor_tets)} poor quality tets")

        if not poor_tets:
            if verbose:
                logger.debug("    No poor quality tets, stopping")
            break

        pass_swaps = 0

        for tet_idx, _ in poor_tets:
            if tet_idx >= len(grid.tetrahedra) // 4:
                continue
            if grid.tetrahedra[4 * tet_idx] == -1:
                continue

            orig_quality = _evaluate_quality(grid, tet_idx, criterion)
            improved = False

            # ── Try edge swaps ────────────────────────────────
            edges = fetch_edges(grid.tetrahedra, tet_idx)

            for edge in edges:
                if improved:
                    break

                # Boundary guard: never swap edges where both
                # endpoints are off the interior (level >= 1): this
                # skips surface-surface, surface-ridge, ridge-ridge and
                # corner pairs alike.
                if (grid.pointLevels[edge[0]] >= 1
                        and grid.pointLevels[edge[1]] >= 1):
                    continue

                try:
                    edge_shell = shell_of_edge(grid, edge[0], edge[1])
                except ValueError:
                    edge_shell = None
                configs = (get_edge_swap_configs(grid, edge, shell=edge_shell)
                           if edge_shell is not None else [])
                best_config = None
                best_min_q = float('-inf')

                if (criterion == "min-dihedral" and configs
                        and all(len(c) == len(configs[0]) for c in configs)):
                    if edge_shell is not None:
                        _tets = grid.tetrahedra
                        _old = np.array(
                            [[int(_tets[4 * t + k]) for k in range(4)]
                             for t in edge_shell],
                            dtype=np.int64).reshape(-1, 4)
                    else:
                        _old = np.zeros((0, 4), dtype=np.int64)
                    _bi, _bq = swap_score_configs_jit(
                        np.asarray(grid.coordinates, dtype=np.float64),
                        np.asarray(grid.dead_mask, dtype=np.bool_),
                        _old, edge_shell is not None,
                        np.array(configs, dtype=np.int64).reshape(
                            len(configs), len(configs[0]), 4),
                        1e-9)
                    if _bi >= 0:
                        best_config = configs[_bi]
                        best_min_q = _bq
                    configs = []

                for config in configs:
                    # Reject folded reconnections up front so a
                    # fold is never chosen as "best": its quality would be
                    # computed on overlapping geometry and look deceptively good.
                    if edge_shell is not None and not _swap_conserves_volume(
                            grid, edge_shell, config):
                        continue
                    config_qualities = []
                    for tet in config:
                        if any(grid.dead_mask[v] for v in tet):
                            config_qualities.append(-1e6)
                            continue
                        pts = [grid.coordinates[v * 3: v * 3 + 3] for v in tet]
                        # get_edge_swap_configs already guarantees signed_volume
                        # > 0; no winding flip here (never mask an
                        # inversion by flipping a negative tet positive).
                        sv = signed_volume(pts[0], pts[1], pts[2], pts[3])
                        if sv <= 1e-15:
                            config_qualities.append(-1e6)
                            continue

                        # Temporarily add tet to evaluate
                        grid.tetrahedra.extend(tet)
                        temp_idx = len(grid.tetrahedra) // 4 - 1
                        q = _evaluate_quality(grid, temp_idx, criterion)
                        config_qualities.append(q)
                        del grid.tetrahedra[temp_idx * 4: temp_idx * 4 + 4]

                    min_q = min(config_qualities) if config_qualities else -1e6
                    if min_q > best_min_q:
                        best_min_q = min_q
                        best_config = config

                # Freitag criterion: new worst > old worst
                if best_config and best_min_q > orig_quality:
                    if perform_edge_swap(grid, edge, best_config, active_tets,
                                         face_to_tets_map=_face_to_tets_map):
                        pass_swaps += 1
                        improved = True

                # Dual quality criterion: if Freitag (dihedral) didn't
                # accept, try metric-quality (tet_quality) acceptance.
                # This catches cases where dihedral angles are marginal
                # but the anisotropic quality metric improves.
                if not improved and best_config:
                    try:
                        old_metric_q = _metric_quality_of_config(
                            grid, tet_idx)
                        new_metric_q = _metric_quality_of_new_config(
                            grid, best_config)
                        if check_loseille_lohner_gate(
                                old_metric_q, new_metric_q, q_r, q_a):
                            if perform_edge_swap(grid, edge, best_config, active_tets,
                                                 face_to_tets_map=_face_to_tets_map):
                                pass_swaps += 1
                                improved = True
                    except (ValueError, IndexError):
                        pass  # Skip if metric evaluation fails

            # ── Try face swaps (2→3) if no edge swap worked ───
            if not improved:
                faces = fetch_faces(grid.tetrahedra, tet_idx)
                for face in faces:
                    if improved:
                        break

                    face_sorted = tuple(sorted(face))

                    # O(N) face-to-tet lookup via pre-built dictionary
                    # replaces the previous O(N²) scan of all tets
                    tets_sharing = _face_to_tets_map.get(face_sorted, [])
                    # Filter out dead tets
                    tets_sharing = [t for t in tets_sharing
                                    if grid.tetrahedra[4 * t] != -1]

                    if len(tets_sharing) != 2:
                        continue

                    t1, t2 = tets_sharing
                    tet1 = [grid.tetrahedra[4 * t1 + k] for k in range(4)]
                    tet2 = [grid.tetrahedra[4 * t2 + k] for k in range(4)]
                    all_verts = list(set(tet1 + tet2))
                    if len(all_verts) != 5:
                        continue

                    face_verts = set(face_sorted)
                    remaining = [v for v in all_verts if v not in face_verts]
                    if len(remaining) != 2:
                        continue

                    v1, v2 = remaining
                    # Check if edge (v1, v2) already exists in the mesh
                    ball_v1 = grid.ball_of(v1)
                    ball_v2 = grid.ball_of(v2)
                    if set(ball_v1) & set(ball_v2):
                        continue

                    f_list = list(face_sorted)

                    # Create 3 new tets from the 2→3 swap
                    new_config = []
                    valid_config = True
                    for i in range(3):
                        new_tet = [v1, v2, f_list[i], f_list[(i + 1) % 3]]
                        if any(grid.dead_mask[v] for v in new_tet):
                            valid_config = False
                            break
                        pts = [grid.coordinates[v * 3: v * 3 + 3] for v in new_tet]
                        sv = signed_volume(pts[0], pts[1], pts[2], pts[3])
                        if sv < 0:
                            new_tet[2], new_tet[3] = new_tet[3], new_tet[2]
                            pts = [grid.coordinates[v * 3: v * 3 + 3] for v in new_tet]
                            sv = signed_volume(pts[0], pts[1], pts[2], pts[3])
                        if sv <= 1e-15:
                            valid_config = False
                            break
                        new_config.append(new_tet)

                    if not valid_config:
                        continue

                    # Geometric guard: the 3 new tets must tile
                    # the 2-tet cavity without folding.  When the tet pair is
                    # reflex (segment v1-v2 misses the shared face interior) the
                    # 2->3 swap yields 3 overlapping tets — each individually
                    # positive after the winding flip above, yet double-covering
                    # the cavity.  Reject by cavity-volume conservation.
                    if not _swap_conserves_volume(grid, [t1, t2], new_config):
                        continue

                    # Check for duplicate tetrahedra
                    deleted_tets = {tuple(sorted(tet1)), tuple(sorted(tet2))}
                    duplicate_found = False
                    for tet in new_config:
                        sorted_verts = tuple(sorted(tet))
                        if sorted_verts in active_tets and sorted_verts not in deleted_tets:
                            duplicate_found = True
                            break
                    if duplicate_found:
                        continue

                    # Evaluate quality
                    config_qs = []
                    for tet in new_config:
                        grid.tetrahedra.extend(tet)
                        temp_idx = len(grid.tetrahedra) // 4 - 1
                        q = _evaluate_quality(grid, temp_idx, criterion)
                        config_qs.append(q)
                        del grid.tetrahedra[temp_idx * 4: temp_idx * 4 + 4]

                    min_q = min(config_qs) if config_qs else -1e6
                    if min_q > orig_quality:
                        if __debug__:
                            _mapped_before_fs = _mapped_boundary_before(
                                grid, [t1, t2])
                        # Remove old tet faces from face_to_tets_map
                        for t_idx_old in sorted([t1, t2], reverse=True):
                            old_verts = grid.tetrahedra[4 * t_idx_old: 4 * t_idx_old + 4]
                            if old_verts[0] != -1:
                                for _face_old in fetch_faces(grid.tetrahedra, t_idx_old):
                                    fkey_old = tuple(sorted(_face_old))
                                    if fkey_old in _face_to_tets_map:
                                        _face_to_tets_map[fkey_old] = [
                                            t for t in _face_to_tets_map[fkey_old]
                                            if t != t_idx_old
                                        ]

                        ball = grid.ballOfNode
                        for t_idx in sorted([t1, t2], reverse=True):
                            verts = grid.tetrahedra[4 * t_idx: 4 * t_idx + 4]
                            for v in verts:
                                if v in ball and ball[v] is not None and t_idx in ball[v]:
                                    ball[v].remove(t_idx)
                            grid.tetrahedra[4 * t_idx: 4 * t_idx + 4] = [-1, -1, -1, -1]

                        for deleted in deleted_tets:
                            active_tets.discard(deleted)

                        for tet in new_config:
                            # Check orientation and flip if needed
                            pts = [grid.coordinates[v * 3 : v * 3 + 3] for v in tet]
                            if signed_volume(pts[0], pts[1], pts[2], pts[3]) < 0:
                                tet[2], tet[3] = tet[3], tet[2]
                            grid.tetrahedra.extend(tet)
                            new_idx = len(grid.tetrahedra) // 4 - 1
                            for v in tet:
                                if v not in ball or ball[v] is None:
                                    ball[v] = []
                                ball[v].append(new_idx)
                            active_tets.add(tuple(sorted(tet)))

                            # Add new tet faces to face_to_tets_map
                            for _face_new in [(tet[1], tet[2], tet[3]),
                                              (tet[0], tet[2], tet[3]),
                                              (tet[0], tet[1], tet[3]),
                                              (tet[0], tet[1], tet[2])]:
                                fkey_new = tuple(sorted(_face_new))
                                _face_to_tets_map[fkey_new].append(new_idx)

                        if __debug__:
                            _assert_boundary_preserved(
                                _mapped_before_fs, new_config,
                                f"face {face_sorted}")

                        # Bump mesh version
                        grid._mesh_version += 1
                        grid._topo_version += 1
                        pass_swaps += 1
                        improved = True

        if verbose:
            logger.debug(f"    Pass {pass_num + 1} completed {pass_swaps} swaps")

        total_swaps += pass_swaps
        if pass_swaps == 0:
            if verbose:
                logger.debug("    No improvements found, stopping")
            break

    grid.tetrahedra = np.array(grid.tetrahedra, dtype=np.int32)
    return total_swaps
