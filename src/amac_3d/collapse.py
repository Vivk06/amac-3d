"""Coarsen metric-short edges while preserving mesh validity.

Collapse obeys boundary hierarchy, duplicate-cell and manifold guards,
signed-volume and quality limits, and local intersection checks. Repeated
passes are coordinated by adapt; edge lengths follow Michal & Krakos (2012).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from typing import List, Tuple, TYPE_CHECKING

import math
import numpy as np

from .metric import (
    det_metric, tet_quality, face_quality,
    signed_volume,
    face_normal, dot_product,
    COLLAPSE_THRESHOLD,
)
from .cavity import (
    collect_edges, shell_of_edge, nodes_around_node, edges_around_node,
)
from .adj import pack_face
from .kernels import (
    edge_length_nodes_jit, duplicate_cell_jit,
    collapse_len_quality_jit, collapse_tet_local_jit,
)


def _int_array(values, dtype=np.int64) -> np.ndarray:
    """Node or tet index list/array as a contiguous integer array."""
    return np.asarray(values, dtype=dtype)


def _tets_array(grid: 'Grid') -> np.ndarray:
    """``grid.tetrahedra`` as a numpy int32 array (no copy when it already is)."""
    t = grid.tetrahedra
    return t if isinstance(t, np.ndarray) else np.asarray(t, dtype=np.int32)


ALPHA_LEN = 2.8284
L_UP = math.sqrt(2.0)
MAX_COLLAPSE_NODE_AGE = 20

if TYPE_CHECKING:
    from .grid import Grid


def _is_bdy_edge(grid: 'Grid', p1: int, p2: int) -> bool:
    return p1 in grid.bdyPointsSet and p2 in grid.bdyPointsSet


# ──────────────── Boundary Crease Check ───────────────────────────

def check_boundary_crease(
    grid: 'Grid', pc: int, po: int,
    ball_pc: List[int], crease_threshold: float = 0.866,
) -> bool:
    """Reject a collapse pc->po that folds any incident boundary face.

    Compares pre- and post-collapse Euclidean face normals for every
    boundary face incident to pc (identified via faceSurfMap membership,
    not a linear tet scan). Returns False if any normal dot
    product drops below crease_threshold (cos 30° = 0.866).
    """
    coords = grid.coordinates
    tets = grid.tetrahedra

    for tet_idx in ball_pc:
        tet_nodes = tets[4 * tet_idx: 4 * tet_idx + 4]
        if tet_nodes[0] == -1:
            continue
        for tri in ((tet_nodes[0], tet_nodes[1], tet_nodes[2]),
                    (tet_nodes[0], tet_nodes[1], tet_nodes[3]),
                    (tet_nodes[0], tet_nodes[2], tet_nodes[3]),
                    (tet_nodes[1], tet_nodes[2], tet_nodes[3])):
            if pc not in tri:
                continue
            if pack_face(tri[0], tri[1], tri[2]) not in grid.faceSurfMap:
                continue
            if po in tri:
                continue  # face collapses to an edge — skip
            others = [v for v in tri if v != pc]
            if any(grid.dead_mask[v] for v in (pc, po, others[0], others[1])):
                continue
            p_pc = coords[pc * 3: pc * 3 + 3]
            p_po = coords[po * 3: po * 3 + 3]
            p_o0 = coords[others[0] * 3: others[0] * 3 + 3]
            p_o1 = coords[others[1] * 3: others[1] * 3 + 3]
            n_old = face_normal(p_pc, p_o0, p_o1)
            n_new = face_normal(p_po, p_o0, p_o1)
            if dot_product(n_old, n_new) < crease_threshold:
                return False
    return True


# ──────────────── Collapse Quality Check ──────────────────────────

class CollapseMemo:
    """Gate-6 rejection memo for one ``_collapse_to_quiescence`` call.

    ``rejected[(pc, po)]`` is the commit counter at which the local part of
    ``collapse_tet_quality`` rejected the candidate. ``topo_stamp[n]`` is the
    counter of the last commit whose ``ball_pc`` tets contained ``n`` (its ball
    or a tet's node list changed); ``geo_stamp[n]`` is the counter of the last
    commit with ``n`` as its pc or po (coordinates, metric, dead_mask changed).
    A rejection is replayed only while ``topo_stamp[pc]`` and ``geo_stamp`` of
    the ball nodes and po are unchanged since.
    Valid only while node and tet indices are stable (no ``cleanDeadElements``)
    and ``q_r``, ``q_a`` are constant.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.topo_stamp: dict = {}
        self.geo_stamp: dict = {}
        self.rejected: dict = {}


def _collapse_tet_quality_local(
    grid: 'Grid', pc: int, po: int,
    ball_pc: List[int], shell: List[int],
    q_r: float = 1.0, q_a: float = 1e-3,
) -> int:
    """Survivor-tet part of ``collapse_tet_quality``. Reads only the node lists,
    coordinates, metric, dead_mask and faceSurfMap faces of the survivor tets.
    Returns 0 on a local rejection, 1 otherwise."""
    coords = grid.coordinates
    tets = grid.tetrahedra
    mm = grid.metricMesh

    tets_of_interest = [t for t in ball_pc if t not in shell]

    # Tet gates (old worst, volume, per-tet quality, Loseille-Lohner) in the kernel
    if not collapse_tet_local_jit(
            coords, mm, _tets_array(grid), grid.dead_mask,
            _int_array(ball_pc), _int_array(shell),
            int(pc), int(po), float(q_r), float(q_a)):
        return 0

    face_set: set = set()
    face_qualities: List[float] = []

    for tet_idx in tets_of_interest:
        tet_nodes = list(tets[tet_idx * 4: tet_idx * 4 + 4])
        # Replace pc with po
        new_nodes = [po if n == pc else n for n in tet_nodes]

        # Face qualities
        faces = [
            tuple(sorted([new_nodes[0], new_nodes[1], new_nodes[2]])),
            tuple(sorted([new_nodes[0], new_nodes[1], new_nodes[3]])),
            tuple(sorted([new_nodes[0], new_nodes[2], new_nodes[3]])),
            tuple(sorted([new_nodes[1], new_nodes[2], new_nodes[3]])),
        ]
        for face in faces:
            if pack_face(face[0], face[1], face[2]) not in grid.faceSurfMap:
                continue  # Krakos (d): interior face — skip
            if face in face_set:
                continue
            face_set.add(face)
            fn = list(face)
            fm = [mm[n * 6: n * 6 + 6] for n in fn]
            if any(grid.dead_mask[n] for n in fn):
                return 0
            fp = [coords[n * 3: n * 3 + 3] for n in fn]
            qf = face_quality(fp[0], fp[1], fp[2], fm)
            face_qualities.append(qf)

    if face_qualities and min(face_qualities) < 1e-8:
        return 0
    return 1


def collapse_tet_quality(
    grid: 'Grid', pc: int, po: int,
    ball_pc: List[int], shell: List[int],
    q_r: float = 1.0, q_a: float = 1e-3,
    _local_done: bool = False,
) -> float:
    """
    Guards 3 and 4 of the collapse: does merging pc → po produce a valid,
    non-intersecting neighbourhood?

    For each tet in ball(pc) \\ shell (the tets that survive the collapse),
    replace pc with po and check:
        1. signed_volume > 1e-12   — inversion / near-degeneracy guard
        2. tet_quality  > 1e-8     — Knupp algebraic quality floor
        3. face_quality > 1e-8     — per-face quality floor (deduplicated by
                                     sorted vertex triple)
    Then run the self-intersection guard: rebuild the merged tets with
    pc→po and call ``intersection.check_local_intersection`` with po moved to
    ``po_coords``. For a boundary edge, po_coords is first re-projected onto po's
    surface (Level 1) or curve (Level 2) via ``geom``, so the intersection test
    sees the true post-collapse position rather than the pre-projection one.

    ``_local_done=True`` skips the survivor-tet part when the caller has already
    run ``_collapse_tet_quality_local`` and it passed.

    Returns 1 if every check passes, 0 on the first failure.
    """
    coords = grid.coordinates
    tets = grid.tetrahedra

    tets_of_interest = [t for t in ball_pc if t not in shell]

    if not _local_done and not _collapse_tet_quality_local(
            grid, pc, po, ball_pc, shell, q_r, q_a):
        return 0

    # Krakos (e): boundary crease check
    if not check_boundary_crease(grid, pc, po, ball_pc):
        return 0

    # Intersection prevention check: skip for pure-interior cavities
    # where positive signed volume suffices; retained for any boundary node)
    from .intersection import check_local_intersection
    tets_to_check = []
    all_interior = True
    levels = grid.pointLevels
    for tet_idx in tets_of_interest:
        tet_nodes = list(tets[tet_idx * 4: tet_idx * 4 + 4])
        new_nodes = [po if n == pc else n for n in tet_nodes]
        tets_to_check.append(new_nodes)
        if all_interior:
            for n in new_nodes:
                if n < len(levels) and levels[n] != 0:
                    all_interior = False
                    break

    if not all_interior:
        po_coords = coords[po * 3: po * 3 + 3]
        if _is_bdy_edge(grid, pc, po):
            from .geom import project_to_surface, project_to_curve
            level = levels[po] if po < len(levels) else 0
            x, y, z = po_coords
            if level == 1:
                surfid = grid.bdyFaceMap.get(po, -1)
                if surfid != -1:
                    po_coords = project_to_surface([x, y, z], grid, surfid)
            elif level == 2:
                ednr = grid.bdyEdgeMap.get(po, -1)
                if ednr != -1:
                    po_coords = project_to_curve([x, y, z], grid, ednr)

        if check_local_intersection(
            grid, tets_to_check,
            coords_override={po: po_coords},
            ignored_tets=set(ball_pc) | set(shell)
        ):
            return 0

    return 1


def collapse_len_quality(
    grid: 'Grid', edges: List[Tuple[int, int]],
    u_idx: int, pc: int,
    post_min_ratio: float = 1e-3,
    post_max_ratio: float = ALPHA_LEN * L_UP,
) -> int:
    """
    Post-collapse edge-length bound. After merging pc → po, every edge from po
    to pc's former neighbours must have Riemannian length
    ℓ_M ∈ [post_min_ratio, post_max_ratio].

    The "no-worse" clause (refine ref_collapse.c:868) also accepts when
    new_min >= old_min and new_max <= old_max, even if outside the absolute
    band — the collapse does not worsen the local edge-length extremes.

    Returns 1 if every incident edge is within bounds, 0 otherwise.
    """
    po = edges[u_idx][1] if edges[u_idx][0] == pc else edges[u_idx][0]

    return int(collapse_len_quality_jit(
        grid.coordinates, grid.metricMesh, _tets_array(grid), grid.dead_mask,
        _int_array(grid.ball_of(pc)), int(pc), int(po),
        float(post_min_ratio), float(post_max_ratio)))


# ──────────────── Edge Collapse ────────────────────────────────────

def collapse_edge(
    grid: 'Grid', pc: int, po: int,
    ball_pc: List[int], ball_po: List[int],
    shell: List[int],
) -> bool:
    """
    Collapse edge (pc, po) by merging pc into po.

    Post-conditions:
        - pc is dead (coords = None, metric = -1)
        - All references to pc become po
        - Shell tets are deleted (marked [-1,-1,-1,-1])
        - ballOfNode updated for all affected nodes
    """
    coords = grid.coordinates
    tets = grid.tetrahedra
    ball = grid.ballOfNode
    mm = grid.metricMesh
    ml = grid.metricLog
    levels = grid.pointLevels

    # Capture faceSurfMap entries touching pc before the merge mutates tets.
    # Only faces of pc's incident tets (ball_pc) can change — no whole-map scan.
    _affected_faces: dict = {}
    for _t in ball_pc:
        _v = tets[4 * _t: 4 * _t + 4]
        if _v[0] == -1:
            continue
        for _f in ((_v[1], _v[2], _v[3]), (_v[0], _v[2], _v[3]),
                   (_v[0], _v[1], _v[3]), (_v[0], _v[1], _v[2])):
            if pc in _f:
                _key = pack_face(*_f)
                if _key in grid.faceSurfMap:
                    _affected_faces[_key] = (_f, grid.faceSurfMap[_key])

    ball_minus_shell = [t for t in ball_pc if t not in shell]

    # Replace pc → po in surviving tets
    for tet_idx in ball_minus_shell:
        for k in range(4):
            if tets[4 * tet_idx + k] == pc:
                tets[4 * tet_idx + k] = po

    # Update ball[po]
    for t in ball_minus_shell:
        ball[po].append(t)

    # Remove shell tets from incident nodes' balls
    for shell_idx in shell:
        verts = tets[4 * shell_idx: 4 * shell_idx + 4]
        for v in verts:
            if v == pc:
                continue
            if v in ball and shell_idx in ball[v]:
                ball[v].remove(shell_idx)

    ball[pc] = []

    # Mark shell tets dead
    for shell_idx in shell:
        tets[4 * shell_idx: 4 * shell_idx + 4] = [-1, -1, -1, -1]

    # Mark pc dead
    coords[3 * pc: 3 * pc + 3] = np.nan
    grid.dead_mask[pc] = True
    grid._topo_version += 1
    grid._mesh_version += 1
    mm[6 * pc: 6 * pc + 6] = [-1] * 6
    ml[6 * pc: 6 * pc + 6] = [-1] * 6

    # Boundary cleanup
    if _is_bdy_edge(grid, pc, po):
        _bdy_edge_collapse(grid, pc)
        # Reconcile the surviving node's classification with the post-collapse
        # topology, then re-project it onto its geometry.
        grid.reclassify_node(po)
        _reproject_surviving_node(grid, po)
    else:
        levels[pc] = -1
        # po absorbed pc's incident tets; reconcile its classification too.
        grid.reclassify_node(po)

    # Remap faceSurfMap keys pc→po; drop the collapsed edge's own faces
    # (pc,po,x) → (po,po,x) as degenerate. On a merge collision (a surviving
    # po-face already carries a tag) keep the existing tag.
    for _key, (_f, _sid) in _affected_faces.items():
        grid.faceSurfMap.pop(_key, None)
        _remapped = tuple(po if v == pc else v for v in _f)
        if len(set(_remapped)) == 3:
            grid.faceSurfMap.setdefault(pack_face(*_remapped), _sid)

    return True


def _reproject_surviving_node(grid: 'Grid', po: int) -> None:
    """
    After collapse, ensure the surviving boundary node *po* lies
    exactly on its geometric entity.

    If po is a face node (Level 1), re-project onto its surface.
    If po is an edge node (Level 2), re-project onto its curve.
    Corner nodes (Level 3) are not modified.
    """
    from .geom import project_to_surface, project_to_curve

    levels = grid.pointLevels
    coords = grid.coordinates

    if po >= len(levels):
        return

    level = levels[po]
    if level <= 0 or level >= 3:
        return  # Interior or corner — no projection needed

    x = coords[po * 3]
    y = coords[po * 3 + 1]
    z = coords[po * 3 + 2]

    if x is None:
        return

    def _inverts(new):
        # True when moving po to *new* leaves any live tet in its ball flat or inverted.
        for t in grid.ball_of(po):
            n = grid.tetrahedra[4 * t: 4 * t + 4]
            if n[0] == -1:
                continue
            pts = [list(new) if v == po else list(coords[3 * v: 3 * v + 3]) for v in n]
            if signed_volume(*pts) < 1e-12:
                return True
        return False

    if level == 1:
        surfid = grid.bdyFaceMap.get(po, -1)
        if surfid != -1:
            proj = project_to_surface([x, y, z], grid, surfid)
            if _inverts(proj):
                return
            coords[po * 3: po * 3 + 3] = proj
    elif level == 2:
        ednr = grid.bdyEdgeMap.get(po, -1)
        if ednr != -1:
            proj = project_to_curve([x, y, z], grid, ednr)
            if _inverts(proj):
                return
            coords[po * 3: po * 3 + 3] = proj


def _bdy_edge_collapse(grid: 'Grid', pc: int) -> None:
    """Clean up boundary data structures when a boundary node is removed."""
    levels = grid.pointLevels

    if levels[pc] == 1:
        if pc in grid.bdyPoints:
            grid.bdyPoints[grid.bdyPoints.index(pc)] = -1
        if pc in grid.bdyFacePoints:
            grid.bdyFacePoints[grid.bdyFacePoints.index(pc)] = -1
        grid.bdyFaceMap.pop(pc, None)
        levels[pc] = -1
        return

    if levels[pc] == 2:
        if pc in grid.bdyPoints:
            grid.bdyPoints[grid.bdyPoints.index(pc)] = -1
        for idx, entry in enumerate(grid.bdyEdgePoints):
            if isinstance(entry, (list, tuple)):
                p, _ = entry
                if p == pc:
                    grid.bdyEdgePoints[idx] = (-1, -1)
                    break
        if pc in grid.bdyEdgeMap:
            grid.bdyEdgeMap[pc] = None
        levels[pc] = -1
        return


def check_duplicate_cell(grid: 'Grid', pc: int, po: int, shell: List[int]) -> bool:
    """Refine-style manifold test (ref_collapse.c:444-470).

    Substitute po for pc in each surviving tet of ball(pc).  If any
    resulting sorted 4-tuple already exists in ball(po), the collapse
    would create a duplicate cell — reject.  The test also rejects a collapse
    whose substituted tet recreates a face of a live tet of ball(po) outside
    the shell, which would leave an interior face shared by four tets.
    """
    return bool(duplicate_cell_jit(
        _tets_array(grid), _int_array(grid.ball_of(po)),
        _int_array(grid.ball_of(pc)), _int_array(shell), int(pc), int(po)))


def check_link_condition(grid: 'Grid', u: int, v: int, shell: List[int]) -> bool:
    """Verify the topological link condition for collapsing edge ``(u, v)``.

    The collapse is topologically safe iff ``Lk(u) ∩ Lk(v) == Lk(uv)`` — the
    only simplices common to the links of *u* and *v* are those of the shared
    edge itself.  Violations would merge the two vertices into a non-manifold
    edge or vertex.

    Args:
        grid: The mesh.
        u, v: Endpoints of the candidate collapse edge.
        shell: Tetrahedra sharing edge ``(u, v)`` (its open or closed fan).

    Returns:
        True if the link condition holds (the collapse is topologically admissible).
    """
    tets = grid.tetrahedra
    
    # 1. Gather neighbors of u and v
    u_neighbors = set()
    for t_idx in grid.ball_of(u):
        if tets[4 * t_idx] == -1: continue
        for val in tets[4 * t_idx : 4 * t_idx + 4]:
            if val != u:
                u_neighbors.add(val)
                
    v_neighbors = set()
    for t_idx in grid.ball_of(v):
        if tets[4 * t_idx] == -1: continue
        for val in tets[4 * t_idx : 4 * t_idx + 4]:
            if val != v:
                v_neighbors.add(val)
                
    common_neighbors = u_neighbors.intersection(v_neighbors)
    
    # 2. Gather ring vertices (Vertex Link of edge uv)
    ring_vertices = set()
    shell_opposite_edges = set()
    for t_idx in shell:
        if tets[4 * t_idx] == -1: continue
        tet = tets[4 * t_idx : 4 * t_idx + 4]
        others = [val for val in tet if val != u and val != v]
        if len(others) == 2:
            ring_vertices.update(others)
            shell_opposite_edges.add(tuple(sorted(others)))
            
    # Vertex Link Condition: common neighbors must match ring vertices exactly
    if common_neighbors != ring_vertices:
        return False
        
    # 3. Gather common faces (edges forming a face with both u and v)
    u_faces = set()
    for t_idx in grid.ball_of(u):
        if tets[4 * t_idx] == -1: continue
        tet = tets[4 * t_idx : 4 * t_idx + 4]
        for f in [(tet[1], tet[2], tet[3]), (tet[0], tet[2], tet[3]),
                  (tet[0], tet[1], tet[3]), (tet[0], tet[1], tet[2])]:
            if u in f:
                u_faces.add(tuple(sorted([val for val in f if val != u])))
                
    v_faces = set()
    for t_idx in grid.ball_of(v):
        if tets[4 * t_idx] == -1: continue
        tet = tets[4 * t_idx : 4 * t_idx + 4]
        for f in [(tet[1], tet[2], tet[3]), (tet[0], tet[2], tet[3]),
                  (tet[0], tet[1], tet[3]), (tet[0], tet[1], tet[2])]:
            if v in f:
                v_faces.add(tuple(sorted([val for val in f if val != v])))
                
    common_faces = u_faces.intersection(v_faces)
    
    # Edge Link Condition: common faces must match shell opposite edges exactly
    if common_faces != shell_opposite_edges:
        return False
        
    return True


# ──────────────── Collapse Direction ───────────────────────────────

def _normalize_collapse_direction(grid: 'Grid', pc_a: int, po: int):
    """Apply the boundary-hierarchy rules (Krakos (a)) to a candidate edge.

    Rule: always remove the lower-level node to preserve boundary topology,
    and verify surface/curve compatibility for boundary pairs. Returns the
    normalized ``(dying, surviving)`` pair, or ``None`` when the edge cannot
    legally collapse in either direction (corner endpoint, or incompatible
    surfid/curve membership).
    """
    levels = grid.pointLevels
    if levels[pc_a] > levels[po]:
        # pc_a is higher-level (more constrained) — swap
        # direction: collapse po into pc_a instead
        pc_a, po = po, pc_a
    if levels[pc_a] == 3:
        # Corner points are immovable
        return None
    if levels[pc_a] >= 1 and levels[po] == 0:
        # pc_a is boundary, po is interior — swap direction:
        # we want interior (pc_new=po) to die, boundary to survive
        pc_a, po = po, pc_a

    # Check geometric compatibility for boundary points
    corner_ednrs = {cp: ednrs for cp, ednrs in grid.cornerPoints}
    if levels[pc_a] == 1:
        sf_c = grid.bdyFaceMap.get(pc_a)
        if sf_c is not None:
            if levels[po] == 1:
                sf_o = grid.bdyFaceMap.get(po)
                if sf_c != sf_o:
                    return None
            elif levels[po] in (2, 3):
                if sf_c not in grid.cornerPointSurfMap.get(po, set()):
                    return None
    elif levels[pc_a] == 2:
        ed_c = grid.bdyEdgeMap.get(pc_a)
        if ed_c is not None:
            if levels[po] == 2:
                ed_o = grid.bdyEdgeMap.get(po)
                if ed_c != ed_o:
                    return None
            elif levels[po] == 3:
                if ed_c not in corner_ednrs.get(po, set()):
                    return None
    return pc_a, po


# ──────────────── Collapse Pass ────────────────────────────────────

def _build_edges_by_node(edges):
    """Build node -> incident-edge-index map in O(Ne).

    Lists are ascending by construction (enumeration order), matching
    what ``edges_around_node`` returns via an O(Ne) filter.
    """
    from collections import defaultdict
    edges_by_node = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        edges_by_node[u].append(i)
        edges_by_node[v].append(i)
    return edges_by_node


def _apply_collapses(grid, edges, edges_by_node, removal_flags, min_node_length,
                     short_edges, n_points, show_rejections,
                     q_r, q_a, collapse_threshold,
                     post_min_ratio, post_max_ratio, memo=None) -> int:
    """Apply loop shared by serial and parallel paths — all guards verbatim."""
    coords = grid.coordinates
    mm = grid.metricMesh
    tets = grid.tetrahedra

    _reject = {
        "dead_or_out_of_bounds": 0,
        "no_removal_flag": 0,
        "boundary_hierarchy": 0,
        # This diagnostic key records duplicate-cell rejections.
        "link_condition": 0,
        "collapse_tet_quality": 0,
        "collapse_len_quality": 0,
        "length_reverify": 0,
    }

    removal_nodes: List[Tuple[int, float]] = [
        (i, min_node_length[i])
        for i in range(n_points) if removal_flags[i] == 1
    ]
    removal_nodes.sort(key=lambda x: x[1])

    n_collapses = 0
    ages = grid.nodeAge

    for pc, _ in removal_nodes:
        if grid.dead_mask[pc]:
            continue
        if removal_flags[pc] == 0:
            continue
        if pc < len(ages) and ages[pc] > MAX_COLLAPSE_NODE_AGE:
            continue

        adj_edges = edges_by_node.get(pc, [])

        adj_with_len: List[Tuple[int, float]] = []
        for u_idx in adj_edges:
            p1, p2 = edges[u_idx]
            if p1 >= n_points or p2 >= n_points:
                continue
            if grid.dead_mask[p1] or grid.dead_mask[p2]:
                continue
            m1 = mm[6 * p1: 6 * p1 + 6]
            m2 = mm[6 * p2: 6 * p2 + 6]
            if len(m1) == 0 or len(m2) == 0 or m1[0] == -1 or m2[0] == -1:
                continue
            L = edge_length_nodes_jit(coords, mm, p1, p2)
            if L < collapse_threshold:
                adj_with_len.append((u_idx, L))

        adj_with_len.sort(key=lambda x: x[1])

        for u_idx, _ in adj_with_len:
            p1, p2 = edges[u_idx]
            pc_a = pc
            po = p2 if p1 == pc_a else p1

            if grid.dead_mask[po]:
                _reject["dead_or_out_of_bounds"] += 1
                continue
            if removal_flags[po] == 0:
                _reject["no_removal_flag"] += 1
                continue

            # Gate 1: length re-verify (staleness check, 3.4 µs)
            L = edge_length_nodes_jit(coords, mm, p1, p2)
            if L > collapse_threshold:
                _reject["length_reverify"] += 1
                continue

            # Gate 2: boundary hierarchy
            direction = _normalize_collapse_direction(grid, pc_a, po)
            if direction is None:
                _reject["boundary_hierarchy"] += 1
                continue
            pc_a, po = direction

            ball_pc = grid.ball_of(pc_a)
            ball_po = grid.ball_of(po)
            if len(ball_pc) == 0:
                continue

            try:
                shell = shell_of_edge(grid, pc_a, po)
            except ValueError:
                continue

            # Gate 3: duplicate-cell manifold test (refine-style)
            if not check_duplicate_cell(grid, pc_a, po, shell):
                _reject["link_condition"] += 1
                continue

            # Gate 5: post-collapse edge-length bound
            lq = collapse_len_quality(grid, edges, u_idx, pc_a,
                                      post_min_ratio=post_min_ratio,
                                      post_max_ratio=post_max_ratio)
            if lq == 0:
                _reject["collapse_len_quality"] += 1
                continue

            # Gate 6: tet quality + intersection (most expensive, last)
            if memo is not None:
                prev = memo.rejected.get((pc_a, po))
                if prev is not None and memo.topo_stamp.get(pc_a, -1) <= prev:
                    geo = memo.geo_stamp
                    last = geo.get(po, -1)
                    for t in ball_pc:
                        tn = tets[4 * t: 4 * t + 4]
                        if tn[0] == -1:
                            continue
                        for n in tn:
                            g_n = geo.get(n, -1)
                            if g_n > last:
                                last = g_n
                    if last <= prev:
                        _reject["collapse_tet_quality"] += 1
                        continue
                if not _collapse_tet_quality_local(grid, pc_a, po, ball_pc,
                                                   shell, q_r, q_a):
                    memo.rejected[(pc_a, po)] = memo.counter
                    _reject["collapse_tet_quality"] += 1
                    continue

            tq = collapse_tet_quality(grid, pc_a, po, ball_pc, shell,
                                     q_r=q_r, q_a=q_a,
                                     _local_done=memo is not None)
            if tq == 0:
                _reject["collapse_tet_quality"] += 1
                continue

            if memo is not None:
                touched = set()
                for t in ball_pc:
                    tn = tets[4 * t: 4 * t + 4]
                    if tn[0] != -1:
                        touched.update(tn)
            success = collapse_edge(grid, pc_a, po, ball_pc, ball_po, shell)
            if success:
                if memo is not None:
                    memo.counter += 1
                    for n in touched:
                        memo.topo_stamp[n] = memo.counter
                    memo.geo_stamp[pc_a] = memo.counter
                    memo.geo_stamp[po] = memo.counter
                n_collapses += 1
                removal_flags[pc_a] = 0
                if po < len(ages):
                    ages[po] = 0
                # 1-ring poisoning: suppress neighbours of the surviving
                # node from being re-collapsed in this pass (refine-style)
                removal_flags[po] = 0
                for nb in nodes_around_node(grid, po):
                    if nb < n_points:
                        removal_flags[nb] = 0
                break
        else:
            if pc < len(ages):
                ages[pc] += 1

    if show_rejections:
        logger.debug(f"  Collapse rejection breakdown: {_reject}")
    return n_collapses


def collapse_pass(grid: 'Grid', show_rejections: bool = False,
                          q_r: float = 1.0, q_a: float = 1e-3,
                          collapse_threshold: float = COLLAPSE_THRESHOLD,
                          post_min_ratio: float = 1e-3,
                          post_max_ratio: float = ALPHA_LEN * L_UP,
                          memo=None) -> int:
    """Collapse metric-short edges while preserving boundary and quality guards.

    Candidate edges are scanned by the kernel and processed shortest first.
    The post-collapse band and optional memo constrain accepted edits.
    """
    coords = grid.coordinates
    mm = grid.metricMesh

    n_points = len(coords) // 3
    n_tets = len(grid.tetrahedra) // 4

    if len(mm) != n_points * 6:
        logger.warning(f"Collapse arrays out of sync: {n_points} nodes but {len(mm)} metric entries (expected {n_points * 6})")
        if len(mm) < n_points * 6:
            padding = [-1.0] * (n_points * 6 - len(mm))
            mm.extend(padding)

    edges = collect_edges(grid)

    from .kernels import collapse_candidate_scan_jit
    edges_arr = np.array(edges, dtype=np.int32)
    rf, mnl, se, sl, n_short = collapse_candidate_scan_jit(
        coords, mm, grid.dead_mask, edges_arr, n_points, collapse_threshold
    )
    removal_flags = rf.tolist()
    min_node_length = mnl.tolist()
    short_edges = [(int(se[i]), float(sl[i])) for i in range(n_short)]

    edges_by_node = _build_edges_by_node(edges)

    return _apply_collapses(grid, edges, edges_by_node, removal_flags,
                            min_node_length, short_edges, n_points,
                            show_rejections, q_r, q_a, collapse_threshold,
                            post_min_ratio, post_max_ratio, memo)




# ──────────────── Flat-Tet Repair ──────────────────────────────────

def repair_flat_tets(grid: 'Grid', flat_tets: List[int]) -> int:
    """Remove flat (zero-volume) tets by collapsing one of their edges.

    Flat tets are invisible to the normal passes: their edges are usually
    metric-conforming (never a collapse candidate), their degenerate face
    can lie on the boundary (off-limits to swap), and their vertices are
    often ridge/corner-constrained (immovable for smoothing) — observed as
    three collinear-on-ridge tets surviving four full ABCD cycles on the
    cube run.

    For each flat tet, attempt to collapse one of its six edges, shortest
    Euclidean first (the merge that distorts surrounding geometry least),
    with the metric-length candidate gate bypassed. Every safety guard is
    kept: boundary hierarchy and surface/curve compatibility
    (``_normalize_collapse_direction``), duplicate-cell test,
    ``collapse_tet_quality`` (positive volume, Knupp floor, boundary
    crease, self-intersection — ``q_a = 1e-8`` so no new degenerate tet is
    accepted) and the post-collapse edge-length bound.

    The caller must run ``grid.cleanDeadElements()`` after a non-zero
    return.

    Returns the number of repair collapses performed.
    """
    from .report import tet_is_flat

    coords = grid.coordinates
    tets = grid.tetrahedra
    n_repaired = 0

    for t in flat_tets:
        if tets[4 * t] == -1:
            continue        # removed by an earlier repair collapse
        if not tet_is_flat(grid, t):
            continue        # reshaped (fixed) by an earlier repair collapse
        verts = list(tets[4 * t: 4 * t + 4])

        tet_edges: List[Tuple[float, int, int]] = []
        for i in range(4):
            for j in range(i + 1, 4):
                pa, pb = verts[i], verts[j]
                ca = coords[3 * pa: 3 * pa + 3]
                cb = coords[3 * pb: 3 * pb + 3]
                d2 = sum((ca[k] - cb[k]) ** 2 for k in range(3))
                tet_edges.append((d2, pa, pb))
        tet_edges.sort()

        for _, pa, pb in tet_edges:
            direction = _normalize_collapse_direction(grid, pa, pb)
            if direction is None:
                continue
            pc, po = direction

            ball_pc = grid.ball_of(pc)
            ball_po = grid.ball_of(po)
            try:
                shell = shell_of_edge(grid, pc, po)
            except ValueError:
                continue
            if not check_duplicate_cell(grid, pc, po, shell):
                continue
            # q_r=inf disables the relative gate; q_a=1e-8 still rejects
            # any resulting degenerate tet (tet_quality returns 0.0).
            if not collapse_tet_quality(grid, pc, po, ball_pc, shell,
                                        q_r=float('inf'), q_a=1e-8):
                continue
            if not collapse_len_quality(grid, [(pc, po)], 0, pc):
                continue
            if collapse_edge(grid, pc, po, ball_pc, ball_po, shell):
                n_repaired += 1
                break

    return n_repaired
