"""Project and classify boundary nodes on their geometric entities.

Use EGADS when available, analytic plane or sphere projection when defined,
then a PN-triangle discrete fallback. Surface and curve projection preserve
boundary features. See Vlachos et al. (2001), Loseille & Lohner (2009), and
NASA refine ref_geom.c for the discrete fallback.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

from .metric import (
    cross_product, dot_product, norm_of_vec, unit_vec,
    NOISE_FLOOR,
)
from .adj import pack_face

if TYPE_CHECKING:
    from .grid import Grid


# ──────────────── Data Structures ──────────────────────────────────

class BezierPatch:
    """
    Cubic triangular Bézier patch (PN-triangle) defined by 10 control
    points b300, b030, b003, b210, b120, b021, b012, b102, b201, b111.

    The patch smoothly interpolates positions at the three corners and
    approximates normals at them.
    """
    __slots__ = (
        'b300', 'b030', 'b003',
        'b210', 'b120', 'b021', 'b012', 'b102', 'b201',
        'b111',
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            setattr(self, slot, [0.0, 0.0, 0.0])

    def evaluate(self, u: float, v: float) -> List[float]:
        """
        Evaluate the patch at barycentric parameters (u, v, w=1-u-v).
        Returns [x, y, z].
        """
        w = 1.0 - u - v
        x = (w * w * w * self.b300[0]
             + u * u * u * self.b030[0]
             + v * v * v * self.b003[0]
             + 3 * w * w * u * self.b210[0]
             + 3 * w * u * u * self.b120[0]
             + 3 * u * u * v * self.b021[0]
             + 3 * u * v * v * self.b012[0]
             + 3 * w * v * v * self.b102[0]
             + 3 * w * w * v * self.b201[0]
             + 6 * w * u * v * self.b111[0])
        y = (w * w * w * self.b300[1]
             + u * u * u * self.b030[1]
             + v * v * v * self.b003[1]
             + 3 * w * w * u * self.b210[1]
             + 3 * w * u * u * self.b120[1]
             + 3 * u * u * v * self.b021[1]
             + 3 * u * v * v * self.b012[1]
             + 3 * w * v * v * self.b102[1]
             + 3 * w * w * v * self.b201[1]
             + 6 * w * u * v * self.b111[1])
        z = (w * w * w * self.b300[2]
             + u * u * u * self.b030[2]
             + v * v * v * self.b003[2]
             + 3 * w * w * u * self.b210[2]
             + 3 * w * u * u * self.b120[2]
             + 3 * u * u * v * self.b021[2]
             + 3 * u * v * v * self.b012[2]
             + 3 * w * v * v * self.b102[2]
             + 3 * w * w * v * self.b201[2]
             + 6 * w * u * v * self.b111[2])
        return [x, y, z]


# ──────────── Vertex Normal Computation ────────────────────────────

def compute_vertex_normals(grid: 'Grid') -> Dict[int, List[float]]:
    """
    Compute area-weighted vertex normals from boundary faces.

    For each boundary triangle, computes the unnormalised cross product
    (whose magnitude is twice the triangle area), and accumulates it at
    each vertex.  The final result is normalised per vertex.

    Returns
    -------
    dict[int, list[float]]
        Mapping from boundary node index to unit normal [nx, ny, nz].
    """
    coords = grid.coordinates
    tets = grid.tetrahedra

    from collections import defaultdict
    face_count: Dict[Tuple[int, ...], int] = defaultdict(int)
    n_tets = len(tets) // 4

    for t in range(n_tets):
        base = 4 * t
        tet = [tets[base + k] for k in range(4)]
        for fi in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]:
            f = tuple(sorted([tet[fi[0]], tet[fi[1]], tet[fi[2]]]))
            face_count[f] += 1

    bdy_faces = [f for f, c in face_count.items() if c == 1]

    normal_acc: Dict[int, List[float]] = {}

    for (a, b, c) in bdy_faces:
        pa = coords[a * 3: a * 3 + 3]
        pb = coords[b * 3: b * 3 + 3]
        pc = coords[c * 3: c * 3 + 3]

        ab = [pb[i] - pa[i] for i in range(3)]
        ac = [pc[i] - pa[i] for i in range(3)]
        n = cross_product(ab, ac)  # magnitude = 2 * area

        for v in (a, b, c):
            if v not in normal_acc:
                normal_acc[v] = [0.0, 0.0, 0.0]
            for k in range(3):
                normal_acc[v][k] += n[k]

    result: Dict[int, List[float]] = {}
    for v, n in normal_acc.items():
        mag = norm_of_vec(n)
        if mag > NOISE_FLOOR:
            result[v] = [n[k] / mag for k in range(3)]
        else:
            result[v] = [0.0, 0.0, 1.0]

    return result


# ──────────── PN-Triangle Patch Construction ───────────────────────

def _w_ij(pi: List[float], pj: List[float],
           ni: List[float]) -> List[float]:
    """
    Compute the control-point displacement for edge (i→j):
        w_ij = dot(pj - pi, ni) * ni   (scalar projection onto normal)
    The interior control point is: b_ij = (2*pi + pj - w_ij) / 3
    """
    dp = dot_product([pj[k] - pi[k] for k in range(3)], ni)
    return [dp * ni[k] for k in range(3)]


def build_bezier_patch(
    p0: List[float], p1: List[float], p2: List[float],
    n0: List[float], n1: List[float], n2: List[float],
) -> BezierPatch:
    """
    Build a cubic triangular Bézier patch (PN-triangle) from three
    vertex positions and normals.

    The corner control points coincide with the input positions.
    Edge control points are computed via normal-based displacement
    (Vlachos et al. 2001).

    Parameters
    ----------
    p0, p1, p2 : list[float]
        Triangle vertex positions [x, y, z].
    n0, n1, n2 : list[float]
        Unit vertex normals [nx, ny, nz].

    Returns
    -------
    BezierPatch
    """
    bp = BezierPatch()

    bp.b300 = list(p0)
    bp.b030 = list(p1)
    bp.b003 = list(p2)

    w01 = _w_ij(p0, p1, n0)
    w10 = _w_ij(p1, p0, n1)
    w12 = _w_ij(p1, p2, n1)
    w21 = _w_ij(p2, p1, n2)
    w20 = _w_ij(p2, p0, n2)
    w02 = _w_ij(p0, p2, n0)

    bp.b210 = [(2 * p0[k] + p1[k] - w01[k]) / 3.0 for k in range(3)]
    bp.b120 = [(2 * p1[k] + p0[k] - w10[k]) / 3.0 for k in range(3)]
    bp.b021 = [(2 * p1[k] + p2[k] - w12[k]) / 3.0 for k in range(3)]
    bp.b012 = [(2 * p2[k] + p1[k] - w21[k]) / 3.0 for k in range(3)]
    bp.b102 = [(2 * p2[k] + p0[k] - w20[k]) / 3.0 for k in range(3)]
    bp.b201 = [(2 * p0[k] + p2[k] - w02[k]) / 3.0 for k in range(3)]

    E = [(bp.b210[k] + bp.b120[k] + bp.b021[k]
          + bp.b012[k] + bp.b102[k] + bp.b201[k]) / 6.0
         for k in range(3)]
    V = [(p0[k] + p1[k] + p2[k]) / 3.0 for k in range(3)]
    bp.b111 = [E[k] + 0.5 * (E[k] - V[k]) for k in range(3)]

    return bp


# ──────────── Closest Point on Triangle ────────────────────────────

def _closest_point_on_triangle(
    point: List[float],
    a: List[float], b: List[float], c: List[float],
) -> Tuple[List[float], float, float]:
    """
    Find the closest point on triangle (a, b, c) to *point*.
    Returns (closest_pt, u, v) where u,v are barycentric coords
    such that closest = (1-u-v)*a + u*b + v*c.
    """
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    ap = [point[i] - a[i] for i in range(3)]

    d1 = dot_product(ab, ap)
    d2 = dot_product(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return list(a), 0.0, 0.0

    bp_ = [point[i] - b[i] for i in range(3)]
    d3 = dot_product(ab, bp_)
    d4 = dot_product(ac, bp_)
    if d3 >= 0.0 and d4 <= d3:
        return list(b), 1.0, 0.0

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v_ = d1 / (d1 - d3) if abs(d1 - d3) > NOISE_FLOOR else 0.0
        return [a[i] + v_ * ab[i] for i in range(3)], v_, 0.0

    cp_ = [point[i] - c[i] for i in range(3)]
    d5 = dot_product(ab, cp_)
    d6 = dot_product(ac, cp_)
    if d6 >= 0.0 and d5 <= d6:
        return list(c), 0.0, 1.0

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w_ = d2 / (d2 - d6) if abs(d2 - d6) > NOISE_FLOOR else 0.0
        return [a[i] + w_ * ac[i] for i in range(3)], 0.0, w_

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w_ = (d4 - d3) / ((d4 - d3) + (d5 - d6)) if abs((d4 - d3) + (d5 - d6)) > NOISE_FLOOR else 0.0
        bc = [c[i] - b[i] for i in range(3)]
        return [b[i] + w_ * bc[i] for i in range(3)], 1.0 - w_, w_

    denom = 1.0 / (va + vb + vc) if abs(va + vb + vc) > NOISE_FLOOR else 1.0
    v_ = vb * denom
    w_ = vc * denom
    return [a[i] + v_ * ab[i] + w_ * ac[i] for i in range(3)], v_, w_


# ──────────── Analytical Projection Helpers ─────────────────────────

def project_to_plane(
    point: List[float], origin: List[float], normal: List[float],
) -> List[float]:
    """
    Project *point* onto the plane defined by *origin* and *normal*.

    Formula:  p' = p − ((p − o) · n̂) · n̂
    """
    n_mag = norm_of_vec(normal)
    if n_mag < NOISE_FLOOR:
        return list(point)
    n_hat = [normal[k] / n_mag for k in range(3)]
    dp = [point[k] - origin[k] for k in range(3)]
    dist = dot_product(dp, n_hat)
    return [point[k] - dist * n_hat[k] for k in range(3)]


def _project_to_sphere(
    point: List[float], center: List[float], radius: float,
) -> List[float]:
    """
    Project *point* radially onto the sphere (center, radius).

    Formula:  p' = c + R · (p − c) / |p − c|
    """
    dp = [point[k] - center[k] for k in range(3)]
    d = norm_of_vec(dp)
    if d < NOISE_FLOOR:
        return [center[0] + radius, center[1], center[2]]
    return [center[k] + radius * dp[k] / d for k in range(3)]


# ──────────── Surface Projection ───────────────────────────────────

def project_to_surface(
    point: List[float], grid: 'Grid', surfid: int,
) -> List[float]:
    """
    Project *point* onto the boundary surface identified by *surfid*.

    Strategy (priority order):
        1. EGADS: if an EGADS model is loaded and *surfid* is mapped,
           project via EG_invEvaluate (exact B-rep).
        2. Analytical: if solidData contains a plane/sphere definition
           for *surfid*, project analytically.
        3. Discrete: build a local PN-triangle Bézier patch from the
           nearest boundary triangle and project onto it.

    Returns
    -------
    list[float]
        Projected [x, y, z] coordinates.
    """
    # ── 0. Try EGADS B-rep projection ────────────────────────────
    _egads_model = getattr(grid, 'egadsModel', None)
    _topo_map = getattr(grid, 'topoMap', None)
    if _egads_model is not None and _topo_map is not None:
        from .egads_adapter import project_to_surface_egads
        result = project_to_surface_egads(point, _egads_model, _topo_map, surfid)
        if result is not None:
            return result

    # ── 1. Try analytical projection ──────────────────────────────
    if grid.solidData and 'surfaceList' in grid.solidData:
        for surf in grid.solidData['surfaceList']:
            if surf['surfId'] == surfid:
                if surf['type'] == 'plane':
                    return project_to_plane(
                        point, surf['origin'], surf['normal'])
                elif surf['type'] == 'sphere':
                    return _project_to_sphere(
                        point, surf['center'], surf['radius'])

    # ── 2. Discrete PN-triangle fallback ──────────────────────────
    coords = grid.coordinates

    from collections import defaultdict
    tets = grid.tetrahedra
    face_count: Dict[Tuple[int, ...], int] = defaultdict(int)
    n_tets = len(tets) // 4
    for t in range(n_tets):
        base = 4 * t
        tet = [tets[base + k] for k in range(4)]
        if tet[0] == -1:
            continue
        for fi in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]:
            f = tuple(sorted([tet[fi[0]], tet[fi[1]], tet[fi[2]]]))
            face_count[f] += 1

    bdy_faces = [f for f, c in face_count.items() if c == 1]

    # Filter to faces where all vertices belong to surfid
    surf_faces: List[Tuple[int, ...]] = []
    for f in bdy_faces:
        # Exact per-face tag decides membership when the face is mapped.
        _fs = grid.faceSurfMap.get(pack_face(*f)) if hasattr(grid, 'faceSurfMap') else None
        if _fs is not None:
            if _fs == surfid:
                surf_faces.append(f)
            continue
        on_surf = True
        for v in f:
            if v >= len(grid.pointLevels):
                on_surf = False
                break
            lvl = grid.pointLevels[v]
            if lvl == 1:
                if grid.bdyFaceMap.get(v, -1) != surfid:
                    on_surf = False
                    break
            elif lvl >= 2:
                pass  # ridge/corner may be on any adjacent surface
            else:
                on_surf = False
                break
        if on_surf:
            surf_faces.append(f)

    if not surf_faces:
        surf_faces = bdy_faces

    # Find nearest face
    best_dist = float('inf')
    best_uv = (0.0, 0.0)
    best_face = surf_faces[0] if surf_faces else (0, 1, 2)

    for f in surf_faces:
        pa = coords[f[0] * 3: f[0] * 3 + 3]
        pb = coords[f[1] * 3: f[1] * 3 + 3]
        pc = coords[f[2] * 3: f[2] * 3 + 3]
        cp, u, v = _closest_point_on_triangle(point, pa, pb, pc)
        d = sum((point[i] - cp[i]) ** 2 for i in range(3))
        if d < best_dist:
            best_dist = d
            best_uv = (u, v)
            best_face = f

    # Build Bézier patch on the nearest face — use cached normals
    cache = grid._geom_cache
    ver = grid._mesh_version
    if cache.get('_version') != ver:
        cache['normals'] = compute_vertex_normals(grid)
        cache['_version'] = ver
    v_normals = cache['normals']
    pa = coords[best_face[0] * 3: best_face[0] * 3 + 3]
    pb = coords[best_face[1] * 3: best_face[1] * 3 + 3]
    pc = coords[best_face[2] * 3: best_face[2] * 3 + 3]
    na = v_normals.get(best_face[0], [0.0, 0.0, 1.0])
    nb = v_normals.get(best_face[1], [0.0, 0.0, 1.0])
    nc = v_normals.get(best_face[2], [0.0, 0.0, 1.0])

    patch = build_bezier_patch(pa, pb, pc, na, nb, nc)
    u, v = best_uv
    return patch.evaluate(u, v)


# ──────────── Ridge / Curve Projection ─────────────────────────────

def project_to_curve(
    point: List[float], grid: 'Grid', ednr: int,
) -> List[float]:
    """
    Snap *point* to the ridge curve identified by *ednr*.

    Strategy (priority order):
        1. EGADS: if an EGADS model is loaded and *ednr* is mapped,
           project via EG_invEvaluate (exact parametric curve).
        2. Discrete: quadratic Bézier interpolation along the nearest
           ridge edge.

    Returns
    -------
    list[float]
        Projected [x, y, z] coordinates.
    """
    # ── 0. Try EGADS B-rep curve projection ──────────────────────
    _egads_model = getattr(grid, 'egadsModel', None)
    _topo_map = getattr(grid, 'topoMap', None)
    if _egads_model is not None and _topo_map is not None:
        from .egads_adapter import project_to_curve_egads
        result = project_to_curve_egads(point, _egads_model, _topo_map, ednr)
        if result is not None:
            return result

    coords = grid.coordinates

    edge_set = {p for p, _ in grid.bdyEdgePoints}
    corner_set = {p for p, _ in grid.cornerPoints}
    ridge_set = edge_set | corner_set

    nodes_on_ednr = set()
    for p, e in grid.bdyEdgeMap.items():
        if e == ednr:
            nodes_on_ednr.add(p)

    for p, ednrs in grid.cornerPoints:
        if ednr in ednrs:
            nodes_on_ednr.add(p)

    ridge_edges: List[Tuple[int, int]] = []
    if hasattr(grid, 'ball_of'):
        # Local construction: every qualifying edge has an endpoint on the
        # curve, so it lies in that endpoint's ball.  Ordered by the first
        # (tet, pair) that contains it, matching collect_edges order.
        tets = grid.tetrahedra
        n_tets = len(tets) // 4
        first: dict = {}
        for p in nodes_on_ednr:
            for t in grid.ball_of(p):
                t = int(t)
                if t >= n_tets or int(tets[4 * t]) == -1:
                    continue
                v4 = [int(tets[4 * t + m]) for m in range(4)]
                k = 0
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = v4[i], v4[j]
                        if ((a in nodes_on_ednr or b in nodes_on_ednr)
                                and a in ridge_set and b in ridge_set):
                            key = (a, b) if a < b else (b, a)
                            tk = (t, k)
                            old = first.get(key)
                            if old is None or tk < old:
                                first[key] = tk
                        k += 1
        ridge_edges = sorted(first, key=first.__getitem__)
    else:
        from .cavity import collect_edges
        for (u, v) in collect_edges(grid):
            if u in nodes_on_ednr or v in nodes_on_ednr:
                if u in ridge_set and v in ridge_set:
                    ridge_edges.append((u, v))

    if not ridge_edges:
        return list(point)

    # Find nearest ridge edge
    best_dist = float('inf')
    best_t = 0.5
    best_edge = ridge_edges[0]

    for (u, v) in ridge_edges:
        pu = coords[u * 3: u * 3 + 3]
        pv = coords[v * 3: v * 3 + 3]

        seg = [pv[k] - pu[k] for k in range(3)]
        seg_len_sq = dot_product(seg, seg)
        if seg_len_sq < NOISE_FLOOR:
            continue
        t = dot_product([point[k] - pu[k] for k in range(3)], seg) / seg_len_sq
        t = max(0.0, min(1.0, t))

        proj = [pu[k] + t * seg[k] for k in range(3)]
        d = sum((point[k] - proj[k]) ** 2 for k in range(3))
        if d < best_dist:
            best_dist = d
            best_t = t
            best_edge = (u, v)

    u, v = best_edge
    pu = coords[u * 3: u * 3 + 3]
    pv = coords[v * 3: v * 3 + 3]

    # Quadratic Bézier whose control point is the chord midpoint
    ctrl = [(pu[k] + pv[k]) / 2.0 for k in range(3)]

    t = best_t
    result = [
        (1 - t) ** 2 * pu[k] + 2 * (1 - t) * t * ctrl[k] + t ** 2 * pv[k]
        for k in range(3)
    ]
    return result


# ──────────── Discrete Surface Classifier ──────────────────────────

def _boundary_face_list(grid: 'Grid') -> List[Tuple[int, ...]]:
    """Faces owned by exactly one tet, in scan order.

    Cached on the grid as ``_bdy_faces_cache`` keyed on
    ``(_topo_version, len(tetrahedra))``; grids without ``_topo_version``
    are rescanned on every call.
    """
    topo = getattr(grid, "_topo_version", None)
    key = (topo, len(grid.tetrahedra))
    if topo is not None:
        cached = getattr(grid, "_bdy_faces_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

    import numpy as np
    T = np.asarray(grid.tetrahedra).reshape(-1, 4).astype(np.int64)
    # Row 4*t + fi is local face fi of tet t, in the scan order below.
    faces = np.stack((T[:, [1, 2, 3]], T[:, [0, 2, 3]],
                      T[:, [0, 1, 3]], T[:, [0, 1, 2]]), axis=1).reshape(-1, 3)
    faces.sort(axis=1)
    if len(faces) == 0:
        bdy_faces = []
    else:
        base = int(faces.max()) + 2
        if base ** 3 < 2 ** 63:
            f1 = faces + 1
            keys = (f1[:, 0] * base + f1[:, 1]) * base + f1[:, 2]
            _, first, counts = np.unique(keys, return_index=True,
                                         return_counts=True)
        else:
            _, first, counts = np.unique(faces, axis=0, return_index=True,
                                         return_counts=True)
        first = np.sort(first[counts == 1])
        bdy_faces = list(map(tuple, faces[first].tolist()))
    if topo is not None:
        grid._bdy_faces_cache = (key, bdy_faces)
    return bdy_faces


def classify_point_to_surface(
    x: float, y: float, z: float, grid: 'Grid',
) -> int:
    """
    Classify point (x,y,z) to the nearest boundary surface.

    Strategy: EGADS (if loaded) → discrete nearest-triangle.

    Returns the surfid of the nearest boundary face, or -1 if
    no boundary faces exist.
    """
    # ── 0. Try EGADS classification ──────────────────────────────
    _egads_model = getattr(grid, 'egadsModel', None)
    _topo_map = getattr(grid, 'topoMap', None)
    if _egads_model is not None and _topo_map is not None:
        from .egads_adapter import classify_point_egads
        result = classify_point_egads(x, y, z, _egads_model, _topo_map)
        if result >= 0:
            return result

    coords = grid.coordinates
    point = [x, y, z]

    bdy_faces = _boundary_face_list(grid)
    if not bdy_faces:
        return -1

    best_dist = float('inf')
    best_surfid = -1

    for f in bdy_faces:
        pa = coords[f[0] * 3: f[0] * 3 + 3]
        pb = coords[f[1] * 3: f[1] * 3 + 3]
        pc = coords[f[2] * 3: f[2] * 3 + 3]
        cp, _, _ = _closest_point_on_triangle(point, pa, pb, pc)
        d = sum((point[i] - cp[i]) ** 2 for i in range(3))
        if d < best_dist:
            best_dist = d
            # Prefer the face's own tag; else use the surfaces of its vertices.
            _fs = grid.faceSurfMap.get(pack_face(*f)) if hasattr(grid, 'faceSurfMap') else None
            if _fs is not None and _fs >= 0:
                best_surfid = _fs
            else:
                # Untagged face: candidates are the surfaces its vertices lie on.
                cand = set()
                for v in f:
                    # A node created by the caller may not have a level yet.
                    lv = grid.pointLevels[v] if v < len(grid.pointLevels) else 0
                    if lv == 1:
                        if v in grid.bdyFaceMap:
                            cand.add(grid.bdyFaceMap[v])
                    elif lv >= 2:
                        cand |= grid.cornerPointSurfMap.get(v, set())
                if len(cand) == 1:
                    best_surfid = next(iter(cand))
                elif cand:
                    best_surfid = min(sorted(cand), key=lambda s: sum(
                        (point[i] - project_to_surface(point, grid, s)[i]) ** 2
                        for i in range(3)))

    return best_surfid
