"""Provide Numba kernels equivalent to the Python geometry and operator paths.

Leaf kernels use cache=True and nogil=True. Whole-loop kernels use prange where
their work is independent. Fastmath stays disabled; nested parallel regions are
avoided so a parallel caller does not oversubscribe threads.
"""

import math

import logging

logger = logging.getLogger(__name__)
import numpy as np
from numba import njit, prange

# ──────────────────────── Constants ────────────────────────────────
NOISE_FLOOR = 1e-15
TAYLOR_THRESHOLD = 1e-6
QUALITY_CONSTANT = 36.0 / (3.0 ** (1.0 / 3.0))

# Edge/face index tables for tet intersection tests (module-level → compile-time)
# 6 edges of a tet: vertex pairs
_EDGE_I = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
_EDGE_J = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
# 4 faces of a tet: vertex triples
_FACE_I = np.array([1, 0, 0, 0], dtype=np.int32)
_FACE_J = np.array([2, 2, 1, 1], dtype=np.int32)
_FACE_K = np.array([3, 3, 3, 2], dtype=np.int32)


# ──────────────────────── Leaf kernels ─────────────────────────────

@njit(cache=True, nogil=True)
def _signed_volume(a, b, c, d):
    """Signed volume of tet (a,b,c,d). a,b,c,d: float64 array views of length 3."""
    Ki = ((a[0] - d[0]) * ((b[1] - d[1]) * (c[2] - d[2])
                            - (b[2] - d[2]) * (c[1] - d[1]))
          - (b[0] - d[0]) * ((a[1] - d[1]) * (c[2] - d[2])
                              - (a[2] - d[2]) * (c[1] - d[1]))
          + (c[0] - d[0]) * ((a[1] - d[1]) * (b[2] - d[2])
                              - (a[2] - d[2]) * (b[1] - d[1])))
    return Ki / 6.0


@njit(cache=True, nogil=True)
def _gamma_m_gamma(gamma, metric):
    """sqrt(gamma^T M gamma). gamma: float64[3], metric: float64[6] packed symmetric."""
    gx, gy, gz = gamma[0], gamma[1], gamma[2]
    val = (metric[0] * gx * gx
           + 2.0 * metric[1] * gx * gy
           + 2.0 * metric[2] * gx * gz
           + metric[3] * gy * gy
           + 2.0 * metric[4] * gy * gz
           + metric[5] * gz * gz)
    if val < NOISE_FLOOR:
        val = 0.0
    return math.sqrt(val)


@njit(cache=True, nogil=True)
def _gamma_m_gamma_scalar(gx, gy, gz, m0, m1, m2, m3, m4, m5):
    """Scalar version of gamma_m_gamma — no array allocation."""
    val = (m0 * gx * gx
           + 2.0 * m1 * gx * gy
           + 2.0 * m2 * gx * gz
           + m3 * gy * gy
           + 2.0 * m4 * gy * gz
           + m5 * gz * gz)
    if val < NOISE_FLOOR:
        val = 0.0
    return math.sqrt(val)


@njit(cache=True, nogil=True)
def _det_metric(m):
    """Determinant of 3x3 symmetric metric stored as 6 floats."""
    return (m[0] * m[3] * m[5]
            - m[0] * m[4] * m[4]
            - m[1] * m[1] * m[5]
            + 2.0 * m[1] * m[2] * m[4]
            - m[2] * m[2] * m[3])


@njit(cache=True, nogil=True)
def _segment_triangle_intersect(p0, p1, a, b, c, epsilon):
    """Moller-Trumbore segment-triangle test. All args float64[3] except epsilon (float)."""
    ex = p1[0] - p0[0]; ey = p1[1] - p0[1]; ez = p1[2] - p0[2]
    e1x = b[0] - a[0]; e1y = b[1] - a[1]; e1z = b[2] - a[2]
    e2x = c[0] - a[0]; e2y = c[1] - a[1]; e2z = c[2] - a[2]
    tx = p0[0] - a[0]; ty = p0[1] - a[1]; tz = p0[2] - a[2]

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


@njit(cache=True, nogil=True)
def _point_in_tet(p, a, b, c, d, epsilon):
    """Check if point p is inside tet (a,b,c,d). All float64[3]."""
    v_total = _signed_volume(a, b, c, d)
    if abs(v_total) < 1e-12:
        return False

    v0 = _signed_volume(p, b, c, d)
    v1 = _signed_volume(a, p, c, d)
    v2 = _signed_volume(a, b, p, d)
    v3 = _signed_volume(a, b, c, p)

    sign = 1.0 if v_total > 0.0 else -1.0
    if v0 * sign < epsilon:
        return False
    if v1 * sign < epsilon:
        return False
    if v2 * sign < epsilon:
        return False
    if v3 * sign < epsilon:
        return False
    return True


# ──────────────────────── Metric kernels ───────────────────────────

@njit(cache=True, nogil=True)
def _length_in_metric(m1, m2, p1, p2):
    """Continuous metric edge length. m1,m2: float64[6], p1,p2: float64[3]."""
    if p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]:
        return 0.0

    gx = p2[0] - p1[0]
    gy = p2[1] - p1[1]
    gz = p2[2] - p1[2]

    l_a = _gamma_m_gamma_scalar(gx, gy, gz,
                                m1[0], m1[1], m1[2], m1[3], m1[4], m1[5])
    l_b = _gamma_m_gamma_scalar(gx, gy, gz,
                                m2[0], m2[1], m2[2], m2[3], m2[4], m2[5])

    l1 = l_a if l_a > l_b else l_b  # max
    l2 = l_b if l_a > l_b else l_a  # min

    if l2 < NOISE_FLOOR:
        if l1 < 1e-6:
            return 0.0
        return l1

    a = l1 / l2

    # Taylor expansion near alpha ~ 1 (numerical stability, NOT optimization)
    if abs(a - 1.0) < TAYLOR_THRESHOLD:
        return l1 * (2.0 / 3.0) * (a * a + a + 1.0) / (a * (a + 1.0))

    # Explicit branch replaces try/except (ZeroDivisionError, ValueError)
    log_a = math.log(a)
    denom = a * log_a
    if abs(denom) < 1e-30:
        L = 0.5 * (l_a + l_b)
    else:
        L = l1 * (a - 1.0) / denom

    EPSILON = 1e-12
    return L if L > EPSILON else 0.0


@njit(cache=True, nogil=True)
def _tet_quality(a, b, c, d, metrics, dets):
    """Knupp algebraic tet quality. a,b,c,d: float64[3]. metrics: float64[4,6]. dets: float64[4]."""
    sv = _signed_volume(a, b, c, d)
    V = abs(sv)
    if V <= 0.0:
        return 0.0

    min_det = dets[0]
    for i in range(1, 4):
        if dets[i] < min_det:
            min_det = dets[i]
    if min_det <= 0.0:
        return 0.0

    V0 = math.sqrt(min_det) * V

    L01 = _length_in_metric(metrics[0], metrics[1], a, b)
    L02 = _length_in_metric(metrics[0], metrics[2], a, c)
    L03 = _length_in_metric(metrics[0], metrics[3], a, d)
    L12 = _length_in_metric(metrics[1], metrics[2], b, c)
    L13 = _length_in_metric(metrics[1], metrics[3], b, d)
    L23 = _length_in_metric(metrics[2], metrics[3], c, d)

    sum_sq = (L01*L01 + L02*L02 + L03*L03
              + L12*L12 + L13*L13 + L23*L23)
    if sum_sq < NOISE_FLOOR:
        return 0.0

    return QUALITY_CONSTANT * (V0 ** (2.0 / 3.0)) / sum_sq


# ──────────────────────── Whole-loop kernels ───────────────────────

@njit(cache=True, nogil=True)
def check_local_intersection_jit(
    coords,          # float64[:] flat, 3*Np
    tets,            # int32[:] flat, 4*Nt
    ball_ptr,        # int32[:] CSR pointer, Np+1
    ball_idx,        # int32[:] CSR indices
    dead_mask,       # bool[:] Np
    tets_to_check,   # int32[:,:] shape (M, 4) — proposed tets
    override_node,   # int32 — node with overridden coords (-1 if none)
    override_xyz,    # float64[:] shape (3,) — coords for override_node
    ignored_mask,    # bool[:] shape Nt — True = skip this tet
):
    """
    Check if any tet in tets_to_check intersects any non-ignored grid tet.
    Returns True on first intersection found.
    """
    n_check = tets_to_check.shape[0]
    n_total_tets = len(tets) // 4
    n_nodes = len(dead_mask)

    # 1. Gather unique nodes from tets_to_check (small set, O(M^2) dedup OK)
    unique_nodes = np.empty(n_check * 4, dtype=np.int32)
    n_unique = 0
    for i in range(n_check):
        for j in range(4):
            v = tets_to_check[i, j]
            if v < 0:
                continue
            found = False
            for k in range(n_unique):
                if unique_nodes[k] == v:
                    found = True
                    break
            if not found:
                unique_nodes[n_unique] = v
                n_unique += 1

    # 2. Gather candidate tets from balls of those nodes (deduped)
    cap = 0
    for ui in range(n_unique):
        v = unique_nodes[ui]
        if v < 0 or v >= n_nodes:
            continue
        cap += ball_ptr[v + 1] - ball_ptr[v]
    cand_list = np.empty(cap, dtype=np.int32)
    n_cand = 0

    for ui in range(n_unique):
        v = unique_nodes[ui]
        if v < 0 or v >= n_nodes:
            continue
        for bi in range(ball_ptr[v], ball_ptr[v + 1]):
            ti = ball_idx[bi]
            if ti < 0 or ti >= n_total_tets:
                continue
            dup = False
            for ci in range(n_cand):
                if cand_list[ci] == ti:
                    dup = True
                    break
            if dup:
                continue
            if tets[ti * 4] == -1:
                continue
            if ignored_mask[ti]:
                continue
            cand_list[n_cand] = ti
            n_cand += 1

    if n_cand == 0:
        return False

    # 3. Pre-compute coordinates for tets_to_check, respecting override
    #    Also compute union AABB and mark skippable tets
    pts_check = np.empty((n_check, 4, 3), dtype=np.float64)
    skip_check = np.zeros(n_check, dtype=np.bool_)
    u_min_x = np.inf;  u_max_x = -np.inf
    u_min_y = np.inf;  u_max_y = -np.inf
    u_min_z = np.inf;  u_max_z = -np.inf

    for i in range(n_check):
        do_skip = False
        for j in range(4):
            v = tets_to_check[i, j]
            if v == override_node:
                pts_check[i, j, 0] = override_xyz[0]
                pts_check[i, j, 1] = override_xyz[1]
                pts_check[i, j, 2] = override_xyz[2]
            elif v >= 0 and v < n_nodes and not dead_mask[v]:
                pts_check[i, j, 0] = coords[v * 3]
                pts_check[i, j, 1] = coords[v * 3 + 1]
                pts_check[i, j, 2] = coords[v * 3 + 2]
            else:
                do_skip = True
                break
        if do_skip:
            skip_check[i] = True
            continue
        for j in range(4):
            px = pts_check[i, j, 0]
            py = pts_check[i, j, 1]
            pz = pts_check[i, j, 2]
            if px < u_min_x: u_min_x = px
            if px > u_max_x: u_max_x = px
            if py < u_min_y: u_min_y = py
            if py > u_max_y: u_max_y = py
            if pz < u_min_z: u_min_z = pz
            if pz > u_max_z: u_max_z = pz

    # 4. Filter candidates against union AABB, store filtered indices + AABBs
    filt_idx = np.empty(n_cand, dtype=np.int32)
    filt_aabb = np.empty((n_cand, 6), dtype=np.float64)  # min_x,max_x,min_y,max_y,min_z,max_z
    filt_verts = np.empty((n_cand, 4), dtype=np.int32)
    n_filt = 0

    for ci in range(n_cand):
        ti = cand_list[ci]
        base = ti * 4
        v0 = tets[base]; v1 = tets[base+1]; v2 = tets[base+2]; v3 = tets[base+3]

        if (v0 < 0 or v0 >= n_nodes or v1 < 0 or v1 >= n_nodes or
            v2 < 0 or v2 >= n_nodes or v3 < 0 or v3 >= n_nodes):
            continue
        if dead_mask[v0] or dead_mask[v1] or dead_mask[v2] or dead_mask[v3]:
            continue

        x0 = coords[v0*3]; y0 = coords[v0*3+1]; z0 = coords[v0*3+2]
        x1 = coords[v1*3]; y1 = coords[v1*3+1]; z1 = coords[v1*3+2]
        x2 = coords[v2*3]; y2 = coords[v2*3+1]; z2 = coords[v2*3+2]
        x3 = coords[v3*3]; y3 = coords[v3*3+1]; z3 = coords[v3*3+2]

        # Candidate AABB
        min_x = min(x0, x1, x2, x3)
        max_x = max(x0, x1, x2, x3)
        min_y = min(y0, y1, y2, y3)
        max_y = max(y0, y1, y2, y3)
        min_z = min(z0, z1, z2, z3)
        max_z = max(z0, z1, z2, z3)

        # Overlap with union AABB?
        if (min_x > u_max_x or max_x < u_min_x or
            min_y > u_max_y or max_y < u_min_y or
            min_z > u_max_z or max_z < u_min_z):
            continue

        filt_idx[n_filt] = ti
        filt_aabb[n_filt, 0] = min_x; filt_aabb[n_filt, 1] = max_x
        filt_aabb[n_filt, 2] = min_y; filt_aabb[n_filt, 3] = max_y
        filt_aabb[n_filt, 4] = min_z; filt_aabb[n_filt, 5] = max_z
        filt_verts[n_filt, 0] = v0; filt_verts[n_filt, 1] = v1
        filt_verts[n_filt, 2] = v2; filt_verts[n_filt, 3] = v3
        n_filt += 1

    if n_filt == 0:
        return False

    # 5. Detailed pair-wise intersection checks
    eps = 1e-7

    # Pre-compute per-check-tet AABBs and node IDs (serial, small)
    check_aabb = np.empty((n_check, 6), dtype=np.float64)
    check_nodes = np.empty((n_check, 4), dtype=np.int32)
    active_checks = np.empty(n_check, dtype=np.int32)
    n_active = 0
    for idx1 in range(n_check):
        if skip_check[idx1]:
            continue
        active_checks[n_active] = idx1
        n_active += 1
        min1_x = pts_check[idx1, 0, 0]; max1_x = min1_x
        min1_y = pts_check[idx1, 0, 1]; max1_y = min1_y
        min1_z = pts_check[idx1, 0, 2]; max1_z = min1_z
        for j in range(1, 4):
            px = pts_check[idx1, j, 0]
            py = pts_check[idx1, j, 1]
            pz = pts_check[idx1, j, 2]
            if px < min1_x: min1_x = px
            if px > max1_x: max1_x = px
            if py < min1_y: min1_y = py
            if py > max1_y: max1_y = py
            if pz < min1_z: min1_z = pz
            if pz > max1_z: max1_z = pz
        check_aabb[idx1, 0] = min1_x; check_aabb[idx1, 1] = max1_x
        check_aabb[idx1, 2] = min1_y; check_aabb[idx1, 3] = max1_y
        check_aabb[idx1, 4] = min1_z; check_aabb[idx1, 5] = max1_z
        check_nodes[idx1, 0] = tets_to_check[idx1, 0]
        check_nodes[idx1, 1] = tets_to_check[idx1, 1]
        check_nodes[idx1, 2] = tets_to_check[idx1, 2]
        check_nodes[idx1, 3] = tets_to_check[idx1, 3]

    if n_active == 0:
        return False

    # Flatten (active_check, candidate) pairs
    n_pairs = n_active * n_filt
    found = np.zeros(1, dtype=np.bool_)

    for pair in range(n_pairs):
        if found[0]:
            continue
        ai = pair // n_filt
        ci = pair % n_filt
        idx1 = active_checks[ai]

        # AABB overlap check
        if (check_aabb[idx1, 0] > filt_aabb[ci, 1] or check_aabb[idx1, 1] < filt_aabb[ci, 0] or
            check_aabb[idx1, 2] > filt_aabb[ci, 3] or check_aabb[idx1, 3] < filt_aabb[ci, 2] or
            check_aabb[idx1, 4] > filt_aabb[ci, 5] or check_aabb[idx1, 5] < filt_aabb[ci, 4]):
            continue

        # Count shared vertices
        t1n0 = check_nodes[idx1, 0]; t1n1 = check_nodes[idx1, 1]
        t1n2 = check_nodes[idx1, 2]; t1n3 = check_nodes[idx1, 3]
        t2n0 = filt_verts[ci, 0]; t2n1 = filt_verts[ci, 1]
        t2n2 = filt_verts[ci, 2]; t2n3 = filt_verts[ci, 3]

        shared = 0
        if (t1n0 == t2n0 or t1n0 == t2n1 or t1n0 == t2n2 or t1n0 == t2n3):
            shared += 1
        if (t1n1 == t2n0 or t1n1 == t2n1 or t1n1 == t2n2 or t1n1 == t2n3):
            shared += 1
        if (t1n2 == t2n0 or t1n2 == t2n1 or t1n2 == t2n2 or t1n2 == t2n3):
            shared += 1
        if (t1n3 == t2n0 or t1n3 == t2n1 or t1n3 == t2n2 or t1n3 == t2n3):
            shared += 1

        if shared == 4:
            found[0] = True
            continue
        if shared >= 1:
            continue

        # Get tet2 coordinates
        p2 = np.empty((4, 3), dtype=np.float64)
        p2[0, 0] = coords[t2n0*3]; p2[0, 1] = coords[t2n0*3+1]; p2[0, 2] = coords[t2n0*3+2]
        p2[1, 0] = coords[t2n1*3]; p2[1, 1] = coords[t2n1*3+1]; p2[1, 2] = coords[t2n1*3+2]
        p2[2, 0] = coords[t2n2*3]; p2[2, 1] = coords[t2n2*3+1]; p2[2, 2] = coords[t2n2*3+2]
        p2[3, 0] = coords[t2n3*3]; p2[3, 1] = coords[t2n3*3+1]; p2[3, 2] = coords[t2n3*3+2]

        # Edge-face intersection tests: tet1 edges vs tet2 faces
        pair_found = False
        for ei in range(6):
            if pair_found:
                break
            ep0 = pts_check[idx1, _EDGE_I[ei]]
            ep1 = pts_check[idx1, _EDGE_J[ei]]
            for fi in range(4):
                fa = p2[_FACE_I[fi]]
                fb = p2[_FACE_J[fi]]
                fc = p2[_FACE_K[fi]]
                if _segment_triangle_intersect(ep0, ep1, fa, fb, fc, eps):
                    pair_found = True
                    break

        if not pair_found:
            # Edge-face intersection tests: tet2 edges vs tet1 faces
            for ei in range(6):
                if pair_found:
                    break
                ep0 = p2[_EDGE_I[ei]]
                ep1 = p2[_EDGE_J[ei]]
                for fi in range(4):
                    fa = pts_check[idx1, _FACE_I[fi]]
                    fb = pts_check[idx1, _FACE_J[fi]]
                    fc = pts_check[idx1, _FACE_K[fi]]
                    if _segment_triangle_intersect(ep0, ep1, fa, fb, fc, eps):
                        pair_found = True
                        break

        if not pair_found:
            # Centroid containment: tet1 centroid in tet2?
            c1 = np.empty(3, dtype=np.float64)
            c1[0] = (pts_check[idx1,0,0] + pts_check[idx1,1,0] + pts_check[idx1,2,0] + pts_check[idx1,3,0]) * 0.25
            c1[1] = (pts_check[idx1,0,1] + pts_check[idx1,1,1] + pts_check[idx1,2,1] + pts_check[idx1,3,1]) * 0.25
            c1[2] = (pts_check[idx1,0,2] + pts_check[idx1,1,2] + pts_check[idx1,2,2] + pts_check[idx1,3,2]) * 0.25
            if _point_in_tet(c1, p2[0], p2[1], p2[2], p2[3], eps):
                pair_found = True

        if not pair_found:
            # Centroid containment: tet2 centroid in tet1?
            c2 = np.empty(3, dtype=np.float64)
            c2[0] = (p2[0,0] + p2[1,0] + p2[2,0] + p2[3,0]) * 0.25
            c2[1] = (p2[0,1] + p2[1,1] + p2[2,1] + p2[3,1]) * 0.25
            c2[2] = (p2[0,2] + p2[1,2] + p2[2,2] + p2[3,2]) * 0.25
            if _point_in_tet(c2, pts_check[idx1,0], pts_check[idx1,1], pts_check[idx1,2], pts_check[idx1,3], eps):
                pair_found = True

        if pair_found:
            found[0] = True

    return found[0]


# ──────────────────────── Candidate scan kernels ───────────────────

@njit(cache=True, nogil=True, parallel=True)
def collapse_candidate_scan_jit(coords, mm, dead_mask, edges, n_points, threshold):
    """
    Scan all edges, compute metric lengths, identify short ones for collapse.

    coords: float64[:] flat (3*Np)
    mm: float64[:] flat (6*Np)
    dead_mask: bool[:] (Np)
    edges: int32[:,:] shape (Ne, 2)
    n_points: int
    threshold: float64

    Returns (removal_flags, min_node_length, short_edges, short_lengths, n_short).
    Caller must use short_edges[:n_short] and short_lengths[:n_short].
    """
    n_edges = edges.shape[0]
    is_short = np.zeros(n_edges, dtype=np.bool_)
    edge_lengths = np.empty(n_edges, dtype=np.float64)

    for i in prange(n_edges):
        p1 = edges[i, 0]
        p2 = edges[i, 1]
        if p1 >= n_points or p2 >= n_points:
            continue
        if dead_mask[p1] or dead_mask[p2]:
            continue
        m1 = mm[6*p1:6*p1+6]
        m2 = mm[6*p2:6*p2+6]
        if m1[0] == -1.0 or m2[0] == -1.0:
            continue
        s = coords[3*p1:3*p1+3]
        e = coords[3*p2:3*p2+3]
        L = _length_in_metric(m1, m2, s, e)
        if L < threshold:
            is_short[i] = True
            edge_lengths[i] = L

    removal_flags = np.zeros(n_points, dtype=np.int32)
    min_node_length = np.full(n_points, 2.0)
    short_edges = np.empty(n_edges, dtype=np.int32)
    short_lengths = np.empty(n_edges, dtype=np.float64)
    n_short = 0

    for i in range(n_edges):
        if is_short[i]:
            short_edges[n_short] = i
            short_lengths[n_short] = edge_lengths[i]
            n_short += 1
            p1 = edges[i, 0]
            p2 = edges[i, 1]
            removal_flags[p1] = 1
            removal_flags[p2] = 1
            L = edge_lengths[i]
            if L < min_node_length[p1]:
                min_node_length[p1] = L
            if L < min_node_length[p2]:
                min_node_length[p2] = L

    return removal_flags, min_node_length, short_edges, short_lengths, n_short


@njit(cache=True, nogil=True, parallel=True)
def split_candidate_scan_jit(coords, mm, dead_mask, edges, n_points, threshold):
    """
    Scan all edges, identify long ones for splitting.

    Returns (long_edges, long_lengths, n_long).
    Caller must use long_edges[:n_long] and long_lengths[:n_long].
    """
    n_edges = edges.shape[0]
    is_long = np.zeros(n_edges, dtype=np.bool_)
    edge_lengths = np.empty(n_edges, dtype=np.float64)

    for i in prange(n_edges):
        p1 = edges[i, 0]
        p2 = edges[i, 1]
        if p1 >= n_points or p2 >= n_points:
            continue
        if dead_mask[p1] or dead_mask[p2]:
            continue
        m1 = mm[6*p1:6*p1+6]
        m2 = mm[6*p2:6*p2+6]
        s = coords[3*p1:3*p1+3]
        e = coords[3*p2:3*p2+3]
        L = _length_in_metric(m1, m2, s, e)
        if L > threshold:
            is_long[i] = True
            edge_lengths[i] = L

    long_edges = np.empty(n_edges, dtype=np.int32)
    long_lengths = np.empty(n_edges, dtype=np.float64)
    n_long = 0

    for i in range(n_edges):
        if is_long[i]:
            long_edges[n_long] = i
            long_lengths[n_long] = edge_lengths[i]
            n_long += 1

    return long_edges, long_lengths, n_long


@njit(cache=True, nogil=True, parallel=True)
def compute_edge_active_jit(coords, mm, dead_mask, edges, n_points):
    """Mark edges whose metric length is outside [1/sqrt(2), sqrt(2)].

    Returns (active_mask, n_active) where active_mask[i] is True when
    edge i is non-conforming.
    """
    lo = 0.7071067811865476   # 1/sqrt(2)
    hi = 1.4142135623730951   # sqrt(2)
    n_edges = edges.shape[0]
    active = np.zeros(n_edges, dtype=np.bool_)

    for i in prange(n_edges):
        p1 = edges[i, 0]
        p2 = edges[i, 1]
        if p1 >= n_points or p2 >= n_points:
            active[i] = True
            continue
        if dead_mask[p1] or dead_mask[p2]:
            continue
        m1 = mm[6*p1:6*p1+6]
        m2 = mm[6*p2:6*p2+6]
        if m1[0] == -1.0 or m2[0] == -1.0:
            active[i] = True
            continue
        s = coords[3*p1:3*p1+3]
        e = coords[3*p2:3*p2+3]
        L = _length_in_metric(m1, m2, s, e)
        if L < lo or L > hi:
            active[i] = True

    n_active = 0
    for i in range(n_edges):
        if active[i]:
            n_active += 1

    return active, n_active


@njit(cache=True, nogil=True)
def compact_arrays_jit(coords, metric, metric_log, tets, dead_mask):
    """
    Compact per-node and per-tet arrays, dropping dead nodes/tets.

    Prefix-sum + scatter compaction for Grid.cleanDeadElements (its node and
    tet remap step): builds a dense point_map (old index -> new
    index, -1 for dead nodes), scatters coordinates/metric/metric_log for
    live nodes, and remaps + compacts tetrahedra (dropping all-(-1) dead
    tets).

    Returns (point_map, new_coords, new_metric, new_metric_log, new_tets).
    """
    n_points = dead_mask.shape[0]
    point_map = np.full(n_points, -1, dtype=np.int64)

    n_live = 0
    for i in range(n_points):
        if not dead_mask[i]:
            point_map[i] = n_live
            n_live += 1

    new_coords = np.empty(3 * n_live, dtype=np.float64)
    has_metric = metric.shape[0] > 0
    has_metric_log = metric_log.shape[0] > 0
    new_metric = np.empty(6 * n_live if has_metric else 0, dtype=np.float64)
    new_metric_log = np.empty(6 * n_live if has_metric_log else 0, dtype=np.float64)

    for i in range(n_points):
        j = point_map[i]
        if j == -1:
            continue
        new_coords[3*j:3*j+3] = coords[3*i:3*i+3]
        if has_metric:
            new_metric[6*j:6*j+6] = metric[6*i:6*i+6]
        if has_metric_log:
            new_metric_log[6*j:6*j+6] = metric_log[6*i:6*i+6]

    n_tets_raw = tets.shape[0] // 4
    n_live_tets = 0
    for t in range(n_tets_raw):
        base = 4 * t
        a = tets[base]
        b = tets[base + 1]
        c = tets[base + 2]
        d = tets[base + 3]
        if a == -1 and b == -1 and c == -1 and d == -1:
            continue
        n_live_tets += 1

    new_tets = np.empty(4 * n_live_tets, dtype=np.int32)
    k = 0
    for t in range(n_tets_raw):
        base = 4 * t
        a = tets[base]
        b = tets[base + 1]
        c = tets[base + 2]
        d = tets[base + 3]
        if a == -1 and b == -1 and c == -1 and d == -1:
            continue
        new_tets[4*k] = point_map[a]
        new_tets[4*k + 1] = point_map[b]
        new_tets[4*k + 2] = point_map[c]
        new_tets[4*k + 3] = point_map[d]
        k += 1

    return point_map, new_coords, new_metric, new_metric_log, new_tets


@njit(cache=True, nogil=True)
def build_ball_csr_jit(tets, n_pts):
    """CSR ``(ball_ptr, ball_idx)`` of live tets per node from flat tets.
    A tet with ``tets[4*t] == -1`` is dead and skipped."""
    n_tets = len(tets) // 4
    ball_ptr = np.zeros(n_pts + 1, dtype=np.int32)
    for t in range(n_tets):
        base = 4 * t
        if tets[base] == -1:
            continue
        for k in range(4):
            ball_ptr[tets[base + k] + 1] += 1
    for v in range(n_pts):
        ball_ptr[v + 1] += ball_ptr[v]
    ball_idx = np.empty(ball_ptr[n_pts], dtype=np.int32)
    pos = ball_ptr[:n_pts].copy()
    for t in range(n_tets):
        base = 4 * t
        if tets[base] == -1:
            continue
        for k in range(4):
            v = tets[base + k]
            ball_idx[pos[v]] = t
            pos[v] += 1
    return ball_ptr, ball_idx


@njit(cache=True, nogil=True)
def _bary_coords_tet_jit(p0, p1, p2, p3, point, noise_floor):
    """Barycentric coordinates of *point* in tet (p0..p3); float64[4].
    Exact port of metric.py::bary_coords_tet (same sub-volume argument order)."""
    out = np.empty(4, dtype=np.float64)
    v_full = abs(_signed_volume(p0, p1, p2, p3))
    if v_full < noise_floor:
        out[0] = 0.25
        out[1] = 0.25
        out[2] = 0.25
        out[3] = 0.25
        return out
    out[0] = abs(_signed_volume(p1, p2, p3, point)) / v_full
    out[1] = abs(_signed_volume(p0, p2, p3, point)) / v_full
    out[2] = abs(_signed_volume(p0, p1, p3, point)) / v_full
    out[3] = abs(_signed_volume(p0, p1, p2, point)) / v_full
    for k in range(4):
        if abs(out[k]) < 1e-12:
            out[k] = 0.0
    return out


@njit(cache=True, nogil=True)
def traverse_jit(coords, tets, adj, start_tet, goal, noise_floor):
    """Barycentric ray-walk; returns ``(found, tet)``.
    Exact port of search.py::traverse: exit face is the first index equal to
    the minimum coordinate, sum is left-to-right, cycle test is
    ``neighbour == prev or current visited``."""
    tol = 1e-10
    n_tets = len(tets) // 4
    visited = np.zeros(n_tets, dtype=np.bool_)
    current = start_tet
    prev = -1
    for _ in range(n_tets):
        base = current * 4
        v0 = tets[base]
        v1 = tets[base + 1]
        v2 = tets[base + 2]
        v3 = tets[base + 3]
        bary = _bary_coords_tet_jit(coords[v0 * 3:v0 * 3 + 3],
                                    coords[v1 * 3:v1 * 3 + 3],
                                    coords[v2 * 3:v2 * 3 + 3],
                                    coords[v3 * 3:v3 * 3 + 3],
                                    goal, noise_floor)
        total = ((bary[0] + bary[1]) + bary[2]) + bary[3]
        mn = min(min(bary[0], bary[1]), min(bary[2], bary[3]))
        if abs(total - 1.0) < tol and mn >= -tol:
            return True, current
        j = 0
        for k in range(4):
            if bary[k] == mn:
                j = k
                break
        neighbour = adj[current * 4 + j]
        if neighbour == -1:
            return False, current
        if neighbour == prev or visited[current]:
            return False, current
        visited[current] = True
        prev = current
        current = neighbour
    return False, current


@njit(cache=True, nogil=True)
def brute_force_locate_jit(coords, tets, goal, noise_floor):
    """First tet t with min(bary) >= -tol and |sum(bary) - 1| < tol, else -1.
    Exact port of step 3 of search.py::locate. Like the Python version it does
    NOT skip tets whose vertices are -1 sentinels. There the Python scan fed
    empty coordinate slices to the volume kernel (undefined); here a -1 vertex
    reads wrapped-around coordinates, so results for dead tets are not
    comparable and callers must pass a compacted mesh."""
    tol = 1e-10
    n_tets = len(tets) // 4
    for t in range(n_tets):
        base = t * 4
        v0 = tets[base]
        v1 = tets[base + 1]
        v2 = tets[base + 2]
        v3 = tets[base + 3]
        bary = _bary_coords_tet_jit(coords[v0 * 3:v0 * 3 + 3],
                                    coords[v1 * 3:v1 * 3 + 3],
                                    coords[v2 * 3:v2 * 3 + 3],
                                    coords[v3 * 3:v3 * 3 + 3],
                                    goal, noise_floor)
        total = ((bary[0] + bary[1]) + bary[2]) + bary[3]
        mn = min(min(bary[0], bary[1]), min(bary[2], bary[3]))
        if mn >= -tol and abs(total - 1.0) < tol:
            return t
    return -1


@njit(cache=True, nogil=True)
def ball_min_quality_jit(coords, tets, metrics, dead_mask, node, ball,
                         has_override, override_xyz):
    """Min tet quality over a ball; exact port of smooth._evaluate_quality_arrays.

    Returns 1.0 for an empty ball and -1.0 on a dead node, a dead-metric
    node or a tet with signed volume < 1e-12.
    """
    if len(ball) == 0:
        return 1.0
    min_q = np.inf
    pts = np.empty((4, 3), dtype=np.float64)
    mets = np.empty((4, 6), dtype=np.float64)
    dets = np.empty(4, dtype=np.float64)
    for bi in range(len(ball)):
        base = ball[bi] * 4
        if tets[base] == -1:
            continue
        for j in range(4):
            n = tets[base + j]
            if n == node and has_override:
                for k in range(3):
                    pts[j, k] = override_xyz[k]
            else:
                if dead_mask[n]:
                    return -1.0
                for k in range(3):
                    pts[j, k] = coords[3 * n + k]
            if 6 * n + 6 > len(metrics):
                return -1.0
            if metrics[6 * n] == -1:
                return -1.0
            for k in range(6):
                mets[j, k] = metrics[6 * n + k]
            dets[j] = _det_metric(mets[j])
        sv = _signed_volume(pts[0], pts[1], pts[2], pts[3])
        if sv < 1e-12:
            return -1.0
        q = _tet_quality(pts[0], pts[1], pts[2], pts[3], mets, dets)
        min_q = min(min_q, q)
    return min_q


@njit(cache=True, nogil=True)
def tet_volume_quality_jit(coords, metrics, n0, n1, n2, n3,
                           ovr_node, ovr_xyz, ovr_metric, has_ovr_metric):
    """(signed volume, quality) of tet (n0..n3). No dead checks.

    A vertex equal to ovr_node takes ovr_xyz (and ovr_metric when
    has_ovr_metric). Use ovr_node = -2 for no override.
    """
    pts = np.empty((4, 3), dtype=np.float64)
    mets = np.empty((4, 6), dtype=np.float64)
    dets = np.empty(4, dtype=np.float64)
    for j in range(4):
        if j == 0:
            n = n0
        elif j == 1:
            n = n1
        elif j == 2:
            n = n2
        else:
            n = n3
        if n == ovr_node:
            for k in range(3):
                pts[j, k] = ovr_xyz[k]
        else:
            for k in range(3):
                pts[j, k] = coords[3 * n + k]
        if n == ovr_node and has_ovr_metric:
            for k in range(6):
                mets[j, k] = ovr_metric[k]
        else:
            for k in range(6):
                mets[j, k] = metrics[6 * n + k]
        dets[j] = _det_metric(mets[j])
    sv = _signed_volume(pts[0], pts[1], pts[2], pts[3])
    q = _tet_quality(pts[0], pts[1], pts[2], pts[3], mets, dets)
    return sv, q


@njit(cache=True, nogil=True)
def edge_length_nodes_jit(coords, metrics, a, b):
    """Metric edge length between nodes a and b."""
    return _length_in_metric(metrics[6 * a:6 * a + 6], metrics[6 * b:6 * b + 6],
                             coords[3 * a:3 * a + 3], coords[3 * b:3 * b + 3])


@njit(cache=True, nogil=True)
def _sort4_jit(a, b, c, d, out):
    """Write the ascending order of (a, b, c, d) into out[0:4]."""
    if b < a:
        a, b = b, a
    if d < c:
        c, d = d, c
    if c < a:
        a, c = c, a
    if d < b:
        b, d = d, b
    if c < b:
        b, c = c, b
    out[0] = a
    out[1] = b
    out[2] = c
    out[3] = d


@njit(cache=True, nogil=True)
def duplicate_cell_jit(tets, ball_po, ball_pc, shell, pc, po):
    """Exact port of collapse.check_duplicate_cell: False when substituting po
    for pc in a surviving tet of ball(pc) reproduces a live tet of ball(po),
    or reproduces a face of a live tet of ball(po) outside the shell (which
    would leave an interior face shared by four tets)."""
    n_po = len(ball_po)
    existing = np.empty((n_po, 4), dtype=np.int64)
    n_ex = 0
    row = np.empty(4, dtype=np.int64)
    for i in range(n_po):
        t = ball_po[i]
        if tets[4 * t] == -1:
            continue
        _sort4_jit(tets[4 * t], tets[4 * t + 1], tets[4 * t + 2],
                   tets[4 * t + 3], row)
        for k in range(4):
            existing[n_ex, k] = row[k]
        n_ex += 1

    for i in range(len(ball_pc)):
        t = ball_pc[i]
        in_shell = False
        for j in range(len(shell)):
            if shell[j] == t:
                in_shell = True
                break
        if in_shell:
            continue
        if tets[4 * t] == -1:
            continue
        n0 = np.int64(tets[4 * t])
        n1 = np.int64(tets[4 * t + 1])
        n2 = np.int64(tets[4 * t + 2])
        n3 = np.int64(tets[4 * t + 3])
        if n0 == pc:
            n0 = po
        if n1 == pc:
            n1 = po
        if n2 == pc:
            n2 = po
        if n3 == pc:
            n3 = po
        _sort4_jit(n0, n1, n2, n3, row)
        for e in range(n_ex):
            if (existing[e, 0] == row[0] and existing[e, 1] == row[1]
                    and existing[e, 2] == row[2] and existing[e, 3] == row[3]):
                return False

        # Duplicate-face test: each face {po, x, y} of the substituted tet
        # must not already belong to a live tet of ball(po) outside the shell
        # (shell tets die in the collapse). The exception is a face whose
        # opposite pair {x, y} lies on a shell tet {pc, po, x, y}: there the
        # outside tet is the shell tet's neighbour and the face is shared by
        # exactly two tets afterwards. The face nodes are distinct, so
        # containment in the other tet equals face equality.
        nn = np.empty(4, dtype=np.int64)
        nn[0] = n0
        nn[1] = n1
        nn[2] = n2
        nn[3] = n3
        for skip in range(4):
            if nn[skip] == po:
                continue
            a = np.int64(-1)
            b = np.int64(-1)
            c = np.int64(-1)
            cnt = 0
            for m in range(4):
                if m == skip:
                    continue
                if cnt == 0:
                    a = nn[m]
                elif cnt == 1:
                    b = nn[m]
                else:
                    c = nn[m]
                cnt += 1
            # x, y are the two face nodes other than po
            x = a
            y = b
            if a == po:
                x = b
                y = c
            elif b == po:
                y = c
            on_shell = False
            for j in range(len(shell)):
                st = shell[j]
                hx = False
                hy = False
                for m in range(4):
                    v = np.int64(tets[4 * st + m])
                    if v == x:
                        hx = True
                    if v == y:
                        hy = True
                if hx and hy:
                    on_shell = True
                    break
            if on_shell:
                continue
            for k in range(n_po):
                u = ball_po[k]
                if tets[4 * u] == -1:
                    continue
                u_shell = False
                for j in range(len(shell)):
                    if shell[j] == u:
                        u_shell = True
                        break
                if u_shell:
                    continue
                hit = 0
                for m in range(4):
                    v = np.int64(tets[4 * u + m])
                    if v == a or v == b or v == c:
                        hit += 1
                if hit == 3:
                    return False
    return True


@njit(cache=True, nogil=True)
def collapse_len_quality_jit(coords, mm, tets, dead_mask, ball_pc, pc, po,
                             post_min_ratio, post_max_ratio):
    """Exact port of collapse.collapse_len_quality. The neighbours of pc are
    gathered from ball(pc) in tet order (deduplicated); the 0/1 result does
    not depend on their order."""
    nbrs = np.empty(4 * len(ball_pc), dtype=np.int64)
    n_nb = 0
    for i in range(len(ball_pc)):
        t = ball_pc[i]
        for k in range(4):
            v = np.int64(tets[4 * t + k])
            if v == pc:
                continue
            seen = False
            for j in range(n_nb):
                if nbrs[j] == v:
                    seen = True
                    break
            if not seen:
                nbrs[n_nb] = v
                n_nb += 1

    old_min = np.inf
    old_max = 0.0
    for j in range(n_nb):
        pn = nbrs[j]
        if dead_mask[pn] or dead_mask[pc]:
            continue
        L_old = edge_length_nodes_jit(coords, mm, pc, pn)
        if L_old < old_min:
            old_min = L_old
        if L_old > old_max:
            old_max = L_old

    new_min = np.inf
    new_max = 0.0
    for j in range(n_nb):
        pn = nbrs[j]
        if pn == po:
            continue
        if dead_mask[pn] or dead_mask[po]:
            return 0
        L = edge_length_nodes_jit(coords, mm, po, pn)
        if L < new_min:
            new_min = L
        if L > new_max:
            new_max = L
        if L < post_min_ratio or L > post_max_ratio:
            if new_min < old_min or new_max > old_max:
                return 0

    if new_min < post_min_ratio or new_max > post_max_ratio:
        if new_min < old_min or new_max > old_max:
            return 0
    return 1


@njit(cache=True, nogil=True)
def collapse_tet_local_jit(coords, mm, tets, dead_mask, ball_pc, shell,
                           pc, po, q_r, q_a):
    """Exact port of the tet part of collapse._collapse_tet_quality_local
    (old-worst, signed-volume, per-tet and Loseille-Lohner gates). Returns 0
    on a rejection, 1 otherwise. Face qualities are checked by the caller."""
    no_xyz = np.zeros(6, dtype=np.float64)
    n_b = len(ball_pc)
    keep = np.empty(n_b, dtype=np.bool_)
    for i in range(n_b):
        t = ball_pc[i]
        in_shell = False
        for j in range(len(shell)):
            if shell[j] == t:
                in_shell = True
                break
        keep[i] = not in_shell

    old_worst = np.inf
    for i in range(n_b):
        if not keep[i]:
            continue
        t = ball_pc[i]
        n0 = np.int64(tets[4 * t])
        n1 = np.int64(tets[4 * t + 1])
        n2 = np.int64(tets[4 * t + 2])
        n3 = np.int64(tets[4 * t + 3])
        if n0 == -1:
            continue
        if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
            continue
        q = tet_volume_quality_jit(coords, mm, n0, n1, n2, n3,
                                   -2, no_xyz, no_xyz, False)[1]
        if q < old_worst:
            old_worst = q

    have = False
    new_worst = 0.0
    for i in range(n_b):
        if not keep[i]:
            continue
        t = ball_pc[i]
        n0 = np.int64(tets[4 * t])
        n1 = np.int64(tets[4 * t + 1])
        n2 = np.int64(tets[4 * t + 2])
        n3 = np.int64(tets[4 * t + 3])
        if n0 == pc:
            n0 = po
        if n1 == pc:
            n1 = po
        if n2 == pc:
            n2 = po
        if n3 == pc:
            n3 = po
        if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
            return 0
        res = tet_volume_quality_jit(coords, mm, n0, n1, n2, n3,
                                     -2, no_xyz, no_xyz, False)
        if res[0] < 1e-12:
            return 0
        q = res[1]
        if q < old_worst / q_r or q < q_a:
            return 0
        if not have:
            new_worst = q
            have = True
        elif q < new_worst:
            new_worst = q

    if not (new_worst >= (old_worst / q_r) and new_worst >= q_a):
        return 0
    return 1


@njit(cache=True, nogil=True)
def pliant_force_sum_jit(coords, metrics, dead_mask, center, nbrs, noise_floor):
    """Sum of pliant forces on *center* over *nbrs*, accumulated in list order.

    Bitwise port of smooth._pliant_force_arrays summed from 0.0. Skipped
    neighbours still add +0.0, as the Python accumulation does.
    """
    total = np.zeros(3, dtype=np.float64)
    for i in range(nbrs.shape[0]):
        n = nbrs[i]
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        if not (dead_mask[center] or dead_mask[n]):
            m_c = metrics[6 * center:6 * center + 6]
            m_n = metrics[6 * n:6 * n + 6]
            if not (m_c[0] == -1 or m_n[0] == -1):
                p_c = coords[3 * center:3 * center + 3]
                p_n = coords[3 * n:3 * n + 3]
                ratio = _length_in_metric(m_c, m_n, p_c, p_n)
                if not (ratio < noise_floor):
                    l4 = ratio * ratio * ratio * ratio
                    force = (1.0 - l4) * math.exp(-l4)
                    if ratio < 1.0:
                        force += (1.0 - ratio) * (1.0 - ratio)
                    f0 = force * ((p_c[0] - p_n[0]) / ratio)
                    f1 = force * ((p_c[1] - p_n[1]) / ratio)
                    f2 = force * ((p_c[2] - p_n[2]) / ratio)
        total[0] += f0
        total[1] += f1
        total[2] += f2
    return total


@njit(cache=True, nogil=True)
def node_neighbours_csr_jit(tets, ball_ptr, ball_idx, n_pts):
    """CSR ``(nbr_ptr, nbr_idx)`` of vertices sharing a live tet with each node.

    Each row is sorted, unique and excludes the node itself.
    """
    nbr_ptr = np.zeros(n_pts + 1, dtype=np.int32)
    for v in range(n_pts):
        cnt = ball_ptr[v + 1] - ball_ptr[v]
        tmp = np.empty(4 * cnt, dtype=np.int32)
        m = 0
        for bi in range(ball_ptr[v], ball_ptr[v + 1]):
            base = 4 * ball_idx[bi]
            for k in range(4):
                u = tets[base + k]
                if u != v:
                    tmp[m] = u
                    m += 1
        tmp = np.unique(tmp[:m])
        nbr_ptr[v + 1] = nbr_ptr[v] + len(tmp)
    nbr_idx = np.empty(nbr_ptr[n_pts], dtype=np.int32)
    for v in range(n_pts):
        cnt = ball_ptr[v + 1] - ball_ptr[v]
        tmp = np.empty(4 * cnt, dtype=np.int32)
        m = 0
        for bi in range(ball_ptr[v], ball_ptr[v + 1]):
            base = 4 * ball_idx[bi]
            for k in range(4):
                u = tets[base + k]
                if u != v:
                    tmp[m] = u
                    m += 1
        tmp = np.unique(tmp[:m])
        p = nbr_ptr[v]
        for i in range(len(tmp)):
            nbr_idx[p + i] = tmp[i]
    return nbr_ptr, nbr_idx


@njit(cache=True, nogil=True)
def greedy_colour_jit(cand, nbr_ptr, nbr_idx, n_pts):
    """Greedy distance-2 colouring of the candidate nodes in index order.

    Returns int32 colour per node (-1 for non-candidates). Each candidate
    takes the smallest colour unused by every already-coloured candidate in
    its 1-ring or 2-ring. check_local_intersection_jit for node A tests A's
    ball tets against tets touching A or its neighbours; those tets contain
    only nodes within distance 2 of A. With this colouring no node moving in
    the same colour appears in them, and no metric in A's ball changes, so a
    trial verdict is evaluated on exactly the state A commits into.
    """
    colour = np.full(n_pts, -1, dtype=np.int32)
    for v in range(n_pts):
        if not cand[v]:
            continue
        bound = 1
        for i in range(nbr_ptr[v], nbr_ptr[v + 1]):
            u = nbr_idx[i]
            bound += 1 + nbr_ptr[u + 1] - nbr_ptr[u]
        used = np.zeros(bound + 1, dtype=np.bool_)
        for i in range(nbr_ptr[v], nbr_ptr[v + 1]):
            u = nbr_idx[i]
            c = colour[u]
            if c >= 0 and c <= bound:
                used[c] = True
            for j in range(nbr_ptr[u], nbr_ptr[u + 1]):
                w = nbr_idx[j]
                c = colour[w]
                if c >= 0 and c <= bound:
                    used[c] = True
        c = 0
        while used[c]:
            c += 1
        colour[v] = c
    return colour


@njit(cache=True, nogil=True)
def _clamp_trial_jit(trial, planes_o, planes_n, has_bbox, bb_min, bb_max):
    """In-place domain clamp: bbox when has_bbox, else the plane list."""
    if has_bbox:
        for k in range(3):
            trial[k] = max(bb_min[k], min(bb_max[k], trial[k]))
    else:
        for p in range(planes_o.shape[0]):
            dist = (planes_n[p, 0] * (trial[0] - planes_o[p, 0])
                    + planes_n[p, 1] * (trial[1] - planes_o[p, 1])
                    + planes_n[p, 2] * (trial[2] - planes_o[p, 2]))
            if dist > 0:
                trial[0] -= dist * planes_n[p, 0]
                trial[1] -= dist * planes_n[p, 1]
                trial[2] -= dist * planes_n[p, 2]


@njit(cache=True, nogil=True)
def _smooth_one_node_jit(coords, tets, metrics, dead_mask, ball_ptr, ball_idx,
                         nbr_ptr, nbr_idx, node, planes_o, planes_n, has_bbox,
                         bb_min, bb_max, q_a, alpha, tol, backoff_tries,
                         noise_floor, ignored, out_xyz):
    """Interior smoothing trial for one node (smooth.smooth_tet_node logic).

    Writes the accepted position to out_xyz and returns True, else False.
    *ignored* is a scratch mask over tets, all False on entry and on exit.
    """
    nbrs = nbr_idx[nbr_ptr[node]:nbr_ptr[node + 1]]
    if len(nbrs) == 0:
        return False
    ball = ball_idx[ball_ptr[node]:ball_ptr[node + 1]]
    force = pliant_force_sum_jit(coords, metrics, dead_mask, node, nbrs,
                                 noise_floor)
    no_ovr = np.zeros(3, dtype=np.float64)
    old_q = ball_min_quality_jit(coords, tets, metrics, dead_mask, node, ball,
                                 False, no_ovr)

    n_live = 0
    for bi in range(len(ball)):
        if tets[4 * ball[bi]] != -1:
            n_live += 1
    check = np.empty((n_live, 4), dtype=np.int32)
    m = 0
    for bi in range(len(ball)):
        base = 4 * ball[bi]
        if tets[base] != -1:
            for k in range(4):
                check[m, k] = tets[base + k]
            m += 1
    for bi in range(len(ball)):
        ignored[ball[bi]] = True

    trial = np.empty(3, dtype=np.float64)
    ok = False
    backoff = 1.0
    for _ in range(backoff_tries):
        for k in range(3):
            trial[k] = coords[3 * node + k] + backoff * alpha * force[k]
        _clamp_trial_jit(trial, planes_o, planes_n, has_bbox, bb_min, bb_max)
        trial_q = ball_min_quality_jit(coords, tets, metrics, dead_mask, node,
                                       ball, True, trial)
        if trial_q >= q_a and trial_q > old_q - tol:
            if not check_local_intersection_jit(
                    coords, tets, ball_ptr, ball_idx, dead_mask, check,
                    node, trial, ignored):
                for k in range(3):
                    out_xyz[k] = trial[k]
                ok = True
                break
        backoff *= 0.5

    for bi in range(len(ball)):
        ignored[ball[bi]] = False
    return ok


@njit(cache=True, nogil=True, parallel=True)
def smooth_colour_trials_jit(coords, tets, metrics, dead_mask, ball_ptr,
                             ball_idx, nbr_ptr, nbr_idx, nodes, planes_o,
                             planes_n, has_bbox, bb_min, bb_max, q_a, alpha,
                             tol, backoff_tries, noise_floor, out_xyz, out_ok):
    """Trial positions for one colour class of interior nodes, in parallel.

    Writes out_xyz[i] / out_ok[i] for nodes[i]; no shared array is modified.
    Nodes of one colour are non-adjacent, so each ball is evaluated against
    exactly the state its node commits into. Results do not depend on the
    thread count: nodes are processed in fixed blocks, one scratch mask each.
    """
    n = len(nodes)
    n_tets = len(tets) // 4
    n_blocks = min(n, 256)
    for b in prange(n_blocks):
        lo = (b * n) // n_blocks
        hi = ((b + 1) * n) // n_blocks
        ignored = np.zeros(n_tets, dtype=np.bool_)
        for i in range(lo, hi):
            out_ok[i] = _smooth_one_node_jit(
                coords, tets, metrics, dead_mask, ball_ptr, ball_idx,
                nbr_ptr, nbr_idx, nodes[i], planes_o, planes_n, has_bbox,
                bb_min, bb_max, q_a, alpha, tol, backoff_tries, noise_floor,
                ignored, out_xyz[i])


@njit(cache=True, nogil=True)
def tet_qualities_jit(coords, tets, metrics):
    """Knupp quality of every tet with tets[4t] != -1, in tet order."""
    n_tets = len(tets) // 4
    out = np.empty(n_tets, dtype=np.float64)
    count = 0
    mk = np.empty((4, 6), dtype=np.float64)
    dets = np.empty(4, dtype=np.float64)
    for t in range(n_tets):
        base = 4 * t
        if tets[base] == -1:
            continue
        for k in range(4):
            v = tets[base + k]
            for j in range(6):
                mk[k, j] = metrics[v * 6 + j]
            dets[k] = _det_metric(mk[k])
        v0 = tets[base]
        v1 = tets[base + 1]
        v2 = tets[base + 2]
        v3 = tets[base + 3]
        out[count] = _tet_quality(coords[v0 * 3: v0 * 3 + 3],
                                  coords[v1 * 3: v1 * 3 + 3],
                                  coords[v2 * 3: v2 * 3 + 3],
                                  coords[v3 * 3: v3 * 3 + 3], mk, dets)
        count += 1
    return out[:count].copy()


@njit(cache=True, nogil=True)
def _metric_bilinear(u, v, M):
    """u^T M v for 3-vectors and a packed 6-component metric."""
    return (u[0] * (M[0] * v[0] + M[1] * v[1] + M[2] * v[2])
            + u[1] * (M[1] * v[0] + M[3] * v[1] + M[4] * v[2])
            + u[2] * (M[2] * v[0] + M[4] * v[1] + M[5] * v[2]))


@njit(cache=True, nogil=True)
def ma_weights_jit(coords, tets, live_idx, mk_all):
    """Marcum-Alauzet edge weight of each tet in live_idx (inf if degenerate).

    mk_all[i] is the element mean metric of tet live_idx[i].
    """
    n = len(live_idx)
    out = np.empty(n, dtype=np.float64)
    ed = np.empty((9, 3), dtype=np.float64)
    for i in range(n):
        base = 4 * live_idx[i]
        v0 = tets[base]
        v1 = tets[base + 1]
        v2 = tets[base + 2]
        v3 = tets[base + 3]
        a = coords[v0 * 3: v0 * 3 + 3]
        b = coords[v1 * 3: v1 * 3 + 3]
        c = coords[v2 * 3: v2 * 3 + 3]
        d = coords[v3 * 3: v3 * 3 + 3]
        mk = mk_all[i]
        det_mk = _det_metric(mk)
        six_V = 6.0 * _signed_volume(a, b, c, d)
        if abs(six_V) < 1e-30:
            out[i] = np.inf
            continue
        denom = math.sqrt(abs(det_mk)) * abs(six_V)
        for k in range(3):
            ed[0, k] = b[k] - a[k]      # ea
            ed[1, k] = c[k] - a[k]      # eb
            ed[2, k] = d[k] - a[k]      # ec
            ed[3, k] = c[k] - b[k]      # ed
            ed[4, k] = d[k] - b[k]      # ef
            ed[5, k] = d[k] - c[k]      # eg
            ed[6, k] = -ed[0, k]        # neg_ea
            ed[7, k] = -ed[1, k]        # neg_eb
            ed[8, k] = -ed[3, k]        # neg_ed
        w_max = -np.inf
        for q in range(6):
            if q == 0:
                s = ed[0]; j = ed[1]; kk = ed[2]
            elif q == 1:
                s = ed[1]; j = ed[0]; kk = ed[2]
            elif q == 2:
                s = ed[2]; j = ed[0]; kk = ed[1]
            elif q == 3:
                s = ed[3]; j = ed[6]; kk = ed[4]
            elif q == 4:
                s = ed[4]; j = ed[6]; kk = ed[3]
            else:
                s = ed[5]; j = ed[7]; kk = ed[8]
            sj = _metric_bilinear(s, j, mk)
            sk = _metric_bilinear(s, kk, mk)
            ss = _metric_bilinear(s, s, mk)
            jk = _metric_bilinear(j, kk, mk)
            w = (sj * sk - ss * jk) / denom
            if w > w_max:
                w_max = w
        out[i] = w_max
    return out


@njit(cache=True, nogil=True)
def complexity_jit(coords, tets, metrics, dead_mask):
    """Continuous-mesh complexity, sum over live tets and vertices of sqrt(det) * V / 4."""
    complexity = 0.0
    for i in range(0, len(tets), 4):
        v0 = tets[i]
        if v0 == -1:
            continue
        v1 = tets[i + 1]
        v2 = tets[i + 2]
        v3 = tets[i + 3]
        if dead_mask[v0] or dead_mask[v1] or dead_mask[v2] or dead_mask[v3]:
            continue
        vol = abs(_signed_volume(coords[v0 * 3: v0 * 3 + 3],
                                 coords[v1 * 3: v1 * 3 + 3],
                                 coords[v2 * 3: v2 * 3 + 3],
                                 coords[v3 * 3: v3 * 3 + 3]))
        for k in range(4):
            v = tets[i + k]
            m = metrics[v * 6: v * 6 + 6]
            if len(m) == 0 or m[0] == -1:
                continue
            det = _det_metric(m)
            if det > 0.0:
                complexity += math.sqrt(det) * vol / 4.0
    return complexity


@njit(cache=True, nogil=True)
def flat_tet_mask_jit(coords, tets, dead_mask):
    """True for each flat tet (report.tet_is_flat), over all tets."""
    n_tets = len(tets) // 4
    out = np.zeros(n_tets, dtype=np.bool_)
    for t in range(n_tets):
        i = 4 * t
        ai = tets[i]
        bi = tets[i + 1]
        ci = tets[i + 2]
        di = tets[i + 3]
        if ai == -1:
            continue
        if dead_mask[ai] or dead_mask[bi] or dead_mask[ci] or dead_mask[di]:
            continue
        a = coords[ai * 3: ai * 3 + 3]
        b = coords[bi * 3: bi * 3 + 3]
        c = coords[ci * 3: ci * 3 + 3]
        d = coords[di * 3: di * 3 + 3]
        det_val = 6.0 * abs(_signed_volume(a, b, c, d))
        max_edge_sq = 0.0
        for q in range(6):
            if q == 0:
                p = a; r = b
            elif q == 1:
                p = b; r = c
            elif q == 2:
                p = c; r = a
            elif q == 3:
                p = a; r = d
            elif q == 4:
                p = b; r = d
            else:
                p = c; r = d
            e = ((p[0] - r[0]) * (p[0] - r[0])
                 + (p[1] - r[1]) * (p[1] - r[1])
                 + (p[2] - r[2]) * (p[2] - r[2]))
            if q == 0 or e > max_edge_sq:
                max_edge_sq = e
        if max_edge_sq < NOISE_FLOOR:
            out[t] = True
        else:
            out[t] = det_val / max_edge_sq <= 1e-15
    return out


# ──────────────────────── Warmup ───────────────────────────────────

# ──────────────── Per-tet angle measures ───────────────────────────
# Exact ports of the swap.py dihedral / solid-angle logic: every floating-point
# operation keeps the order and association of the Python code, and min/max/clamp
# follow CPython's comparison semantics, so results are bit-identical.

@njit(cache=True, nogil=True)
def _clamp_unit_jit(x):
    """CPython max(-1.0, min(1.0, x)), including its NaN behaviour."""
    y = x if x < 1.0 else 1.0
    return y if y > -1.0 else -1.0


@njit(cache=True, nogil=True)
def _face_normal_jit(coords, a, b, c):
    """Unit normal of triangle (a, b, c); zeros below NOISE_FLOOR."""
    ax = coords[3 * a]
    ay = coords[3 * a + 1]
    az = coords[3 * a + 2]
    u0 = coords[3 * b] - ax
    u1 = coords[3 * b + 1] - ay
    u2 = coords[3 * b + 2] - az
    v0 = coords[3 * c] - ax
    v1 = coords[3 * c + 1] - ay
    v2 = coords[3 * c + 2] - az
    n0 = u1 * v2 - u2 * v1
    n1 = u2 * v0 - u0 * v2
    n2 = u0 * v1 - u1 * v0
    mag = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
    if mag < NOISE_FLOOR:
        return 0.0, 0.0, 0.0
    return n0 / mag, n1 / mag, n2 / mag


@njit(cache=True, nogil=True)
def _angle_between_jit(u0, u1, u2, v0, v1, v2):
    """Angle in radians between two 3D vectors; 0.0 if either is degenerate."""
    d = u0 * v0 + u1 * v1 + u2 * v2
    mu = math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    mv = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    if mu < NOISE_FLOOR or mv < NOISE_FLOOR:
        return 0.0
    return math.acos(_clamp_unit_jit(d / (mu * mv)))


@njit(cache=True, nogil=True)
def dihedral_angles_jit(coords, n0, n1, n2, n3):
    """Six dihedral angles of tet (n0, n1, n2, n3), swap.dihedral_angle order."""
    a0, a1, a2 = _face_normal_jit(coords, n1, n2, n3)
    b0, b1, b2 = _face_normal_jit(coords, n0, n2, n3)
    c0, c1, c2 = _face_normal_jit(coords, n0, n1, n3)
    d0, d1, d2 = _face_normal_jit(coords, n0, n1, n2)
    out = np.empty(6, dtype=np.float64)
    out[0] = _angle_between_jit(a0, a1, a2, b0, b1, b2)
    out[1] = _angle_between_jit(a0, a1, a2, c0, c1, c2)
    out[2] = _angle_between_jit(a0, a1, a2, d0, d1, d2)
    out[3] = _angle_between_jit(b0, b1, b2, c0, c1, c2)
    out[4] = _angle_between_jit(b0, b1, b2, d0, d1, d2)
    out[5] = _angle_between_jit(c0, c1, c2, d0, d1, d2)
    return out


@njit(cache=True, nogil=True)
def _unit_from_jit(coords, vp, q):
    """unit_vec(coords[q] - coords[vp]) with the NOISE_FLOOR guard."""
    x = coords[3 * q] - coords[3 * vp]
    y = coords[3 * q + 1] - coords[3 * vp + 1]
    z = coords[3 * q + 2] - coords[3 * vp + 2]
    n = math.sqrt(x * x + y * y + z * z)
    if n < NOISE_FLOOR:
        return 0.0, 0.0, 0.0
    return x / n, y / n, z / n


@njit(cache=True, nogil=True)
def solid_angle_jit(coords, n0, n1, n2, n3, vertex):
    """Solid angle at *vertex* of tet (n0..n3) by the spherical excess formula."""
    o0 = -1
    o1 = -1
    o2 = -1
    cnt = 0
    for k in range(4):
        if k == 0:
            v = n0
        elif k == 1:
            v = n1
        elif k == 2:
            v = n2
        else:
            v = n3
        if v != vertex:
            if cnt == 0:
                o0 = v
            elif cnt == 1:
                o1 = v
            elif cnt == 2:
                o2 = v
            cnt += 1
    if cnt != 3:
        return 0.0

    x0, y0, z0 = _unit_from_jit(coords, vertex, o0)
    x1, y1, z1 = _unit_from_jit(coords, vertex, o1)
    x2, y2, z2 = _unit_from_jit(coords, vertex, o2)

    cos_a = _clamp_unit_jit(x1 * x2 + y1 * y2 + z1 * z2)
    cos_b = _clamp_unit_jit(x0 * x2 + y0 * y2 + z0 * z2)
    cos_c = _clamp_unit_jit(x0 * x1 + y0 * y1 + z0 * z1)

    A = math.acos(cos_a)
    B = math.acos(cos_b)
    C = math.acos(cos_c)
    s = (A + B + C) / 2.0

    m = A
    if B > m:
        m = B
    if C > m:
        m = C
    if s <= m:
        return 0.0
    p = (math.tan(s / 2) * math.tan((s - A) / 2)
         * math.tan((s - B) / 2) * math.tan((s - C) / 2))
    if p < 0.0:
        return 0.0
    return 4.0 * math.atan(math.sqrt(p))


@njit(cache=True, nogil=True)
def tet_angle_measures_jit(coords, n0, n1, n2, n3):
    """(min dihedral, max dihedral, max solid angle) of tet (n0..n3)."""
    d = dihedral_angles_jit(coords, n0, n1, n2, n3)
    mn = d[0]
    mx = d[0]
    for i in range(1, 6):
        if d[i] < mn:
            mn = d[i]
        if d[i] > mx:
            mx = d[i]
    ms = solid_angle_jit(coords, n0, n1, n2, n3, n0)
    x = solid_angle_jit(coords, n0, n1, n2, n3, n1)
    if x > ms:
        ms = x
    x = solid_angle_jit(coords, n0, n1, n2, n3, n2)
    if x > ms:
        ms = x
    x = solid_angle_jit(coords, n0, n1, n2, n3, n3)
    if x > ms:
        ms = x
    return mn, mx, ms


@njit(cache=True, nogil=True)
def mesh_angle_extrema_jit(coords, tets, dead_mask):
    """(found, min dihedral, max dihedral, max solid angle) over live tets
    of non-negligible volume, in tet order."""
    found = False
    mesh_min = 0.0
    mesh_max = 0.0
    mesh_max_solid = 0.0
    for t in range(len(tets) // 4):
        n0 = tets[4 * t]
        if n0 == -1:
            continue
        n1 = tets[4 * t + 1]
        n2 = tets[4 * t + 2]
        n3 = tets[4 * t + 3]
        if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
            continue
        vol = abs(_signed_volume(coords[3 * n0:3 * n0 + 3],
                                 coords[3 * n1:3 * n1 + 3],
                                 coords[3 * n2:3 * n2 + 3],
                                 coords[3 * n3:3 * n3 + 3]))
        if vol > 1e-12:
            mn, mx, ms = tet_angle_measures_jit(coords, n0, n1, n2, n3)
            if not found:
                found = True
                mesh_min = mn
                mesh_max = mx
                mesh_max_solid = ms
            else:
                if mn < mesh_min:
                    mesh_min = mn
                if mx > mesh_max:
                    mesh_max = mx
                if ms > mesh_max_solid:
                    mesh_max_solid = ms
    return found, mesh_min, mesh_max, mesh_max_solid


@njit(cache=True, nogil=True)
def edge_swap_configs_jit(coords, dead_mask, surround, v_top, v_bot, patterns):
    """Valid edge-swap configurations of a shell ring: (out, n_valid), the
    first n_valid rows of out in pattern, rotation order. Each row holds two
    tets per pattern triangle, apex v_top then v_bot."""
    n_shell = len(surround)
    n_pat = patterns.shape[0]
    n_tri = patterns.shape[1]
    out = np.empty((n_pat * n_shell, 2 * n_tri, 4), dtype=np.int64)
    n_valid = 0
    for p in range(n_pat):
        for r in range(n_shell):
            valid = True
            for ti in range(n_tri):
                for a in range(2):
                    row = 2 * ti + a
                    out[n_valid, row, 0] = v_top if a == 0 else v_bot
                    for k in range(3):
                        out[n_valid, row, 1 + k] = surround[(patterns[p, ti, k] + r) % n_shell]
                    n0 = out[n_valid, row, 0]
                    n1 = out[n_valid, row, 1]
                    n2 = out[n_valid, row, 2]
                    n3 = out[n_valid, row, 3]
                    if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
                        valid = False
                        break
                    sv = _signed_volume(coords[3 * n0:3 * n0 + 3],
                                        coords[3 * n1:3 * n1 + 3],
                                        coords[3 * n2:3 * n2 + 3],
                                        coords[3 * n3:3 * n3 + 3])
                    if sv <= 0.0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                n_valid += 1
    return out, n_valid


@njit(cache=True, nogil=True)
def swap_conserves_volume_jit(coords, dead_mask, old_nodes, new_nodes, tol):
    """True iff the new tets (rows of node ids) tile the volume of the old
    tets without gap or overlap. A row starting at -1 is a deleted old tet."""
    v_old = 0.0
    for r in range(old_nodes.shape[0]):
        n0 = old_nodes[r, 0]
        n1 = old_nodes[r, 1]
        n2 = old_nodes[r, 2]
        n3 = old_nodes[r, 3]
        if n0 == -1:
            return False
        if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
            return False
        v_old += abs(_signed_volume(coords[3 * n0:3 * n0 + 3],
                                    coords[3 * n1:3 * n1 + 3],
                                    coords[3 * n2:3 * n2 + 3],
                                    coords[3 * n3:3 * n3 + 3]))
    v_new = 0.0
    for r in range(new_nodes.shape[0]):
        n0 = new_nodes[r, 0]
        n1 = new_nodes[r, 1]
        n2 = new_nodes[r, 2]
        n3 = new_nodes[r, 3]
        if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
            return False
        v_new += abs(_signed_volume(coords[3 * n0:3 * n0 + 3],
                                    coords[3 * n1:3 * n1 + 3],
                                    coords[3 * n2:3 * n2 + 3],
                                    coords[3 * n3:3 * n3 + 3]))
    return abs(v_new - v_old) <= tol * max(1.0, v_old)


@njit(cache=True, nogil=True)
def config_min_quality_jit(coords, metrics, nodes):
    """Minimum metric quality over the rows of node ids, in row order."""
    ovr_xyz = np.zeros(3, dtype=np.float64)
    ovr_metric = np.zeros(6, dtype=np.float64)
    min_q = np.inf
    for r in range(nodes.shape[0]):
        q = tet_volume_quality_jit(coords, metrics, nodes[r, 0], nodes[r, 1],
                                   nodes[r, 2], nodes[r, 3], -2,
                                   ovr_xyz, ovr_metric, False)[1]
        if q < min_q:
            min_q = q
    return min_q


@njit(cache=True, nogil=True)
def swap_score_configs_jit(coords, dead_mask, old_nodes, has_shell, configs,
                           tol):
    """(best_index, best_min_q) over candidate reconnections, min-dihedral.

    configs: (n_cfg, m, 4) node ids. A config is skipped when has_shell and
    it does not tile the volume of old_nodes. A config scores the minimum
    tet quality (-1e6 for a dead node or a non-positive volume). The first
    config with the strictly greatest score wins; best_index is -1 when none
    was kept."""
    best_index = -1
    best_min_q = -np.inf
    for ci in range(configs.shape[0]):
        cfg = configs[ci]
        if has_shell and not swap_conserves_volume_jit(
                coords, dead_mask, old_nodes, cfg, tol):
            continue
        min_q = -1e6
        for r in range(cfg.shape[0]):
            n0 = cfg[r, 0]
            n1 = cfg[r, 1]
            n2 = cfg[r, 2]
            n3 = cfg[r, 3]
            if dead_mask[n0] or dead_mask[n1] or dead_mask[n2] or dead_mask[n3]:
                q = -1e6
            else:
                sv = _signed_volume(coords[3 * n0:3 * n0 + 3],
                                    coords[3 * n1:3 * n1 + 3],
                                    coords[3 * n2:3 * n2 + 3],
                                    coords[3 * n3:3 * n3 + 3])
                if sv <= 1e-15:
                    q = -1e6
                else:
                    q = tet_angle_measures_jit(coords, n0, n1, n2, n3)[0]
            if r == 0 or q < min_q:
                min_q = q
        if min_q > best_min_q:
            best_min_q = min_q
            best_index = ci
    return best_index, best_min_q


def warmup():
    """Trigger compilation of all JIT kernels. Call once at adapt() entry.

    Not @njit itself — it's a plain Python function that calls each kernel
    with tiny valid inputs to force compilation. Numba typing errors surface
    before a long adaptation run begins.
    """
    import time
    t0 = time.perf_counter()

    a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    m = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    gamma = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    _signed_volume(a, b, c, d)
    _gamma_m_gamma(gamma, m)
    _gamma_m_gamma_scalar(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    _det_metric(m)
    _segment_triangle_intersect(a, b, c, d, np.array([0.5, 0.5, 0.0], dtype=np.float64), 1e-7)
    _point_in_tet(np.array([0.1, 0.1, 0.1], dtype=np.float64), a, b, c, d, 1e-7)
    _length_in_metric(m, m, a, b)

    metrics = np.array([[1,0,0,1,0,1],[1,0,0,1,0,1],[1,0,0,1,0,1],[1,0,0,1,0,1]], dtype=np.float64)
    dets = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _tet_quality(a, b, c, d, metrics, dets)

    # quality-evaluation ports
    wm = np.array([1,0,0,1,0,1]*4, dtype=np.float64)
    wc = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1], dtype=np.float64)
    wt = np.array([0,1,2,3], dtype=np.int32)
    wd = np.zeros(4, dtype=np.bool_)
    ball_min_quality_jit(wc, wt, wm, wd, 0, np.array([0], dtype=np.int64),
                         False, np.zeros(3, dtype=np.float64))
    tet_volume_quality_jit(wc, wm, 0, 1, 2, 3, -2, np.zeros(3, dtype=np.float64),
                           np.zeros(6, dtype=np.float64), False)
    edge_length_nodes_jit(wc, wm, 0, 1)
    wtets = np.array([0, 1, 2, 3, 0, 1, 2, 4, -1, -1, -1, -1], dtype=np.int32)
    wbp = np.array([0, 1], dtype=np.int64)
    wbo = np.array([0, 1], dtype=np.int64)
    wsh = np.array([0], dtype=np.int64)
    wm5b = np.array([1,0,0,1,0,1]*5, dtype=np.float64)
    wc5b = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1], dtype=np.float64)
    wd5b = np.zeros(5, dtype=np.bool_)
    duplicate_cell_jit(wtets, wbo, wbp, wsh, 0, 4)
    collapse_len_quality_jit(wc5b, wm5b, wtets, wd5b, wbp, 0, 1, 1e-3, 4.0)
    collapse_tet_local_jit(wc5b, wm5b, wtets, wd5b, wbp, wsh, 0, 1, 1.0, 1e-3)
    pliant_force_sum_jit(wc, wm, wd, 0, np.array([1, 2], dtype=np.int64), 1e-15)

    # coloured smoothing kernels
    wc4 = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1, 0.25,0.25,0.25], dtype=np.float64)
    wt4 = np.array([4,1,2,3, 0,4,2,3, 0,1,4,3, 0,1,2,4], dtype=np.int32)
    wm5 = np.array([1,0,0,1,0,1]*5, dtype=np.float64)
    wd5 = np.zeros(5, dtype=np.bool_)
    bp, bi = build_ball_csr_jit(wt4, 5)
    npt, nix = node_neighbours_csr_jit(wt4, bp, bi, 5)
    cand5 = np.zeros(5, dtype=np.bool_)
    cand5[4] = True
    greedy_colour_jit(cand5, npt, nix, 5)
    out_xyz_w = np.zeros((1, 3), dtype=np.float64)
    out_ok_w = np.zeros(1, dtype=np.bool_)
    smooth_colour_trials_jit(wc4, wt4, wm5, wd5, bp, bi, npt, nix,
                             np.array([4], dtype=np.int64),
                             np.zeros((0, 3), dtype=np.float64),
                             np.zeros((0, 3), dtype=np.float64), False,
                             np.zeros(3, dtype=np.float64),
                             np.zeros(3, dtype=np.float64), 1e-3, 0.2, 1e-8,
                             4, NOISE_FLOOR, out_xyz_w, out_ok_w)

    # report diagnostics
    tet_qualities_jit(wc, wt, wm)
    ma_weights_jit(wc, wt, np.array([0], dtype=np.int64),
                   np.array([[1,0,0,1,0,1]], dtype=np.float64))
    complexity_jit(wc, wt, wm, wd)
    flat_tet_mask_jit(wc, wt, wd)

    # check_local_intersection_jit
    coords = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1, 2,0,0, 2,1,0, 2,0,1, 2,1,1], dtype=np.float64)
    tets_arr = np.array([0,1,2,3, 4,5,6,7], dtype=np.int32)
    ball_ptr = np.array([0,1,1,1,1,2,2,2,2], dtype=np.int32)
    ball_idx = np.array([0, 1], dtype=np.int32)
    dead = np.zeros(8, dtype=np.bool_)
    ttc = np.array([[0,1,2,3]], dtype=np.int32)
    override = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ign = np.zeros(2, dtype=np.bool_)
    check_local_intersection_jit(coords, tets_arr, ball_ptr, ball_idx, dead,
                                  ttc, np.int32(-1), override, ign)

    # Candidate scans
    edges_arr = np.array([[0,1],[1,2],[2,3]], dtype=np.int32)
    mm = np.array([1,0,0,1,0,1]*4, dtype=np.float64)
    dead4 = np.zeros(4, dtype=np.bool_)
    collapse_candidate_scan_jit(coords[:12], mm, dead4, edges_arr, 4, 0.7071)
    split_candidate_scan_jit(coords[:12], mm, dead4, edges_arr, 4, 1.4142)
    compute_edge_active_jit(coords[:12], mm, dead4, edges_arr, 4)

    # compact_arrays_jit
    tets_compact = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
    dead_compact = np.zeros(4, dtype=np.bool_)
    compact_arrays_jit(coords[:12], mm, mm, tets_compact, dead_compact)

    # build_ball_csr_jit
    build_ball_csr_jit(tets_compact, 4)

    # point location
    adj_w = np.array([-1, 1, -1, -1, 0, -1, -1, -1], dtype=np.int32)
    goal_w = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    _bary_coords_tet_jit(a, b, c, d, goal_w, NOISE_FLOOR)
    traverse_jit(coords, tets_arr, adj_w, 0, goal_w, NOISE_FLOOR)
    brute_force_locate_jit(coords, tets_arr, goal_w, NOISE_FLOOR)

    # per-tet angle measures
    dihedral_angles_jit(coords, 0, 1, 2, 3)
    solid_angle_jit(coords, 0, 1, 2, 3, 0)
    tet_angle_measures_jit(coords, 0, 1, 2, 3)
    mesh_angle_extrema_jit(coords, np.array([0, 1, 2, 3], dtype=np.int64),
                           dead)
    one_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    swap_conserves_volume_jit(wc, dead, one_tet, one_tet, 1e-9)
    config_min_quality_jit(wc, wm, one_tet)
    swap_score_configs_jit(wc, dead, one_tet, True,
                           one_tet.reshape(1, 1, 4), 1e-9)

    t1 = time.perf_counter()
    logger.debug(f"  JIT warm-up: {t1 - t0:.1f}s")
