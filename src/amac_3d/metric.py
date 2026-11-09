"""Compute metric lengths, quality, geometry, and tensor interpolation.

Metrics are positive-definite tensors stored as six components. Interpolation
uses the Log-Euclidean form; signed volume is the orientation oracle.
References: Michal & Krakos (2012); Alauzet (2010); Knupp (2001); Park (2018).
"""

from __future__ import annotations

import math
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .grid import Grid

# ──────────────────────────── Constants ─────────────────────────────

EIG_FLOOR: float = 1e-300
NOISE_FLOOR: float = 1e-15
TAYLOR_THRESHOLD: float = 1e-6
QUALITY_CONSTANT: float = 36.0 / (3.0 ** (1.0 / 3.0))
FACE_QUALITY_CONSTANT: float = 4.0 / (3.0 ** 0.5)
SPLIT_THRESHOLD: float = math.sqrt(2.0)
COLLAPSE_THRESHOLD: float = 1.0 / math.sqrt(2.0)


# ──────────────────── Vector / Tensor Helpers ───────────────────────

def dot_product(u: List[float], v: List[float]) -> float:
    """Dot product of two 3-vectors."""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross_product(u: List[float], v: List[float]) -> List[float]:
    """Cross product of two 3-vectors."""
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def norm_of_vec(v: List[float]) -> float:
    """Euclidean norm of a 3-vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def unit_vec(v: List[float]) -> List[float]:
    """Normalised direction vector."""
    n = norm_of_vec(v)
    if n < NOISE_FLOOR:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def face_normal(p0: List[float], p1: List[float],
                p2: List[float]) -> List[float]:
    """Unit outward normal of triangle (p0, p1, p2)."""
    v1 = [p1[i] - p0[i] for i in range(3)]
    v2 = [p2[i] - p0[i] for i in range(3)]
    n = cross_product(v1, v2)
    mag = norm_of_vec(n)
    if mag < NOISE_FLOOR:
        return [0.0, 0.0, 0.0]
    return [n[0] / mag, n[1] / mag, n[2] / mag]


# ───────────────── Symmetric 3×3 Metric Utilities ──────────────────
# Packed format: [M11, M12, M13, M22, M23, M33]
#   |M11 M12 M13|
#   |M12 M22 M23|
#   |M13 M23 M33|

def det_metric(m: List[float]) -> float:
    """Determinant of a symmetric 3×3 metric stored as 6 floats."""
    from .kernels import _det_metric as _dm_jit
    return _dm_jit(np.asarray(m, dtype=np.float64))


# ──────────────── Quadratic Form: γ'ᵀ M γ' ────────────────────────

def gamma_m_gamma(gamma: List[float], metric: List[float]) -> float:
    """
    Compute √(γ'ᵀ M γ') where M is a symmetric 3×3 metric
    and γ' = b − a is a difference vector.

    Applies a noise floor of 1e-15 before the square-root to prevent
    negative values due to floating-point arithmetic.

    Reference: Michal & Krakos (2012), Eq. (3).
    """
    from .kernels import _gamma_m_gamma as _gmg_jit
    return _gmg_jit(np.asarray(gamma, dtype=np.float64), np.asarray(metric, dtype=np.float64))


# ──────────── Continuous Edge Length in Metric ──────────────────────

def length_in_metric(m1: List[float], m2: List[float],
                     p1: List[float], p2: List[float]) -> float:
    """
    Compute the continuous metric edge length between *p1* and *p2*
    using the metric tensors *m1* (at p1) and *m2* (at p2).

    Formula (geometric):
        ℓ_eff = ℓ₁ · (α − 1) / (α · ln α)
        where α = ℓ₁ / ℓ₂,  ℓ₁ = max(ℓ_a, ℓ_b),  ℓ₂ = min(ℓ_a, ℓ_b)

    When |α − 1| < 1e-6 a rational Taylor approximation is used:
        ℓ_eff = ℓ₁ · (2/3) · (α² + α + 1) / (α(α + 1))

    Reference: Michal & Krakos (2012), §4.
    """
    from .kernels import _length_in_metric as _lim_jit
    return _lim_jit(np.asarray(m1, dtype=np.float64), np.asarray(m2, dtype=np.float64),
                    np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64))


def length_in_node_metric(node_metric: List[float],
                          seg_start: List[float],
                          seg_end: List[float]) -> float:
    """Edge length measured in a single node's metric tensor."""
    if seg_start[0] == seg_end[0] and seg_start[1] == seg_end[1] and seg_start[2] == seg_end[2]:
        return 0.0
    gamma = [seg_end[i] - seg_start[i] for i in range(3)]
    return gamma_m_gamma(gamma, node_metric)


# ──────────────── Log / Exp of Symmetric Metric ────────────────────

def log_metric(m6: List[float]) -> List[float]:
    """
    Matrix logarithm of a 3×3 symmetric positive-definite metric
    stored as 6 floats.  Uses the spectral decomposition identity
    V diag(log λ) Vᵀ, which is exact for real symmetric SPD matrices.

    Near-singular metrics are handled via EIG_FLOOR clamping on
    eigenvalues before the log.

    Returns 6 real floats (upper-triangular packed).

    Reference: Alauzet (2010), Eq. (12).
    """
    a, b, c, d, e, f = m6
    M = np.array([[a, b, c],
                   [b, d, e],
                   [c, e, f]])
    w, V = np.linalg.eigh(M)
    L = (V * np.log(np.maximum(w, EIG_FLOOR))) @ V.T
    return [float(L[0, 0]), float(L[0, 1]), float(L[0, 2]),
            float(L[1, 1]), float(L[1, 2]), float(L[2, 2])]


def log_metric_batch(m6_flat: np.ndarray) -> np.ndarray:
    """Batched matrix logarithm over an (N, 6) array of packed SPD metrics.

    Equivalent to calling ``log_metric`` on each row, but performed as a
    single stacked ``np.linalg.eigh`` for ~10× throughput.  Returns an
    (N, 6) array of packed log-metrics.
    """
    N = m6_flat.shape[0]
    M = np.empty((N, 3, 3), dtype=np.float64)
    M[:, 0, 0] = m6_flat[:, 0]
    M[:, 0, 1] = m6_flat[:, 1]
    M[:, 0, 2] = m6_flat[:, 2]
    M[:, 1, 0] = m6_flat[:, 1]
    M[:, 1, 1] = m6_flat[:, 3]
    M[:, 1, 2] = m6_flat[:, 4]
    M[:, 2, 0] = m6_flat[:, 2]
    M[:, 2, 1] = m6_flat[:, 4]
    M[:, 2, 2] = m6_flat[:, 5]

    w, V = np.linalg.eigh(M)
    log_w = np.log(np.maximum(w, EIG_FLOOR))
    L = (V * log_w[:, np.newaxis, :]) @ np.swapaxes(V, -2, -1)

    out = np.empty((N, 6), dtype=np.float64)
    out[:, 0] = L[:, 0, 0]
    out[:, 1] = L[:, 0, 1]
    out[:, 2] = L[:, 0, 2]
    out[:, 3] = L[:, 1, 1]
    out[:, 4] = L[:, 1, 2]
    out[:, 5] = L[:, 2, 2]
    return out


def exp_metric(m6: List[float]) -> List[float]:
    """
    Matrix exponential of a 3×3 symmetric log-metric via spectral
    decomposition: V diag(exp λ) Vᵀ.

    Reference: Alauzet (2010), Eq. (13).
    """
    a, b, c, d, e, f = m6
    M = np.array([[a, b, c],
                   [b, d, e],
                   [c, e, f]])
    w, V = np.linalg.eigh(M)
    E = (V * np.exp(w)) @ V.T
    return [float(E[0, 0]), float(E[0, 1]), float(E[0, 2]),
            float(E[1, 1]), float(E[1, 2]), float(E[2, 2])]


def exp_metric_batch(m6_flat: np.ndarray) -> np.ndarray:
    """Batched matrix exponential over an (N, 6) array of packed symmetric
    log-metrics; the stacked counterpart of ``exp_metric``."""
    N = m6_flat.shape[0]
    M = np.empty((N, 3, 3), dtype=np.float64)
    M[:, 0, 0] = m6_flat[:, 0]
    M[:, 0, 1] = m6_flat[:, 1]
    M[:, 0, 2] = m6_flat[:, 2]
    M[:, 1, 0] = m6_flat[:, 1]
    M[:, 1, 1] = m6_flat[:, 3]
    M[:, 1, 2] = m6_flat[:, 4]
    M[:, 2, 0] = m6_flat[:, 2]
    M[:, 2, 1] = m6_flat[:, 4]
    M[:, 2, 2] = m6_flat[:, 5]

    w, V = np.linalg.eigh(M)
    E = (V * np.exp(w)[:, np.newaxis, :]) @ np.swapaxes(V, -2, -1)

    out = np.empty((N, 6), dtype=np.float64)
    out[:, 0] = E[:, 0, 0]
    out[:, 1] = E[:, 0, 1]
    out[:, 2] = E[:, 0, 2]
    out[:, 3] = E[:, 1, 1]
    out[:, 4] = E[:, 1, 2]
    out[:, 5] = E[:, 2, 2]
    return out


# ──────────────── Log-Euclidean Interpolation ──────────────────────

def edge_metric_interpolate(grid: 'Grid', p1: int, p2: int,
                            point: List[float]) -> List[float]:
    """
    Interpolate the metric at *point* along edge (p1, p2) using
    Log-Euclidean weighting based on Euclidean arc-length ratio.

    M(x) = exp( (1-t)·log(M₁) + t·log(M₂) )

    Reference: Alauzet (2010), §3.
    """
    coords = grid.coordinates
    ml = grid.metricLog

    s = [coords[p1 * 3], coords[p1 * 3 + 1], coords[p1 * 3 + 2]]
    e = [coords[p2 * 3], coords[p2 * 3 + 1], coords[p2 * 3 + 2]]

    # Euclidean parameter along the edge
    dist_se_sq = sum((e[i] - s[i]) ** 2 for i in range(3))
    if dist_se_sq < NOISE_FLOOR:
        return [grid.metricMesh[p1 * 6 + k] for k in range(6)]

    dist_sp_sq = sum((point[i] - s[i]) ** 2 for i in range(3))
    t_end = math.sqrt(dist_sp_sq / dist_se_sq)
    t_start = 1.0 - t_end

    lm_s = ml[p1 * 6: p1 * 6 + 6]
    lm_e = ml[p2 * 6: p2 * 6 + 6]

    blended = [t_start * lm_s[k] + t_end * lm_e[k] for k in range(6)]
    return exp_metric(blended)


# ──────────────────── Volume Computations ──────────────────────────

def signed_volume(a: List[float], b: List[float],
                  c: List[float], d: List[float]) -> float:
    """
    Signed volume of tetrahedron (a, b, c, d).

    Returns Ki / 6  **without** taking abs().
    Positive ⟹ correct orientation (CCW face normal outward).
    Negative ⟹ inverted element.

    The sign lets split, collapse, and swap reject inverted tets.
    """
    from .kernels import _signed_volume as _sv_jit
    return _sv_jit(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                   np.asarray(c, dtype=np.float64), np.asarray(d, dtype=np.float64))


def volume_abs(a: List[float], b: List[float],
               c: List[float], d: List[float]) -> float:
    """Absolute volume (always ≥ 0).  Convenience wrapper."""
    return abs(signed_volume(a, b, c, d))


def area_of_triangle(a: List[float], b: List[float],
                     c: List[float]) -> float:
    """Area of triangle (a, b, c) via cross-product magnitude / 2."""
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    cp = cross_product(ab, ac)
    return 0.5 * norm_of_vec(cp)


# ──────────── Barycentric Coordinates ──────────────────────────────

def bary_coords_tet(p0: List[float], p1: List[float],
                    p2: List[float], p3: List[float],
                    point: List[float]) -> List[float]:
    """
    Barycentric coordinates of *point* w.r.t. tetrahedron (p0..p3).
    Each λ_i is the sub-tet volume ratio.

    Returns [λ0, λ1, λ2, λ3].
    """
    v_full = volume_abs(p0, p1, p2, p3)
    if v_full < NOISE_FLOOR:
        return [0.25, 0.25, 0.25, 0.25]

    v0 = volume_abs(p1, p2, p3, point)
    v1 = volume_abs(p0, p2, p3, point)
    v2 = volume_abs(p0, p1, p3, point)
    v3 = volume_abs(p0, p1, p2, point)

    bary = [v0 / v_full, v1 / v_full, v2 / v_full, v3 / v_full]
    return [0.0 if abs(v) < 1e-12 else v for v in bary]


# ──────────── Element Quality Measures ─────────────────────────────

def tet_quality(a: List[float], b: List[float],
                c: List[float], d: List[float],
                metrics: List[List[float]],
                dets: List[float]) -> float:
    """
    Anisotropic tetrahedral quality:

        Q = C · V₀^(2/3) / Σ ℓ²_M

    where V₀ = √min(det M_j) · |V|,
          C  = 36 / ∛3.

    Returns 0.0 for degenerate or inverted elements.

    Reference: Knupp (2001); Michal & Krakos (2012), Eq. (11).
    """
    from .kernels import _tet_quality as _tq_jit
    return _tq_jit(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                   np.asarray(c, dtype=np.float64), np.asarray(d, dtype=np.float64),
                   np.asarray(metrics, dtype=np.float64),
                   np.asarray(dets, dtype=np.float64))


def face_quality(a: List[float], b: List[float],
                 c: List[float],
                 metrics: List[List[float]]) -> float:
    """
    Anisotropic triangle quality:

        Q = (4/√3) · 3 · A₀ / Σ ℓ²_M

    where A₀ = det(M)^(1/3) · A.
    """
    A = area_of_triangle(a, b, c)
    if A <= 0.0:
        return 0.0

    min_d = min(det_metric(m) for m in metrics)
    if min_d <= 1e-14:
        min_d = 1e-14

    A0 = (min_d ** (1.0 / 3.0)) * A

    L01 = length_in_metric(metrics[0], metrics[1], a, b)
    L02 = length_in_metric(metrics[0], metrics[2], a, c)
    L12 = length_in_metric(metrics[1], metrics[2], b, c)

    sum_sq = L01 * L01 + L02 * L02 + L12 * L12
    if sum_sq < NOISE_FLOOR:
        return 0.0

    return FACE_QUALITY_CONSTANT * 3.0 * A0 / sum_sq


def metric_bilinear(u: List[float], v: List[float],
                    M: List[float]) -> float:
    """Symmetric bilinear form u^T M v for 3-vectors and 6-component SPD M."""
    return (u[0] * (M[0] * v[0] + M[1] * v[1] + M[2] * v[2])
            + u[1] * (M[1] * v[0] + M[3] * v[1] + M[4] * v[2])
            + u[2] * (M[2] * v[0] + M[4] * v[1] + M[5] * v[2]))


def marcum_alauzet_edge_weight(a: List[float], b: List[float],
                               c: List[float], d: List[float],
                               metrics: List[List[float]],
                               det_mk: float,
                               mk: List[float] = None) -> float:
    """Marcum-Alauzet minimum edge weight Q_M(K) = max_i W_i.

    Marcum & Alauzet (2020), §2.3.2, p. 5.
    Higher = worse (degenerate); perfect regular tet with M=I → -√2/4.

    When *mk* is supplied the 4 log_metric + 1 exp_metric calls are
    skipped — the caller has already computed the element mean metric.
    """
    six_V = 6.0 * signed_volume(a, b, c, d)
    if abs(six_V) < 1e-30:
        return float('inf')

    if mk is None:
        log_a = log_metric(metrics[0])
        log_b = log_metric(metrics[1])
        log_c = log_metric(metrics[2])
        log_d = log_metric(metrics[3])
        mk = exp_metric([(log_a[i] + log_b[i] + log_c[i] + log_d[i]) / 4.0
                          for i in range(6)])

    denom = math.sqrt(abs(det_mk)) * abs(six_V)

    ea = [b[i] - a[i] for i in range(3)]
    eb = [c[i] - a[i] for i in range(3)]
    ec = [d[i] - a[i] for i in range(3)]
    ed = [c[i] - b[i] for i in range(3)]
    ef = [d[i] - b[i] for i in range(3)]
    eg = [d[i] - c[i] for i in range(3)]
    neg_ea = [-ea[i] for i in range(3)]
    neg_eb = [-eb[i] for i in range(3)]
    neg_ed = [-ed[i] for i in range(3)]

    triples = [
        (ea, eb, ec),
        (eb, ea, ec),
        (ec, ea, eb),
        (ed, neg_ea, ef),
        (ef, neg_ea, ed),
        (eg, neg_eb, neg_ed),
    ]

    w_max = -float('inf')
    for s, j, k in triples:
        sj = metric_bilinear(s, j, mk)
        sk = metric_bilinear(s, k, mk)
        ss = metric_bilinear(s, s, mk)
        jk = metric_bilinear(j, k, mk)
        w = (sj * sk - ss * jk) / denom
        if w > w_max:
            w_max = w
    return w_max


def check_loseille_lohner_gate(q_old: float, q_new: float,
                               q_r: float, q_a: float) -> bool:
    """Loseille-Löhner acceptance: q_new >= q_old/q_r AND q_new >= q_a.

    q_r >= 1.0 (relative decay tolerance; 1.0 = strict improvement).
    q_a       absolute floor (max(1e-3, current worst mesh quality)).
    """
    return q_new >= (q_old / q_r) and q_new >= q_a
