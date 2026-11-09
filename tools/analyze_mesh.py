r"""
analyze_mesh.py — Mesh diagnostic histogram tool for AMAC.

Loads a Netgen .vol mesh with a Medit .sol metric field and produces the
three-panel histogram figure of mesh quality diagnostics:

    1. Dihedral angle in metric space   (degrees)
    2. Dihedral angle in physical space (degrees)
    3. Edge length in metric space      (l_M)

plus a printed summary table (n_nodes, n_tets, conforming %, mean/min/max Q,
flat tets).

The metric can be read two ways:
  --metric <.sol>              read directly on --mesh (.sol must match --mesh's
                                node count/ordering).
  --metric <.sol> --bg-mesh <.vol>
                                the .sol was sampled on --bg-mesh (a coarser
                                background mesh); interpolated onto --mesh via
                                Log-Euclidean interpolation, reproducing the
                                adaptation pipeline exactly (run/adapt.py).
                                Use this to reproduce adaptation_history.json stats.

Usage:
    python tools/analyze_mesh.py --geo cases/cube_cross/input/geometry/cube.geo \
        --mesh cases/cube_cross/input/mesh/cube1241.vol \
        --metric cases/cube_cross/input/metric/cross.sol

Output: two PNG figures in the case output directory, or --output-dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from amac_3d.grid import Grid
from amac_3d.metric import length_in_metric
from amac_3d.swap import dihedral_angle
from amac_3d.cavity import collect_edges
from amac_3d.report import (
    compute_edge_lengths,
    compute_element_qualities,
    count_flat_tets,
)


# ──────────────── Metric-space geometry ────────────────────────────

def _spectral_sqrt(m6):
    """Spectral square root of a symmetric 3x3 metric packed as 6 floats."""
    M = np.array([
        [m6[0], m6[1], m6[2]],
        [m6[1], m6[3], m6[4]],
        [m6[2], m6[4], m6[5]],
    ])
    w, v = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return v @ np.diag(np.sqrt(w)) @ v.T


def _face_normal_np(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    mag = np.linalg.norm(n)
    if mag < 1e-15:
        return np.zeros(3)
    return n / mag


def _angle_between_np(u, v):
    mu = np.linalg.norm(u)
    mv = np.linalg.norm(v)
    if mu < 1e-15 or mv < 1e-15:
        return 0.0
    cos_theta = np.clip(np.dot(u, v) / (mu * mv), -1.0, 1.0)
    return float(np.arccos(cos_theta))


_FACE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_FACE_LOCAL = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]


def metric_space_dihedral_angles(grid) -> list:
    """All 6 dihedral angles (degrees) per active tet, in metric-transformed space."""
    coords = grid.coordinates
    tets = grid.tetrahedra
    metrics = grid.metricMesh
    n_tets = len(tets) // 4
    angles = []

    for t in range(n_tets):
        base = 4 * t
        verts = [tets[base + k] for k in range(4)]
        if verts[0] == -1:
            continue

        m_avg = [0.0] * 6
        for v in verts:
            m6 = metrics[v * 6: v * 6 + 6]
            for k in range(6):
                m_avg[k] += m6[k] / 4.0

        m_sqrt = _spectral_sqrt(m_avg)
        pts = np.array([coords[v * 3: v * 3 + 3] for v in verts])
        pts_t = pts @ m_sqrt.T

        normals = []
        for a, b, c in _FACE_LOCAL:
            normals.append(_face_normal_np(pts_t[a], pts_t[b], pts_t[c]))

        for i, j in _FACE_PAIRS:
            angles.append(np.degrees(_angle_between_np(normals[i], normals[j])))

    return angles


def physical_space_dihedral_angles(grid) -> list:
    """All 6 dihedral angles (degrees) per active tet, Euclidean."""
    tets = grid.tetrahedra
    n_tets = len(tets) // 4
    angles = []
    for t in range(n_tets):
        if tets[4 * t] == -1:
            continue
        angles.extend(np.degrees(a) for a in dihedral_angle(grid, t))
    return angles


def metric_space_edge_lengths(grid) -> list:
    """l_M for every unique edge in the mesh."""
    coords = grid.coordinates
    metrics = grid.metricMesh
    lengths = []
    for u, v in collect_edges(grid):
        p_u = coords[u * 3: u * 3 + 3]
        p_v = coords[v * 3: v * 3 + 3]
        m_u = metrics[u * 6: u * 6 + 6]
        m_v = metrics[v * 6: v * 6 + 6]
        lengths.append(length_in_metric(m_u, m_v, p_u, p_v))
    return lengths


# ──────────────── Plotting ──────────────────────────────────────────

_IDEAL_DIHEDRAL = np.degrees(np.arccos(1.0 / 3.0))  # 70.53° for regular tet
_CONFORMING_LO = 1.0 / np.sqrt(2.0)                 # 0.707
_CONFORMING_HI = np.sqrt(2.0)                        # 1.414


_ANGLE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
               100, 110, 120, 130, 140, 150, 160, 170, 180]

_LENGTH_BINS = [0.0, 0.25, 0.50, _CONFORMING_LO, 0.90, 1.10,
               _CONFORMING_HI, 2.0, 3.0, 5.0]


def _pct_bar(ax, data, edges, ideal_lo, ideal_hi, color_normal, color_ideal,
             xlabel, ylabel="% of Total"):
    arr = np.array(data)
    counts = np.array([np.sum((arr >= edges[i]) & (arr < edges[i + 1]))
                       for i in range(len(edges) - 1)])
    counts[-1] += np.sum(arr >= edges[-1])
    pcts = 100.0 * counts / max(len(arr), 1)

    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == int(lo) and hi == int(hi):
            labels.append(f"[{int(lo)},{int(hi)})")
        else:
            labels.append(f"[{lo:.2f},{hi:.2f})")

    colors = [color_ideal if edges[i] >= ideal_lo and edges[i + 1] <= ideal_hi
              else color_normal for i in range(len(edges) - 1)]

    bars = ax.bar(range(len(pcts)), pcts, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    y_top = max(pcts) * 1.15 if max(pcts) > 0 else 1.0
    ax.set_ylim(0, y_top)
    for bar, p in zip(bars, pcts):
        if p >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{p:.1f}%", ha="center", va="bottom", fontsize=13,
                    fontweight="bold")


def plot_histograms(angles_metric, angles_physical, lengths, title, output, bins):
    base, ext = os.path.splitext(output)
    ideal_color = "#2ca02c"
    ideal_alpha = 0.15
    len_max = min(max(lengths) * 1.05, 4.0)

    # ── Image 1: density histograms ──
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(42, 14))

    ax1.axvspan(50, 90, color=ideal_color,
                alpha=ideal_alpha, label=f"Ideal region (50°–90°)")
    ax1.axvline(_IDEAL_DIHEDRAL, color=ideal_color, ls="--", lw=1.2)
    ax1.hist(angles_metric, bins=bins, range=(0, 180), density=True,
             color="tab:blue", edgecolor="black", linewidth=0.6,
             label="Edge Primitive")
    ax1.set_xlabel("Dihedral Angle in Metric Space (deg)", fontsize=16)
    ax1.set_ylabel("Normalized Count", fontsize=16)
    ax1.set_xlim(0, 180)
    ax1.set_xticks(np.arange(0, 181, 10))
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize=15)

    ax2.axvspan(50, 90, color=ideal_color,
                alpha=ideal_alpha, label=f"Ideal region (50°–90°)")
    ax2.axvline(_IDEAL_DIHEDRAL, color=ideal_color, ls="--", lw=1.2)
    ax2.hist(angles_physical, bins=bins, range=(0, 180), density=True,
             color="tab:blue", edgecolor="black", linewidth=0.6,
             label="Edge Primitive")
    ax2.set_xlabel("Dihedral Angle in Physical Space (deg)", fontsize=16)
    ax2.set_ylabel("Normalized Count", fontsize=16)
    ax2.set_xlim(0, 180)
    ax2.set_xticks(np.arange(0, 181, 10))
    ax2.tick_params(labelsize=14)
    ax2.legend(fontsize=15)

    ax3.axvspan(_CONFORMING_LO, _CONFORMING_HI, color=ideal_color,
                alpha=ideal_alpha,
                label=f"Conforming [{_CONFORMING_LO:.3f}, {_CONFORMING_HI:.3f}]")
    ax3.hist(lengths, bins=bins, range=(0, len_max), density=True,
             color="tab:blue", edgecolor="black", linewidth=0.6,
             label="Edge Primitive")
    ax3.set_xlabel("Edge Length in Metric Space (ℓ_M)", fontsize=16)
    ax3.set_ylabel("Normalized Count", fontsize=16)
    ax3.set_xlim(0, len_max)
    ax3.set_xticks(np.arange(0, len_max + 0.1, 0.2))
    ax3.tick_params(labelsize=14)
    ax3.legend(fontsize=15)

    hist_path = f"{base}_hist{ext}"
    fig1.suptitle(title, fontsize=20)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Image 2: percentage-in-group bar charts ──
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(42, 14))

    _pct_bar(ax4, angles_metric, _ANGLE_BINS,
             50, 90,
             "tab:orange", ideal_color,
             "Dihedral Angle in Metric Space (deg)")

    _pct_bar(ax5, angles_physical, _ANGLE_BINS,
             50, 90,
             "tab:orange", ideal_color,
             "Dihedral Angle in Physical Space (deg)")

    len_edges = [e for e in _LENGTH_BINS if e <= len_max]
    if len_edges[-1] < len_max:
        len_edges.append(len_max)
    _pct_bar(ax6, lengths, len_edges,
             _CONFORMING_LO, _CONFORMING_HI,
             "tab:orange", ideal_color,
             "Edge Length in Metric Space (ℓ_M)")

    pct_path = f"{base}_pct{ext}"
    fig2.suptitle(f"{title} — Distribution by Group", fontsize=20)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(pct_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return hist_path, pct_path


# ──────────────── Summary stats ─────────────────────────────────────

def print_summary(grid):
    n_nodes = len(grid.coordinates) // 3
    n_tets = len(grid.tetrahedra) // 4
    edge_stats = compute_edge_lengths(grid)
    quality_stats = compute_element_qualities(grid)
    n_flat = count_flat_tets(grid)

    print("──────── Summary ────────")
    print(f"  n_nodes:            {n_nodes}")
    print(f"  n_tets:             {n_tets}")
    print(f"  conforming edges:   {edge_stats['fraction_conforming']:.4%}")
    print(f"  edge length mean:   {edge_stats['mean']:.4f}")
    print(f"  quality mean:       {quality_stats['mean']:.4f}")
    print(f"  quality min/max:    {quality_stats['min']:.4e} / {quality_stats['max']:.4f}")
    print(f"  flat tets:          {n_flat}")
    print("──────────────────────────")


# ──────────────── CLI ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geo", required=True, help=".geo geometry file")
    ap.add_argument("--mesh", required=True, help=".vol mesh file")
    ap.add_argument("--metric", required=True, help=".sol metric file")
    ap.add_argument("--bg-mesh",
                     help="background/prev .vol mesh that --metric's .sol was sampled on "
                          "(reproduces the adaptation pipeline's Log-Euclidean interpolation "
                          "via Grid.metric, instead of a pointwise --metric read on --mesh itself)")
    ap.add_argument("--output", help="output PNG file name (default: mesh stem)")
    ap.add_argument("--output-dir", help="output directory (default: case output/ or mesh directory; input/ refused)")
    ap.add_argument("--title", help="figure suptitle")
    ap.add_argument("--bins", type=int, default=60, help="number of histogram bins")
    args = ap.parse_args()

    mesh_path = Path(args.mesh).expanduser().resolve()
    if args.output_dir is None:
        input_dir = next((part for part in mesh_path.parents
                          if part.name == "input"), None)
        output_dir = input_dir.parent / "output" if input_dir else mesh_path.parent
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir = output_dir.resolve()
    if "input" in output_dir.parts:
        ap.error(f"refused output directory inside input/: {output_dir}")
    if args.output is not None and Path(args.output).name != args.output:
        ap.error("--output must be a file name; use --output-dir for its directory")

    metric_path = None
    metric_function = None
    if args.bg_mesh:
        # Reproduce the adaptation pipeline's background metric interpolation:
        # the .sol is sampled on the coarse background mesh, and every node of the
        # target mesh gets its metric via Log-Euclidean interpolation inside the
        # background mesh's tets (Grid.metric -> search.locate + edge_metric_interpolate).
        bg_grid = Grid.from_mesh(args.geo, args.bg_mesh, metric_path=args.metric)
        metric_function = bg_grid.metric
    else:
        metric_path = args.metric

    grid = Grid.from_mesh(args.geo, args.mesh, metric_path=metric_path,
                           metric_function=metric_function)

    angles_metric = metric_space_dihedral_angles(grid)
    angles_physical = physical_space_dihedral_angles(grid)
    lengths = metric_space_edge_lengths(grid)

    print_summary(grid)

    mesh_stem = os.path.splitext(os.path.basename(args.mesh))[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = str(output_dir / (args.output or f"{mesh_stem}_histograms.png"))
    title = args.title or mesh_stem

    hist_path, pct_path = plot_histograms(angles_metric, angles_physical,
                                          lengths, title, output, args.bins)
    print(f"Saved density histograms:  {hist_path}")
    print(f"Saved percentage breakdown: {pct_path}")


if __name__ == "__main__":
    main()
