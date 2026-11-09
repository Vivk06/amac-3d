"""Measure mesh quality and record adaptation history without mutating the grid.

Compute metric edge-length and element-quality distributions, flat-tet counts,
and node classifications. AdaptationTracker writes iteration records and an
advisory convergence warning; adapt controls termination.
"""

from __future__ import annotations

import math
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid

from .metric import length_in_metric, volume_abs, NOISE_FLOOR, exp_metric_batch
from .kernels import (tet_qualities_jit, ma_weights_jit, complexity_jit,
                      flat_tet_mask_jit)
from .cavity import collect_edges

logger = logging.getLogger(__name__)


# ── Serial diagnostic implementations ─────────────────────────────

def _compute_edge_lengths_serial(grid, edges=None):
    if edges is None:
        edges = collect_edges(grid)
    lengths = []

    for u, v in edges:
        m1 = grid.metricMesh[u * 6: u * 6 + 6]
        m2 = grid.metricMesh[v * 6: v * 6 + 6]
        p1 = grid.coordinates[u * 3: u * 3 + 3]
        p2 = grid.coordinates[v * 3: v * 3 + 3]
        l_eff = length_in_metric(m1, m2, p1, p2)
        lengths.append(l_eff)

    return _build_edge_length_result(lengths)


def _compute_element_qualities_serial(grid):
    tets = grid.tetrahedra
    qualities = tet_qualities_jit(grid.coordinates, tets,
                                  grid.metricMesh).tolist()

    live = np.flatnonzero(tets[0::4] != -1)
    verts = tets.reshape(-1, 4)[live]
    ml = grid.metricLog.reshape(-1, 6)
    mk_log = (ml[verts[:, 0]] + ml[verts[:, 1]]
              + ml[verts[:, 2]] + ml[verts[:, 3]]) / 4.0
    mk_all = exp_metric_batch(mk_log)
    w = ma_weights_jit(grid.coordinates, tets, live, mk_all)
    ma_weights = list(w[~np.isinf(w)])

    return _build_quality_result(qualities, ma_weights)


# ── Result builders (shared by serial and parallel paths) ───────────

def _build_edge_length_result(lengths):
    n = len(lengths)
    if n == 0:
        return {
            "values": [],
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "histogram": {},
            "fraction_short": 0.0,
            "fraction_conforming": 0.0,
            "fraction_long": 0.0,
        }

    lengths_arr = np.array(lengths)
    val_min = float(np.min(lengths_arr))
    val_max = float(np.max(lengths_arr))
    val_mean = float(np.mean(lengths_arr))
    val_std = float(np.std(lengths_arr))

    bins_definition = [
        ("[0.000, 0.250)", lambda x: 0.0 <= x < 0.25),
        ("[0.250, 0.500)", lambda x: 0.25 <= x < 0.50),
        ("[0.500, 0.707)", lambda x: 0.50 <= x < 0.7071067811865475),
        ("[0.707, 0.900)", lambda x: 0.7071067811865475 <= x < 0.90),
        ("[0.900, 1.100)", lambda x: 0.90 <= x < 1.10),
        ("[1.100, 1.414)", lambda x: 1.10 <= x < 1.4142135623730951),
        ("[1.414, 2.000)", lambda x: 1.4142135623730951 <= x < 2.00),
        ("[2.000, 3.000)", lambda x: 2.00 <= x < 3.00),
        ("[3.000, 5.000)", lambda x: 3.00 <= x < 5.00),
        ("[5.000,   inf)", lambda x: x >= 5.00),
    ]

    hist_counts = {name: 0 for name, _ in bins_definition}
    for val in lengths:
        for name, cond in bins_definition:
            if cond(val):
                hist_counts[name] += 1
                break

    collapse_limit = 1.0 / math.sqrt(2.0)
    split_limit = math.sqrt(2.0)

    short_count = sum(1 for x in lengths if x < collapse_limit)
    long_count = sum(1 for x in lengths if x > split_limit)
    conforming_count = n - short_count - long_count

    return {
        "values": lengths,
        "count": n,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
        "histogram": hist_counts,
        "fraction_short": float(short_count / n),
        "fraction_conforming": float(conforming_count / n),
        "fraction_long": float(long_count / n),
    }


def _build_quality_result(qualities, ma_weights):
    inf = float('inf')
    n = len(qualities)
    if n == 0:
        return {
            "values": [],
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "histogram": {},
            "fraction_below_0001": 0.0,
            "fraction_above_05": 0.0,
            "marcum_alauzet_max": inf,
            "marcum_alauzet_mean": inf,
        }

    qualities_arr = np.array(qualities)
    val_min = float(np.min(qualities_arr))
    val_max = float(np.max(qualities_arr))
    val_mean = float(np.mean(qualities_arr))
    val_std = float(np.std(qualities_arr))

    bins_definition = [
        ("[0.00, 0.01)", lambda x: 0.0 <= x < 0.01),
        ("[0.01, 0.10)", lambda x: 0.01 <= x < 0.10),
        ("[0.10, 0.20)", lambda x: 0.10 <= x < 0.20),
        ("[0.20, 0.30)", lambda x: 0.20 <= x < 0.30),
        ("[0.30, 0.40)", lambda x: 0.30 <= x < 0.40),
        ("[0.40, 0.50)", lambda x: 0.40 <= x < 0.50),
        ("[0.50, 0.60)", lambda x: 0.50 <= x < 0.60),
        ("[0.60, 0.70)", lambda x: 0.60 <= x < 0.70),
        ("[0.70, 0.80)", lambda x: 0.70 <= x < 0.80),
        ("[0.80, 0.90)", lambda x: 0.80 <= x < 0.90),
        ("[0.90, 1.00]", lambda x: x >= 0.90),
    ]

    hist_counts = {name: 0 for name, _ in bins_definition}
    for val in qualities:
        for name, cond in bins_definition:
            if cond(val):
                hist_counts[name] += 1
                break

    below_0001 = sum(1 for x in qualities if x < 0.001)
    above_05 = sum(1 for x in qualities if x >= 0.5)

    if ma_weights:
        ma_max = max(ma_weights)
        ma_mean = sum(ma_weights) / len(ma_weights)
    else:
        ma_max = inf
        ma_mean = inf

    return {
        "values": qualities,
        "count": n,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
        "histogram": hist_counts,
        "fraction_below_0001": float(below_0001 / n),
        "fraction_above_05": float(above_05 / n),
        "marcum_alauzet_max": ma_max,
        "marcum_alauzet_mean": ma_mean,
    }


# ── Public dispatchers ──────────────────────────────────────────────

def compute_edge_lengths(grid: Grid) -> dict:
    """
    Collect all unique edges, compute their continuous metric lengths,
    and bin them into 10 categories with statistical summary.

    Results are memoised on ``grid._mesh_version`` — a single cached entry
    is overwritten on each recomputation, so memory is bounded.
    """
    cache = grid._diag_cache
    ver = grid._mesh_version
    cached = cache.get('edge_lengths')
    if cached is not None and cached[0] == ver:
        return cached[1]

    result = _compute_edge_lengths_serial(grid)

    cache['edge_lengths'] = (ver, result)
    return result


def compute_element_qualities(grid: Grid) -> dict:
    """
    Compute tet quality for all active tets using normalized determinant.
    Bins qualities in [0, 1] into 11 categories with statistical summary.
    Also computes Marcum-Alauzet edge-weight stats per element.

    Results are memoised on ``grid._mesh_version`` — a single cached entry
    is overwritten on each recomputation, so memory is bounded.
    """
    cache = grid._diag_cache
    ver = grid._mesh_version
    cached = cache.get('element_qualities')
    if cached is not None and cached[0] == ver:
        return cached[1]

    result = _compute_element_qualities_serial(grid)

    cache['element_qualities'] = (ver, result)
    return result


def min_element_quality(grid: Grid) -> float:
    """Return the minimum positive Knupp tet quality, skipping Marcum-Alauzet.

    This is the fast path for ``_get_worst_mesh_quality`` in ``adapt.py``,
    which only needs min(q); it skips the Marcum-Alauzet stack that
    ``compute_element_qualities`` evaluates on every tet.
    """
    qualities = tet_qualities_jit(grid.coordinates, grid.tetrahedra,
                                  grid.metricMesh)
    worst = float('inf')

    for q in qualities.tolist():
        if 0.0 < q < worst:
            worst = q

    return worst if worst < float('inf') else 0.0


def compute_complexity(grid: Grid) -> float:
    """
    Continuous-mesh complexity C = Σ_tets Σ_nodes √det(M_node) · V/4
    (NASA Refine ``ref_adapt_parameter``).

    C is the node budget the metric field implies: a metric-conforming
    quasi-unit mesh has nodes/C ≈ O(1). Refine throttles splitting when
    nodes/C > 3 (node-count overshoot); here it is reported per iteration
    so over/under-resolution relative to the metric is visible.
    """
    return complexity_jit(grid.coordinates, grid.tetrahedra,
                          grid.metricMesh, grid.dead_mask)


def tet_is_flat(grid: Grid, t: int) -> bool:
    """
    Flatness predicate for a single tet index: normalised determinant
    det / max(edge_length²) ≤ 1e-15. Dead tets and tets with dead
    vertices are not flat.
    """
    coords = grid.coordinates
    tets = grid.tetrahedra
    EPSILON = 1e-15

    i = 4 * t
    a_idx, b_idx, c_idx, d_idx = tets[i], tets[i + 1], tets[i + 2], tets[i + 3]
    if a_idx == -1:
        return False

    if grid.dead_mask[a_idx] or grid.dead_mask[b_idx] or grid.dead_mask[c_idx] or grid.dead_mask[d_idx]:
        return False

    a = coords[a_idx * 3: a_idx * 3 + 3]
    b = coords[b_idx * 3: b_idx * 3 + 3]
    c = coords[c_idx * 3: c_idx * 3 + 3]
    d = coords[d_idx * 3: d_idx * 3 + 3]

    vol = volume_abs(a, b, c, d)
    det_val = 6.0 * vol

    edges = [
        sum((a[k] - b[k]) ** 2 for k in range(3)),
        sum((b[k] - c[k]) ** 2 for k in range(3)),
        sum((c[k] - a[k]) ** 2 for k in range(3)),
        sum((a[k] - d[k]) ** 2 for k in range(3)),
        sum((b[k] - d[k]) ** 2 for k in range(3)),
        sum((c[k] - d[k]) ** 2 for k in range(3)),
    ]
    max_edge_sq = max(edges)
    if max_edge_sq < NOISE_FLOOR:
        return True

    return det_val / max_edge_sq <= EPSILON


def flat_tet_indices(grid: Grid) -> list:
    """
    Indices of all flat (nearly co-planar) tetrahedra, for targeted repair.
    """
    return np.flatnonzero(flat_tet_mask_jit(
        grid.coordinates, grid.tetrahedra, grid.dead_mask)).tolist()


def count_flat_tets(grid: Grid) -> int:
    """
    Check for flat (nearly co-planar) tetrahedra without raising.
    """
    return len(flat_tet_indices(grid))


def count_node_classifications(grid: Grid) -> dict:
    """
    Count points in each of the 6 boundary classification levels.
    """
    interior = 0
    surface = 0
    ridge = 0
    corner = 0
    other = 0
    
    for lv in grid.pointLevels:
        if lv == 0:
            interior += 1
        elif lv == 1:
            surface += 1
        elif lv == 2:
            ridge += 1
        elif lv == 3:
            corner += 1
        else:
            other += 1
            
    return {
        "interior": interior,
        "surface": surface,
        "ridge": ridge,
        "corner": corner,
        "other": other,
    }


def print_histogram(histogram: Dict[str, int], title: str, mark_out_of_bounds: bool = False) -> None:
    """
    Prints a textual representation of a histogram.
    If mark_out_of_bounds is True, bins outside [0.707, 1.414] will be flagged with a '*'.
    """
    logger.debug(f"\n  Distribution: {title}")
    
    # List of edge bins that are non-conforming
    non_conforming_bins = {
        "[0.000, 0.250)",
        "[0.250, 0.500)",
        "[0.500, 0.707)",
        "[1.414, 2.000)",
        "[2.000, 3.000)",
        "[3.000, 5.000)",
        "[5.000,   inf)"
    }
    
    for bin_name, count in histogram.items():
        flag = ""
        if mark_out_of_bounds and bin_name in non_conforming_bins:
            flag = " *" if count > 0 else ""
        logger.debug(f"    {bin_name:<16}: {count:10d}{flag}")


@dataclass
class IterationRecord:
    """Per-iteration adaptation statistics captured by ``AdaptationTracker``.

    Holds the node/tet counts, operation counts (splits, collapses, swaps,
    smooth moves), flat-tet count, the node-classification breakdown, the
    edge-length and quality histogram-stat dicts, and the wall time for a
    single outer adaptation iteration.
    """
    iteration: int
    n_nodes: int
    n_tets: int
    n_splits: int
    n_collapses: int
    n_swaps_total: int
    n_smooth_moves_total: int
    n_flat_tets: int
    node_classification: dict
    edge_length_stats: dict
    quality_stats: dict
    q_a_tolerance: float
    marcum_alauzet_max: float
    marcum_alauzet_mean: float
    wall_time_s: float
    max_dihedral_deg: float = 0.0     # display only; excluded from history JSON
    complexity: float = 0.0            # Σ√det(M)·V — metric node budget
    nodes_per_complexity: float = 0.0  # nnode/C; Refine flags overshoot at >3
    max_node_age: int = 0              # worst stuck-node age


class AdaptationTracker:
    """
    Tracks adaptation statistics across iterations, logs to terminal,
    and saves to a JSON file.
    """
    def __init__(self) -> None:
        self.history: List[IterationRecord] = []
        
    def record_iteration(self, record: IterationRecord) -> None:
        """Append *record* to the iteration history."""
        self.history.append(record)

    def print_summary(self, record: IterationRecord) -> None:
        """Print a one-iteration summary for *record* to stdout.

        Includes node/tet/operation counts, the node-level breakdown, and the
        edge-length and element-quality histograms with their summary stats.
        """
        logger.debug(f"\n--- Iteration {record.iteration} Summary ---")
        logger.debug(f"  Nodes: {record.n_nodes} | Tets: {record.n_tets} | Flat tets: {record.n_flat_tets}")
        logger.debug(f"  Operations: Splits={record.n_splits}, Collapses={record.n_collapses}, "
              f"Swaps={record.n_swaps_total}, Smooths={record.n_smooth_moves_total}")
        logger.debug(f"  Nodes by level: Interior={record.node_classification['interior']}, "
              f"Surface={record.node_classification['surface']}, Ridge={record.node_classification['ridge']}, "
              f"Corner={record.node_classification['corner']}")
        logger.debug(f"  Execution time: {record.wall_time_s:.2f} s")
        logger.debug(f"  Complexity: {record.complexity:.1f} | "
              f"nodes/C: {record.nodes_per_complexity:.2f} | "
              f"max node age: {record.max_node_age}")
        
        # Edge Length details
        e_stats = record.edge_length_stats
        print_histogram(e_stats["histogram"], "Edge lengths (conforming range [0.707, 1.414])", mark_out_of_bounds=True)
        logger.debug(f"    min: {e_stats['min']:.4f} | max: {e_stats['max']:.4f} | mean: {e_stats['mean']:.4f} | std: {e_stats['std']:.4f}")
        logger.debug(f"    Fraction: Short={e_stats['fraction_short']:.4%}, "
              f"Conforming={e_stats['fraction_conforming']:.4%}, Long={e_stats['fraction_long']:.4%}")
              
        # Quality details
        q_stats = record.quality_stats
        print_histogram(q_stats["histogram"], "Element qualities (higher is better)")
        logger.debug(f"    min: {q_stats['min']:.4e} | max: {q_stats['max']:.4f} | mean: {q_stats['mean']:.4f} | std: {q_stats['std']:.4f}")
        logger.debug(f"    Fraction: Quality < 0.001 = {q_stats['fraction_below_0001']:.4%}, "
              f"Quality >= 0.5 = {q_stats['fraction_above_05']:.4%}")
        logger.debug(f"  LL absolute floor q_a = {record.q_a_tolerance:.4f}")
        logger.debug(f"  Marcum-Alauzet edge weight: max={record.marcum_alauzet_max:.4f}, mean={record.marcum_alauzet_mean:.4f}")
        if record.iteration == 1:
            logger.info("")
            logger.info("in band = edges with metric length in [0.71, 1.41]; Q = element quality, 1 is ideal")
            logger.info("")
            logger.info("  iter   nodes    tets   in band   short    long   min Q   mean Q   flat   time s")
        logger.info(f"{record.iteration:6d} {record.n_nodes:7d} {record.n_tets:7d} "
                    f"{e_stats['fraction_conforming']:8.1%} {e_stats['fraction_short']:7.1%} "
                    f"{e_stats['fraction_long']:7.1%} {q_stats['min']:7.3f} "
                    f"{q_stats['mean']:8.3f} {record.n_flat_tets:6d} {record.wall_time_s:8.1f}")

    def print_final_report(self, *, converged: bool = False,
                           enable_swap: bool = True) -> None:
        """Print the final convergence diagnostic across all recorded iterations.

        Reports start→final deltas (node/tet counts, edge conformance, quality,
        flat tets), total wall time, and stall warnings (conforming-% plateau,
        declining mean quality, accumulating flat tets) once at least three
        iterations exist.  Does nothing if no iterations were recorded.
        """
        if not self.history:
            return
        first = self.history[0]
        last = self.history[-1]
        
        logger.debug("\n" + "=" * 50)
        logger.debug("         FINAL CONVERGENCE DIAGNOSTIC REPORT")
        logger.debug("=" * 50)
        logger.debug(f"Total Iterations: {len(self.history)}")
        logger.debug(f"Node count change: {first.n_nodes} -> {last.n_nodes}")
        logger.debug(f"Tet count change:  {first.n_tets} -> {last.n_tets}")
        
        logger.debug("\nEdge Length Conformance:")
        logger.debug(f"  Start: Conforming={first.edge_length_stats['fraction_conforming']:.2%}, "
              f"Short={first.edge_length_stats['fraction_short']:.2%}, Long={first.edge_length_stats['fraction_long']:.2%}")
        logger.debug(f"  Final: Conforming={last.edge_length_stats['fraction_conforming']:.2%}, "
              f"Short={last.edge_length_stats['fraction_short']:.2%}, Long={last.edge_length_stats['fraction_long']:.2%}")
              
        logger.debug("\nElement Quality:")
        logger.debug(f"  Start: min={first.quality_stats['min']:.4e}, mean={first.quality_stats['mean']:.4f}, <0.001={first.quality_stats['fraction_below_0001']:.2%}")
        logger.debug(f"  Final: min={last.quality_stats['min']:.4e}, mean={last.quality_stats['mean']:.4f}, <0.001={last.quality_stats['fraction_below_0001']:.2%}")

        logger.debug("\nMarcum-Alauzet Edge Weight (higher = worse):")
        logger.debug(f"  Start: max={first.marcum_alauzet_max:.4f}, mean={first.marcum_alauzet_mean:.4f}")
        logger.debug(f"  Final: max={last.marcum_alauzet_max:.4f}, mean={last.marcum_alauzet_mean:.4f}")

        logger.debug("\nFlat tetrahedra:")
        logger.debug(f"  Start: {first.n_flat_tets} | Final: {last.n_flat_tets}")
        
        total_time = sum(r.wall_time_s for r in self.history)
        logger.debug(f"\nTotal adaptation loop wall time: {total_time:.2f} s")
        
        # ── Convergence stall detection ────────────────────────────
        warnings = []
        if len(self.history) >= 3:
            recent = self.history[-3:]
            conforming_trend = [
                r.edge_length_stats['fraction_conforming'] for r in recent
            ]
            quality_trend = [r.quality_stats['mean'] for r in recent]
            flat_trend = [r.n_flat_tets for r in recent]
            
            # Conforming % stall
            conf_range = max(conforming_trend) - min(conforming_trend)
            if conf_range < 0.005:
                hint = "; consider enabling swap" if not enable_swap else ""
                warnings.append("Edge conformance stalled "
                                f"(range {conf_range:.4f} over the last 3 iterations){hint}")
            
            # Quality decline
            if all(q2 <= q1 for q1, q2
                   in zip(quality_trend, quality_trend[1:])):
                warnings.append("Mean quality declined over the last 3 iterations; check collapse and split quality gates")
            
            # Flat tet accumulation (strict rise to a nonzero count — an
            # all-zero trend is healthy, not "increasing")
            if flat_trend[-1] > 0 and all(f2 >= f1 for f1, f2
                   in zip(flat_trend, flat_trend[1:])):
                warnings.append("Flat tet count increased over the last 3 iterations; check topology repair")

            # Marcum-Alauzet max increasing (dihedral degradation)
            ma_trend = [r.marcum_alauzet_max for r in recent]
            if all(m2 > m1 for m1, m2
                   in zip(ma_trend, ma_trend[1:])):
                warnings.append("Marcum-Alauzet max edge weight increased; dihedral angles degraded")

        if warnings:
            logger.debug("\nConvergence Diagnostics:")

        logger.debug("=" * 50)
        logger.info("")
        if converged:
            logger.info("Stopped: converged at iteration %d (edge-length extremes stable)", last.iteration)
        else:
            logger.info("Stopped: iteration limit (%d) reached", last.iteration)
        logger.info("")
        logger.info("  %-13s %8s  %8s", "", "iter 1", "final")
        logger.info("  %-13s %8d  %8d", "nodes", first.n_nodes, last.n_nodes)
        logger.info("  %-13s %8d  %8d", "tets", first.n_tets, last.n_tets)
        logger.info("  %-13s %8s  %8s", "in band",
                    f"{first.edge_length_stats['fraction_conforming']:.1%}",
                    f"{last.edge_length_stats['fraction_conforming']:.1%}")
        logger.info("  %-13s %8.3f  %8.3f", "mean Q", first.quality_stats['mean'], last.quality_stats['mean'])
        logger.info("  %-13s %8.3f  %8.3f", "min Q", first.quality_stats['min'], last.quality_stats['min'])
        logger.info("  %-13s %8s  %8s", "max dihedral",
                    f"{first.max_dihedral_deg:.1f}°", f"{last.max_dihedral_deg:.1f}°")
        logger.info("  %-13s %8d  %8d", "flat tets", first.n_flat_tets, last.n_flat_tets)
        logger.info("  %-13s %8s", "loop time", f"{total_time:.1f} s")
        logger.info("")
        logger.info("Warnings: %s", f"{len(warnings)} warnings; see adaptation.log" if warnings else "none")
        for warning in warnings:
            logger.warning("  %s", warning)
        
    def export_json(self, output_dir: str) -> None:
        """Write the iteration history to ``adaptation_history.json``.

        The file is created in *output_dir*; raw per-element value lists are
        stripped from the histogram-stat dicts to keep the JSON lightweight.
        """
        import os
        filepath = os.path.join(output_dir, "adaptation_history.json")
        
        data_to_save = []
        for record in self.history:
            d = asdict(record)
            del d["max_dihedral_deg"]
            # Remove raw values lists to keep JSON lightweight
            if "values" in d["edge_length_stats"]:
                del d["edge_length_stats"]["values"]
            if "values" in d["quality_stats"]:
                del d["quality_stats"]["values"]
            data_to_save.append(d)
            
        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=2)
        logger.debug(f"\n  Adaptation history saved to: {filepath}")
