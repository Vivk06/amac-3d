"""Coordinate split, collapse, swap, and smooth passes on a Grid.

The ABCD cycle preserves mesh validity, writes bridge meshes, and reports
quality. Iterations cap a convergence-controlled loop. Its cycle follows
NASA refine ref_adapt.c::ref_adapt_pass.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

from typing import List, TYPE_CHECKING

from .metric import (volume_abs, NOISE_FLOOR,
                     COLLAPSE_THRESHOLD, SPLIT_THRESHOLD)
from .split import split_pass
from .collapse import collapse_pass, repair_flat_tets, CollapseMemo
from .swap import local_reconnection, compute_adaptive_thresholds
from .smooth import smooth_pass
from .io import export_mesh, export_vol
from .report import (
    AdaptationTracker,
    compute_complexity,
    compute_edge_lengths,
    compute_element_qualities,
    count_flat_tets,
    count_node_classifications,
    flat_tet_indices,
    IterationRecord,
    min_element_quality,
)

if TYPE_CHECKING:
    from .grid import Grid


# Safety cap: with tet_colour removed and 1-ring poisoning in
# place, each sweep does much more work. Expect <= 6 sweeps; the cap bounds a
# pathological loop only.
MAX_COLLAPSE_SWEEPS = 30
COLLAPSE_QUIESCENCE_FRACTION = 0.02   # stop when a sweep yields < 2% of the first sweep

# Split/collapse hysteresis: the post-split collapse (Phase C1b) only removes
# edges below this floor, sparing near-band edges [POST_SPLIT_COLLAPSE_THRESHOLD,
# 1/√2) that a split just created. Leaving them for smoothing to equilibrate
# breaks the split→collapse→re-split churn that stalls conforming-% convergence.
POST_SPLIT_COLLAPSE_THRESHOLD = 0.6

# ── Refine-calibrated annealing and termination ──────────────────────
# Q_A_CAP: the annealed absolute LL floor never exceeds this — NASA Refine's
# target_quality = max(min(0.1, worst_Q), 1e-3) (ref_adapt.c:300). Without the
# cap, a good mesh raises its own floor and blocks legitimate operations.
Q_A_CAP = 0.1
# Convergence termination (ref_adapt_parameter + ref_subcommand.c): stop when
# the min/max edge-length extremes change < RATIO_STABLE_TOL relatively for
# two consecutive iterations, at least MIN_ITERATIONS have run, and no node
# has been stuck unmoved for MAX_NODE_AGE smooth passes (unless the band is
# already healthy).
RATIO_STABLE_TOL = 0.01
MIN_ITERATIONS = 5
MAX_NODE_AGE = 50


# ──────────────── Worst Mesh Quality ─────────────────────────────

def _get_worst_mesh_quality(grid: 'Grid') -> float:
    """Return the worst (minimum) positive Knupp tet quality across the mesh.

    Exact-zero qualities (flat/degenerate tets) are excluded: a handful of
    degenerate tets would otherwise drag the annealed LL floor
    ``q_a = max(1e-3, worst_Q)`` down to 1e-3 for the whole mesh, letting
    every subsequent collapse/swap/smooth accept near-degenerate results
    (as seen on the cube run, iterations 6-8). Flat tets are handled
    by the dedicated repair pass, not by relaxing the global floor.
    """
    return min_element_quality(grid)


# ──────────────── Annealed Gate Parameters ────────────────────────

def _anneal_parameters(grid: 'Grid') -> dict:
    """Compute the per-call annealed gates (NASA Refine ref_adapt_parameter).

    Returns a dict with:
      q_a             — absolute LL floor: max(1e-3, min(Q_A_CAP, worst_Q)).
                        Capped so a good mesh cannot raise its own floor and
                        block legitimate operations (Refine caps at 0.1).
      post_min_ratio, post_max_ratio — the *ratcheting* post-operation
                        edge-length band: never stricter than the current
                        mesh extremes (min(worst_short, 1/√2) /
                        max(worst_long, √2)), so operations may not worsen
                        the extremes but are not held to the ideal band
                        while the mesh is far from it. Two Refine easing
                        rules then lower the min bound so a high floor
                        cannot block splits (ref_adapt.c:317-327).
      min_ratio, max_ratio — the raw current extremes (for termination).

    Called at iteration start AND again before Phase C (split): coarsening
    and smoothing change worst_Q and the extremes, so the refinement phases
    gate on fresh values (Refine re-runs ref_adapt_parameter mid-pass).
    """
    q_a = max(1e-3, min(Q_A_CAP, _get_worst_mesh_quality(grid)))

    stats = compute_edge_lengths(grid)
    min_ratio = stats['min']
    max_ratio = stats['max']

    post_min = min(min_ratio, COLLAPSE_THRESHOLD)
    post_max = max(max_ratio, SPLIT_THRESHOLD)

    # Easing rule 1: extremes far apart — allow shorter post-op edges so
    # splitting the very long edges is not blocked by the short-side bound.
    if post_max > 4.0 and post_min > 0.4:
        post_min *= 4.0 / post_max
    # Easing rule 2: near convergence — "prevent high lower limit from
    # stopping split" (Refine comment verbatim).
    elif 1.6 < post_max < 3.5 and post_min > 0.4:
        post_min *= 1.4 / post_max

    return {
        'q_a': q_a,
        'post_min_ratio': post_min,
        'post_max_ratio': post_max,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
    }


# ──────────────── Flatness Check ──────────────────────────────────

def check_flat_tets(grid: 'Grid') -> List[int]:
    """
    Check for flat (nearly co-planar) tetrahedra.

    Uses normalised determinant:  det / max(edge_length²)
    A tet is flat if this ratio ≤ EPSILON (1e-15).

    Raises
    ------
    RuntimeError
        If any flat tetrahedra are found.
    """
    coords = grid.coordinates
    tets = grid.tetrahedra
    EPSILON = 1e-15
    flat_indices: List[int] = []

    for i in range(0, len(tets), 4):
        a_idx, b_idx, c_idx, d_idx = tets[i], tets[i + 1], tets[i + 2], tets[i + 3]

        # Skip dead tets
        if a_idx == -1:
            continue

        if grid.dead_mask[a_idx] or grid.dead_mask[b_idx] or grid.dead_mask[c_idx] or grid.dead_mask[d_idx]:
            continue

        a = coords[a_idx * 3: a_idx * 3 + 3]
        b = coords[b_idx * 3: b_idx * 3 + 3]
        c = coords[c_idx * 3: c_idx * 3 + 3]
        d = coords[d_idx * 3: d_idx * 3 + 3]

        vol = volume_abs(a, b, c, d)
        det_val = 6.0 * vol

        # Six edge lengths squared
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
            flat_indices.append(i // 4)
            continue

        normalised = det_val / max_edge_sq
        if normalised <= EPSILON:
            flat_indices.append(i // 4)

    if flat_indices:
        raise RuntimeError(
            f"Flat tetrahedra found at indices: {flat_indices[:20]}"
            f"{'...' if len(flat_indices) > 20 else ''}"
        )

    return []


# ──────────────── Collapse Helper ─────────────────────────────────

def _collapse_to_quiescence(grid: 'Grid', max_sweeps: int,
                            show_rejections: bool,
                            q_r: float = 1.0,
                            q_a: float = 1e-3,
                            collapse_threshold: float = COLLAPSE_THRESHOLD,
                            post_min_ratio: float = 1e-3,
                            post_max_ratio: float = 4.0) -> List[int]:
    """Run bounded collapse sweeps until quiescence.

    Dead elements are NOT cleaned between sweeps — collapse_edge maintains
    ballOfNode incrementally, and collect_edges / shell_of_edge skip dead
    tets via the [-1] sentinel.  The caller cleans once after this returns
    the boundary maps. Stops when a sweep collapses nothing or falls below
    ``COLLAPSE_QUIESCENCE_FRACTION`` of the first sweep.
    """
    sweep_counts: List[int] = []
    n_first = 0
    memo = CollapseMemo()
    for sweep in range(max_sweeps):
        n_sweep = collapse_pass(grid, show_rejections=show_rejections,
                                q_r=q_r, q_a=q_a,
                                collapse_threshold=collapse_threshold,
                                post_min_ratio=post_min_ratio,
                                post_max_ratio=post_max_ratio, memo=memo)
        sweep_counts.append(n_sweep)
        if sweep == 0:
            n_first = n_sweep
        if n_sweep == 0 or n_sweep < COLLAPSE_QUIESCENCE_FRACTION * n_first:
            break
    return sweep_counts


def _print_collapse_summary(grid: 'Grid', sweep_counts: List[int]) -> None:
    """Print the sweep sequence and the resulting short-edge fraction."""
    total = sum(sweep_counts)
    seq = " + ".join(str(n) for n in sweep_counts)
    logger.debug(f"       Sweeps: {seq} = {total} collapses "
          f"(dead elements cleaned after quiescence)")
    stats = compute_edge_lengths(grid)
    logger.debug(f"       Shorter edges fraction: {stats['fraction_short']:.6f}")


# ──────────────── Swap Helper ─────────────────────────────────────

def _run_swap(grid: 'Grid', label: str, verbose: bool = True,
              q_r: float = 1.0, q_a: float = 1e-3) -> int:
    """Run a single swap pass and clean dead elements."""
    logger.debug(f"  [{label}] Swap pass")
    swaps = local_reconnection(grid, criterion="min-dihedral",
                                max_passes=1, verbose=False,
                                q_r=q_r, q_a=q_a)
    if swaps > 0:
        grid.cleanDeadElements()
    if verbose:
        logger.debug(f"       Swaps: {swaps}")
    return swaps


# ──────────────── Adaptation Driver ───────────────────────────────

def adapt(grid: 'Grid', iterations: int,
          output_dir: str = 'output',
          enable_swap: bool = True,
          bridge_format: str = 'msh',
          collapse_sweeps: int = MAX_COLLAPSE_SWEEPS,
          show_collapse_rejections: bool = False,
          show_split_rejections: bool = False) -> 'Grid':
    """
    Run the edge-primitive mesh adaptation loop.

    Cycle order (modelled after NASA Refine's ``ref_adapt_pass``):

        Phase A — Coarsening:
            swap → collapse-to-quiescence (multi-sweep) → swap
        Phase B — Post-coarsen smoothing:
            smooth → swap
        Phase C — Refinement:
            split → collapse-to-quiescence (multi-sweep) → swap
        Phase D — Final smoothing:
            smooth → swap

    Parameters
    ----------
    grid : Grid
    iterations : int
        Maximum number of outer adaptation passes (a cap, not a
        duration). The loop self-terminates when the min/max edge-length
        extremes change < 1% for two consecutive iterations (Refine-style
        convergence test; see the constants block).
    output_dir : str
        Directory for bridging/output files.
    enable_swap : bool
        If True, run the local reconnection (swap) pass at each stage.
        Set to False for faster runs on large meshes where the swap
        cost is prohibitive.
    bridge_format : str
        Bridge output format: ``'msh'``, ``'vol'``, or ``'both'``.
    collapse_sweeps : int
        Maximum collapse sweeps per multi-sweep loop (Phases A2 and C1b).
    show_collapse_rejections : bool
        If True, print the per-sweep collapse rejection breakdown.
    show_split_rejections : bool
        If True, print the per-pass split rejection breakdown (Phase C1).

    Returns
    -------
    Grid
        The adapted grid.
    """
    import os
    import time
    os.makedirs(output_dir, exist_ok=True)

    from .kernels import warmup as _jit_warmup
    _jit_warmup()

    logger.debug(f"Adaptation started (swap={'ON' if enable_swap else 'OFF'}).")

    import numba
    logger.debug(f"  Parallel: numba threads={numba.get_num_threads()}, "
          f"threading layer={numba.config.THREADING_LAYER}")

    tracker = AdaptationTracker()

    # Convergence-termination state: the loop stops early when
    # the edge-length extremes stabilise for two consecutive iterations.
    last_min_ratio: float = None
    last_max_ratio: float = None
    converged_prev = False
    converged = False
    for i in range(iterations):
        t_start = time.perf_counter()
        logger.debug(f"\n=== Iteration {i + 1}/{iterations} ===")

        # ── Loseille-Löhner annealed gating parameters ───────────
        params = _anneal_parameters(grid)
        q_a = params['q_a']
        post_min_ratio = params['post_min_ratio']
        post_max_ratio = params['post_max_ratio']
        logger.debug(f"  Annealed gates: q_a={q_a:.4f}, "
              f"post-edge band [{post_min_ratio:.3f}, {post_max_ratio:.2f}]")
        q_a_split = 1e-3
        q_r_split = float('inf')
        q_r_collapse = 2.0
        q_r_coarsen  = 1.05
        q_r_smooth_1 = 1.03
        q_r_refine   = 1.01
        q_r_smooth_2 = 1.00

        n_swaps_total = 0
        n_smooth_moves_total = 0

        # ── Active-edge restriction ──────────────────────────────
        grid.update_active_edges()
        if grid._active_edge_frac < 1.0:
            logger.debug(f"  Active-edge scan: {grid._active_edge_frac:.1%} of edges non-conforming")

        # ── Phase A: Coarsening (swap → collapse → clean → swap) ──
        if enable_swap:
            n_swaps_total += _run_swap(grid, "A1",
                                       q_r=q_r_coarsen, q_a=q_a)

        logger.debug("  [A2] Collapse pass (multi-sweep)")
        sweep_counts = _collapse_to_quiescence(grid, collapse_sweeps,
                                               show_collapse_rejections,
                                               q_r=q_r_collapse, q_a=q_a,
                                               post_min_ratio=post_min_ratio,
                                               post_max_ratio=post_max_ratio)
        grid.cleanDeadElements()
        _print_collapse_summary(grid, sweep_counts)
        n_collapses = sum(sweep_counts)

        if enable_swap:
            n_swaps_total += _run_swap(grid, "A4",
                                       q_r=q_r_coarsen, q_a=q_a)

        # ── Phase B: Post-coarsen smoothing (smooth → swap) ───────
        logger.debug("  [B1] Smooth pass (post-coarsen)")
        n_smooth_moves_total += smooth_pass(grid,
                                            q_a=q_a)

        if enable_swap:
            n_swaps_total += _run_swap(grid, "B2",
                                       q_r=q_r_smooth_1, q_a=q_a)

        # ── Re-anneal before refinement (Refine mid-pass param) ─────────
        # Coarsening + smoothing changed worst_Q and the edge extremes;
        # the refinement phases gate on fresh values.
        params = _anneal_parameters(grid)
        q_a = params['q_a']
        post_min_ratio = params['post_min_ratio']
        post_max_ratio = params['post_max_ratio']
        logger.debug(f"  Re-annealed gates: q_a={q_a:.4f}, "
              f"post-edge band [{post_min_ratio:.3f}, {post_max_ratio:.2f}]")

        # Refresh active-edge mask after coarsening changed topology
        grid.update_active_edges()
        if grid._active_edge_frac < 1.0:
            logger.debug(f"  Active-edge scan (re-annealed): {grid._active_edge_frac:.1%} non-conforming")

        # ── Phase C: Refinement (split → collapse → swap) ──
        logger.debug("  [C1] Split pass")
        n_splits = split_pass(grid, q_r=q_r_split, q_a=q_a_split,
                              show_rejections=show_split_rejections,
                              post_min_ratio=post_min_ratio,
                              post_max_ratio=post_max_ratio)
        grid.clear_active_edges()

        logger.debug("  [C1b] Post-split collapse (multi-sweep)")
        sweep_counts = _collapse_to_quiescence(grid, collapse_sweeps,
                                               show_collapse_rejections,
                                               q_r=q_r_collapse, q_a=q_a,
                                               collapse_threshold=POST_SPLIT_COLLAPSE_THRESHOLD,
                                               post_min_ratio=post_min_ratio,
                                               post_max_ratio=post_max_ratio)
        grid.cleanDeadElements()
        _print_collapse_summary(grid, sweep_counts)
        n_collapses += sum(sweep_counts)

        if enable_swap:
            n_swaps_total += _run_swap(grid, "C2",
                                       q_r=q_r_refine, q_a=q_a)

        # ── Phase D: Final smoothing (smooth → swap) ──────────────
        logger.debug("  [D1] Smooth pass (final)")
        n_smooth_moves_total += smooth_pass(grid,
                                            q_a=q_a)

        if enable_swap:
            n_swaps_total += _run_swap(grid, "D2",
                                       q_r=q_r_smooth_2, q_a=q_a)

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        # ── Quality check ─────────────────────────────────────────
        logger.debug("  [QC] Flatness check")
        flat_tets = flat_tet_indices(grid)
        n_flat = len(flat_tets)
        if n_flat == 0:
            logger.debug("  ✓ No flat tetrahedra")
        else:
            # Flat tets self-perpetuate: their edges are usually
            # metric-conforming (invisible to collapse), their degenerate
            # face can be a boundary face (off-limits to swap), and their
            # nodes ridge-constrained (immovable). Repair them here so
            # they never survive into the next iteration.
            logger.debug(f"  ⚠ Found {n_flat} flat tetrahedra")
            logger.debug("  [QC-R] Flat-tet repair pass")
            n_repaired = repair_flat_tets(grid, flat_tets)
            if n_repaired > 0:
                grid.cleanDeadElements()
            n_flat = count_flat_tets(grid)
            if n_flat == 0:
                logger.debug(f"       Repaired via {n_repaired} collapses — "
                      f"no flat tetrahedra remain")
            else:
                logger.warning(f"{n_flat} flat tetrahedra remain after {n_repaired} repair collapses this iteration")

        # ── Bridging output ───────────────────────────────────────
        bridge_path = os.path.join(output_dir, f'bridge_{i}')
        if bridge_format in ('msh', 'both'):
            export_mesh(grid, bridge_path, 'msh')
            logger.debug(f"  Bridge file written: {bridge_path}.msh")
        if bridge_format in ('vol', 'both'):
            export_vol(grid, bridge_path + '.vol')
            logger.debug(f"  Bridge file written: {bridge_path}.vol")

        # ── Metrics collection and summary output ─────────────────
        edge_stats = compute_edge_lengths(grid)
        quality_stats = compute_element_qualities(grid)
        angle_limits = compute_adaptive_thresholds(grid)
        max_dihedral_deg = (math.degrees(angle_limits['meshMaxDihedral'])
                            if angle_limits is not None else 0.0)
        node_class = count_node_classifications(grid)
        n_nodes = len(grid.coordinates) // 3
        complexity = compute_complexity(grid)
        nodes_per_complexity = (n_nodes / complexity
                                if complexity > 0.0 else 0.0)
        max_node_age = max(grid.nodeAge) if grid.nodeAge else 0

        record = IterationRecord(
            iteration=i + 1,
            n_nodes=n_nodes,
            n_tets=len(grid.tetrahedra) // 4,
            n_splits=n_splits,
            n_collapses=n_collapses,
            n_swaps_total=n_swaps_total,
            n_smooth_moves_total=n_smooth_moves_total,
            n_flat_tets=n_flat,
            node_classification=node_class,
            edge_length_stats=edge_stats,
            quality_stats=quality_stats,
            q_a_tolerance=q_a,
            marcum_alauzet_max=quality_stats["marcum_alauzet_max"],
            marcum_alauzet_mean=quality_stats["marcum_alauzet_mean"],
            wall_time_s=elapsed,
            max_dihedral_deg=max_dihedral_deg,
            complexity=complexity,
            nodes_per_complexity=nodes_per_complexity,
            max_node_age=max_node_age,
        )
        tracker.record_iteration(record)
        tracker.print_summary(record)

        # ── Convergence termination ────────────────────────
        # Refine-style: stop when the min/max edge-length extremes change
        # < 1% relatively for two consecutive iterations, at least
        # MIN_ITERATIONS have run, and no movable node has been stuck
        # for MAX_NODE_AGE smooth passes (unless the band is healthy).
        min_r = edge_stats['min']
        max_r = edge_stats['max']
        converged_now = False
        if last_min_ratio is not None and min_r > 0.0 and max_r > 0.0:
            stable = (abs(min_r - last_min_ratio)
                      < RATIO_STABLE_TOL * last_min_ratio
                      and abs(max_r - last_max_ratio)
                      < RATIO_STABLE_TOL * last_max_ratio)
            unstuck = (max_node_age < MAX_NODE_AGE
                       or (min_r > 0.1 and max_r < 3.0))
            converged_now = stable and unstuck
        if (converged_now and converged_prev
                and (i + 1) >= min(MIN_ITERATIONS, iterations)):
            logger.debug(f"\n  ✓ Converged: edge-length extremes stable "
                  f"(<{RATIO_STABLE_TOL:.0%} change) for 2 consecutive "
                  f"iterations — terminating at iteration "
                  f"{i + 1}/{iterations}")
            converged = True
            break
        last_min_ratio, last_max_ratio = min_r, max_r
        converged_prev = converged_now

    # ── Final report and export ───────────────────────────────────
    tracker.export_json(output_dir)
    tracker.print_final_report(converged=converged, enable_swap=enable_swap)

    logger.debug("\nAdaptation complete.")
    return grid
