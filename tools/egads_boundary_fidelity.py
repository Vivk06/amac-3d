#!/usr/bin/env python3
"""
egads_boundary_fidelity.py — Measure boundary node fidelity against CAD geometry.

For each boundary node in the adapted mesh, measures the distance from the node
to its corresponding EGADS face/edge. Reports max/mean/RMS deviation and
projection coverage.

Usage:
    python tools/egads_boundary_fidelity.py \
        --geo cases/cube_egads/input/geometry/cube.geo \
        --mesh cases/cube_egads/input/mesh/cube1241.vol \
        --step cases/cube_egads/input/geometry/cube.step \
        --metric cases/cube_egads/input/metric/cross.sol

Input: Netgen .vol mesh, .geo geometry, STEP CAD, optional .sol metric.
Output: fidelity report on stdout; no files are written. Requires pyEGADS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def measure_boundary_fidelity(grid, model, topo):
    """Measure distance from each boundary node to its EGADS face/edge."""
    from amac_3d.egads_adapter import project_to_surface_egads, project_to_curve_egads

    results = {
        'surface_deviations': [],
        'ridge_deviations': [],
        'egads_hits': 0,
        'egads_misses': 0,
    }

    for node in range(len(grid.pointLevels)):
        level = grid.pointLevels[node]
        if level == 1:
            surfid = grid.bdyFaceMap.get(node, -1)
            if surfid < 0:
                continue
            pt = grid.coordinates[node * 3: node * 3 + 3].tolist()
            proj = project_to_surface_egads(pt, model, topo, surfid)
            if proj is not None:
                dev = np.linalg.norm(np.array(proj) - np.array(pt))
                results['surface_deviations'].append(dev)
                results['egads_hits'] += 1
            else:
                results['egads_misses'] += 1
        elif level == 2:
            ednr = grid.bdyEdgeMap.get(node, -1)
            if ednr < 0:
                continue
            pt = grid.coordinates[node * 3: node * 3 + 3].tolist()
            proj = project_to_curve_egads(pt, model, topo, ednr)
            if proj is not None:
                dev = np.linalg.norm(np.array(proj) - np.array(pt))
                results['ridge_deviations'].append(dev)
                results['egads_hits'] += 1
            else:
                results['egads_misses'] += 1

    return results


def print_report(results):
    """Print the fidelity report."""
    print("\n" + "=" * 60)
    print("EGADS Boundary Fidelity Report")
    print("=" * 60)

    total = results['egads_hits'] + results['egads_misses']
    if total > 0:
        hit_rate = results['egads_hits'] / total * 100
        print(f"\nProjection coverage: {results['egads_hits']}/{total} "
              f"({hit_rate:.1f}%)")
        if results['egads_misses'] > 0:
            print(f"  Fallbacks: {results['egads_misses']}")

    for label, devs in [
        ("Surface nodes", results['surface_deviations']),
        ("Ridge nodes", results['ridge_deviations']),
    ]:
        if not devs:
            print(f"\n{label}: no data")
            continue
        arr = np.array(devs)
        print(f"\n{label} ({len(arr)} nodes):")
        print(f"  Max deviation:  {arr.max():.2e}")
        print(f"  Mean deviation: {arr.mean():.2e}")
        print(f"  RMS deviation:  {np.sqrt(np.mean(arr**2)):.2e}")
        print(f"  Median:         {np.median(arr):.2e}")

    all_devs = results['surface_deviations'] + results['ridge_deviations']
    if all_devs:
        max_dev = max(all_devs)
        gate = "PASS" if max_dev < 1e-6 else "FAIL"
        print(f"\nGate (max deviation < 1e-6): {gate} (max={max_dev:.2e})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Measure boundary node fidelity against EGADS CAD geometry.',
        epilog=('Output: report on stdout; no files are written. Example: '
                'python tools/egads_boundary_fidelity.py '
                '--geo cases/cube_egads/input/geometry/cube.geo '
                '--mesh cases/cube_egads/input/mesh/cube1241.vol '
                '--step cases/cube_egads/input/geometry/cube.step'))
    parser.add_argument('--geo', required=True, help='Path to .geo file')
    parser.add_argument('--mesh', required=True, help='Path to .vol mesh file')
    parser.add_argument('--step', required=True, help='Path to STEP CAD file')
    parser.add_argument('--metric', help='Optional: path to .sol metric file')
    args = parser.parse_args()

    from amac_3d.grid import Grid
    from amac_3d.egads_adapter import EgadsModel, build_topo_map

    print(f"Loading mesh: {args.mesh}")
    kwargs = {'metric_path': args.metric} if args.metric else {}
    grid = Grid.from_mesh(args.geo, args.mesh, **kwargs)

    print(f"Loading EGADS model: {args.step}")
    model = EgadsModel(args.step)
    topo = build_topo_map(model, grid)

    print("Measuring boundary fidelity...")
    results = measure_boundary_fidelity(grid, model, topo)
    print_report(results)


if __name__ == "__main__":
    main()
