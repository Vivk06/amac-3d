#!/usr/bin/env python3
"""
analytic_metric_generator.py — Generate Medit .sol metric files from analytic functions.

Generates anisotropic Riemannian metric tensor fields on Netgen .vol meshes
using analytic formulas from Gokula Krishna Tavva's thesis (Chapter 4,
Eqs 4.1–4.6, reference [32]: Loseille & Alauzet).

Each metric function maps a spatial coordinate (x, y, z) to a 6-component
symmetric positive-definite tensor [M11, M12, M13, M22, M23, M33].

Usage:
  python tools/analytic_metric_generator.py generate \
      --mesh cases/cube_cross/input/mesh/cube1241.vol --metric cross
  python tools/analytic_metric_generator.py validate \
      --mesh cases/cube_cross/input/mesh/cube1241.vol --metric cross \
      --reference cases/cube_cross/input/metric/cross.sol
  python tools/analytic_metric_generator.py list

Adding a new metric:
  1. Define a function: f(x, y, z, **params) -> [M11, M12, M13, M22, M23, M33]
  2. Decorate with @register_metric(name, description, {param: default, ...})
  3. No other changes needed — CLI, generator, and validator work via the registry.
"""

import math
import argparse
import os
import sys
from pathlib import Path


# ──────────────── Metric Registry ────────────────────────────────────

METRIC_REGISTRY = {}


def register_metric(name, description, param_defaults):
    """Decorator to register an analytic metric function.

    The decorated function must accept (x, y, z, **params) and return
    [M11, M12, M13, M22, M23, M33] (upper-triangular packed SPD tensor).
    """
    def decorator(func):
        METRIC_REGISTRY[name] = {
            'function': func,
            'description': description,
            'params': param_defaults,
        }
        return func
    return decorator


# ──────────────── Cross Metric (Thesis Eq. 4.1–4.3) ─────────────────

@register_metric(
    'cross',
    'Diagonal cross metric — three anisotropic linear features aligned '
    'with coordinate planes, intersecting at domain center '
    '(Tavva thesis Eq. 4.1-4.3, from Loseille & Alauzet)',
    {
        'h_min': 0.003125,
        'h_max': 0.0625,
        'alpha_scale': 40.0,
        'center_x': 0.5,
        'center_y': 0.5,
        'center_z': 0.5,
        'scale': 0.1,
    },
)
def cross_metric(x, y, z, *, h_min=0.003125, h_max=0.0625,
                 alpha_scale=40.0, center_x=0.5, center_y=0.5,
                 center_z=0.5, scale=0.1):
    """M = scale * diag(h_x^{-2}, h_y^{-2}, h_z^{-2})

    h_dir = min(2^{alpha_dir} * h_min, h_max)
    alpha_x = alpha_scale * |x - center_x|, etc.
    """
    ax = alpha_scale * abs(x - center_x)
    ay = alpha_scale * abs(y - center_y)
    az = alpha_scale * abs(z - center_z)

    hx = min(2.0 ** ax * h_min, h_max)
    hy = min(2.0 ** ay * h_min, h_max)
    hz = min(2.0 ** az * h_min, h_max)

    return [scale / (hx * hx), 0.0, 0.0,
            scale / (hy * hy), 0.0,
            scale / (hz * hz)]


# ──────────────── Quarter Cylinder Metric (Thesis Eq. 4.4–4.6) ──────

@register_metric(
    'cylinder',
    'Quarter cylinder metric — curved anisotropic feature at r=r_center '
    'with rotation-based anisotropy in the xy-plane and constant h_z '
    '(Tavva thesis Eq. 4.4-4.6, from Loseille & Alauzet)',
    {
        'h1_base': 0.002,
        'h1_growth': 5.0,
        'h2_base': 0.05,
        'h2_growth': 2.0,
        'h_max': 0.1,
        'h_z': 0.003125,
        'alpha_scale': 10.0,
        'r_center': 0.75,
        'scale': 0.03,
    },
)
def cylinder_metric(x, y, z, *, h1_base=0.002, h1_growth=5.0,
                    h2_base=0.05, h2_growth=2.0, h_max=0.1,
                    h_z=0.003125, alpha_scale=10.0, r_center=0.75,
                    scale=0.03):
    """M = scale * R^T diag(h1^{-2}, h2^{-2}, h_z^{-2}) R

    h1 = min(h1_base * h1_growth^alpha, h_max)   [finer; eigenvector (y, x)/r, equal to the
                                      radial direction only on the plane x = y]
    h2 = min(h2_base * h2_growth^alpha, h_max)    [tangential, coarser]
    theta = atan2(x, y)   [thesis convention]
    alpha = alpha_scale * |r_center - sqrt(x^2 + y^2)|
    h_z is constant (extruded 2D feature).
    """
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(x, y)
    alpha = alpha_scale * abs(r_center - r)

    h1 = min(h1_base * h1_growth ** alpha, h_max)
    h2 = min(h2_base * h2_growth ** alpha, h_max)

    h1_inv2 = scale / (h1 * h1)
    h2_inv2 = scale / (h2 * h2)
    hz_inv2 = scale / (h_z * h_z)

    c = math.cos(theta)
    s = math.sin(theta)

    M11 = h1_inv2 * c * c + h2_inv2 * s * s
    M12 = (h1_inv2 - h2_inv2) * c * s
    M22 = h1_inv2 * s * s + h2_inv2 * c * c

    return [M11, M12, 0.0, M22, 0.0, hz_inv2]


# ──────────────── Uniform Isotropic Metric (example) ─────────────────

@register_metric(
    'uniform',
    'Uniform isotropic metric — constant element size everywhere '
    '(useful for testing and as a template for new metrics)',
    {
        'h': 0.1,
        'scale': 1.0,
    },
)
def uniform_metric(x, y, z, *, h=0.1, scale=1.0):
    """M = scale * (1/h^2) * I"""
    val = scale / (h * h)
    return [val, 0.0, 0.0, val, 0.0, val]


# ──────────────── .vol Coordinate Reader ─────────────────────────────

def read_vol_coordinates(vol_path):
    """Parse a Netgen .vol file and extract node coordinates.

    Returns list of (x, y, z) tuples in file order (0-indexed).
    Only reads the 'points' section. Standalone — no dependency on src/amac_3d/.
    """
    coords = []
    with open(vol_path, 'r') as f:
        lines = [line.strip().split() for line in f]

    i = 0
    while i < len(lines):
        if lines[i] and lines[i][0] == 'points':
            n = int(lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                x = float(lines[j][0])
                y = float(lines[j][1])
                z = float(lines[j][2])
                coords.append((x, y, z))
            break
        i += 1

    if not coords:
        print(f"Error: no 'points' section found in {vol_path}", file=sys.stderr)
        sys.exit(1)

    return coords


# ──────────────── .sol Writer ────────────────────────────────────────

def write_sol(sol_path, metrics):
    """Write a Medit .sol metric file.

    Format matches existing .sol files byte-for-byte:
    5 header lines, 1 blank, N data lines, 1 blank, 'End'.
    """
    n = len(metrics)
    with open(sol_path, 'w') as f:
        f.write('MeshVersionFormatted 2\n')
        f.write('Dimension 3\n')
        f.write('SolAtVertices\n')
        f.write(f'{n}\n')
        f.write('1 3\n')
        f.write('\n')
        for m in metrics:
            vals = '   '.join(f'{v:.16e}' for v in m)
            f.write(f' {vals}\n')
        f.write('\n')
        f.write('End\n')


# ──────────────── .sol Reader ────────────────────────────────────────

def read_sol(sol_path):
    """Parse a Medit .sol file and return per-node metric tensors.

    Returns list of [M11, M12, M13, M22, M23, M33] per node.
    Skips header (6 lines) and footer (2 lines), matching src/amac_3d/io.import_metric.
    """
    metrics = []
    with open(sol_path, 'r') as f:
        lines = f.readlines()
    for line in lines[6:-2]:
        stripped = line.strip()
        if stripped:
            values = list(map(float, stripped.split()))
            if len(values) == 6:
                metrics.append(values)
    return metrics


# ──────────────── Validation / Comparison ────────────────────────────

def compare_metrics(generated, reference, coords, verbose=False):
    """Compare generated metrics against a reference .sol.

    Reports per-component and aggregate error statistics. When verbose=True,
    prints the 10 worst nodes with their coordinates and metric values.
    """
    n = len(generated)
    if n != len(reference):
        print(f"Error: node count mismatch: generated={n}, "
              f"reference={len(reference)}", file=sys.stderr)
        return None

    comp_names = ['M11', 'M12', 'M13', 'M22', 'M23', 'M33']
    max_rel = 0.0
    max_abs = 0.0
    sum_rel = 0.0
    sum_abs = 0.0
    worst_idx = 0
    n_exact = 0
    n_total = n * 6
    per_comp_max_rel = [0.0] * 6
    errors = []

    for i in range(n):
        node_max_rel = 0.0
        for c in range(6):
            g = generated[i][c]
            r = reference[i][c]
            abs_err = abs(g - r)
            rel_err = abs_err / abs(r) if abs(r) > 1e-15 else abs_err

            if rel_err < 1e-10:
                n_exact += 1
            if rel_err > per_comp_max_rel[c]:
                per_comp_max_rel[c] = rel_err
            if rel_err > node_max_rel:
                node_max_rel = rel_err

            sum_rel += rel_err
            sum_abs += abs_err
            if abs_err > max_abs:
                max_abs = abs_err
            if rel_err > max_rel:
                max_rel = rel_err
                worst_idx = i

        errors.append((node_max_rel, i))

    result = {
        'n_nodes': n,
        'max_rel_error': max_rel,
        'mean_rel_error': sum_rel / n_total,
        'max_abs_error': max_abs,
        'mean_abs_error': sum_abs / n_total,
        'worst_node_idx': worst_idx,
        'worst_node_coord': coords[worst_idx],
        'per_component_max_rel': per_comp_max_rel,
        'n_exact_match': n_exact,
        'match_fraction': n_exact / n_total,
    }

    print(f"\nValidation: {n} nodes")
    print(f"  Max relative error:  {max_rel:.6e}")
    print(f"  Mean relative error: {sum_rel / n_total:.6e}")
    print(f"  Max absolute error:  {max_abs:.6e}")
    print(f"  Exact matches:       {n_exact}/{n_total} "
          f"({100.0 * n_exact / n_total:.1f}%)")
    print(f"\n  Per-component max relative error:")
    for c in range(6):
        print(f"    {comp_names[c]}: {per_comp_max_rel[c]:.6e}")
    cx, cy, cz = coords[worst_idx]
    print(f"\n  Worst node: #{worst_idx} at "
          f"({cx:.6f}, {cy:.6f}, {cz:.6f})")

    if verbose:
        errors.sort(reverse=True)
        print(f"\n  Top 10 worst nodes:")
        print(f"  {'Node':>6}  {'Coord':>38}  {'Max Rel Err':>12}")
        for rel_err, idx in errors[:10]:
            x, y, z = coords[idx]
            print(f"  {idx:6d}  ({x:10.6f}, {y:10.6f}, {z:10.6f})  "
                  f"{rel_err:12.6e}")
            print(f"           Gen: "
                  f"{' '.join(f'{v:12.4e}' for v in generated[idx])}")
            print(f"           Ref: "
                  f"{' '.join(f'{v:12.4e}' for v in reference[idx])}")

    return result


# ──────────────── CLI ────────────────────────────────────────────────

def _parse_params(param_list, metric_name):
    """Parse --param KEY=VALUE arguments and validate against defaults."""
    defaults = METRIC_REGISTRY[metric_name]['params']
    overrides = {}
    if param_list:
        for p in param_list:
            if '=' not in p:
                print(f"Error: --param must be KEY=VALUE, got '{p}'",
                      file=sys.stderr)
                sys.exit(1)
            key, val = p.split('=', 1)
            if key not in defaults:
                valid = ', '.join(sorted(defaults.keys()))
                print(f"Error: unknown parameter '{key}' for metric "
                      f"'{metric_name}'. Valid: {valid}", file=sys.stderr)
                sys.exit(1)
            overrides[key] = float(val)
    return {**defaults, **overrides}


def _auto_output_name(metric_name, n_nodes, params):
    """Generate default output filename following existing convention."""
    scale = params.get('scale', 1.0)
    return f'{metric_name}_{n_nodes}_{scale}.sol'


def _output_directory(mesh, override):
    """Use the case output directory for case inputs, or the mesh directory."""
    mesh_path = Path(mesh).expanduser().resolve()
    if override is None:
        input_dir = next((part for part in mesh_path.parents
                          if part.name == 'input'), None)
        directory = input_dir.parent / 'output' if input_dir else mesh_path.parent
    else:
        directory = Path(override).expanduser()
    directory = directory.resolve()
    if 'input' in directory.parts:
        raise SystemExit(f'ERROR: refused output directory inside input/: {directory}')
    return directory


def _cmd_generate(args):
    output_dir = _output_directory(args.mesh, args.output_dir)
    if args.output is not None and Path(args.output).name != args.output:
        raise SystemExit('ERROR: --output must be a file name; use --output-dir for its directory')
    coords = read_vol_coordinates(args.mesh)
    params = _parse_params(args.param, args.metric)
    func = METRIC_REGISTRY[args.metric]['function']

    metrics = [func(x, y, z, **params) for x, y, z in coords]

    output = args.output
    if output is None:
        output = _auto_output_name(args.metric, len(coords), params)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output
    write_sol(output_path, metrics)
    print(f"Generated {output_path}: {len(coords)} nodes, metric='{args.metric}'")


def _cmd_validate(args):
    coords = read_vol_coordinates(args.mesh)
    params = _parse_params(args.param, args.metric)
    func = METRIC_REGISTRY[args.metric]['function']

    generated = [func(x, y, z, **params) for x, y, z in coords]
    reference = read_sol(args.reference)
    result = compare_metrics(generated, reference, coords,
                             verbose=args.verbose)

    if result and args.tolerance is not None:
        if result['max_rel_error'] > args.tolerance:
            print(f"\nFAIL: max relative error {result['max_rel_error']:.6e} "
                  f"exceeds tolerance {args.tolerance:.6e}")
            sys.exit(1)
        else:
            print(f"\nPASS: max relative error within tolerance "
                  f"{args.tolerance:.6e}")


def _cmd_list(_args):
    print("Available analytic metrics:\n")
    for name, info in METRIC_REGISTRY.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Parameters:")
        for pname, pval in info['params'].items():
            print(f"      {pname} = {pval}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Medit .sol metric files from analytic functions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('Input: Netgen .vol mesh; validate also reads a reference .sol.\n'
                'Output: generate writes a .sol to --output-dir (default: case output/); '
                'validate and list print to stdout.\n'
                'Example: python tools/analytic_metric_generator.py generate '
                '--mesh cases/cube_cross/input/mesh/cube1241.vol --metric cross\n'
                'Run "<command> --help" for subcommand details.'))
    sub = parser.add_subparsers(dest='command')

    gen = sub.add_parser('generate', help='Generate a .sol metric file')
    gen.add_argument('--mesh', required=True,
                     help='Input Netgen .vol mesh file')
    gen.add_argument('--metric', required=True,
                     choices=list(METRIC_REGISTRY.keys()),
                     help='Analytic metric function name')
    gen.add_argument('--output', '-o',
                     help='Output .sol file name (default: auto-named)')
    gen.add_argument('--output-dir',
                     help='Output directory (default: case output/ or mesh directory; input/ refused)')
    gen.add_argument('--param', '-p', action='append',
                     help='Override parameter: KEY=VALUE (repeatable)')

    val = sub.add_parser('validate',
                         help='Validate against a reference .sol file')
    val.add_argument('--mesh', required=True,
                     help='Input Netgen .vol mesh file')
    val.add_argument('--metric', required=True,
                     choices=list(METRIC_REGISTRY.keys()),
                     help='Analytic metric function name')
    val.add_argument('--reference', required=True,
                     help='Reference .sol file to compare against')
    val.add_argument('--verbose', '-v', action='store_true',
                     help='Show worst nodes with full metric values')
    val.add_argument('--tolerance', '-t', type=float, default=None,
                     help='Max acceptable relative error (exit 1 if exceeded)')
    val.add_argument('--param', '-p', action='append',
                     help='Override parameter: KEY=VALUE (repeatable)')

    sub.add_parser('list', help='List available metrics with parameters')

    args = parser.parse_args()
    if args.command == 'generate':
        _cmd_generate(args)
    elif args.command == 'validate':
        _cmd_validate(args)
    elif args.command == 'list':
        _cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
