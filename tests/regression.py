"""Small, case based release regression suite.

Run ``python tests/regression.py --lane fast`` for checks without adaptation.
The full lane performs adaptation and is intended for an explicit user run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import pkgutil
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class TestCase:
    name: str
    case: str
    lane: str
    check: object
    reference: object
    tolerance: object


class SkipCase(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def paths(name):
    base = ROOT / "cases" / name
    return base, base / "case.toml"


def check_imports(_work, _threads, _spec):
    import amac_3d

    names = [m.name for m in pkgutil.walk_packages(amac_3d.__path__, "amac_3d.")]
    require(names, "no package modules found")
    for name in names:
        importlib.import_module(name)
    return f"{len(names)} modules"


def check_io(work, _threads, spec):
    from amac_3d.grid import Grid
    from amac_3d.io import (export_mesh, export_vol, import_geo, import_mesh,
                            import_metric, validate_msh_file, validate_vol_file)

    base, _ = paths(spec.case)
    mesh = base / "input/mesh/cube1241.vol"
    metric = base / "input/metric/cross.sol"
    geo = base / "input/geometry/cube.geo"
    raw = import_mesh(str(mesh))
    require(len(import_metric(str(metric))) == len(raw["points"]) * 2,
            "metric and point counts differ")
    require(import_geo(str(geo)), "geometry did not load")
    grid = Grid.from_mesh(str(geo), str(mesh), metric_path=str(metric))
    vol = work / "roundtrip.vol"
    export_vol(grid, str(vol))
    export_mesh(grid, str(work / "roundtrip"), "both")
    reread = import_mesh(str(vol))
    require(validate_vol_file(str(vol))["valid"], "written vol invalid")
    require(validate_msh_file(str(work / "roundtrip.msh"))["valid"], "written msh invalid")
    require(len(reread["points"]) == len(raw["points"]), "vol point count changed")
    require(len(reread["tetrahedra"]) == len(raw["tetrahedra"]), "vol tet count changed")
    require(np.allclose(reread["points"], raw["points"], atol=spec.tolerance),
            "vol coordinates changed")
    msh_text = (work / "roundtrip.msh").read_text()
    nodes = re.search(r"\$Nodes\s+(\d+)\s+(.*?)\$EndNodes", msh_text, re.S)
    require(nodes is not None and int(nodes[1]) == len(raw["points"]) // 3,
            "msh point count changed")
    msh_coords = [float(v) for line in nodes[2].splitlines() for v in line.split()[1:4]]
    require(np.allclose(msh_coords, raw["points"], atol=spec.tolerance),
            "msh coordinates changed")
    vtk_text = (work / "roundtrip.vtk").read_text()
    vtk = re.search(r"POINTS\s+(\d+)\s+\w+\s+(.*?)\s+CELLS\s", vtk_text, re.S)
    require(vtk is not None and int(vtk[1]) == len(raw["points"]) // 3,
            "vtk point count changed")
    vtk_coords = [float(v) for v in vtk[2].split()]
    require(np.allclose(vtk_coords, raw["points"], atol=spec.tolerance),
            "vtk coordinates changed")
    cells = re.search(r"CELL_TYPES\s+(\d+)\s+(.*)", vtk_text, re.S)
    require(cells is not None and cells[2].split().count("10") == len(raw["tetrahedra"]) // 4,
            "vtk tet count changed")
    return f"{len(raw['points']) // 3} points, {len(raw['tetrahedra']) // 4} tets"


def check_metric(_work, _threads, spec):
    from amac_3d.metric import edge_metric_interpolate, length_in_metric, log_metric

    a = [1.0, 0.0, 0.0, 4.0, 0.0, 9.0]
    b = [9.0, 0.0, 0.0, 16.0, 0.0, 25.0]
    grid = SimpleNamespace(coordinates=[0., 0., 0., 1., 0., 0.],
                           metricMesh=a + b, metricLog=log_metric(a) + log_metric(b))
    got = edge_metric_interpolate(grid, 0, 1, [0.5, 0., 0.])
    require(np.allclose(got, [3., 0., 0., 8., 0., 15.], atol=spec.tolerance),
            f"Log-Euclidean midpoint differs: {got}")
    mat = np.array([[got[0], got[1], got[2]],
                    [got[1], got[3], got[4]], [got[2], got[4], got[5]]])
    require(np.min(np.linalg.eigvalsh(mat)) > 0, "interpolated tensor is not SPD")
    unit = [1., 0., 0., 1., 0., 1.]
    require(abs(length_in_metric(unit, unit, [0., 0., 0.], [1., 0., 0.]) - 1.)
            <= spec.tolerance, "unit metric edge length differs")
    return "SPD midpoint and unit edge length"


def check_generator(_work, _threads, _spec):
    # Cylinder is excluded: its bundled file has no recorded generating formula.
    for name in ("cube_cross", "cube_cross_coarse", "cube_egads"):
        base, _ = paths(name)
        refs = sorted((base / "input/metric").glob("*.sol"))
        require(len(refs) == 1, f"{name}: expected one metric file")
        cmd = [sys.executable, str(ROOT / "tools/analytic_metric_generator.py"),
               "validate", "--mesh", str(base / "input/mesh/cube1241.vol"),
               "--metric", "cross", "--reference", str(refs[0]),
               "--tolerance", "1e-12"]
        if name == "cube_cross_coarse":
            cmd += ["--param", "h_min=0.0125"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        require(result.returncode == 0, f"{name}: {result.stdout[-300:]} {result.stderr[-300:]}")
    return "3 cross metrics; cylinder provenance exception"


def check_operators(_work, _threads, _spec):
    from amac_3d.grid import Grid, validate_mesh
    from amac_3d.metric import signed_volume
    from amac_3d.report import count_flat_tets
    from amac_3d.split import split_pass
    from amac_3d.collapse import collapse_pass
    from amac_3d.swap import local_reconnection
    from amac_3d.smooth import smooth_pass

    base, _ = paths("cube_cross")
    grid = Grid.from_mesh(str(base / "input/geometry/cube.geo"),
                          str(base / "input/mesh/cube1241.vol"),
                          metric_path=str(base / "input/metric/cross.sol"))
    for label, operation in (("split", lambda: split_pass(grid)),
                             ("collapse", lambda: collapse_pass(grid)),
                             ("swap", lambda: local_reconnection(grid, max_passes=1, verbose=False)),
                             ("smooth", lambda: smooth_pass(grid, iterations=1, verbose=False))):
        before = grid._mesh_version
        changed = operation()
        require(grid._mesh_version >= before + int(changed > 0),
                f"{label}: mesh version did not increase")
        for tet in np.asarray(grid.tetrahedra).reshape(-1, 4):
            if tet[0] == -1:
                continue
            pts = [grid.coordinates[3 * int(v):3 * int(v) + 3] for v in tet]
            require(signed_volume(*pts) > 1e-12, f"{label}: inverted or flat tet")
        require(count_flat_tets(grid) == 0, f"{label}: flat tets")
    grid.cleanDeadElements()
    require(not np.any(grid.dead_mask), "dead nodes survived compaction")
    require(len(grid.dead_mask) * 3 == len(grid.coordinates), "node arrays disagree")
    require(not np.any(np.asarray(grid.tetrahedra).reshape(-1, 4)[:, 0] == -1),
            "dead tets survived compaction")
    require(all(v == 0 for v in validate_mesh(grid).values()), "mesh integrity failed")
    return "split/collapse/swap/smooth and compaction"


def check_case_loader(work, _threads, _spec):
    from run.adapt import CaseError, load_case

    for case_file in sorted((ROOT / "cases").glob("*/case.toml")):
        out = work / case_file.parent.name
        loaded = load_case(case_file, iterations=3, threads=1, output_dir=str(out))
        require(Path(loaded["run"]["output_dir"]) == out.resolve(),
                "output directory resolution changed")
        require(loaded["adapt"]["iterations"] == 3, "iteration override lost")
    try:
        load_case(paths("cube_cross")[1], output_dir="input/refused")
    except CaseError:
        pass
    else:
        raise AssertionError("input directory output was accepted")
    return "4 case files, overrides and output refusal"


def check_egads(_work, _threads, _spec):
    try:
        from pyEGADS import egads  # noqa: F401
    except ImportError as error:
        raise SkipCase(f"pyEGADS unavailable: {error}") from error
    from amac_3d.egads_adapter import EgadsModel, _project_point_to_face

    base, _ = paths("cube_egads")
    model = EgadsModel(str(base / "input/geometry/cube.step"))
    require(len(model.faces) == 6, "cube CAD face count differs")
    point = np.array([0.5, 0.5, 0.0])
    distances = [_project_point_to_face(face, point)[1] for face in model.faces]
    require(min(distances) <= 1e-8, "cube face projection differs")
    return "six CAD faces and surface projection"


def run_adaptation(case_name, iterations, threads, out):
    command = [sys.executable, "-m", "run.adapt", str(paths(case_name)[1]),
               "--iterations", str(iterations), "--threads", str(threads),
               "--output-dir", str(out), "--force"]
    with (out.parent / f"{out.name}.stdout.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"{case_name} adaptation failed; see {out.name}.stdout.log")
    return out


def check_quality(work, threads, spec):
    out = run_adaptation(spec.case, 12, threads, work / "quality")
    history = json.loads((out / "adaptation_history.json").read_text())
    require(history, "empty adaptation history")
    final = history[-1]
    observed = {"conforming": final["edge_length_stats"]["fraction_conforming"],
                "mean_quality": final["quality_stats"]["mean"],
                "flat_tets": final["n_flat_tets"],
                "below_0001": final["quality_stats"]["fraction_below_0001"]}
    bands = spec.reference
    require(observed["conforming"] >= bands["conforming"] - spec.tolerance,
            f"edge conformance {observed}")
    require(observed["mean_quality"] >= bands["mean_quality"] - spec.tolerance,
            f"mean quality {observed}")
    require(observed["flat_tets"] == 0 and observed["below_0001"] == 0,
            f"flat or very poor tets {observed}")
    return json.dumps(observed, sort_keys=True)


def mesh_hash(path):
    text = path.read_text()
    nodes = re.search(r"\$Nodes\s+\d+\s+(.*?)\$EndNodes", text, re.S)
    elements = re.search(r"\$Elements\s+\d+\s+(.*?)\$EndElements", text, re.S)
    require(nodes is not None and elements is not None, "missing msh sections")
    coords = {}
    for line in nodes[1].splitlines():
        fields = line.split()
        if len(fields) == 4:
            coords[int(fields[0])] = tuple(round(float(x), 12) for x in fields[1:])
    tets = []
    for line in elements[1].splitlines():
        fields = line.split()
        if len(fields) > 4 and fields[1] == "4":
            tets.append(tuple(sorted(map(int, fields[3 + int(fields[2]):]))))
    payload = json.dumps((sorted(coords.items()), sorted(tets)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_threads(work, _threads, _spec):
    hashes = []
    for threads in (1, 4):
        out = run_adaptation("cube_cross", 3, threads, work / f"threads-{threads}")
        bridges = sorted(out.glob("bridge_*.msh"), key=lambda p: int(p.stem.split("_")[-1]))
        require(bridges, "no final bridge mesh")
        hashes.append(mesh_hash(bridges[-1]))
    require(hashes[0] == hashes[1], f"thread hashes differ: {hashes}")
    return hashes[0]


def check_cad_run(work, threads, spec):
    try:
        from pyEGADS import egads  # noqa: F401
    except ImportError as error:
        raise SkipCase(f"ESP/pyEGADS unavailable: {error}") from error
    from amac_3d.egads_adapter import EgadsModel, _project_point_to_face

    out = run_adaptation(spec.case, 3, threads, work / "cad")
    base, _ = paths(spec.case)
    model = EgadsModel(str(base / "input/geometry/cube.step"))
    bridges = sorted(out.glob("bridge_*.msh"), key=lambda p: int(p.stem.split("_")[-1]))
    require(bridges, "CAD run wrote no bridge mesh")
    msh = bridges[-1]
    text = msh.read_text()
    nodes = re.search(r"\$Nodes\s+\d+\s+(.*?)\$EndNodes", text, re.S)
    require(nodes is not None, "CAD run mesh has no nodes")
    boundary = []
    for line in nodes[1].splitlines():
        fields = line.split()
        if len(fields) == 4:
            point = np.array([float(v) for v in fields[1:]])
            if np.any(np.isclose(point, 0., atol=1e-5)) or np.any(np.isclose(point, 1., atol=1e-5)):
                boundary.append(point)
    require(boundary, "no cube boundary nodes found")
    worst = max(min(_project_point_to_face(face, p)[1] for face in model.faces)
                for p in boundary)
    require(worst <= spec.tolerance, f"CAD distance {worst} exceeds {spec.tolerance}")
    return f"maximum CAD distance {worst:.3e}"


CASES = (
    TestCase("imports", "all", "fast", check_imports, "all amac_3d modules", None),
    TestCase("mesh-io", "cube_cross", "fast", check_io, (1241, 6144), 1e-10),
    TestCase("log-metric", "analytic", "fast", check_metric, (3., 8., 15., 1.), 1e-12),
    TestCase("metric-files", "cross cases", "fast", check_generator, "three cross files; cylinder exception", 1e-12),
    TestCase("operators", "cube_cross", "fast", check_operators, "positive volumes and clean compaction", 1e-12),
    TestCase("case-loader", "all", "fast", check_case_loader, "four cases and refusal", None),
    TestCase("egads-projection", "cube_egads", "fast", check_egads, "six faces", 1e-8),
    TestCase("quality-bands", "cube_cross_coarse", "full", check_quality,
             {"conforming": 0.86, "mean_quality": 0.52, "flat_tets": 0, "below_0001": 0}, 0.),
    TestCase("thread-hash", "cube_cross", "full", check_threads, None, None),  # filled by user run
    TestCase("cad-boundary", "cube_egads", "full", check_cad_run, None, 1e-6),  # filled by user run
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("fast", "full"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-references", action="store_true")
    args = parser.parse_args()
    parser.error("--threads must be positive") if args.threads < 1 else None
    selected = [case for case in CASES if case.lane == args.lane]
    if args.lane == "fast":
        with tempfile.TemporaryDirectory(prefix="amac-regression-") as temp:
            return execute(selected, Path(temp), args.threads, args.print_references)
    if args.output_dir is None:
        parser.error("--output-dir is required for the full lane")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return execute(selected, args.output_dir, args.threads, args.print_references)


def execute(selected, work, threads, print_references):
    failures = 0
    for spec in selected:
        try:
            observed = spec.check(work, threads, spec)
        except SkipCase as error:
            print(f"SKIP {spec.name}: {error}", flush=True)
        except Exception as error:
            failures += 1
            print(f"FAIL {spec.name}: {type(error).__name__}: {error}", flush=True)
        else:
            print(f"PASS {spec.name}: {observed}", flush=True)
            if print_references:
                print(f"REFERENCE {spec.name}: {observed}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
