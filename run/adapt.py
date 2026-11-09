"""Run a validated mesh adaptation case from a TOML file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import tomllib


DEFAULTS = {
    "input": {"geometry": "input/geometry/cube.geo", "mesh_prev": "input/mesh/cube1241.vol",
              "mesh": "input/mesh/cube1241.vol", "metric": "input/metric/cross.sol"},
    "adapt": {"iterations": 20, "enable_swap": True, "bridge_format": "msh",
              "collapse_sweeps": 30, "show_collapse_rejections": False,
              "show_split_rejections": False},
    "run": {"output_dir": "output", "output_name": "adapted_mesh",
            "output_format": "vtk", "log_file": "adaptation.log"},
}
INPUT_KEYS = set(DEFAULTS["input"]) | {"egads"}
FORMATS = {"msh", "vol", "both"}
OUTPUT_FORMATS = {"msh", "vtk", "both"}


class CaseError(ValueError):
    """A case file or requested output path is invalid."""


class _TimestampEachLine(logging.Formatter):
    """Keep a timestamp on each line of a multiline log message."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return message.replace("\n", "\n" + self.formatTime(record) + " ")


def _validate(section: str, key: str, value: object) -> None:
    name = f"{section}.{key}"
    if section == "input":
        if not isinstance(value, str) or not value.strip():
            raise CaseError(f"{name}: expected a nonempty path string")
    elif section == "adapt":
        if key in {"iterations", "collapse_sweeps"}:
            if type(value) is not int or not 1 <= value <= 10000:
                raise CaseError(f"{name}: expected integer in 1..10000")
        elif key in {"enable_swap", "show_collapse_rejections", "show_split_rejections"}:
            if type(value) is not bool:
                raise CaseError(f"{name}: expected true or false")
        elif key == "bridge_format" and (type(value) is not str or value not in FORMATS):
            raise CaseError(f"{name}: valid choices: {', '.join(sorted(FORMATS))}")
    elif section == "run":
        if key == "output_format":
            if type(value) is not str or value not in OUTPUT_FORMATS:
                raise CaseError(f"{name}: valid choices: {', '.join(sorted(OUTPUT_FORMATS))}")
        elif not isinstance(value, str) or not value.strip():
            raise CaseError(f"{name}: expected a nonempty path string")
        elif key in {"output_name", "log_file"} and Path(value).name != value:
            raise CaseError(f"{name}: expected a file name without directories")


def load_case(case_file: str | Path, *, iterations: int | None = None,
              threads: int | None = None, output_dir: str | None = None,
              force: bool = False) -> dict:
    """Validate a case without importing or starting the engine."""
    path = Path(case_file).expanduser().resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if not isinstance(raw, dict):
        raise CaseError("case: expected TOML tables input, adapt, run")
    for section, values in raw.items():
        if section not in DEFAULTS:
            raise CaseError(f"{section}: unknown table; valid tables: input, adapt, run")
        if not isinstance(values, dict):
            raise CaseError(f"{section}: expected a table")
        allowed = INPUT_KEYS if section == "input" else set(DEFAULTS[section])
        for key, value in values.items():
            if key not in allowed:
                raise CaseError(f"{section}.{key}: unknown key; valid choices: {', '.join(sorted(allowed))}")
            _validate(section, key, value)
    resolved = {section: {**defaults, **raw.get(section, {})}
                for section, defaults in DEFAULTS.items()}
    if "egads" in raw.get("input", {}):
        resolved["input"]["egads"] = raw["input"]["egads"]
    if iterations is not None:
        _validate("adapt", "iterations", iterations)
        resolved["adapt"]["iterations"] = iterations
    if threads is not None and (type(threads) is not int or threads < 1):
        raise CaseError("threads: expected positive integer")
    case_dir = path.parent
    if output_dir is not None:
        _validate("run", "output_dir", output_dir)
        resolved["run"]["output_dir"] = output_dir
    # Schema errors above precede all input existence checks.
    for key, value in resolved["input"].items():
        _validate("input", key, value)
        target = (case_dir / value).expanduser().resolve()
        if not target.is_file():
            raise CaseError(f"input.{key}: missing file: {target}")
        resolved["input"][key] = str(target)
    out = (case_dir / resolved["run"]["output_dir"]).expanduser().resolve()
    if "input" in out.parts:
        raise CaseError(f"run.output_dir: refused path inside input/: {out}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        raise CaseError(f"run.output_dir: existing non-empty directory: {out}; use --force")
    resolved["run"]["output_dir"] = str(out)
    resolved["case_file"] = str(path)
    resolved["threads"] = threads
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    resolved["engine_commit"] = commit
    return resolved


def run_case(case: dict, *, verbose: bool = False) -> None:
    out = Path(case["run"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("amac_3d")
    logger.setLevel(logging.DEBUG)
    for old_handler in logger.handlers:
        old_handler.close()
    logger.handlers.clear()
    file_handler = logging.FileHandler(out / case["run"]["log_file"])
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_TimestampEachLine("%(asctime)s %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    from amac_3d.grid import Grid
    from amac_3d.adapt import adapt
    from amac_3d.io import export_mesh
    import numba

    if case["threads"] is not None:
        numba.set_num_threads(min(case["threads"], numba.config.NUMBA_NUM_THREADS))
    case["threads"] = numba.get_num_threads()
    (out / "resolved_case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
    data = case["input"]
    bg_mesh = Grid.from_mesh(data["geometry"], data["mesh_prev"],
                             metric_path=data["metric"], egads_path=data.get("egads"))
    start_mesh = Grid.from_mesh(data["geometry"], data["mesh"],
                                metric_function=bg_mesh.metric, egads_path=data.get("egads"))
    case_name = Path(case["case_file"]).parent.name
    logger.debug("Output: %s", out)
    logger.info("amac-3d  %s", case_name)
    logger.info("")
    logger.info("  mesh      %s  (%d nodes, %d tets)", Path(data["mesh"]).name,
                len(start_mesh.coordinates) // 3, len(start_mesh.tetrahedra) // 4)
    logger.info("  metric    %s", Path(data["metric"]).name)
    geometry = Path(data["geometry"]).name
    if data.get("egads"):
        geometry += " + CAD " + Path(data["egads"]).name
    logger.info("  geometry  %s", geometry)
    logger.info("  limit     %d iterations, swap %s, %d threads",
                case["adapt"]["iterations"], "on" if case["adapt"]["enable_swap"] else "off",
                case["threads"])
    logger.info("  output    %s", out)
    logger.info("  log       %s  (full detail)", case["run"]["log_file"])
    adapt(start_mesh, case["adapt"]["iterations"], output_dir=str(out),
          enable_swap=case["adapt"]["enable_swap"],
          bridge_format=case["adapt"]["bridge_format"],
          collapse_sweeps=case["adapt"]["collapse_sweeps"],
          show_collapse_rejections=case["adapt"]["show_collapse_rejections"],
          show_split_rejections=case["adapt"]["show_split_rejections"])
    export_mesh(start_mesh, str(out / case["run"]["output_name"]),
                case["run"]["output_format"])
    logger.info("")
    logger.info("Written:")
    for fmt in ("vtk", "msh"):
        if case["run"]["output_format"] in (fmt, "both"):
            logger.info("  %s.%s   final mesh", case["run"]["output_name"], fmt)
    history = json.loads((out / "adaptation_history.json").read_text())
    last_bridge = history[-1]["iteration"] - 1
    bridge_format = case["adapt"]["bridge_format"]
    for fmt in ("msh", "vol"):
        if bridge_format in (fmt, "both"):
            logger.info("  bridge_0..%d.%s   mesh after each iteration", last_bridge, fmt)
    logger.info("  adaptation_history.json   per-iteration statistics")
    logger.info("  %s   full log", case["run"]["log_file"])
    logger.info("  resolved_case.json   the case as run")
    logger.debug("Adaptation finished.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_file")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show full adaptation detail on the console")
    parser.add_argument("--check", action="store_true", help="validate and print the resolved case without running")
    args = parser.parse_args()
    try:
        case = load_case(args.case_file, iterations=args.iterations, threads=args.threads,
                         output_dir=args.output_dir, force=args.force)
    except (CaseError, OSError, tomllib.TOMLDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(case, indent=2, sort_keys=True))
        return 0
    try:
        run_case(case, verbose=args.verbose)
    except Exception:
        logging.getLogger("amac_3d").debug("Adaptation failed", exc_info=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
