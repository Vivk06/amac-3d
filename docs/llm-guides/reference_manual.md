# Reference manual

AMAC adapts a tetrahedral mesh toward unit edge length in a spatial metric. This manual states the numerical model and the public file and command interfaces. The [code manual](code_manual.md) describes implementation ownership and invariants.

## Metric and quality

A metric field assigns each point a symmetric positive-definite tensor $M(x)$. AMAC stores its upper triangle at each node as $(M_{11},M_{12},M_{13},M_{22},M_{23},M_{33})$. For a constant tensor, edge vector $e$ has length $\ell_M(e)=\sqrt{e^T M e}$. An isotropic tensor $M=h^{-2}I$ asks for physical spacing $h$. A metric-conforming edge lies in $[1/\sqrt2,\sqrt2]$, about $[0.707,1.414]$.

For endpoint lengths $\ell_a$ and $\ell_b$, `length_in_metric` uses the logarithmic mean $(\ell_b-\ell_a)/\log(\ell_b/\ell_a)$, with a near-equality branch to avoid cancellation. It approximates the continuous path integral of metric length; it need not equal direct quadrature of a tensor that rotates along the edge. This distinction matters for strongly anisotropic fields.

For positive-definite tensors $M_1,M_2$, interpolation uses $M(t)=\exp((1-t)\log M_1+t\log M_2)$. `Grid.metric` uses barycentric weights inside a background tet and exponentiates the weighted node logs. The eigenvalue floor in `log_metric` is $10^{-300}$ for numerical finiteness. A supplied metric should still have strictly positive eigenvalues; the floor does not validate bad input. `Grid.resample_node_metric` refreshes both tensor and log tensor after a split or accepted move.

The tetrahedron quality is $Q=C V_0^{2/3}/\sum_{i=1}^{6}\ell_{M,i}^2$, where $C=36/\sqrt[3]{3}$, $V_0=|V|\sqrt{\min_j\det M_j}$, and $V$ is signed Euclidean volume. A regular isotropic tetrahedron has $Q=1$. An exactly flat tet has $Q=0$. Because the formula uses $|V|$, operator validity uses a separate positive-signed-volume check. Triangle quality uses metric-scaled area and a three-edge denominator; collapse checks it on boundary faces. The Marcum-Alauzet edge weight diagnoses wide metric dihedral angles; higher is worse, and the history reports its maximum and mean without using them as stop criteria.

A local cavity operation passes the Loseille-Löhner gate when its new worst quality satisfies $q_{new}\ge q_{old}/q_r$ and $q_{new}\ge q_a$. Here $q_r\ge1$ bounds relative decrease and $q_a$ is an absolute floor. Split uses an effectively unbounded relative factor and a local floor. Collapse uses $q_r=2$ and an annealed floor. Swap combines dihedral improvement with the relevant phase gate. Smooth accepts only a safe ball move with no material quality loss and the same floor.

## ABCD iteration

| Step | Operation | Main acceptance feature |
| --- | --- | --- |
| A1, A4 | Swap before and after collapse | Improve a poor local configuration; preserve volume |
| A2 | Collapse below $1/\sqrt2$ until quiescent | Duplicate-cell, volume, quality, length and boundary checks |
| B1, B2 | Smooth, then swap | Constrained movement followed by reconnection |
| C1 | Split long edges above $\sqrt2$ | Metric midpoint, valid sub-tets and projected boundary |
| C1b | Collapse below 0.6 until quiescent | Post-split hysteresis prevents immediate reversal |
| C2 | Swap | Restore local quality |
| D1, D2 | Smooth, then swap | Equilibrate after topology changes |
| Quality and output | Check flat tets, try guarded repair, write bridge and history | Report unresolved flat tets |

`adapt` anneals the floor at the start and before C as $q_a=\max(10^{-3},\min(0.1,Q_{worst}))$, using the smallest positive quality. A ratcheting edge band starts no stricter than current extremes and tightens as the mesh improves. Collapse to quiescence stops when a sweep finds no collapse or fewer than 2% of the first sweep's count, with `collapse_sweeps` as a cap. The iteration cap is an upper bound: after at least five iterations, two consecutive comparisons with less than 1% change in both minimum and maximum edge length can stop the loop, subject to the `nodeAge` guard. The exact rule is in `src/amac_3d/adapt.py`.

The complexity estimate is $\mathcal C\approx\sum_K |K|\,\frac14\sum_{v\in K}\sqrt{\det M_v}$. `compute_complexity` reports it and the ratio of nodes to complexity. The engine reports the ratio; it does not enforce a node-budget throttle.

## Bundled analytic fields

`tools/analytic_metric_generator.py` samples an analytic tensor at each node of a Netgen mesh. Its `cross` field sets $\alpha_d=a|x_d-c_d|$, $h_d=\min(2^{\alpha_d}h_{min},h_{max})$, and $M=s\,\mathrm{diag}(h_x^{-2},h_y^{-2},h_z^{-2})$. Defaults from the current function are $h_{min}=0.003125$, $h_{max}=0.0625$, $a=40$, centers $(0.5,0.5,0.5)$, and scale $s=0.1$. Three intersecting planes receive fine spacing normal to themselves. The global scale multiplies tensor entries, so the physical unit-length spacing is $h_d/\sqrt{s}$.

For `cylinder`, let $r=\sqrt{x^2+y^2}$, $\theta=\operatorname{atan2}(x,y)$, and $\alpha=a|r_c-r|$. Set $h_1=\min(h_{1,b}g_1^\alpha,h_{max})$ and $h_2=\min(h_{2,b}g_2^\alpha,h_{max})$. With $c=\cos\theta$ and $t=\sin\theta$, the in-plane tensor has $M_{11}=s(h_1^{-2}c^2+h_2^{-2}t^2)$, $M_{12}=s(h_1^{-2}-h_2^{-2})ct$, and $M_{22}=s(h_1^{-2}t^2+h_2^{-2}c^2)$; $M_{33}=s h_z^{-2}$ and the other entries vanish. Defaults are $h_{1,b}=0.002$, $g_1=5$, $h_{2,b}=0.05$, $g_2=2$, $h_{max}=0.1$, $h_z=0.003125$, $a=10$, $r_c=0.75$, and $s=0.03$. The fine eigenvector is $(y,x,0)/r$, which need not align with the radial vector. The fixed bundled cylinder file has separate provenance from these defaults.

The `uniform` field returns $M=s h^{-2}I$, with $h=0.1$ and $s=1$ by default. Each field exposes its parameters through the generator's `list` subcommand; `--param KEY=VALUE` overrides them for generation or validation.

## Case file schema

`run/adapt.py` merges each listed default with the corresponding TOML table. These 15 rows come from the current loader schema and validation. Paths in `[input]` resolve relative to the case file and must name existing files. The optional CAD path has no default.

| Key | Type | Default | Allowed | Effect |
| --- | --- | --- | --- | --- |
| `input.geometry` | path string | `input/geometry/cube.geo` | nonempty; existing file | selects input |
| `input.mesh_prev` | path string | `input/mesh/cube1241.vol` | nonempty; existing file | selects input |
| `input.mesh` | path string | `input/mesh/cube1241.vol` | nonempty; existing file | selects input |
| `input.metric` | path string | `input/metric/cross.sol` | nonempty; existing file | selects input |
| `input.egads` | path string | `(absent)` | nonempty; existing file | selects input |
| `adapt.iterations` | integer | `20` | 1..10000 | maximum ABCD cycles |
| `adapt.enable_swap` | boolean | `true` | true or false | local reconnection |
| `adapt.bridge_format` | string | `msh` | msh, vol, both | per-iteration bridge format |
| `adapt.collapse_sweeps` | integer | `30` | 1..10000 | collapse pass limit |
| `adapt.show_collapse_rejections` | boolean | `false` | true or false | collapse debug messages |
| `adapt.show_split_rejections` | boolean | `false` | true or false | split debug messages |
| `run.output_dir` | path string | `output` | nonempty; outside input/ | run directory |
| `run.output_name` | string | `adapted_mesh` | nonempty basename | final mesh stem |
| `run.output_format` | string | `vtk` | msh, vtk, both | final mesh format |
| `run.log_file` | string | `adaptation.log` | nonempty basename | full log name |

Count: 15

The loader refuses any table beyond `[input]`, `[adapt]`, `[run]`, and any key absent from the table. It rejects wrong types, empty paths, missing files, unsupported format values, integer counts outside 1..10000, and an existing nonempty output directory without `--force`. It refuses any output path with an `input/` component. `--threads` must be a positive integer. Worker count, threading layer, `fastmath`, kernel selection, quality thresholds and numerical tolerances are deliberately not case settings; the engine owns those contracts. A setting becomes a case key only when a user needs it and its default reproduces the bundled reference cases.

## Command line

The table below is generated from `argparse` calls in `run/adapt.py` and `tools/*.py`. A command may have subcommands: `analytic_metric_generator.py generate`, `validate`, and `list`. `-h` and `--help` are supplied by `argparse` to each parser and subparser. The table contains 57 names, counting positional arguments, explicit flags and these help aliases. `--check` validates a case without importing the engine or writing output. `--force` permits writing into an existing nonempty directory; older files not replaced by this run may remain. `--iterations` and `--output-dir` override case values; `--threads` selects Numba threads, capped at its launch limit. `--verbose` changes the console log level, while the file log retains full detail.

| Command | Argument | Default / meaning |
| --- | --- | --- |
| `python run/adapt.py` | `case_file` |  |
| `python run/adapt.py` | `--iterations` |  |
| `python run/adapt.py` | `--threads` |  |
| `python run/adapt.py` | `--output-dir` |  |
| `python run/adapt.py` | `--force` |  |
| `python run/adapt.py` | `--verbose` | show full adaptation detail on the console |
| `python run/adapt.py` | `--check` | validate and print the resolved case without running |
| `python run/adapt.py` | `-h, --help` | Print usage and exit. |
| `python tools/analytic_metric_generator.py generate` | `--mesh` | Input Netgen .vol mesh file |
| `python tools/analytic_metric_generator.py generate` | `--metric` | Analytic metric function name |
| `python tools/analytic_metric_generator.py generate` | `--output, -o` | Output .sol file name (default: auto-named) |
| `python tools/analytic_metric_generator.py generate` | `--output-dir` | Output directory (default: case output/ or mesh directory; input/ refused) |
| `python tools/analytic_metric_generator.py generate` | `--param, -p` | Override parameter: KEY=VALUE (repeatable) |
| `python tools/analytic_metric_generator.py validate` | `--mesh` | Input Netgen .vol mesh file |
| `python tools/analytic_metric_generator.py validate` | `--metric` | Analytic metric function name |
| `python tools/analytic_metric_generator.py validate` | `--reference` | Reference .sol file to compare against |
| `python tools/analytic_metric_generator.py validate` | `--verbose, -v` | Show worst nodes with full metric values |
| `python tools/analytic_metric_generator.py validate` | `--tolerance, -t` | Max acceptable relative error (exit 1 if exceeded) |
| `python tools/analytic_metric_generator.py validate` | `--param, -p` | Override parameter: KEY=VALUE (repeatable) |
| `python tools/analytic_metric_generator.py` | `-h, --help` | Print usage and exit. |
| `python tools/analytic_metric_generator.py generate` | `-h, --help` | Print usage and exit. |
| `python tools/analytic_metric_generator.py list` | `-h, --help` | Print usage and exit. |
| `python tools/analytic_metric_generator.py validate` | `-h, --help` | Print usage and exit. |
| `python tools/analyze_mesh.py` | `--geo` | .geo geometry file |
| `python tools/analyze_mesh.py` | `--mesh` | .vol mesh file |
| `python tools/analyze_mesh.py` | `--metric` | .sol metric file |
| `python tools/analyze_mesh.py` | `--bg-mesh` | background/prev .vol mesh that --metric's .sol was sampled on (reproduces the adaptation pipeline's Log-Euclidean interpolation via Grid.metric, instead of a pointwise --metric read on --mesh itself) |
| `python tools/analyze_mesh.py` | `--output` | output PNG file name (default: mesh stem) |
| `python tools/analyze_mesh.py` | `--output-dir` | output directory (default: case output/ or mesh directory; input/ refused) |
| `python tools/analyze_mesh.py` | `--title` | figure suptitle |
| `python tools/analyze_mesh.py` | `--bins` | default `60`; number of histogram bins |
| `python tools/analyze_mesh.py` | `-h, --help` | Print usage and exit. |
| `python tools/convert_mesh.py` | `input_msh` | Input .msh file |
| `python tools/convert_mesh.py` | `output_vol` | Output .vol file name under --output-dir (default: input stem + .vol) |
| `python tools/convert_mesh.py` | `--output-dir` | Output directory (default: case output/ for case input meshes, otherwise the mesh directory; never inside input/) |
| `python tools/convert_mesh.py` | `--geo` | Path to .geo file for surface reclassification |
| `python tools/convert_mesh.py` | `--with-edges` | Generate edgesegmentsgi2 section |
| `python tools/convert_mesh.py` | `--verbose` | Print detailed diagnostics |
| `python tools/convert_mesh.py` | `-h, --help` | Print usage and exit. |
| `python tools/egads_boundary_fidelity.py` | `--geo` | Path to .geo file |
| `python tools/egads_boundary_fidelity.py` | `--mesh` | Path to .vol mesh file |
| `python tools/egads_boundary_fidelity.py` | `--step` | Path to STEP CAD file |
| `python tools/egads_boundary_fidelity.py` | `--metric` | Optional: path to .sol metric file |
| `python tools/egads_boundary_fidelity.py` | `-h, --help` | Print usage and exit. |

Names: 57

The metric generator's `list` subcommand prints registered fields; `generate` writes a Medit tensor file; `validate` compares a generated field with a supplied reference. `--param KEY=VALUE` repeats to change field parameters. The bundled analytic fields are `cross`, `cylinder`, and `uniform`. The supplied `cylinder.sol` does not match the generator's current default cylinder parameters; retain it as a fixed case input and supply explicit parameters when producing a new metric.

## Files and output layout

Internal node indices are zero-based; mesh-file indices start at one. Use the reader and writer named below when exchanging a file.

| Format | Content and constraints | Reader or writer |
| --- | --- | --- |
| `.geo` | Netgen CSG. Analytical import recognizes `plane` and `sphere`; other CSG statements do not create analytical projection surfaces. Surface IDs follow parsed primitive order. | `import_geo` in `src/amac_3d/io.py` |
| `.vol` | Netgen keyword sections: `points`, `volumeelements`, `surfaceelements`, `facedescriptors`, optional `edgesegmentsgi2`. Tet rows have four nodes. Face tags come from geometry classification, with file descriptors as fallback. Ridge and corner classification needs edge segments. | `import_mesh`, `export_vol` |
| `.sol`, `.mtr` | Medit `SolAtVertices` with one six-component symmetric tensor per `.vol` point in order. The reader expects the project's exact header and trailer layout; check SPD and component order before import. | `import_metric`, `tools/analytic_metric_generator.py` |
| `.msh` | Gmsh 2.2 ASCII, physical surface names and tet elements; bridge export and optional final export. | `export_mesh`, `tools/convert_mesh.py` |
| `.vtk` | Legacy VTK visualization mesh; final export only. | `export_mesh` |
| `.step`, `.iges` | CAD geometry for optional pyEGADS projection, paired with a `.geo` and mesh. | `src/amac_3d/egads_adapter.py` |

`tools/convert_mesh.py` converts a Gmsh 2.2 `.msh` to Netgen `.vol`. `--geo` can reclassify surface faces; `--with-edges` writes `edgesegmentsgi2` so ridge and corner levels survive reimport. A `.vol` exported without edge segments cannot reconstruct those levels from that file alone.

The case's `[run].output_dir` defaults to `output` under its case directory. Relative output paths resolve there; absolute paths stay absolute; `--output-dir` overrides both. A normal run writes:

| File | Meaning |
| --- | --- |
| `resolved_case.json` | Absolute inputs, effective settings, chosen thread count, output path and engine commit when available. |
| `adaptation.log` by default | Full time-stamped engine and driver detail. |
| `bridge_0.msh`, `bridge_1.msh`, ... | Mesh after each completed iteration; extension follows `adapt.bridge_format`. |
| `adaptation_history.json` | Array of per-iteration records, including counts, edge-length and quality statistics, complexity, and stop information. |
| `<run.output_name>.vtk` by default | Final visualization mesh; extension follows `run.output_format`. |

The console gives one row per iteration, a stop reason, first/final comparison and written-file list. The JSON history keeps its stable metric keys; the displayed maximum dihedral is a display measure, not a new JSON key. `tools/analyze_mesh.py` writes metric and physical dihedral and metric edge-length histograms to a chosen directory. `tools/egads_boundary_fidelity.py` reports CAD boundary distance on stdout. Each writing tool accepts output-directory control and refuses output inside `input/`.

## Quality bands and known limits

| Measure | Interpretation or comparison band |
| --- | --- |
| Conforming edges | Fraction with metric length in $[1/\sqrt2,\sqrt2]$. Reported, not a fixed 95% pass criterion. |
| Mean Knupp $Q$ | A project quality target is above 0.3 for conditioned study cases. |
| Minimum $Q$ | A value below 0.01 identifies a poor tet; exact zero and negative signed volume need separate checks. |
| Flat tets | Desired count zero; flatness uses $6|V|/\max_e\|e\|^2\le10^{-15}$ or all edges essentially zero. |
| Matched engine comparison | In the historical five-case, one-process, four-thread, 12-iteration comparison: conformity within 3 percentage points, mean $Q$ no more than 0.05 lower, zero flat tets, complexity within 10%. These are comparison bands, not universal output guarantees. |

The historical maximum-size cross metric ($h_{max}=0.125$) was run for 20 iterations on 2026-11-05, in one process with four Numba threads, using the study engine at `4da7d28` with the documented surface-label and reprojection fixes. It ended with 6,896 tets, 76.47% conforming edges, mean $Q=0.2778$, minimum $Q=2.690\times10^{-3}$ and zero flat tets. Its mean remains below 0.3 and some tets remain below $Q=0.01$. This is a historical sensitivity setting rather than one of the four shipped case files; the metric can be generated from the cross field with $h_{max}=0.125$. These numbers do not describe the shipped `cube_cross` default metric or a new run at HEAD.

CAD projection is optional and depends on a working pyEGADS/ESP installation. `EgadsModel` accepts one body; it does not build an initial mesh. A topology mapping below 95% face-triangle confidence warns and proceeds. Failed inverse evaluation can fall back without an adaptation-time fallback count or snap tolerance. The analytical `.geo` importer handles planes and spheres, not arbitrary CSG primitives; the mesh-based curved-boundary fallback loses fidelity on coarse meshes. CAD boundary fidelity depends on the supplied STEP or IGES model and topology map. Conversion without edge segments loses ridge classification. The engine does not generate a curvature-aware metric from CAD. The cylinder metric's original generator settings were not recorded; current defaults have a measured maximum relative mismatch of 0.97 against that bundled file.

## References

- Loseille and Alauzet (2011), continuous mesh framework and complexity.
- Loseille and Löhner (2010), cavity operators and local quality gates.
- Knupp (2001), algebraic tetrahedron quality.
- Michal and Krakos (2012), metric edge length and tetrahedron quality equations.
- Vlachos et al. (2001), PN triangles for the mesh-based curved-boundary fallback.
- Balan, Park and Anderson, NASA Refine, for the operator-cycle model and implementation comparisons.
- Tavva, thesis Chapter 4, for the cross and cylinder analytic metric equations.
