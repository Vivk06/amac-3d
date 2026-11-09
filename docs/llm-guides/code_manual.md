# Code manual

AMAC adapts a tetrahedral mesh to a spatially varying, symmetric positive-definite metric. The public driver is `run/adapt.py`; the engine is the `amac_3d` package under `src/amac_3d/`. Read the [reference manual](reference_manual.md) for equations, case keys, file formats, and quality measures.

## Package map and call flow

| Owner | Responsibility | Main entry points |
| --- | --- | --- |
| `run/adapt.py` | Validate a TOML case, establish output and logging, call the engine | `load_case`, `run_case` |
| `src/amac_3d/grid.py` | Mutable mesh, metrics, boundary classification, compaction and caches | `Grid`, `Grid.from_mesh`, `Grid.metric`, `Grid.cleanDeadElements` |
| `src/amac_3d/io.py` | Netgen, Gmsh, VTK and Medit readers and writers | `import_mesh`, `import_metric`, `export_mesh` |
| `src/amac_3d/metric.py` | SPD arithmetic, metric lengths, local quality and acceptance gate | `length_in_metric`, `edge_metric_interpolate`, `tet_quality` |
| `src/amac_3d/adj.py`, `src/amac_3d/cavity.py`, `src/amac_3d/search.py`, `src/amac_3d/octree.py` | Face adjacency, local shells and balls, background point location | `build_adjacency`, `search.locate` |
| `src/amac_3d/split.py`, `src/amac_3d/collapse.py`, `src/amac_3d/swap.py`, `src/amac_3d/smooth.py` | Local mesh operations | `split_pass`, `collapse_pass`, `local_reconnection`, `smooth_pass` |
| `src/amac_3d/geom.py`, `src/amac_3d/egads_adapter.py`, `src/amac_3d/intersection.py` | Analytic or CAD projection and collision checks | `check_local_intersection` |
| `src/amac_3d/kernels.py` | Compiled array and geometric kernels | Numba functions called by the modules above |
| `src/amac_3d/adapt.py`, `src/amac_3d/report.py` | ABCD cycle, termination, diagnostics and history | `adapt`, `AdaptationTracker` |

`run_case` builds a background `Grid` with the case's metric, then a working `Grid` whose `metric_function` points to `bg_mesh.metric`. The background grid remains fixed. As operators add or move working nodes, `Grid.resample_node_metric` evaluates that background field and refreshes both metric representations. `adapt` mutates the working grid through the ABCD cycle, writes a bridge and history after each iteration, and `export_mesh` writes the final mesh. The driver makes exactly two `Grid.from_mesh` calls, one `adapt` call, and one `export_mesh` call. It sets the Numba thread count only when the CLI requests it.

### Callable entry points

The public calls that usually cross a module boundary are below. The operator functions mutate `Grid`; readers and diagnostics return data. Read their current signatures in the named modules before adding a caller.

| Module | Calls and return role |
| --- | --- |
| `grid.py` | `Grid.from_mesh` constructs from geometry, volume mesh and either metric file or callable; `Grid.metric` samples a point; `Grid.cleanDeadElements` compacts; `Grid.resample_node_metric` refreshes a node; `Grid.update_active_edges` and `Grid.get_scan_edges` manage split candidates. |
| `io.py` | `import_geo`, `import_mesh`, `import_metric` read source formats; `export_mesh`, `export_bridge`, `export_vol` write meshes; `convert_msh_to_vol`, `validate_msh_file`, `validate_vol_file` support conversion and checks. |
| `metric.py` | `length_in_metric`, `length_in_node_metric`, `log_metric`, `exp_metric`, `edge_metric_interpolate`, `tet_quality`, `face_quality`, `check_loseille_lohner_gate` define the numeric contract. |
| `split.py`, `collapse.py` | `split_pass`, `collapse_pass`, `repair_flat_tets` run local changes; `count_hanging_nodes` diagnoses a split mesh. |
| `swap.py`, `smooth.py` | `local_reconnection`, `smooth_pass` run the other two operators; `dihedral_angle` reports six angles. |
| `geom.py`, `egads_adapter.py` | `project_to_surface`, `project_to_curve` dispatch projection; `EgadsModel` and `build_topo_map` load optional CAD topology. |
| `report.py` | `compute_edge_lengths`, `compute_element_qualities`, `compute_complexity`, `count_flat_tets` observe the grid; `AdaptationTracker` records iterations. |

The main library sequence is `Grid.from_mesh` for the fixed background, `Grid.from_mesh` with that grid's `metric` callable for the working mesh, `adapt`, then `export_mesh`. Direct callers own logging and output-directory policy; `run/adapt.py` supplies those around the same sequence.

## Grid state

All node arrays include dead slots until `cleanDeadElements` compacts them. `Np` counts node slots; `Nt` counts tetrahedron slots. Internal indices start at zero; mesh files use one-based indices.

| Field | Shape or role | Contract |
| --- | --- | --- |
| `coordinates` | `float64[3Np]` | Three Cartesian coordinates per node; dead slots may contain NaN. |
| `tetrahedra` | `int32[4Nt]` | Four node indices per tet; a dead tet uses four `-1` values. Loaded tets have positive signed volume after orientation normalization. |
| `metricMesh`, `metricLog` | `float64[6Np]` each | Upper triangular SPD tensor and its matrix logarithm, packed as 11, 12, 13, 22, 23, 33. |
| `bgMetricFunction` | callable or absent | Samples the fixed background field after movement or insertion. |
| `dead_mask` | `bool[Np]` | Authoritative dead-node test; coordinate NaN and metric sentinels are storage details. |
| `ballOfNode`, `ball_ptr`, `ball_idx` | mutable dictionary and CSR | Incident tet sets agree after each committed topology operation. The CSR refreshes when its topology version is stale. |
| `adjacents`, `adjPointTets`, `octree` | adjacency and search | Built at load or compaction; background point location uses them on a fixed grid. |
| `pointLevels`, `nodeAge` | `int[Np]` | Boundary mobility class and smoothing or collapse retry age. |
| `faceSurfMap` | packed face to surface ID | Authoritative boundary-face tag; node maps cannot replace it. |
| `bdyFaceMap`, `bdyEdgeMap`, `cornerPointSurfMap`, `cornerEdgeMap` | boundary maps | Keep surface, ridge and corner classification consistent with face tags. |
| `_mesh_version`, `_topo_version` | mutation counters | Geometry and diagnostic caches key on the former; the incident-tet CSR keys on the latter. |
| `_active_scan_edges` | split candidate cache | Restricts split scans when a proper subset of edges is active; collapse scans all edges. |

`Grid.from_mesh` imports geometry and mesh, normalizes winding, classifies boundary faces, builds adjacency, and installs the metric. `Grid.metric` locates a background tet, takes barycentric weights, and exponentiates a weighted sum of node log metrics. `cleanDeadElements` renumbers live nodes and tets and remaps every node-keyed field and boundary dictionary. A new per-node array must grow at split and compact here.

## Cycle and operator boundaries

Each iteration follows A: swap, collapse, swap; B: smooth, swap; C: split, collapse at the narrower post-split threshold, swap; D: smooth, swap. The loop anneals its quality floor and active-edge set at the start and before C. A quality check and optional flat-tet repair follow D. The final bridge and `adaptation_history.json` record the state after the pass. `iterations` caps the loop; the stable-edge-extremes rule may stop it earlier.

| Mutation | Required checks before commit | State after commit |
| --- | --- | --- |
| Split | Metric-length candidate, valid new tets, positive signed volume, quality and local intersection; project constrained nodes | Add coordinates, metric and log metric, age, boundary classification and incident tets; invalidate caches |
| Collapse | Boundary compatibility, duplicate-cell test, positive volume, quality and metric-length bands; local intersection for boundary cavities | Mark the removed node and tets dead; update incident tets and face tags; compact after the collapse phase |
| Swap | Valid local reconnection, signed volume, volume conservation and dihedral or quality criterion | Update tet connectivity and face adjacency; compact after a changing swap phase |
| Smooth | Restrict movement by boundary class, project to geometry, enforce no-worse ball quality and local intersection | Commit safe moves in a stable order; resample metric; bump mesh version |

Pure-interior collapse cavities skip the local intersection scan after the positive-volume and duplicate-cell checks. A boundary cavity still needs it. The skip belongs at the collapse call site; it is not a general exemption for moves. Active-edge restriction belongs to split only: collapse needs every short edge at every node. The post-split collapse threshold of 0.6 leaves a hysteresis interval below the ordinary `1/sqrt(2)` threshold, reducing split-collapse churn.

## Compiled kernels and threads

Numba kernels use `cache=True` and `nogil=True`; they do not use `fastmath`. The environment selects TBB with `NUMBA_THREADING_LAYER=tbb`. Read-only candidate scans and diagnostics can run in parallel. Topology commits remain ordered. `smooth_pass` uses its serial interior path below 2,000 interior nodes; above that count it uses distance-two colouring. Nodes of one colour evaluate trials on disjoint neighbourhoods with `prange`, then commits occur in node order. Surface and ridge nodes stay on the serial path. Colouring can change the result relative to serial smoothing because visit order changes; thread count within the coloured path does not change its commit order.

## Engine contracts and failure modes

| Symptom | Cause | Fix | Detection |
| --- | --- | --- | --- |
| Negative starting volumes | File winding varies | Construct through `Grid`, which normalizes orientation on load | Check signed volumes after import; fast regression mesh I/O case |
| Surface IDs change after export | Node classifications substituted for face tags | Carry `faceSurfMap` through split, collapse, compaction and export | Reimport exported boundary faces and compare IDs |
| Folded or overlapping tets | A local operation skipped duplicate-cell, positive-volume, conservation or intersection checks | Keep each operator's full guard set; use the interior intersection exemption only for pure-interior collapse | Local cavity tests and post-run flat, signed-volume and conversion checks |
| Wrong boundary projection | Smoothing mixed neighbours from distinct surfaces or skipped post-collapse projection | Filter ridge and corner neighbours by the node's surface; project and revalidate every boundary move | Boundary distance and surface-tag checks |
| SPD metric changes meaning | Arithmetic tensor interpolation or stale metric after moving a node | Interpolate matrix logarithms; resample after a move or split | Log/exp and background interpolation checks |
| Corrupt dictionaries after cleanup | A new node-keyed field was not remapped | Add it to split growth and `cleanDeadElements` compaction | Compaction and boundary classification checks |
| Stale diagnostics or geometry | Mutation omitted `_mesh_version` or topology omitted `_topo_version` | Bump the appropriate counter on every commit; key caches on those counters | Compare cached and fresh diagnostics after each operator |
| Premature stop | Added a second convergence rule or ignored `nodeAge` | Keep the existing stable-extremes and age guard in `adapt` | Compare reported stop reason and final history |
| Inverted tet hidden by rewinding | Operation changed tet order to mask negative volume | Reject the operation when signed volume is nonpositive | Signed-volume checks after operators |
| Excessive topology cost | Per-face full tet scan or full intersection scan on pure-interior collapse | Use packed-face adjacency and the proven interior cavity skip | Profile candidate and collapse passes |
| Nonreproducible timing or mesh | Interpreter, TBB layer or thread count changed | Activate the pinned environment and record interpreter, threads and engine commit | Compare `resolved_case.json` and thread-count runs |

Additional live failure modes need the same symptom, cause, fix and detection review:

| Symptom | Cause | Fix | Detection |
| --- | --- | --- | --- |
| Unused switches or divergent helper paths | A speculative abstraction duplicates the working path | Add the smallest measured change and remove dead paths | Search callers and compare outcomes before keeping a branch |
| Long edges survive in a poor shell | Split uses a strict relative quality gate | Keep split's permissive relative gate with its local absolute floor | Inspect accepted split count and maximum metric edge length |
| Quality drifts after node movement | New or moved node retains its old metric | Resample the fixed background metric and refresh the log tensor | Compare metric values before and after a known move |
| Wrong conformity from analysis | A metric sampled on the background is read pointwise on a different working mesh | Pass `--bg-mesh` to `tools/analyze_mesh.py` | Compare tool output with adaptation history on the same bridge |
| Ambiguous-array error | A NumPy array is used as a Boolean | Test `size`, a scalar element, or an explicit reduction | Exercise empty, single-element and multi-element arrays |
| Point equality raises or branches incorrectly | Array `==` returns elementwise values | Use an explicit all-coordinate comparison | Exercise equal and unequal coordinate arrays |
| Split produces mismatched array lengths | Per-node array growth or compaction was omitted | Extend at split and remap in `cleanDeadElements` | Check all node-field lengths after split and cleanup |
| Intersection scan hangs or reads stale incident tets | CSR ball was not refreshed after topology mutation | Key it on `_topo_version` and rebuild before the scan | Compare CSR and dictionary incident-tet sets after each operator |
| Split misses nonconforming edges | Active-edge list survived a topology or annealing change | Recompute or clear it at the documented boundaries | Compare cached candidates with a full edge scan |
| Dead-node values leak into arithmetic | Code tested NaN coordinates or a metric sentinel instead of the mask | Use `dead_mask` for all live-node decisions | Seed a dead slot with NaN and inspect every candidate path |
| Boundary collision after a collapse | A boundary cavity was misclassified as pure interior | Confirm `pointLevels` and classify the entire cavity before the intersection exemption | Construct a cavity that touches a boundary face |
| Numba failure arrives after a long run begins | A newly compiled path was never called in the short lane | Warm representative signatures and inspect compiled modes | Run the fast lane on every changed kernel signature |
| Parallel scan has the right count but wrong members | Threads append through a shared counter | Write a per-edge mask and compact after the scan | Compare candidate sets, not only counts, against serial |
| Collapse quality degrades | Candidate scan was restricted to split's active-edge cache | Scan all short edges during collapse | Compare the full edge set and final conformity |
| Wrong CAD face projection | Mesh surface ID was assumed to equal STEP face index | Build the topology map from geometric correspondence | Measure boundary distance by mapped face |
| CAD projection falls back unexpectedly | Inverse evaluation cannot find a valid face point | Treat failure as fallback and inspect its rate and location | Run boundary fidelity on the adapted mesh |
| Duplicate CAD objects compare unequal | Python object identity is used for EGADS topology | Use the adapter's topology equality operation | Compare two handles for the same face |
| Timings change between runs | Numba thread count was not pinned | Set `--threads` and record it | Check `resolved_case.json` and runtime environment |
| A comparison target disappears | `--force` writes into an existing directory | Archive or choose a new output path before rerunning | List the directory and resolved case before launch |
| Start-row node count seems wrong | The report's first row is after iteration one | Use the imported mesh for pre-adaptation counts | Compare initial import with the first bridge |
| A converter appears to cure flat tets | Conversion removed degenerate elements | Read `n_flat_tets` in `adaptation_history.json` | Compare the history with converter diagnostics |
| Metric log contains infinities | Eigenvalues reached zero before logarithm | Apply `EIG_FLOOR` in `log_metric`; reject invalid supplied fields | Inspect eigenvalues and finite log tensors |
| Hash comparison fails across different execution modes | Different worker-era results or algorithm states are compared | Hash only matched engine and thread conditions | Compare `resolved_case.json` before bridge hashes |
| Cycle reorder lowers quality on another metric | One case alone looked acceptable | Compare the conditioned set of coarse, fine and maximum-size cross cases | Inspect conformity, mean quality, flat tets and complexity per case |
| Nonmanifold face after collapse | New tet face duplicates one outside the edge shell | Keep the duplicate-cell guard on ordinary and repair paths | Count incident live tets per face after large collapse sweeps |
| Flat or inverted boundary tet after reprojection | A new face inherited the wrong surface or surviving node moved without volume check | Classify from its own face or geometry; check all live incident volumes before committing reprojection | Audit face labels, signed volumes and boundary distances after each pass |

The regression in `tests/regression.py` covers imports, mesh I/O, metric arithmetic, operators, the case loader, quality, threading and optional CAD paths. It is a compact smoke and contract lane; individual geometry and topology guard review still matters when changing an operator.

## Why some tempting changes were rejected

| Option | Measured or structural reason | Reopen when |
| --- | --- | --- |
| Process workers for smoothing | A serial interior move cost about 0.3 ms in the measured state; snapshot refresh and pickling exceeded the work. | A profile shows enough work per move to amortize transport. |
| Distance-one colouring with commit-time intersection scan | A measured bridge pass took 2.87 s at four threads; the roughly 0.5 ms commit scan dominated. | A new check makes commit cost small without weakening validity. |
| Coloured smoothing for small meshes | A measured coarse cross run stopped 2.11 percentage points below serial conformity. | A matched small-mesh quality gate passes. |
| Arithmetic metric interpolation | It changes tensor size and edge lengths through metric swelling. | The mathematical metric contract changes with validation. |
| PETSc or MPI | The engine solves no linear system; the measured target machine is one node with eight physical cores. | The workload becomes distributed or a solver enters the engine. |
| GPU topology kernels | Local, branch-heavy cavities offer little work per host-device transfer. | A profile identifies a large regular batch. |
| Cython | It adds a compiler and build surface for the same kernel role as Numba. | A measured kernel gap justifies that cost. |
| External adaptation engine | It would replace the adaptation mechanics being maintained here. | The product scope changes. |
| Spatial-tree replacement | The octree was absent from the measured hot path. | Profiling puts point location on the hot path. |

The earlier iteration-based swap taper was retired after the corrected cycle passed the quality comparison. Keep the current order until a matched case comparison supports a change.

## Extending the engine

**Add a metric.** Register an analytic field in `tools/analytic_metric_generator.py` or supply a Medit `.sol` with one SPD six-component tensor per background node. State parameter defaults and units, validate positive eigenvalues, then compare generated values and metric lengths on a known mesh. Keep tensor interpolation in `metric.py` in log space; do not expose kernel tolerances as case keys.

**Add a geometry backend.** Keep `Grid.from_mesh` as the setup boundary and `geom.py` as projection dispatch. Map backend faces and curves to stable surface IDs, preserve `faceSurfMap` through operations and export, and check projected coordinates against the source geometry. `egads_adapter.py` is the optional CAD example; analytical plane and sphere projection remains available without it.

**Add an operator check.** Define the local failure it prevents, put it before mutation, and verify it on a hand-built cavity that previously passed. Preserve positive-volume, boundary-tag and intersection contracts. If the check changes accept/reject decisions, compare conditioned quality measures and bridge hashes against a run from the same cycle. Update cache counters and compaction if the operation adds state.

## Driver policy and working rules

`load_case` accepts only `[input]`, `[adapt]` and `[run]` tables and exact known keys. It validates types, ranges and file existence before importing the engine or writing output. It refuses unknown keys, unsupported formats, nonpositive counts, an output path inside any `input/` component, and an existing nonempty output directory unless `--force` is present. Relative paths resolve against the case directory; `--output-dir` overrides the case value. `--check` prints the resolved case without running adaptation. The driver records absolute inputs, settings, chosen threads, output and engine commit in `resolved_case.json` before adaptation.

Thread count belongs on the CLI. The case schema excludes engine internals such as worker count, threading layer, `fastmath`, kernel selection, numerical tolerances and quality thresholds. Add a key only for a setting a user needs, with a default that reproduces the reference cases. Update validation, all four bundled cases, generated schema documentation and the case-loader regression together.

Before Python work, source `activate.sh` in the shell and use the project's interpreter. Use the fast regression lane for a short local check. Record the case, engine commit, iteration cap and thread count with any reported quality. Long adaptation runs are started by the user with an explicit output directory; inspect their artifacts after completion.
