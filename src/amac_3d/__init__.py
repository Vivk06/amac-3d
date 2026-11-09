"""Anisotropic tetrahedral mesh adaptation engine.

adapt: Coordinate the operator cycle and convergence.
adj: Build tetrahedron face adjacency.
cavity: Query local mesh connectivity.
collapse: Coarsen short metric edges.
egads_adapter: Map CAD topology and project boundary points.
geom: Classify and project boundary geometry.
grid: Own mesh state, connectivity, and classifications.
intersection: Guard against local self-intersection.
io: Read and write mesh, geometry, and metric files.
kernels: Run compiled geometry and operator calculations.
metric: Compute metric lengths, quality, and interpolation.
octree: Find nearby mesh nodes.
report: Measure quality and record adaptation history.
search: Locate points in tetrahedra.
smooth: Relocate nodes under geometry and quality constraints.
split: Refine long metric edges.
swap: Reconnect tetrahedra to improve quality.

Algorithms cite Michal & Krakos (2012), Alauzet (2010), Loseille & Löhner
(2009), Knupp (2001), and NASA refine ref_adapt.c, ref_metric.c, ref_geom.c.
"""
