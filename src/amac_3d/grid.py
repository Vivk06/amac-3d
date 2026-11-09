"""Own tetrahedral mesh state, connectivity caches, and boundary classification.

Grid loads mesh and metric inputs, normalizes tet winding, and keeps flat arrays,
node balls, boundary maps, and dead-element compaction consistent. Mutations
invalidate version-keyed geometry and diagnostic caches.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import math
from typing import Dict, List, Optional, Callable, Set, Tuple

import numpy as np

from .metric import (
    log_metric, log_metric_batch, exp_metric, bary_coords_tet,
    signed_volume,
)
from .adj import pack_face, unpack_face

# ──────────────── Boundary Classification Levels ───────────────────

POINT_INTERIOR: int = 0
POINT_SURFACE: int = 1
POINT_RIDGE: int = 2
POINT_CORNER: int = 3
POINT_INTERFACE: int = 4
POINT_LOCKED: int = 5

# ────────────────── Flatness Check Epsilon ─────────────────────────

FLATNESS_EPSILON: float = 1e-15


class Grid:
    """
    Central mesh data structure for 3D anisotropic mesh adaptation.

    Flat 1-D arrays:
        coordinates : [x0, y0, z0, x1, y1, z1, ...]   length 3·Np
        tetrahedra  : [a0, b0, c0, d0, a1, ...]        length 4·Nt
        metricMesh  : [M11, M12, M13, M22, M23, M33, ...]  length 6·Np
        metricLog   : [L11, L12, L13, L22, L23, L33, ...]  length 6·Np

    Topological maps:
        ball_ptr    : int32 ndarray         — CSR pointer (Np+1), node → offset in ball_idx
        ball_idx    : int32 ndarray         — CSR index, incident tet indices
        pointLevels : list[int]             — 6-level classification
        adjPointTets: list[int]             — node → one tet index
        adjacents   : list[int]             — face adjacency (set by adj.py)

    Boundary bookkeeping:
        bdyPoints, bdyFacePoints, bdyEdgePoints, cornerPoints
        bdyFaceMap, bdyEdgeMap, cornerEdgeMap, cornerPointSurfMap

    Conventions:
        Node indices are 0-based internally; file formats (.msh, .vol) use 1-based
        indices — conversion happens in io.py on import and export.
        Dead sentinels: coordinates[n*3] is None for dead node n; all four entries
        tetrahedra[t*4 : t*4+4] are -1 for dead tet t.
        Array strides: coords stride-3, tets stride-4, metrics stride-6.
        metricMesh/metricLog component order per node: [M11,M12,M13,M22,M23,M33]
        (upper-triangle row-major). metricLog stores log(M) for Log-Euclidean ops.
    """

    def _normalize_tet_orientation(self) -> int:
        """
        Ensure all tets have positive signed volume.
        
        Netgen uses the opposite winding convention from our 
        signed_volume() formula. This flips any negative-volume
        tet by swapping vertices 2 and 3.
        
        Returns the number of tets flipped.
        """
        coords = self.coordinates
        tets = self.tetrahedra
        n_flipped = 0
        for i in range(0, len(tets), 4):
            if tets[i] == -1:
                continue
            a = coords[tets[i]*3 : tets[i]*3+3]
            b = coords[tets[i+1]*3 : tets[i+1]*3+3]
            c = coords[tets[i+2]*3 : tets[i+2]*3+3]
            d = coords[tets[i+3]*3 : tets[i+3]*3+3]
            sv = signed_volume(a, b, c, d)
            if sv < 0:
                tets[i+2], tets[i+3] = tets[i+3], tets[i+2]
                n_flipped += 1
        return n_flipped

    # ───────────────────── Construction ─────────────────────────────

    def __init__(self, mesh_obj: Dict, solid_data: Optional[Dict] = None) -> None:
        """
        Build a Grid from a parsed mesh dictionary (from ``io.import_mesh``).

        Parameters
        ----------
        mesh_obj : dict
            Must contain 'points', 'tetrahedra', 'edgeSegments'.
        solid_data : dict, optional
            Contains surface classification data.  If None, boundary
            projection uses discrete methods via geom.py.
        """
        self.solidData = solid_data

        # ── Flat arrays ───────────────────────────────────────────
        self.coordinates = np.array(mesh_obj['points'], dtype=np.float64)
        self.tetrahedra = np.array(mesh_obj['tetrahedra'], dtype=np.int32)
        
        # Normalize Netgen winding order
        n_flipped = self._normalize_tet_orientation()
        if n_flipped > 0:
            logger.debug(f"  [Winding Normalization] Flipped {n_flipped} tets to positive orientation.")
        
        edge_segments = mesh_obj.get('edgeSegments', [])

        # Metric arrays (populated later via from_mesh)
        self.metricMesh = np.array([], dtype=np.float64)
        self.metricLog = np.array([], dtype=np.float64)

        # Metric callable (background mesh function, optional)
        self.bgMetricFunction: Optional[Callable] = None

        # ── Boundary extraction ───────────────────────────────────
        bdy_faces = _boundary_faces(self.tetrahedra)
        bdy_point_dup = [v for face in bdy_faces for v in face]
        self.bdyPoints: List[int] = list(set(bdy_point_dup))

        point_ednr_map: Dict[int, Set[int]] = {}
        self.cornerPointSurfMap: Dict[int, Set[int]] = {}

        for seg in edge_segments:
            ednr = seg['ednr1']
            surf = seg['surfid']
            p1, p2 = seg['p1'], seg['p2']

            for p in (p1, p2):
                point_ednr_map.setdefault(p, set()).add(ednr)
                self.cornerPointSurfMap.setdefault(p, set()).add(surf)

        # Corner points: ≥ 3 distinct edge numbers
        self.cornerPoints: List[Tuple[int, set]] = [
            (p, ednrs)
            for p, ednrs in point_ednr_map.items()
            if len(ednrs) >= 3
        ]

        # Corner-edge map for 3-3 splits
        self.cornerEdgeMap: Dict[Tuple[int, int], int] = {}
        n = len(self.cornerPoints)
        for i in range(n):
            p1, e1 = self.cornerPoints[i]
            for j in range(i + 1, n):
                p2, e2 = self.cornerPoints[j]
                common = e1 & e2
                if common:
                    ednr = next(iter(common))
                    key = (min(p1, p2), max(p1, p2))
                    self.cornerEdgeMap[key] = ednr

        # Ridge (edge) points: exactly 1 distinct edge number
        self.bdyEdgePoints: List[Tuple[int, int]] = [
            (p, next(iter(ednrs)))
            for p, ednrs in point_ednr_map.items()
            if len(ednrs) == 1
        ]
        corner_set = {p for p, _ in self.cornerPoints}
        edge_set = {p for p, _ in self.bdyEdgePoints}
        self.bdyEdgeMap: Dict[int, int] = {p: ednr for p, ednr in self.bdyEdgePoints}

        # Face points: boundary but not edge or corner
        self.bdyFacePoints: List[int] = [
            p for p in self.bdyPoints
            if p not in edge_set and p not in corner_set
        ]

        # Surface-id map for face points
        self.bdyFaceMap: Dict[int, int] = {}
        if self.solidData and "classifyPointToSurface" in self.solidData:
            classify_fn = self.solidData["classifyPointToSurface"]
            for p in self.bdyFacePoints:
                x = self.coordinates[p * 3]
                y = self.coordinates[p * 3 + 1]
                z = self.coordinates[p * 3 + 2]
                surf_id = classify_fn(x, y, z)
                if surf_id is None:
                    raise RuntimeError(
                        f"Boundary face point {p} ({x},{y},{z}) not on any surface"
                    )
                self.bdyFaceMap[p] = surf_id
        else:
            # Discrete classification deferred (handled via geom.py)
            pass

        # ── Per-face surface map ──────────────────────────────────
        # Packed-face key → surfId, one entry per boundary face. Seeded
        # geometry-first: geometric face classification (``_classify_face_geo``) is the
        # unambiguous ground truth for analytical surfaces. The file's own
        # descriptor tags (parsed additively by ``import_mesh``) are the
        # fallback for faces it cannot place, or when no ``.geo`` is given.
        # (Netgen ``surfnr`` numbering is producer-specific and does not map
        # arithmetically to geo surfId, so it is fallback-only.)
        fd = mesh_obj.get('faceDescriptors') or []
        tags = mesh_obj.get('bdyTriangleTags') or []
        tris = mesh_obj.get('bdytriangles') or []
        file_face_tag: Dict[int, int] = {}
        for t in range(len(tris) // 3):
            a, b, c = tris[3 * t], tris[3 * t + 1], tris[3 * t + 2]
            if t < len(tags):
                fd_idx = tags[t]
                if 1 <= fd_idx <= len(fd) and fd[fd_idx - 1][0] >= 0:
                    file_face_tag[pack_face(a, b, c)] = fd[fd_idx - 1][0] + 1

        self.faceSurfMap: Dict[int, int] = {}
        for face in bdy_faces:
            a, b, c = face[0], face[1], face[2]
            key = pack_face(a, b, c)
            sid = self._classify_face_geo(a, b, c) if self.solidData else None
            if sid is None:
                sid = file_face_tag.get(key)
            if sid is not None:
                self.faceSurfMap[key] = sid

        # ── Attach tets to points ─────────────────────────────────
        self.adjPointTets: List[int] = self._attach_tetrahedra()
        if -1 in self.adjPointTets:
            orphans = [i for i, v in enumerate(self.adjPointTets) if v == -1]
            raise RuntimeError(
                f"Orphan points (no tet): {orphans[:10]}{'...' if len(orphans) > 10 else ''}"
            )

        # ── Dead-node mask ────────────────────────────────────────
        n_points = len(self.coordinates) // 3
        self.dead_mask = np.zeros(n_points, dtype=np.bool_)

        # ── Ball of node (CSR) ────────────────────────────────────
        self.ball_ptr, self.ball_idx = Grid._build_ball_csr(
            self.tetrahedra, n_points)
        self._ball_csr_topo_version: int = 0
        self._ball_dict: Optional[Dict[int, List[int]]] = None

        # ── Adjacency (set externally by adj.build_adjacency) ─────
        self.adjacents: List[int] = []

        # ── Bounding box ──────────────────────────────────────────
        xs = self.coordinates[0::3]
        ys = self.coordinates[1::3]
        zs = self.coordinates[2::3]
        self.xMin, self.xMax = float(np.nanmin(xs)), float(np.nanmax(xs))
        self.yMin, self.yMax = float(np.nanmin(ys)), float(np.nanmax(ys))
        self.zMin, self.zMax = float(np.nanmin(zs)), float(np.nanmax(zs))

        # ── Point classification ──────────────────────────────────
        self.pointLevels: List[int] = self.classifyPoints()

        # ── Node age: smooth passes since the node last moved ──────────
        # Refine-style stuck-node detector (ref_node_age): incremented by
        # smooth_pass on every unmoved attempt, reset to 0 on an accepted
        # move; new split nodes start at 0. max(nodeAge) guards the
        # convergence-termination test in adapt.py.
        self.nodeAge: List[int] = [0] * (len(self.coordinates) // 3)

        # ── Octree (set externally by octree.build_point_octree) ──
        self.octree = None

        # ── Tet localization cache ────────────────────────────────
        self._prev_tet_idx: int = 0

        # ── Mesh-version counter ───────────────────────────────
        # Monotonically increasing; bumped by every topology mutation
        # and coordinate change. Keyed caches (_geom_cache, _diag_cache)
        # invalidate when the version they were built at no longer matches.
        self._mesh_version: int = 0
        # Topology-only counter (OI-B2): bumped by committed tet/ball
        # mutations (split, collapse, swap, compaction, ballOfNode
        # assignment) but NOT by coordinate/metric changes, so it can
        # detect a stale CSR ball without per-smoothing-move rebuilds.
        self._topo_version: int = 0
        self._geom_cache: dict = {}
        self._diag_cache: dict = {}

        # ── Active-edge cache ────────────────────────────────────
        # Compacted array of non-conforming edge indices, updated by
        # update_active_edges() at iteration boundaries.
        self._active_scan_edges: Optional[np.ndarray] = None   # (N_active, 2) int32
        self._active_scan_edge_list: Optional[List[Tuple[int, int]]] = None
        self._active_edge_frac: float = 1.0

        # ── EGADS model (optional) ───────────────────────────────
        self.egadsModel = None   # EgadsModel or None
        self.topoMap = None      # TopoMap or None

        # ── O(1) boundary-point lookup ────────────────────────────
        self.bdyPointsSet: set = set(self.bdyPoints)

    # ───────────── Ball-of-Node (CSR + dict cache) ─────────────────

    @staticmethod
    def _build_ball_csr(tets, n_pts):
        """Build CSR ``(ball_ptr, ball_idx)`` from a flat tetrahedra array."""
        from .kernels import build_ball_csr_jit
        return build_ball_csr_jit(np.asarray(tets, dtype=np.int32), int(n_pts))

    def ball_of(self, node: int):
        """Tets incident to *node* — CSR-direct when dict cache is cold."""
        if self._ball_dict is not None:
            return self._ball_dict.get(node, [])
        if node < 0 or node >= len(self.ball_ptr) - 1:
            return []
        return self.ball_idx[self.ball_ptr[node]:self.ball_ptr[node + 1]]

    @property
    def ballOfNode(self) -> Dict[int, List[int]]:
        """Lazy dict view of the CSR ball — for mutation-heavy callers."""
        if self._ball_dict is None:
            self._ball_dict = self._csr_to_dict()
        return self._ball_dict

    @ballOfNode.setter
    def ballOfNode(self, value: Dict[int, List[int]]) -> None:
        self._ball_dict = value
        self._topo_version += 1

    def csr_is_stale(self) -> bool:
        """True when the CSR ball no longer reflects the topology."""
        return (self._ball_csr_topo_version != self._topo_version
                or len(self.ball_ptr) - 1 != len(self.coordinates) // 3)

    def set_ball_csr(self, ptr, idx) -> None:
        """Cache a CSR derived from the dict at the current topo version.
        The dict stays authoritative and is not cleared."""
        self.ball_ptr, self.ball_idx = ptr, idx
        self._ball_csr_topo_version = self._topo_version

    def refresh_ball_csr(self) -> None:
        """Cache a CSR built from ``tetrahedra``, leaving the dict untouched.
        Valid because operators keep tetrahedra and ballOfNode consistent at commit boundaries."""
        ptr, idx = Grid._build_ball_csr(
            self.tetrahedra, len(self.coordinates) // 3)
        self.set_ball_csr(ptr, idx)

    def _csr_to_dict(self) -> Dict[int, List[int]]:
        n = len(self.ball_ptr) - 1
        return {i: list(self.ball_idx[self.ball_ptr[i]:self.ball_ptr[i + 1]])
                for i in range(n)}

    # ─────────────── Factory: from_mesh ────────────────────────────

    @classmethod
    def from_mesh(
        cls,
        geo_path: str,
        mesh_path: str,
        metric_path: Optional[str] = None,
        metric_function: Optional[Callable] = None,
        egads_path: Optional[str] = None,
    ) -> 'Grid':
        """
        Construct a fully initialised Grid from file paths.

        Parameters
        ----------
        geo_path : str
            Path to geometry description (.geo) or None for discrete.
        mesh_path : str
            Path to Netgen .vol mesh file.
        metric_path : str, optional
            Path to .sol / .mtr metric file.
        metric_function : callable, optional
            Background metric function  f(x,y,z) → [6 floats].
        egads_path : str, optional
            Path to STEP/IGES CAD file for exact B-rep projection via
            EGADS. Requires pyEGADS. When None, uses analytical/discrete
            fallback (unchanged behavior).

        Returns
        -------
        Grid
        """
        from .io import import_mesh, import_metric, import_geo
        from .adj import build_adjacency
        from .octree import build_point_octree

        solid_data = import_geo(geo_path) if geo_path else None
        mesh_obj = import_mesh(mesh_path)

        grid = cls(mesh_obj, solid_data)

        if metric_path:
            grid.metricMesh = np.array(import_metric(metric_path), dtype=np.float64)
            grid.metricLog = grid._assign_node_metric_log()
        elif metric_function:
            grid.bgMetricFunction = metric_function
            grid._assign_node_metric()

        grid.adjacents = build_adjacency(grid.tetrahedra,
                                          len(grid.coordinates) // 3)
        grid.octree = build_point_octree(grid)

        if egads_path:
            from .egads_adapter import EgadsModel, build_topo_map
            grid.egadsModel = EgadsModel(egads_path)
            grid.topoMap = build_topo_map(grid.egadsModel, grid)

        return grid

    # ───────────── Metric evaluation at a point ────────────────────

    def metric(self, x: float, y: float, z: float) -> List[float]:
        """
        Evaluate metric tensor at (x, y, z) via:
            1. background function (if set), or
            2. log-Euclidean interpolation inside the containing tet.
        """
        if self.bgMetricFunction:
            return self.bgMetricFunction(x, y, z)

        from .search import locate
        tet_idx = locate(x, y, z, self, self._prev_tet_idx)
        if tet_idx == -1:
            raise ValueError(f"Point ({x}, {y}, {z}) not inside any tet")
        if isinstance(tet_idx, list):
            tet_idx = tet_idx[0]
        self._prev_tet_idx = tet_idx

        v0 = self.tetrahedra[tet_idx * 4]
        v1 = self.tetrahedra[tet_idx * 4 + 1]
        v2 = self.tetrahedra[tet_idx * 4 + 2]
        v3 = self.tetrahedra[tet_idx * 4 + 3]

        pts = [
            self.coordinates[v * 3: v * 3 + 3]
            for v in (v0, v1, v2, v3)
        ]
        bary = bary_coords_tet(pts[0], pts[1], pts[2], pts[3], [x, y, z])

        # Log-Euclidean blend
        log_metrics = [
            self.metricLog[v * 6: v * 6 + 6]
            for v in (v0, v1, v2, v3)
        ]
        blended = [
            sum(bary[j] * log_metrics[j][k] for j in range(4))
            for k in range(6)
        ]
        return exp_metric(blended)

    # ───────────── Point classification (6 levels) ─────────────────

    def classifyPoints(self) -> List[int]:
        """
        Classify every mesh point:
            0 = Interior
            1 = Surface (on surfid patch)
            2 = Ridge   (on ednr curve)
            3 = Corner   (≥ 3 edge numbers)
            4 = Interface (multi-material boundary — reserved)
            5 = Locked    (prescribed / periodic — reserved)
        """
        n_points = len(self.coordinates) // 3
        levels = [POINT_INTERIOR] * n_points

        for p in self.bdyFacePoints:
            levels[p] = POINT_SURFACE

        for p, _ in self.bdyEdgePoints:
            levels[p] = POINT_RIDGE

        for p, _ in self.cornerPoints:
            levels[p] = POINT_CORNER

        return levels

    # ──────────── Single-node reclassification ─────────────────────

    def reclassify_node(self, node: int) -> int:
        """
        Recompute *node*'s boundary level from the current mesh topology and
        re-register it in the boundary bookkeeping sets accordingly.

        A node is a *boundary node* iff it belongs to at least one boundary
        face — a triangular face shared by exactly one live tetrahedron.  Nodes
        that are not on any boundary face are interior (level 0) and are
        stripped from ``bdyPoints`` / ``bdyFacePoints`` / ``bdyFaceMap`` /
        ``bdyEdgePoints`` / ``bdyEdgeMap`` so they can never be exported as
        surface elements.  A node found on a boundary face but currently marked
        interior is promoted to a surface node (level 1).  Existing ridge (2) /
        corner (3) classifications are preserved while the node stays on the
        boundary.

        This is the single-node repair primitive used by the split and collapse
        operators to keep node classification consistent with the mesh after a
        local change.  Returns the node's level after reclassification.
        """
        self._mesh_version += 1

        coords = self.coordinates
        if node * 3 >= len(coords) or node >= len(self.dead_mask) or self.dead_mask[node]:
            return -1  # dead node — nothing to classify

        levels = self.pointLevels
        on_boundary = self._is_on_boundary_face(node)

        if not on_boundary:
            # Interior in topology — demote and purge stale boundary registration.
            if node < len(levels):
                levels[node] = POINT_INTERIOR
            self._deregister_boundary_node(node)
            return POINT_INTERIOR

        # Genuinely on the boundary.
        cur = levels[node] if node < len(levels) else POINT_INTERIOR
        if cur < POINT_SURFACE:
            # Wrongly-interior boundary node — promote to surface.
            if node < len(levels):
                levels[node] = POINT_SURFACE
            if node not in self.bdyPointsSet:
                self.bdyPoints.append(node)
                self.bdyPointsSet.add(node)
            if node not in self.bdyFacePoints:
                self.bdyFacePoints.append(node)
            if node not in self.bdyFaceMap:
                # Prefer the majority tag of the node's incident mapped faces,
                # keeping the node-map coherent with the per-face map; fall back
                # to geometric classification.
                surf = self._majority_incident_face_tag(node)
                if surf is None:
                    surf = self._classify_surface(node)
                if surf is not None:
                    self.bdyFaceMap[node] = surf
        elif node not in self.bdyPointsSet:
            # Already classified boundary (surface/ridge/corner) but unregistered.
            self.bdyPoints.append(node)
            self.bdyPointsSet.add(node)
        return levels[node] if node < len(levels) else POINT_SURFACE

    def _is_on_boundary_face(self, node: int) -> bool:
        """True iff *node* lies on a face shared by exactly one live tet.

        Every tet sharing a face that contains *node* also contains *node*, so
        counting face occurrences within ``ballOfNode[node]`` gives the true
        global count: a boundary face appears exactly once.
        """
        from collections import defaultdict
        ball = self.ball_of(node)
        if len(ball) == 0:
            return False
        tets = self.tetrahedra
        face_count: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for t in ball:
            base = 4 * t
            if base >= len(tets) or tets[base] == -1:
                continue
            v = tets[base: base + 4]
            if node not in v:
                continue
            for face in ((v[1], v[2], v[3]), (v[0], v[2], v[3]),
                         (v[0], v[1], v[3]), (v[0], v[1], v[2])):
                if node in face:
                    face_count[tuple(sorted(face))] += 1
        return any(c == 1 for c in face_count.values())

    def _deregister_boundary_node(self, node: int) -> None:
        """Strip all boundary bookkeeping for a node that is now interior."""
        self.bdyPointsSet.discard(node)
        if node in self.bdyPoints:
            self.bdyPoints[self.bdyPoints.index(node)] = -1
        if node in self.bdyFacePoints:
            self.bdyFacePoints[self.bdyFacePoints.index(node)] = -1
        self.bdyFaceMap.pop(node, None)
        for idx, entry in enumerate(self.bdyEdgePoints):
            if isinstance(entry, (list, tuple)) and entry and entry[0] == node:
                self.bdyEdgePoints[idx] = (-1, -1)
        self.bdyEdgeMap.pop(node, None)

    def _majority_incident_face_tag(self, node: int) -> Optional[int]:
        """Majority ``faceSurfMap`` tag among the boundary faces incident to *node*.

        Keeps the node→surf map coherent with the per-face map as topology
        changes. Returns None when no incident face carries a tag."""
        ball = self.ball_of(node)
        if len(ball) == 0:
            return None
        from collections import Counter
        tets = self.tetrahedra
        votes: 'Counter[int]' = Counter()
        seen: Set[int] = set()
        for t in ball:
            base = 4 * t
            if base >= len(tets) or tets[base] == -1:
                continue
            v = tets[base: base + 4]
            if node not in v:
                continue
            for face in ((v[1], v[2], v[3]), (v[0], v[2], v[3]),
                         (v[0], v[1], v[3]), (v[0], v[1], v[2])):
                if node in face:
                    key = pack_face(*face)
                    if key in seen:
                        continue
                    seen.add(key)
                    sid = self.faceSurfMap.get(key)
                    if sid is not None:
                        votes[sid] += 1
        return votes.most_common(1)[0][0] if votes else None

    def _classify_surface(self, node: int) -> Optional[int]:
        """Best-effort surface id for a boundary node via geometry, else None."""
        if not self.solidData or 'classifyPointToSurface' not in self.solidData:
            return None
        x = self.coordinates[node * 3]
        y = self.coordinates[node * 3 + 1]
        z = self.coordinates[node * 3 + 2]
        try:
            return self.solidData['classifyPointToSurface'](x, y, z)
        except Exception:
            return None

    def _classify_face_geo(self, a: int, b: int, c: int) -> Optional[int]:
        """Classify a boundary face to an analytical surfId (geometric face classification), else None.

        All three vertices must lie on the surface (within ``tol``); for a
        plane the face normal must also align with the plane normal
        (``|n̂_face · n̂_plane| > 0.99``) — this rejects the edge-on false
        positives that a centroid test produces at feature ridges. The
        smallest surfId breaks a multi-surface tie deterministically.
        """
        if not self.solidData or 'surfaceList' not in self.solidData:
            return None
        co = self.coordinates
        pa = co[3 * a: 3 * a + 3]
        pb = co[3 * b: 3 * b + 3]
        pc = co[3 * c: 3 * c + 3]
        tol, normal_tol = 1e-6, 0.99
        ab = [pb[k] - pa[k] for k in range(3)]
        ac = [pc[k] - pa[k] for k in range(3)]
        fn = [ab[1] * ac[2] - ab[2] * ac[1],
              ab[2] * ac[0] - ab[0] * ac[2],
              ab[0] * ac[1] - ab[1] * ac[0]]
        fmag = math.sqrt(fn[0] ** 2 + fn[1] ** 2 + fn[2] ** 2)
        if fmag > 1e-15:
            fn = [comp / fmag for comp in fn]
        hits: List[int] = []
        for s in self.solidData['surfaceList']:
            if s['type'] == 'plane':
                o, nrm = s['origin'], s['normal']
                if all(abs(nrm[0] * (p[0] - o[0]) + nrm[1] * (p[1] - o[1])
                           + nrm[2] * (p[2] - o[2])) < tol for p in (pa, pb, pc)):
                    nmag = math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)
                    nh = [nrm[k] / nmag for k in range(3)] if nmag > 1e-15 else [0, 0, 0]
                    if abs(fn[0] * nh[0] + fn[1] * nh[1] + fn[2] * nh[2]) > normal_tol:
                        hits.append(s['surfId'])
            elif s['type'] == 'sphere':
                ce, R = s['center'], s['radius']
                if all(abs(math.sqrt(sum((p[k] - ce[k]) ** 2 for k in range(3))) - R)
                       < tol for p in (pa, pb, pc)):
                    hits.append(s['surfId'])
        return min(hits) if hits else None

    # ──────────── Active-edge restriction ───────────────────────────

    def update_active_edges(self) -> None:
        """Recompute the compacted non-conforming edge list for scan restriction."""
        from .cavity import collect_edges
        from .kernels import compute_edge_active_jit

        edges = collect_edges(self)
        if not edges:
            self._active_scan_edges = None
            self._active_scan_edge_list = None
            self._active_edge_frac = 1.0
            return

        edges_arr = np.array(edges, dtype=np.int32)
        n_points = len(self.coordinates) // 3
        active_mask, n_active = compute_edge_active_jit(
            self.coordinates, self.metricMesh, self.dead_mask,
            edges_arr, n_points)

        self._active_edge_frac = n_active / max(len(edges), 1)

        if n_active == 0 or n_active >= len(edges):
            self._active_scan_edges = None
            self._active_scan_edge_list = None
            return

        active_indices = np.where(active_mask)[0]
        self._active_scan_edges = edges_arr[active_indices]
        self._active_scan_edge_list = [edges[i] for i in active_indices]

    def get_scan_edges(self) -> Tuple:
        """Return ``(edge_list, edges_arr, n_edges)`` — compacted if available, else full."""
        if self._active_scan_edges is not None:
            return (self._active_scan_edge_list,
                    self._active_scan_edges,
                    len(self._active_scan_edge_list))
        from .cavity import collect_edges
        edges = collect_edges(self)
        edges_arr = (np.array(edges, dtype=np.int32) if edges
                     else np.empty((0, 2), dtype=np.int32))
        return (edges, edges_arr, len(edges))

    def clear_active_edges(self) -> None:
        """Invalidate the active-edge cache."""
        self._active_scan_edges = None
        self._active_scan_edge_list = None
        self._active_edge_frac = 1.0

    # ──────────── Dead element compaction ──────────────────────────

    def cleanDeadElements(self) -> None:
        """
        Remove dead nodes (coords = None) and dead tets (all -1).
        Also merge coincident nodes created by boundary projection.
        Re-index everything consistently.

        """
        coords = self.coordinates
        tets = self.tetrahedra
        n_points = len(coords) // 3

        # ── Merge coincident nodes ────────────────────────────────
        # Splits near corners/edges can project new nodes exactly onto
        # existing boundary nodes, creating duplicates.  We detect
        # these via spatial hashing and remap tet references to the
        # canonical (lowest-index) node.
        MERGE_TOL = 1e-10
        INV_TOL = 1.0 / MERGE_TOL

        # Build spatial hash → canonical node index
        pos_to_canon: dict = {}
        merge_map: List[int] = list(range(n_points))  # identity

        for i in range(n_points):
            if self.dead_mask[i]:
                continue
            x = coords[i * 3]
            y = coords[i * 3 + 1]
            z = coords[i * 3 + 2]
            key = (round(x * INV_TOL), round(y * INV_TOL), round(z * INV_TOL))
            if key in pos_to_canon:
                canon = pos_to_canon[key]
                merge_map[i] = canon
                # Keep the higher boundary level between canon and i
                if i < len(self.pointLevels) and canon < len(self.pointLevels):
                    if self.pointLevels[i] > self.pointLevels[canon]:
                        self.pointLevels[canon] = self.pointLevels[i]
                        # Also transfer boundary maps
                        if i in self.bdyFaceMap and canon not in self.bdyFaceMap:
                            self.bdyFaceMap[canon] = self.bdyFaceMap[i]
                        if i in self.bdyEdgeMap and canon not in self.bdyEdgeMap:
                            self.bdyEdgeMap[canon] = self.bdyEdgeMap[i]
                # Mark duplicate as dead
                coords[i * 3] = np.nan
                coords[i * 3 + 1] = np.nan
                coords[i * 3 + 2] = np.nan
                self.dead_mask[i] = True
            else:
                pos_to_canon[key] = i

        # Apply merge_map to all tet references and remove duplicates/degenerates
        n_tets_raw = len(tets) // 4
        seen_tets = set()
        for t in range(n_tets_raw):
            base = 4 * t
            if tets[base] == -1:
                continue
            for k in range(4):
                v = tets[base + k]
                if v >= 0 and v < n_points:
                    tets[base + k] = merge_map[v]
            # Kill tets with duplicate vertices (degenerate)
            verts = tets[base: base + 4]
            if len(set(verts)) < 4:
                tets[base: base + 4] = [-1, -1, -1, -1]
                continue
            # Kill duplicate tets
            tet_key = tuple(sorted(verts))
            if tet_key in seen_tets:
                tets[base: base + 4] = [-1, -1, -1, -1]
            else:
                seen_tets.add(tet_key)

        # ── Build node + tet remap (JIT compaction) ───────────────
        from .kernels import compact_arrays_jit
        (point_map_arr, new_coords, new_metric,
         new_metric_log, new_tets) = compact_arrays_jit(
            coords, self.metricMesh, self.metricLog,
            np.asarray(tets, dtype=np.int32), self.dead_mask)
        point_map = point_map_arr.tolist()

        # ── Rebuild ball-of-node (CSR) ────────────────────────────
        new_np = len(new_coords) // 3

        # ── Rebuild point levels and node ages (drop dead nodes) ──
        new_levels: List[int] = []
        new_ages: List[int] = []
        for i in range(n_points):
            if point_map[i] != -1:
                new_levels.append(self.pointLevels[i])
                new_ages.append(self.nodeAge[i]
                                if i < len(self.nodeAge) else 0)

        # ── Rebuild boundary bookkeeping ──────────────────────────
        new_bdy_points: List[int] = [
            point_map[p] for p in self.bdyPoints
            if p != -1 and point_map[p] != -1
        ]
        new_bdy_face_points: List[int] = [
            point_map[p] for p in self.bdyFacePoints
            if p != -1 and point_map[p] != -1
        ]
        new_bdy_edge_points: List[Tuple[int, int]] = []
        for entry in self.bdyEdgePoints:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                p, ednr = entry
                if p != -1 and point_map[p] != -1:
                    new_bdy_edge_points.append((point_map[p], ednr))

        new_bdy_face_map: Dict[int, int] = {}
        for old, surf in self.bdyFaceMap.items():
            if surf is not None and old != -1 and point_map[old] != -1:
                new_bdy_face_map[point_map[old]] = surf

        new_bdy_edge_map: Dict[int, int] = {}
        for old, ednr in self.bdyEdgeMap.items():
            if ednr is not None and old != -1 and point_map[old] != -1:
                new_bdy_edge_map[point_map[old]] = ednr

        # Remap per-face surface map: expand key → remap each vertex through
        # merge_map then point_map → drop faces with any dead/merged-away
        # vertex → re-pack.
        new_face_surf_map: Dict[int, int] = {}
        for key, surf in self.faceSurfMap.items():
            a, b, c = unpack_face(key)
            na = point_map[merge_map[a]] if a < n_points else -1
            nb = point_map[merge_map[b]] if b < n_points else -1
            nc = point_map[merge_map[c]] if c < n_points else -1
            if na != -1 and nb != -1 and nc != -1 and len({na, nb, nc}) == 3:
                new_face_surf_map[pack_face(na, nb, nc)] = surf

        # Remap the corner/ridge classification maps: these are
        # node-keyed like bdyFaceMap/bdyEdgeMap and go stale identically.
        # Left unremapped, every surf/curve compatibility lookup in
        # _normalize_collapse_direction and the smoothing filter reads
        # the wrong node after the first compaction.
        new_corner_surf_map: Dict[int, Set[int]] = {}
        for old, surfs in self.cornerPointSurfMap.items():
            if 0 <= old < n_points and point_map[merge_map[old]] != -1:
                new_corner_surf_map.setdefault(
                    point_map[merge_map[old]], set()).update(surfs)

        new_corner_points: List[Tuple[int, set]] = []
        for p, ednrs in self.cornerPoints:
            if 0 <= p < n_points and point_map[merge_map[p]] != -1:
                new_corner_points.append((point_map[merge_map[p]], ednrs))

        new_corner_edge_map: Dict[Tuple[int, int], int] = {}
        for (a, b), ednr in self.cornerEdgeMap.items():
            if 0 <= a < n_points and 0 <= b < n_points:
                na = point_map[merge_map[a]]
                nb = point_map[merge_map[b]]
                if na != -1 and nb != -1 and na != nb:
                    new_corner_edge_map[(min(na, nb), max(na, nb))] = ednr

        # ── Apply ─────────────────────────────────────────────────
        self.coordinates = np.array(new_coords, dtype=np.float64)
        self.tetrahedra = np.array(new_tets, dtype=np.int32)
        self.metricMesh = np.array(new_metric, dtype=np.float64)
        self.metricLog = np.array(new_metric_log, dtype=np.float64)
        self.ball_ptr, self.ball_idx = Grid._build_ball_csr(
            new_tets, new_np)
        self._topo_version += 1
        self._ball_csr_topo_version = self._topo_version
        self._ball_dict = None
        self.dead_mask = np.zeros(new_np, dtype=np.bool_)
        self.pointLevels = new_levels
        self.nodeAge = new_ages
        self.bdyPoints = new_bdy_points
        self.bdyPointsSet = set(p for p in new_bdy_points if p >= 0)
        self.bdyFacePoints = new_bdy_face_points
        self.bdyEdgePoints = new_bdy_edge_points
        self.bdyFaceMap = new_bdy_face_map
        self.bdyEdgeMap = new_bdy_edge_map
        self.faceSurfMap = new_face_surf_map
        self.cornerPointSurfMap = new_corner_surf_map
        self.cornerPoints = new_corner_points
        self.cornerEdgeMap = new_corner_edge_map

        # Rebuild bounding box
        if len(self.coordinates) > 0:
            xs = self.coordinates[0::3]
            ys = self.coordinates[1::3]
            zs = self.coordinates[2::3]
            self.xMin, self.xMax = float(np.nanmin(xs)), float(np.nanmax(xs))
            self.yMin, self.yMax = float(np.nanmin(ys)), float(np.nanmax(ys))
            self.zMin, self.zMax = float(np.nanmin(zs)), float(np.nanmax(zs))

        # Rebuild adjacency + octree
        from .adj import build_adjacency
        from .octree import build_point_octree
        self.adjacents = build_adjacency(self.tetrahedra,
                                          len(self.coordinates) // 3)
        self.adjPointTets = self._attach_tetrahedra()
        self.octree = build_point_octree(self)

        # Invalidate active-edge cache (edge indices changed by compaction)
        self._active_scan_edges = None
        self._active_scan_edge_list = None
        self._active_edge_frac = 1.0

        # Bump mesh version — invalidates geom and diagnostic caches
        self._mesh_version += 1

    # ──────────── Internal helpers ─────────────────────────────────

    def _attach_tetrahedra(self) -> List[int]:
        """Map each point to one of its adjacent tetrahedra."""
        n_pts = len(self.coordinates) // 3
        adj = [-1] * n_pts
        for tet_idx in range(len(self.tetrahedra) // 4):
            for k in range(4):
                v = self.tetrahedra[4 * tet_idx + k]
                if adj[v] == -1:
                    adj[v] = tet_idx
        return adj

    def _assign_node_metric_log(self):
        """Compute log of each node metric for log-Euclidean ops.

        Uses a single batched eigh over all node metrics for throughput.
        """
        n_nodes = len(self.metricMesh) // 6
        m6_arr = np.asarray(self.metricMesh, dtype=np.float64).reshape(n_nodes, 6)
        return log_metric_batch(m6_arr).ravel()

    def _assign_node_metric(self) -> None:
        """Evaluate bgMetricFunction at every node and store."""
        xs = self.coordinates[0::3]
        ys = self.coordinates[1::3]
        zs = self.coordinates[2::3]
        metric_array: List[float] = []
        for x, y, z in zip(xs, ys, zs):
            metric_array.extend(self.metric(x, y, z))
        self.metricMesh = np.array(metric_array, dtype=np.float64)
        self.metricLog = self._assign_node_metric_log()

    def resample_node_metric(self, node: int) -> None:
        """Re-evaluate a node's metric from the background field after it moves.

        Only acts when a background metric function is set (the adaptation-run
        case, where the metric field is an analytic/interpolated function of
        position). A node that moves — via smoothing — or is created off an
        edge — via split — otherwise keeps the tensor sampled at its *original*
        position, so the working field drifts from the true field across
        iterations. Re-sampling at the new coordinate keeps every subsequent
        length/quality measurement honest. Updates both ``metricMesh`` and its
        cached log ``metricLog``.

        Without a background function the metric stays as-is (re-sampling would
        require a per-node point location and the field is only known at nodes).
        """
        self._mesh_version += 1
        if not self.bgMetricFunction:
            return
        x = self.coordinates[node * 3]
        y = self.coordinates[node * 3 + 1]
        z = self.coordinates[node * 3 + 2]
        if x is None:
            return
        try:
            m6 = self.bgMetricFunction(x, y, z)
        except ValueError:
            # A background function backed by point location (grid.metric)
            # returns -1/raises for a point it cannot locate — e.g. one landing
            # exactly on a background-mesh face. Keep the prior metric rather
            # than crash a multi-hour run; the stale-metric drift is negligible
            # for the rare miss.
            return
        self.metricMesh[node * 6: node * 6 + 6] = m6
        self.metricLog[node * 6: node * 6 + 6] = log_metric(m6)


# ──────────────── Module-level helpers ─────────────────────────────

def _boundary_faces(tetrahedra: List[int]) -> List[Tuple[int, ...]]:
    """
    Extract boundary faces (faces shared by exactly one tet).
    Uses sorted-tuple counting for uniqueness.
    """
    from collections import defaultdict
    face_count: Dict[Tuple[int, ...], int] = defaultdict(int)
    n_tets = len(tetrahedra) // 4

    for t in range(n_tets):
        tet = tetrahedra[4 * t: 4 * t + 4]
        faces = [
            tuple(sorted([tet[1], tet[2], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[1], tet[2]])),
        ]
        for f in faces:
            face_count[f] += 1

    return [f for f, c in face_count.items() if c == 1]


def validate_mesh(grid: 'Grid') -> dict:
    """
    Run mesh integrity checks on the Grid.

    Returns a dict with counts of issues found:
      - duplicate_tets: tets with identical vertex sets
      - degenerate_tets: tets with < 4 unique vertices
      - overlapping_faces: faces shared by > 2 live tets
      - negative_volume_tets: tets with negative signed volume (engine convention)
    """
    from .metric import signed_volume
    from collections import defaultdict

    tets = grid.tetrahedra
    coords = grid.coordinates
    n_tets = len(tets) // 4

    results = {
        'duplicate_tets': 0,
        'degenerate_tets': 0,
        'overlapping_faces': 0,
        'negative_volume_tets': 0,
    }

    seen = set()
    face_count = defaultdict(int)

    for t in range(n_tets):
        tet = tets[4 * t: 4 * t + 4]
        if tet[0] == -1:
            continue

        # Check degenerate
        if len(set(tet)) < 4:
            results['degenerate_tets'] += 1
            continue

        # Check duplicate
        key = tuple(sorted(tet))
        if key in seen:
            results['duplicate_tets'] += 1
            continue
        seen.add(key)

        # Check negative volume
        pts = [coords[v * 3: v * 3 + 3] for v in tet]
        if signed_volume(pts[0], pts[1], pts[2], pts[3]) < 0:
            results['negative_volume_tets'] += 1

        # Count faces
        for face in [tuple(sorted([tet[1], tet[2], tet[3]])),
                     tuple(sorted([tet[0], tet[2], tet[3]])),
                     tuple(sorted([tet[0], tet[1], tet[3]])),
                     tuple(sorted([tet[0], tet[1], tet[2]]))]:
            face_count[face] += 1

    results['overlapping_faces'] = sum(1 for c in face_count.values() if c > 2)

    return results
