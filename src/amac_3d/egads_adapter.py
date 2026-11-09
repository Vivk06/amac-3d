"""Map mesh boundary tags to EGADS topology for exact projection.

Load pyEGADS lazily; map surfaces and curves to a single CAD body, project
points onto faces or edges, and classify points against that body.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # ego types are runtime-only from pyEGADS

logger = logging.getLogger(__name__)


def _import_pyegads():
    """Lazy import of pyEGADS with a clear error message."""
    try:
        from pyEGADS import egads
        return egads
    except ImportError:
        raise ImportError(
            "pyEGADS not found. Install ESP (Engineering Sketch Pad) from "
            "acdl.mit.edu/ESP and build pyEGADS. Run "
            "`bash setup.sh --esp` and `source activate.sh`."
        )


@dataclass
class TopoMap:
    """Bidirectional mapping between mesh boundary tags and EGADS topology."""
    face_map: Dict[int, object] = field(default_factory=dict)   # surfid -> EGADS Face ego
    edge_map: Dict[int, object] = field(default_factory=dict)   # ednr -> EGADS Edge ego
    node_map: Dict[int, object] = field(default_factory=dict)   # corner_node -> EGADS Node ego
    face_map_inv: Dict[int, int] = field(default_factory=dict)  # EGADS Face index -> surfid
    edge_map_inv: Dict[int, int] = field(default_factory=dict)  # EGADS Edge index -> ednr


class EgadsModel:
    """Wrapper around an EGADS context + loaded body."""

    def __init__(self, step_path: str):
        egads = _import_pyegads()
        self.ctx = egads.Context()
        self.model = self.ctx.loadModel(step_path)
        bodies = self.model.getTopology()[4]
        if len(bodies) != 1:
            raise ValueError(
                f"EGADS model has {len(bodies)} bodies; AMAC requires exactly 1. "
                f"Multi-body support is not yet implemented."
            )
        self.body = bodies[0]
        self.faces = self.body.getBodyTopos(egads.FACE)
        self.edges = self.body.getBodyTopos(egads.EDGE)
        self.nodes = self.body.getBodyTopos(egads.NODE)
        self._pyEGADS = egads

    @property
    def n_faces(self) -> int:
        return len(self.faces)


def _face_centroid_and_normal(face) -> Tuple[np.ndarray, np.ndarray]:
    """Get the centroid and outward normal of an EGADS Face at its parametric center."""
    oclass, mtype, geom, limits, children, senses = face.getTopology()
    u_mid = 0.5 * (limits[0] + limits[1])
    v_mid = 0.5 * (limits[2] + limits[3])
    result = face.evaluate([u_mid, v_mid])
    pos = np.array(result[0])
    du = np.array(result[1][0])
    dv = np.array(result[1][1])
    normal = np.cross(du, dv)
    norm = np.linalg.norm(normal)
    if norm > 1e-15:
        normal /= norm
    if mtype < 0:
        normal = -normal
    return pos, normal


def _project_point_to_face(face, point: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project a point onto an EGADS Face. Returns (projected_xyz, distance)."""
    uv, result_xyz = face.invEvaluate(point.tolist())
    proj = np.array(result_xyz)
    dist = np.linalg.norm(proj - point)
    return proj, dist


def _project_point_to_edge(edge, point: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project a point onto an EGADS Edge. Returns (projected_xyz, distance)."""
    t, result_xyz = edge.invEvaluate(point.tolist())
    proj = np.array(result_xyz)
    dist = np.linalg.norm(proj - point)
    return proj, dist


def build_topo_map(model: EgadsModel, grid) -> TopoMap:
    """
    Build the bidirectional mapping between mesh boundary tags and EGADS topology.

    Algorithm:
    1. Face matching: for each mesh surfid, compute centroid of tagged boundary
       triangles, project onto every EGADS Face, pick the closest. Validate by
       checking >95% of triangles land on the same face.
    2. Edge matching: for each mesh ednr, project ridge nodes onto all EGADS Edges,
       pick the Edge with smallest max projection distance.
    3. Node matching: corner nodes (Level 3) map to nearest EGADS Node.
    4. Consistency check: verify EGADS topology adjacency matches mesh adjacency.
    """
    from .adj import pack_face

    topo = TopoMap()
    coords = grid.coordinates

    # ── 1. Face matching by centroid projection ──

    surfid_triangles: Dict[int, List[Tuple[int, int, int]]] = {}
    if hasattr(grid, 'faceSurfMap') and grid.faceSurfMap:
        from collections import defaultdict
        tets = grid.tetrahedra
        face_count = defaultdict(int)
        n_tets = len(tets) // 4
        for t in range(n_tets):
            base = 4 * t
            if tets[base] == -1:
                continue
            tet = [tets[base + k] for k in range(4)]
            for fi in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]:
                f = tuple(sorted([tet[fi[0]], tet[fi[1]], tet[fi[2]]]))
                face_count[f] += 1

        for f, c in face_count.items():
            if c != 1:
                continue
            pk = pack_face(*f)
            sid = grid.faceSurfMap.get(pk, -1)
            if sid >= 0:
                surfid_triangles.setdefault(sid, []).append(f)

    for surfid, triangles in surfid_triangles.items():
        cx, cy, cz = 0.0, 0.0, 0.0
        n_tri = len(triangles)
        for tri in triangles:
            for v in tri:
                cx += coords[v * 3]
                cy += coords[v * 3 + 1]
                cz += coords[v * 3 + 2]
        n_verts = n_tri * 3
        centroid = np.array([cx / n_verts, cy / n_verts, cz / n_verts])

        best_face = None
        best_dist = float('inf')
        for face in model.faces:
            try:
                _, dist = _project_point_to_face(face, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_face = face
            except Exception:
                continue

        if best_face is not None:
            match_count = 0
            for tri in triangles:
                tri_cx = sum(coords[v * 3] for v in tri) / 3.0
                tri_cy = sum(coords[v * 3 + 1] for v in tri) / 3.0
                tri_cz = sum(coords[v * 3 + 2] for v in tri) / 3.0
                tri_centroid = np.array([tri_cx, tri_cy, tri_cz])

                try:
                    _, d = _project_point_to_face(best_face, tri_centroid)
                    is_closest = True
                    for other_face in model.faces:
                        if other_face is best_face:
                            continue
                        try:
                            _, d2 = _project_point_to_face(other_face, tri_centroid)
                            if d2 < d - 1e-10:
                                is_closest = False
                                break
                        except Exception:
                            continue
                    if is_closest:
                        match_count += 1
                except Exception:
                    continue

            confidence = match_count / n_tri if n_tri > 0 else 0.0
            if confidence < 0.95:
                logger.warning(
                    "surfid %d: only %.1f%% of triangles match EGADS Face "
                    "(expected >95%%). Mapping may be incorrect.",
                    surfid, confidence * 100
                )

            face_idx = model.faces.index(best_face)
            topo.face_map[surfid] = best_face
            topo.face_map_inv[face_idx] = surfid

    # ── 2. Edge matching by endpoint projection ──

    ednr_nodes: Dict[int, List[int]] = {}
    if hasattr(grid, 'bdyEdgeMap'):
        for node, ednr in grid.bdyEdgeMap.items():
            ednr_nodes.setdefault(ednr, []).append(node)
    if hasattr(grid, 'cornerPoints'):
        for node, ednrs in grid.cornerPoints:
            if isinstance(ednrs, (set, frozenset)):
                for e in ednrs:
                    ednr_nodes.setdefault(e, []).append(node)
            elif isinstance(ednrs, int):
                ednr_nodes.setdefault(ednrs, []).append(node)

    for ednr, nodes in ednr_nodes.items():
        if not nodes:
            continue
        best_edge = None
        best_max_dist = float('inf')
        for edge in model.edges:
            max_dist = 0.0
            try:
                for n in nodes:
                    pt = np.array(coords[n * 3: n * 3 + 3])
                    _, d = _project_point_to_edge(edge, pt)
                    max_dist = max(max_dist, d)
                if max_dist < best_max_dist:
                    best_max_dist = max_dist
                    best_edge = edge
            except Exception:
                continue

        if best_edge is not None:
            edge_idx = model.edges.index(best_edge)
            topo.edge_map[ednr] = best_edge
            topo.edge_map_inv[edge_idx] = ednr

    # ── 3. Corner node matching ──

    if hasattr(grid, 'cornerPoints'):
        for node, _ in grid.cornerPoints:
            pt = np.array(coords[node * 3: node * 3 + 3])
            best_node_ego = None
            best_dist = float('inf')
            for ego_node in model.nodes:
                oclass, mtype, geom, data, children, senses = ego_node.getTopology()
                node_xyz = np.array(data)
                d = np.linalg.norm(pt - node_xyz)
                if d < best_dist:
                    best_dist = d
                    best_node_ego = ego_node
            if best_node_ego is not None:
                topo.node_map[node] = best_node_ego

    # ── 4. Consistency check ──
    _check_topo_consistency(model, topo)

    logger.info(
        "EGADS topology map: %d faces, %d edges, %d corners mapped",
        len(topo.face_map), len(topo.edge_map), len(topo.node_map)
    )
    return topo


def _check_topo_consistency(model: EgadsModel, topo: TopoMap) -> None:
    """Verify that mapped edges are topologically adjacent to their mapped faces."""
    pyEGADS = model._pyEGADS
    mismatches = 0
    for ednr, edge in topo.edge_map.items():
        adj_faces = model.body.getBodyTopos(pyEGADS.FACE, edge)
        found_adj = False
        for surfid, face in topo.face_map.items():
            if any(face.isSame(af) for af in adj_faces):
                found_adj = True
                break
        if not found_adj and adj_faces:
            logger.warning(
                "Topology mismatch: edge ednr=%d is not adjacent to any "
                "mapped face in EGADS topology.", ednr
            )
            mismatches += 1
    if mismatches > 0:
        logger.warning("Total topology mismatches: %d", mismatches)


# ──────────── Projection Functions ────────────────────────────────


def project_to_surface_egads(
    point: List[float], model: EgadsModel, topo: TopoMap, surfid: int,
) -> Optional[List[float]]:
    """
    Project point onto the EGADS Face mapped to surfid.

    Returns projected [x,y,z] or None if the surfid has no EGADS mapping
    or if EG_invEvaluate fails (the caller falls back to analytical or PN-triangle projection).
    """
    face = topo.face_map.get(surfid)
    if face is None:
        return None
    try:
        pt = np.asarray(point, dtype=np.float64)
        uv, result_xyz = face.invEvaluate(pt.tolist())
        result = face.evaluate(uv)
        return list(result[0])
    except Exception as e:
        logger.debug("EG_invEvaluate failed for surfid %d: %s", surfid, e)
        return None


def project_to_curve_egads(
    point: List[float], model: EgadsModel, topo: TopoMap, ednr: int,
) -> Optional[List[float]]:
    """
    Project point onto the EGADS Edge mapped to ednr.

    Returns projected [x,y,z] or None if the ednr has no EGADS mapping
    or if EG_invEvaluate fails.
    """
    edge = topo.edge_map.get(ednr)
    if edge is None:
        return None
    try:
        pt = np.asarray(point, dtype=np.float64)
        t, result_xyz = edge.invEvaluate(pt.tolist())
        result = edge.evaluate(t)
        return list(result[0])
    except Exception as e:
        logger.debug("EG_invEvaluate failed for ednr %d: %s", ednr, e)
        return None


def classify_point_egads(
    x: float, y: float, z: float, model: EgadsModel, topo: TopoMap,
) -> int:
    """
    Classify point to nearest mapped EGADS Face.

    Returns the surfid of the closest Face, or -1 if no faces are mapped.
    """
    point = np.array([x, y, z])
    best_surfid = -1
    best_dist = float('inf')
    for surfid, face in topo.face_map.items():
        try:
            _, dist = _project_point_to_face(face, point)
            if dist < best_dist:
                best_dist = dist
                best_surfid = surfid
        except Exception:
            continue
    return best_surfid


def get_bounding_box(model: EgadsModel) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) bounding box from the EGADS body."""
    bbox = model.body.getBoundingBox()
    return np.array(bbox[0]), np.array(bbox[1])
