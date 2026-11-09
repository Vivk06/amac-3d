"""Read mesh, geometry, and metric inputs and write adapted meshes.

Import Netgen .vol, .geo, and Medit .sol/.mtr; export Gmsh .msh, VTK, and
Netgen .vol. Conversion and validation preserve physical-group surface IDs
because they identify each boundary facet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import os
import re
import math
from typing import Dict, List, Optional, TYPE_CHECKING

from .adj import pack_face

if TYPE_CHECKING:
    from .grid import Grid


# ──────────────── Geometry Import ──────────────────────────────────

def import_geo(geo_path: str) -> Dict:
    """
    Parse a ``.geo`` geometry description file.

    Supports ``plane(origin; normal)`` and ``sphere(center; radius)``
    directives.  Returns a dict with:
        - ``surfaceList``: list of surface dicts
        - ``classifyPointToSurface``: callable(x,y,z) → surfId or None
    """
    with open(geo_path, 'r') as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]

    surface_list: List[Dict] = []
    surf_id = 1

    plane_re = r"plane\s*\(([^;]+);([^)]+)\)"
    sphere_re = r"sphere\s*\(([^;]+);([^)]+)\)"

    for line in lines:
        for match in re.findall(plane_re, line):
            origin = [float(x) for x in match[0].split(',')]
            normal = [float(x) for x in match[1].split(',')]
            surface_list.append({
                'type': 'plane', 'surfId': surf_id,
                'origin': origin, 'normal': normal,
            })
            surf_id += 1

        for match in re.findall(sphere_re, line):
            center = [float(x) for x in match[0].split(',')]
            radius = float(match[1])
            surface_list.append({
                'type': 'sphere', 'surfId': surf_id,
                'center': center, 'radius': radius,
            })
            surf_id += 1

    def classify_point_to_surface(x: float, y: float, z: float) -> Optional[int]:
        tol = 1e-8
        for surf in surface_list:
            if surf['type'] == 'plane':
                ox, oy, oz = surf['origin']
                nx, ny, nz = surf['normal']
                dist = nx * (x - ox) + ny * (y - oy) + nz * (z - oz)
                if abs(dist) < tol:
                    return surf['surfId']
            elif surf['type'] == 'sphere':
                cx, cy, cz = surf['center']
                R = surf['radius']
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                if abs(dist - R) < tol:
                    return surf['surfId']
        return None

    return {
        'surfaceList': surface_list,
        'classifyPointToSurface': classify_point_to_surface,
    }


# ──────────────── Mesh Import (.vol) ──────────────────────────────

def import_mesh(mesh_path: str) -> Dict[str, List]:
    """
    Parse a Netgen ``.vol`` mesh file.

    Returns dict with:
        - ``points``       : flat [x0,y0,z0, x1,...] (0-indexed)
        - ``tetrahedra``   : flat [a0,b0,c0,d0, ...] (0-indexed)
        - ``bdytriangles`` : flat list of surface triangle vertex indices
        - ``edgeSegments`` : list of edge-segment metadata dicts
        - ``bdyTriangleTags`` : per-triangle face-descriptor index (1-based, parallel
          to ``bdytriangles``) — the file's own surface tag for each face
        - ``faceDescriptors`` : list of ``(surfnr, domin, domout, tlosurf, bcprop)``
    """
    with open(mesh_path, 'r') as f:
        raw_lines = [line.strip().split() for line in f]

    points: List[float] = []
    surface_tris: List[int] = []
    surface_tags: List[int] = []
    tetrahedra: List[int] = []
    edge_segments: List[Dict] = []
    face_descriptors: List[Tuple[int, int, int, int, int]] = []

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if not line:
            i += 1
            continue

        if line[0] == 'points':
            n = int(raw_lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                x, y, z = (float(raw_lines[j][0]),
                           float(raw_lines[j][1]),
                           float(raw_lines[j][2]))
                points.extend([x, y, z])
            i += 2 + n

        elif line[0] == 'facedescriptors':
            n = int(raw_lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                # surfnr may be written as a float ("0", "-1"); coerce via float.
                vals = [int(float(x)) for x in raw_lines[j][:5]]
                while len(vals) < 5:
                    vals.append(0)
                face_descriptors.append(tuple(vals))
            i += 2 + n

        elif line[0] == 'surfaceelements':
            n = int(raw_lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                fd_idx = int(raw_lines[j][0])
                a = int(raw_lines[j][5]) - 1
                b = int(raw_lines[j][6]) - 1
                c = int(raw_lines[j][7]) - 1
                surface_tris.extend([a, b, c])
                surface_tags.append(fd_idx)
            i += 2 + n

        elif line[0] == 'volumeelements':
            n = int(raw_lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                if int(raw_lines[j][1]) == 4:
                    a = int(raw_lines[j][2]) - 1
                    b = int(raw_lines[j][3]) - 1
                    c = int(raw_lines[j][4]) - 1
                    d = int(raw_lines[j][5]) - 1
                    tetrahedra.extend([a, b, c, d])
            i += 2 + n

        elif line[0] == 'edgesegmentsgi2':
            n = int(raw_lines[i + 1][0])
            for j in range(i + 2, i + 2 + n):
                seg = raw_lines[j]
                edge_segments.append({
                    'surfid': int(seg[0]),
                    'p1': int(seg[2]) - 1,
                    'p2': int(seg[3]) - 1,
                    'sf1': int(seg[6]),
                    'sf2': int(seg[7]),
                    'ednr1': int(seg[8]),
                    'dist1': float(seg[9]),
                    'ednr2': int(seg[10]),
                    'dist2': float(seg[11]),
                })
            i += 2 + n
        else:
            i += 1

    return {
        'points': points,
        'bdytriangles': surface_tris,
        'tetrahedra': tetrahedra,
        'edgeSegments': edge_segments,
        'bdyTriangleTags': surface_tags,
        'faceDescriptors': face_descriptors,
    }


# ──────────────── Metric Import (.sol / .mtr) ─────────────────────

def import_metric(metric_path: str) -> List[float]:
    """
    Parse a Medit .sol / .mtr metric file.

    Skips the first 6 header lines and last 2 footer lines.
    Each data line contains 6 floats: [M11, M12, M13, M22, M23, M33].

    Returns flat list of metric components.
    """
    metric: List[float] = []
    with open(metric_path, 'r') as f:
        lines = f.readlines()
        for line in lines[6:-2]:
            stripped = line.strip()
            if stripped:
                values = list(map(float, stripped.split()))
                metric.extend(values)
    return metric


# ──────────────── Mesh Export (.msh / .vtk) ───────────────────────

def _boundary_faces(tetrahedra: List[int]) -> List[Tuple[int, int, int]]:
    """Extract boundary faces (appearing in exactly one tet) and orient them outwards."""
    from collections import defaultdict
    fc = defaultdict(list)
    n_tets = len(tetrahedra) // 4
    for t in range(n_tets):
        tet = tetrahedra[4 * t: 4 * t + 4]
        if tet[0] == -1:
            continue
        # For each face, we store the sorted key, the tet index, and the opposite vertex index in that tet.
        # tet is expected to have positive signed volume orientation (winding normalization).
        f0_key = tuple(sorted([tet[1], tet[2], tet[3]]))
        f1_key = tuple(sorted([tet[0], tet[2], tet[3]]))
        f2_key = tuple(sorted([tet[0], tet[1], tet[3]]))
        f3_key = tuple(sorted([tet[0], tet[1], tet[2]]))
        
        fc[f0_key].append((t, 0))
        fc[f1_key].append((t, 1))
        fc[f2_key].append((t, 2))
        fc[f3_key].append((t, 3))
        
    oriented_faces = []
    for key, occurrences in fc.items():
        if len(occurrences) == 1:
            t, opp_idx = occurrences[0]
            tet = tetrahedra[4 * t: 4 * t + 4]
            # Retrieve correct outward ordering based on positive signed volume orientation of tet
            if opp_idx == 0:
                face = (tet[1], tet[2], tet[3])
            elif opp_idx == 1:
                face = (tet[0], tet[3], tet[2])
            elif opp_idx == 2:
                face = (tet[0], tet[1], tet[3])
            else: # opp_idx == 3
                face = (tet[0], tet[2], tet[1])
            oriented_faces.append(face)
            
    return oriented_faces


def export_mesh(
    grid: 'Grid', filename: str, fmt: str = 'msh',
) -> None:
    """
    Export mesh to Gmsh 2.2 ASCII (.msh) and/or VTK Legacy (.vtk).

    Parameters
    ----------
    grid : Grid
    filename : str
        Output path without extension.
    fmt : str
        ``'msh'``, ``'vtk'``, or ``'both'``.
    """
    from .grid import Grid
    coords = grid.coordinates
    tets = grid.tetrahedra

    # Filter out dead tets
    live_tets: List[List[int]] = []
    for i in range(0, len(tets), 4):
        t = tets[i: i + 4]
        if t[0] != -1:
            live_tets.append(t)

    bdy_tris = _boundary_faces(tets)
    n_pts = len(coords) // 3
    n_tets = len(live_tets)
    n_tris = len(bdy_tris)

    if fmt not in ('msh', 'vtk', 'both'):
        raise ValueError(f"Unsupported format '{fmt}'. Use 'msh', 'vtk', or 'both'.")

    # Ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Boundary classification helper
    classify_fn = None
    if grid.solidData and "classifyPointToSurface" in grid.solidData:
        classify_fn = grid.solidData["classifyPointToSurface"]

    def classify_with_tol(x: float, y: float, z: float, tol: float = 1e-5) -> Optional[int]:
        if not grid.solidData or 'surfaceList' not in grid.solidData:
            return None
        for surf in grid.solidData['surfaceList']:
            if surf['type'] == 'plane':
                ox, oy, oz = surf['origin']
                nx, ny, nz = surf['normal']
                dist = nx * (x - ox) + ny * (y - oy) + nz * (z - oz)
                if abs(dist) < tol:
                    return surf['surfId']
            elif surf['type'] == 'sphere':
                cx, cy, cz = surf['center']
                R = surf['radius']
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                if abs(dist - R) < tol:
                    return surf['surfId']
        return None

    def get_face_surf_id(tri: Tuple[int, int, int]) -> int:
        """
        Determine the surface ID for a boundary face triangle.

        Priority order (authoritative → fallback):
          1. bdyFaceMap consensus — all face-level vertices agree on surfid
          2. bdyFaceMap + cornerPointSurfMap intersection — mixed vertex types
          3. bdyFaceMap any vertex — at least one vertex has a surfid
          4. Analytical geometry (centroid classification with relaxed tolerance)
          5. Fallback: surf_id = 1
        """
        # ── 0. Per-face map (authoritative when present) ───────────
        if hasattr(grid, 'faceSurfMap'):
            _fs = grid.faceSurfMap.get(pack_face(*tri))
            if _fs is not None and _fs >= 0:
                return _fs

        # ── 1. Check bdyFaceMap consensus ──────────────────────────
        # If all 3 vertices are in bdyFaceMap with the same surfid, use it.
        bdy_surfids = []
        corner_surfsets = []
        for v in tri:
            _sid = grid.bdyFaceMap.get(v) if hasattr(grid, 'bdyFaceMap') else None
            if _sid is not None and _sid >= 0:
                bdy_surfids.append(_sid)
            elif hasattr(grid, 'cornerPointSurfMap') and v in grid.cornerPointSurfMap:
                corner_surfsets.append(grid.cornerPointSurfMap[v])
            elif hasattr(grid, 'bdyEdgeMap') and v in grid.bdyEdgeMap:
                # Edge nodes may appear on multiple surfaces; check cornerPointSurfMap
                if hasattr(grid, 'cornerPointSurfMap') and v in grid.cornerPointSurfMap:
                    corner_surfsets.append(grid.cornerPointSurfMap[v])

        # All 3 vertices are face-level nodes with same surfid
        if len(bdy_surfids) == 3 and bdy_surfids[0] == bdy_surfids[1] == bdy_surfids[2]:
            return bdy_surfids[0]

        # ── 2. Intersect surface sets from mixed vertex types ─────
        # Build candidate set from all vertex evidence
        candidate_surfs = None
        if bdy_surfids:
            candidate_surfs = set(bdy_surfids)
        for cs in corner_surfsets:
            if candidate_surfs is None:
                candidate_surfs = set(cs)
            else:
                candidate_surfs.intersection_update(cs)

        if candidate_surfs and len(candidate_surfs) == 1:
            return next(iter(candidate_surfs))

        # If multiple candidates remain, prefer the one most bdy_surfids agree on
        if candidate_surfs and bdy_surfids:
            from collections import Counter
            vote = Counter(s for s in bdy_surfids if s in candidate_surfs)
            if vote:
                return vote.most_common(1)[0][0]

        # ── 3. Any single vertex bdyFaceMap hit ───────────────────
        if bdy_surfids:
            return bdy_surfids[0]

        # ── 4. Analytical geometry fallback ────────────────────────
        x_c = sum(coords[3 * v] for v in tri) / 3.0
        y_c = sum(coords[3 * v + 1] for v in tri) / 3.0
        z_c = sum(coords[3 * v + 2] for v in tri) / 3.0

        if classify_fn:
            surf_id = classify_fn(x_c, y_c, z_c)
            if surf_id is not None:
                return surf_id

        # Relaxed tolerance for adapted meshes
        surf_id = classify_with_tol(x_c, y_c, z_c, tol=1e-4)
        if surf_id is not None:
            return surf_id

        # Individual vertex classification
        for v in tri:
            x, y, z = coords[3 * v], coords[3 * v + 1], coords[3 * v + 2]
            surf_id = classify_with_tol(x, y, z, tol=1e-4)
            if surf_id is not None:
                return surf_id

        # ── 5. Final fallback ─────────────────────────────────────
        return 1

    def _write_msh():
        path = f"{filename}.msh"

        # Pre-compute surf_id for each boundary triangle
        bdy_tri_surfids = [get_face_surf_id(tri) for tri in bdy_tris]
        unique_surf_ids = sorted(set(bdy_tri_surfids))

        with open(path, 'w') as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

            # Write $PhysicalNames so convert_mesh.py receives surface group metadata
            n_phys = len(unique_surf_ids) + 1  # +1 for volume group
            f.write("$PhysicalNames\n")
            f.write(f"{n_phys}\n")
            for sid in unique_surf_ids:
                f.write(f'2 {sid} "surface_{sid}"\n')
            f.write(f'3 1 "volume_1"\n')
            f.write("$EndPhysicalNames\n")

            f.write("$Nodes\n")
            f.write(f"{n_pts}\n")
            for i in range(n_pts):
                x, y, z = coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]
                f.write(f"{i + 1} {x:.16e} {y:.16e} {z:.16e}\n")
            f.write("$EndNodes\n")

            n_elems = n_tris + n_tets
            f.write("$Elements\n")
            f.write(f"{n_elems}\n")

            idx = 1
            for tri, surf_id in zip(bdy_tris, bdy_tri_surfids):
                nodes = [v + 1 for v in tri]
                f.write(f"{idx} 2 2 {surf_id} {surf_id} {' '.join(map(str, nodes))}\n")
                idx += 1
            for tet in live_tets:
                nodes = [v + 1 for v in tet]
                f.write(f"{idx} 4 2 1 1 {' '.join(map(str, nodes))}\n")
                idx += 1

            f.write("$EndElements\n")

    def _write_vtk():
        path = f"{filename}.vtk"
        with open(path, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Anisotropic Mesh Adaptation Engine Output\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")

            f.write(f"POINTS {n_pts} float\n")
            for i in range(n_pts):
                x, y, z = coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]
                f.write(f"{x} {y} {z}\n")

            total_cells = n_tets + n_tris
            total_size = n_tets * 5 + n_tris * 4
            f.write(f"\nCELLS {total_cells} {total_size}\n")

            for tet in live_tets:
                f.write(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
            for tri in bdy_tris:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

            f.write(f"\nCELL_TYPES {total_cells}\n")
            for _ in range(n_tets):
                f.write("10\n")   # VTK_TETRA
            for _ in range(n_tris):
                f.write("5\n")    # VTK_TRIANGLE

    if fmt == 'msh':
        _write_msh()
    elif fmt == 'vtk':
        _write_vtk()
    elif fmt == 'both':
        _write_msh()
        _write_vtk()


def export_bridge(
    grid: 'Grid', filename: str, iteration: int,
) -> None:
    """
    Write iteration-tagged bridging file for intermediate output.

    Generates ``<filename>_bridge_<iteration>.msh``.
    """
    bridge_name = f"{filename}_bridge_{iteration}"
    export_mesh(grid, bridge_name, 'msh')


def _orient_face_outward(face, opp, node_coords):
    """Order face (a,b,c) so its normal points away from the opposite vertex."""
    a, b, c = face
    pa, pb, pc, pd = node_coords[a], node_coords[b], node_coords[c], node_coords[opp]
    nx = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1])
    ny = (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2])
    nz = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
    dx, dy, dz = pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2]
    if nx * dx + ny * dy + nz * dz > 0:
        return (a, c, b)
    return (a, b, c)


def _repair_surface_classification(volume_elements, surface_elements, node_coords):
    """Reconcile the surface-element list with the volume boundary.

    The display surface of a tet mesh is exactly the faces shared by a single
    tet.  Drops surface elements that are not true boundary faces (interior
    nodes/tets in the display segment) and adds boundary faces missing from the
    list, so the .vol surface segment matches the volume exactly.  Existing
    physical tags are preserved; added faces take the majority kept tag (else 1).

    Returns (repaired_surface_elements, n_removed, n_added).
    """
    from collections import defaultdict, Counter
    face_owner = {}
    face_count: Dict = defaultdict(int)
    for (_, _, p1, p2, p3, p4) in volume_elements:
        for opp, face in ((p4, (p1, p2, p3)), (p3, (p1, p2, p4)),
                          (p2, (p1, p3, p4)), (p1, (p2, p3, p4))):
            key = tuple(sorted(face))
            face_count[key] += 1
            face_owner[key] = (face, opp)
    boundary = {k for k, c in face_count.items() if c == 1}

    existing = {}
    for (phys, ent, a, b, c) in surface_elements:
        existing.setdefault(tuple(sorted((a, b, c))), (phys, ent, (a, b, c)))

    n_removed = sum(1 for (_, _, a, b, c) in surface_elements
                    if tuple(sorted((a, b, c))) not in boundary)
    kept_tags = [existing[k][0] for k in boundary if k in existing]
    default_tag = Counter(kept_tags).most_common(1)[0][0] if kept_tags else 1

    repaired = []
    n_added = 0
    for key in boundary:
        if key in existing:
            phys, ent, v = existing[key]
            repaired.append((phys, ent, v[0], v[1], v[2]))
        else:
            face, opp = face_owner[key]
            v = _orient_face_outward(face, opp, node_coords)
            repaired.append((default_tag, 1, v[0], v[1], v[2]))
            n_added += 1
    if n_added:
        # This converter has no .geo; added faces take the majority tag. For
        # geometry-correct retagging use tools/convert_mesh.py with --geo.
        logger.debug(f"  [convert_msh_to_vol] {n_added} added boundary face(s) tagged by "
              f"majority fallback (tag {default_tag})")
    return repaired, n_removed, n_added


def convert_msh_to_vol(msh_path: str, vol_path: str) -> None:
    """
    Pure Python converter to translate Gmsh .msh (2.2) to Netgen .vol format.

    This function bypasses Netgen's internal C++ save bugs and preserves all
    boundary elements and FaceDescriptors without index mismatching.

    Surface elements are repaired against the volume boundary so interior faces
    are never written into the .vol surface display segment (see
    ``_repair_surface_classification``).

    This is the lightweight in-library converter (no ``.geo`` input). For
    heavy-duty conversion with geometric surface classification and repair-mode
    retagging, use the standalone ``tools/convert_mesh.py --geo`` tool.
    """
    physical_names = {} # (dim, physical_id) -> name
    nodes = [] # list of (x, y, z)
    node_id_map = {} # gmsh_node_id -> 1-based index
    
    surface_elements = [] # list of (physical_id, entity_id, p1, p2, p3)
    volume_elements = [] # list of (physical_id, entity_id, p1, p2, p3, p4)
    
    with open(msh_path, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('$'):
                if line.startswith('$End'):
                    section = None
                else:
                    section = line
                continue
                
            if section == '$PhysicalNames':
                parts = line.split()
                if len(parts) >= 3:
                    dim = int(parts[0])
                    phys_id = int(parts[1])
                    name = parts[2].strip('"')
                    physical_names[(dim, phys_id)] = name
            elif section == '$Nodes':
                parts = line.split()
                if len(parts) == 1:
                    continue
                node_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes.append((x, y, z))
                node_id_map[node_id] = len(nodes)
            elif section == '$Elements':
                parts = line.split()
                if len(parts) == 1:
                    continue
                elm_id = int(parts[0])
                elm_type = int(parts[1])
                num_tags = int(parts[2])
                tags = [int(x) for x in parts[3:3+num_tags]]
                node_ids = [int(x) for x in parts[3+num_tags:]]
                
                phys_id = tags[0] if num_tags >= 1 else 1
                entity_id = tags[1] if num_tags >= 2 else 1
                
                if elm_type == 2: # 3-node triangle
                    surface_elements.append((phys_id, entity_id, node_ids[0], node_ids[1], node_ids[2]))
                elif elm_type == 4: # 4-node tet
                    volume_elements.append((phys_id, entity_id, node_ids[0], node_ids[1], node_ids[2], node_ids[3]))

    # Map node IDs to 1-based contiguous indices
    mapped_surface_elements = []
    for phys_id, entity_id, p1, p2, p3 in surface_elements:
        mapped_surface_elements.append((phys_id, entity_id, node_id_map[p1], node_id_map[p2], node_id_map[p3]))
        
    mapped_volume_elements = []
    for phys_id, entity_id, p1, p2, p3, p4 in volume_elements:
        mapped_volume_elements.append((phys_id, entity_id, node_id_map[p1], node_id_map[p2], node_id_map[p3], node_id_map[p4]))

    # Repair surface classification against the volume boundary: drop interior
    # faces wrongly listed as surface (internal nodes/tets in the display
    # segment) and add any boundary face missing from the list.
    node_coords = {i + 1: nodes[i] for i in range(len(nodes))}
    mapped_surface_elements, _n_surf_removed, _n_surf_added = \
        _repair_surface_classification(
            mapped_volume_elements, mapped_surface_elements, node_coords)

    # Guard: a non-positive physical id would leak a negative bcnr/bcprop into
    # the descriptors. This converter has no .geo to retag from, so fail loudly.
    bad_ids = sorted(pid for pid in set(el[0] for el in mapped_surface_elements)
                     if pid <= 0)
    if bad_ids:
        raise ValueError(
            f"Non-positive surface physical id(s) {bad_ids} in .msh. Use the "
            f"standalone tools/convert_mesh.py --geo to retag from geometry, "
            f"or correct the input tags.")

    unique_surf_phys_ids = sorted(list(set(el[0] for el in mapped_surface_elements)))
    unique_vol_phys_ids = sorted(list(set(el[0] for el in mapped_volume_elements)))

    surf_phys_to_fd_idx = {phys_id: idx + 1 for idx, phys_id in enumerate(unique_surf_phys_ids)}
    
    with open(vol_path, 'w') as f:
        f.write("# Generated by custom Python converter\n\n")
        f.write("mesh3d\n")
        f.write("dimension\n3\n")
        f.write("geomtype\n0\n\n")
        
        # Facedescriptors
        f.write("# surfnr\tdomin\tdomout\ttlosurf\tbcprop\n")
        f.write("facedescriptors\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for phys_id in unique_surf_phys_ids:
            fd_idx = surf_phys_to_fd_idx[phys_id]
            f.write(f" {fd_idx} 1 0 -1 {phys_id}\n")
        f.write("\n")
        
        # Surface elements
        f.write("# surfnr    bcnr   domin  domout      np      p1      p2      p3\n")
        f.write("surfaceelements\n")
        f.write(f"{len(mapped_surface_elements)}\n")
        for phys_id, entity_id, p1, p2, p3 in mapped_surface_elements:
            fd_idx = surf_phys_to_fd_idx[phys_id]
            f.write(f" {fd_idx} {phys_id} 1 0 3 {p1} {p2} {p3}\n")
        f.write("\n")
        
        # Volume elements
        f.write("#  matnr      np      p1      p2      p3      p4\n")
        f.write("volumeelements\n")
        f.write(f"{len(mapped_volume_elements)}\n")
        for phys_id, entity_id, p1, p2, p3, p4 in mapped_volume_elements:
            f.write(f" {phys_id} 4 {p1} {p2} {p3} {p4}\n")
        f.write("\n")
        
        # Points
        f.write("#          X             Y             Z\n")
        f.write("points\n")
        f.write(f"{len(nodes)}\n")
        for x, y, z in nodes:
            f.write(f"    {x:.16f}      {y:.16f}      {z:.16f}\n")
        f.write("\n")
        
        # Pointelements
        f.write("#          pnum             index\n")
        f.write("pointelements\n0\n\n")
        
        # Materials
        f.write("materials\n")
        f.write(f"{len(unique_vol_phys_ids)}\n")
        for phys_id in unique_vol_phys_ids:
            name = physical_names.get((3, phys_id), f"vol{phys_id}")
            f.write(f"{phys_id} {name}\n")
        f.write("\n")
        
        # BCNames
        f.write("bcnames\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for phys_id in unique_surf_phys_ids:
            name = physical_names.get((2, phys_id), f"surf{phys_id}")
            f.write(f"{surf_phys_to_fd_idx[phys_id]}\t{name}\n")
        f.write("\n")
        
        # Face colours
        f.write("#   Surfnr     Red     Green     Blue\n")
        f.write("face_colours\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for phys_id in unique_surf_phys_ids:
            fd_idx = surf_phys_to_fd_idx[phys_id]
            f.write(f"       {fd_idx}   0.00000000   1.00000000   0.00000000\n")
            
        # Face transparencies
        f.write("face_transparencies\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for phys_id in unique_surf_phys_ids:
            fd_idx = surf_phys_to_fd_idx[phys_id]
            f.write(f"       {fd_idx}   1.00000000\n")
        f.write("\n")
        
        f.write("endmesh\n")


def export_vol(grid: 'Grid', filename: str) -> None:
    """
    Export Grid directly to Netgen .vol format with full boundary preservation.

    Uses the Grid's internal boundary data (bdyFaceMap, cornerPointSurfMap)
    to write correct face descriptors and surface elements, bypassing the
    lossy .msh roundtrip.

    Parameters
    ----------
    grid : Grid
    filename : str
        Output .vol file path.
    """
    from collections import defaultdict
    coords = grid.coordinates
    tets = grid.tetrahedra

    # Ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Filter out dead tets
    live_tets = []
    for i in range(0, len(tets), 4):
        t = tets[i: i + 4]
        if t[0] != -1:
            live_tets.append(t)

    # Extract boundary faces with surface IDs
    bdy_tris = _boundary_faces(tets)

    # Boundary classification helper (reuse the logic from export_mesh)
    classify_fn = None
    if grid.solidData and "classifyPointToSurface" in grid.solidData:
        classify_fn = grid.solidData["classifyPointToSurface"]

    def _classify_with_tol(x, y, z, tol=1e-4):
        if not grid.solidData or 'surfaceList' not in grid.solidData:
            return None
        for surf in grid.solidData['surfaceList']:
            if surf['type'] == 'plane':
                ox, oy, oz = surf['origin']
                nx, ny, nz = surf['normal']
                dist = nx * (x - ox) + ny * (y - oy) + nz * (z - oz)
                if abs(dist) < tol:
                    return surf['surfId']
            elif surf['type'] == 'sphere':
                cx, cy, cz = surf['center']
                R = surf['radius']
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                if abs(dist - R) < tol:
                    return surf['surfId']
        return None

    def _get_surf_id(tri):
        """Determine surface ID from Grid's internal data."""
        # Per-face map (authoritative when present).
        if hasattr(grid, 'faceSurfMap'):
            _fs = grid.faceSurfMap.get(pack_face(*tri))
            if _fs is not None and _fs >= 0:
                return _fs

        bdy_surfids = []
        corner_surfsets = []
        for v in tri:
            _sid = grid.bdyFaceMap.get(v) if hasattr(grid, 'bdyFaceMap') else None
            if _sid is not None and _sid >= 0:
                bdy_surfids.append(_sid)
            elif hasattr(grid, 'cornerPointSurfMap') and v in grid.cornerPointSurfMap:
                corner_surfsets.append(grid.cornerPointSurfMap[v])

        if len(bdy_surfids) == 3 and bdy_surfids[0] == bdy_surfids[1] == bdy_surfids[2]:
            return bdy_surfids[0]

        candidate_surfs = None
        if bdy_surfids:
            candidate_surfs = set(bdy_surfids)
        for cs in corner_surfsets:
            if candidate_surfs is None:
                candidate_surfs = set(cs)
            else:
                candidate_surfs.intersection_update(cs)

        if candidate_surfs and len(candidate_surfs) == 1:
            return next(iter(candidate_surfs))
        if bdy_surfids:
            return bdy_surfids[0]

        # Analytical fallback
        x_c = sum(coords[3 * v] for v in tri) / 3.0
        y_c = sum(coords[3 * v + 1] for v in tri) / 3.0
        z_c = sum(coords[3 * v + 2] for v in tri) / 3.0
        sid = _classify_with_tol(x_c, y_c, z_c)
        if sid is not None:
            return sid
        return 1

    # Compute surf_id for each boundary triangle
    tri_surfids = [_get_surf_id(tri) for tri in bdy_tris]
    unique_sids = sorted(set(tri_surfids))
    sid_to_fd_idx = {sid: idx + 1 for idx, sid in enumerate(unique_sids)}

    # Geometry surface number mapping
    geo_surfnr_map = {}
    if grid.solidData and 'surfaceList' in grid.solidData:
        for surf in grid.solidData['surfaceList']:
            geo_surfnr_map[surf['surfId']] = surf['surfId'] - 1
    for sid in unique_sids:
        if sid not in geo_surfnr_map:
            geo_surfnr_map[sid] = sid - 1

    n_pts = len(coords) // 3

    with open(filename, 'w') as f:
        f.write("# Generated by Anisotropic Mesh Adaptation Engine\n\n")
        f.write("mesh3d\ndimension\n3\ngeomtype\n0\n\n")

        # Face descriptors
        f.write("# surfnr\tdomin\tdomout\ttlosurf\tbcprop\n")
        f.write("facedescriptors\n")
        f.write(f"{len(unique_sids)}\n")
        for sid in unique_sids:
            geo_sn = geo_surfnr_map[sid]
            f.write(f" {geo_sn} 1 0 0 1\n")
        f.write("\n")

        # Surface elements
        f.write("# surfnr    bcnr   domin  domout      np      p1      p2      p3\n")
        f.write("surfaceelements\n")
        f.write(f"{len(bdy_tris)}\n")
        for tri, sid in zip(bdy_tris, tri_surfids):
            fd_idx = sid_to_fd_idx[sid]
            # 1-based node indices
            f.write(f" {fd_idx} 1 1 0 3 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("\n")

        # Volume elements
        f.write("#  matnr      np      p1      p2      p3      p4\n")
        f.write("volumeelements\n")
        f.write(f"{len(live_tets)}\n")
        for tet in live_tets:
            f.write(f"1 4 {tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1}\n")
        f.write("\n")

        # Points
        f.write("#          X             Y             Z\n")
        f.write("points\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            x, y, z = coords[3*i], coords[3*i+1], coords[3*i+2]
            f.write(f"    {x:.16f}      {y:.16f}      {z:.16f}\n")
        f.write("\n")

        # Point elements
        f.write("#          pnum             index\n")
        f.write("pointelements\n0\n\n")

        # Face colours
        f.write("#   Surfnr     Red     Green     Blue\n")
        f.write("face_colours\n")
        f.write(f"{len(unique_sids)}\n")
        for sid in unique_sids:
            fd_idx = sid_to_fd_idx[sid]
            f.write(f"       {fd_idx}   0.00000000   1.00000000   0.00000000\n")

        f.write("\nendmesh\n")


# ──────────────── Validation Helpers ───────────────────────────────

def validate_msh_file(path: str) -> Dict:
    """
    Validate a Gmsh 2.2 .msh file for structural integrity.

    Returns dict with keys:
        valid (bool), version (str), n_nodes (int), n_tets (int),
        n_tris (int), surf_phys_ids (set[int]), errors (list[str])
    """
    errors: List[str] = []

    if not os.path.isfile(path):
        return {
            'valid': False, 'version': '', 'n_nodes': 0, 'n_tets': 0,
            'n_tris': 0, 'surf_phys_ids': set(),
            'errors': [f'File not found: {path}'],
        }

    with open(path, 'r') as f:
        lines = f.readlines()

    section = None
    version = ''
    n_nodes_declared = 0
    node_ids: set = set()
    n_tets = 0
    n_tris = 0
    surf_phys_ids: set = set()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('$End'):
            section = None
            continue
        if line.startswith('$'):
            section = line
            continue

        if section == '$MeshFormat':
            if not version:
                parts = line.split()
                if parts:
                    version = parts[0]
        elif section == '$Nodes':
            if n_nodes_declared == 0:
                try:
                    n_nodes_declared = int(line)
                except ValueError:
                    pass
            else:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        node_ids.add(int(parts[0]))
                    except ValueError:
                        pass
        elif section == '$Elements':
            parts = line.split()
            if len(parts) < 3:
                continue  # element count line (single token) or malformed
            try:
                elm_type = int(parts[1])
                n_tags = int(parts[2])
                if elm_type == 4:
                    n_tets += 1
                elif elm_type == 2:
                    n_tris += 1
                    if n_tags >= 1 and len(parts) >= 4:
                        surf_phys_ids.add(int(parts[3]))
            except (ValueError, IndexError):
                pass

    n_nodes = len(node_ids)

    if not version:
        errors.append('Missing $MeshFormat section')
    if n_nodes_declared == 0:
        errors.append('Missing or empty $Nodes section')
    elif n_nodes != n_nodes_declared:
        errors.append(
            f'Node count: header={n_nodes_declared}, actual={n_nodes}')
    if n_tets == 0:
        errors.append('No tetrahedral elements (type 4) found')

    return {
        'valid': len(errors) == 0,
        'version': version,
        'n_nodes': n_nodes,
        'n_tets': n_tets,
        'n_tris': n_tris,
        'surf_phys_ids': surf_phys_ids,
        'errors': errors,
    }


def validate_vol_file(path: str) -> Dict:
    """
    Validate a Netgen .vol mesh file for structural integrity.

    Returns dict with keys:
        valid (bool), n_points (int), n_surface_elements (int),
        n_volume_elements (int), has_edge_segments (bool), errors (list[str])
    """
    errors: List[str] = []

    if not os.path.isfile(path):
        return {
            'valid': False, 'n_points': 0, 'n_surface_elements': 0,
            'n_volume_elements': 0, 'has_edge_segments': False,
            'errors': [f'File not found: {path}'],
        }

    with open(path, 'r') as f:
        raw_lines = [line.strip() for line in f]

    # Skip leading comments/blank lines when checking for the mesh3d header
    has_mesh3d = any(
        line == 'mesh3d'
        for line in raw_lines[:10]
        if line and not line.startswith('#')
    )
    if not has_mesh3d:
        errors.append("Missing 'mesh3d' header in file")

    n_pts_declared = n_surf_declared = n_vol_declared = 0
    has_edge_segments = False

    for i, tok in enumerate(raw_lines):
        if tok == 'points' and i + 1 < len(raw_lines):
            try:
                n_pts_declared = int(raw_lines[i + 1].split()[0])
            except (ValueError, IndexError):
                pass
        elif tok == 'surfaceelements' and i + 1 < len(raw_lines):
            try:
                n_surf_declared = int(raw_lines[i + 1].split()[0])
            except (ValueError, IndexError):
                pass
        elif tok == 'volumeelements' and i + 1 < len(raw_lines):
            try:
                n_vol_declared = int(raw_lines[i + 1].split()[0])
            except (ValueError, IndexError):
                pass
        elif tok in ('edgesegmentsgi2', 'edgesegments') and i + 1 < len(raw_lines):
            try:
                has_edge_segments = int(raw_lines[i + 1].split()[0]) > 0
            except (ValueError, IndexError):
                pass

    try:
        data = import_mesh(path)
    except Exception as exc:
        errors.append(f'import_mesh failed: {exc}')
        return {
            'valid': False,
            'n_points': n_pts_declared,
            'n_surface_elements': n_surf_declared,
            'n_volume_elements': n_vol_declared,
            'has_edge_segments': has_edge_segments,
            'errors': errors,
        }

    n_pts_actual = len(data['points']) // 3
    n_surf_actual = len(data['bdytriangles']) // 3
    n_vol_actual = len(data['tetrahedra']) // 4

    if n_pts_declared > 0 and n_pts_actual != n_pts_declared:
        errors.append(
            f'Points: declared={n_pts_declared}, parsed={n_pts_actual}')
    if n_surf_declared > 0 and n_surf_actual != n_surf_declared:
        errors.append(
            f'Surface elements: declared={n_surf_declared}, parsed={n_surf_actual}')
    if n_vol_declared > 0 and n_vol_actual != n_vol_declared:
        errors.append(
            f'Volume elements: declared={n_vol_declared}, parsed={n_vol_actual}')
    if n_pts_actual == 0:
        errors.append('No points parsed')
    if n_vol_actual == 0:
        errors.append('No volume elements parsed')
    if n_surf_actual == 0:
        errors.append('No surface elements parsed')

    return {
        'valid': len(errors) == 0,
        'n_points': n_pts_actual,
        'n_surface_elements': n_surf_actual,
        'n_volume_elements': n_vol_actual,
        'has_edge_segments': has_edge_segments,
        'errors': errors,
    }
