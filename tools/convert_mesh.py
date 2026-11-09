#!/usr/bin/env python3
"""
convert_mesh.py — Robust Gmsh .msh (2.2) to Netgen .vol converter.

Features:
  - Duplicate & degenerate tet removal (log + remove)
  - Overlapping tet detection & removal (log + remove)
  - Geometry-based surface reclassification (--geo)
  - Correct Netgen face descriptor format
  - Hanging node detection (diagnostic)
  - Optional edge segment reconstruction (--with-edges)

Usage:
  python3 tools/convert_mesh.py <input.msh> [output.vol] [--output-dir DIR] [--geo <file.geo>]
"""

import os
import sys
import math
import argparse
from collections import defaultdict, Counter
from pathlib import Path


# ──────────────── Signed Volume (Netgen/Engine Convention) ─────────

def _signed_volume(a, b, c, d):
    """Signed volume using the engine/Netgen convention: (a-d)·((b-d)×(c-d))/6."""
    Ki = ((a[0] - d[0]) * ((b[1] - d[1]) * (c[2] - d[2])
                            - (b[2] - d[2]) * (c[1] - d[1]))
          - (b[0] - d[0]) * ((a[1] - d[1]) * (c[2] - d[2])
                              - (a[2] - d[2]) * (c[1] - d[1]))
          + (c[0] - d[0]) * ((a[1] - d[1]) * (b[2] - d[2])
                              - (a[2] - d[2]) * (b[1] - d[1])))
    return Ki / 6.0


# ──────────────── .geo File Parser ─────────────────────────────────

def _parse_geo_surfaces(geo_path):
    """Parse plane/sphere surface definitions from a Netgen .geo file."""
    surfaces = []
    surf_id = 0
    with open(geo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse: solid name = plane(ox,oy,oz; nx,ny,nz) -bc=N;
            if 'plane' in line.lower():
                surf_id += 1
                try:
                    paren_start = line.index('(')
                    paren_end = line.index(')')
                    inner = line[paren_start + 1:paren_end]
                    parts = inner.split(';')
                    origin = [float(x.strip()) for x in parts[0].split(',')]
                    normal = [float(x.strip()) for x in parts[1].split(',')]
                    surfaces.append({
                        'id': surf_id, 'type': 'plane',
                        'origin': origin, 'normal': normal
                    })
                except (ValueError, IndexError):
                    pass
            elif 'sphere' in line.lower() and 'solid' in line.lower():
                surf_id += 1
                try:
                    paren_start = line.index('(')
                    paren_end = line.index(')')
                    inner = line[paren_start + 1:paren_end]
                    parts = inner.split(';')
                    center = [float(x.strip()) for x in parts[0].split(',')]
                    radius = float(parts[1].strip())
                    surfaces.append({
                        'id': surf_id, 'type': 'sphere',
                        'center': center, 'radius': radius
                    })
                except (ValueError, IndexError):
                    pass
    return surfaces


def _point_on_surface(x, y, z, surf, tol=1e-4):
    """Check if point (x,y,z) lies on the given analytical surface."""
    if surf['type'] == 'plane':
        ox, oy, oz = surf['origin']
        nx, ny, nz = surf['normal']
        dist = nx * (x - ox) + ny * (y - oy) + nz * (z - oz)
        return abs(dist) < tol
    elif surf['type'] == 'sphere':
        cx, cy, cz = surf['center']
        R = surf['radius']
        dist = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        return abs(dist - R) < tol
    return False


def _face_on_surface(pa, pb, pc, surf, tol=1e-6, normal_tol=0.99):
    """Geometric face-level test: does triangle (pa,pb,pc) lie on ``surf``?

    Plane: all three vertices within *tol* of the plane AND the face normal
    aligned with the plane normal (``|n̂_face · n̂_plane| > normal_tol``) — the
    alignment check rejects the edge-on false positives a centroid test admits
    at feature ridges. Sphere: all three vertices within *tol* of the radius.
    """
    if surf['type'] == 'plane':
        ox, oy, oz = surf['origin']
        nx, ny, nz = surf['normal']
        for p in (pa, pb, pc):
            if abs(nx * (p[0] - ox) + ny * (p[1] - oy) + nz * (p[2] - oz)) >= tol:
                return False
        ab = [pb[i] - pa[i] for i in range(3)]
        ac = [pc[i] - pa[i] for i in range(3)]
        fn = [ab[1] * ac[2] - ab[2] * ac[1],
              ab[2] * ac[0] - ab[0] * ac[2],
              ab[0] * ac[1] - ab[1] * ac[0]]
        fmag = math.sqrt(fn[0]**2 + fn[1]**2 + fn[2]**2)
        nmag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if fmag < 1e-15 or nmag < 1e-15:
            return False
        dot = (fn[0] * nx + fn[1] * ny + fn[2] * nz) / (fmag * nmag)
        return abs(dot) > normal_tol
    elif surf['type'] == 'sphere':
        cx, cy, cz = surf['center']
        R = surf['radius']
        for p in (pa, pb, pc):
            if abs(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) - R) >= tol:
                return False
        return True
    return False


def _geo_face_surface(pa, pb, pc, surfaces, tol=1e-6, normal_tol=0.99):
    """Return the unique analytical surface id a face sits on, or None (geometric face classification).

    Collects every matching surface; on a degenerate multi-match (a ridge shared
    by two surfaces) the smallest id is returned deterministically.
    """
    hits = [s['id'] for s in surfaces
            if _face_on_surface(pa, pb, pc, s, tol, normal_tol)]
    return min(hits) if hits else None


# ──────────────── Mesh Cleaning Utilities ──────────────────────────

def _remove_duplicates_and_degenerates(volume_elements, nodes):
    """Remove duplicate tets (same 4 nodes) and degenerate tets (< 4 unique nodes).
    Also removes zero-volume tets. Returns cleaned list and diagnostic counts."""
    seen = set()
    cleaned = []
    n_dup = n_degen = n_zero = 0
    for elem in volume_elements:
        phys_id, entity_id, p1, p2, p3, p4 = elem
        node_set = frozenset([p1, p2, p3, p4])
        if len(node_set) < 4:
            n_degen += 1
            continue
        key = tuple(sorted([p1, p2, p3, p4]))
        if key in seen:
            n_dup += 1
            continue
        # Check zero volume
        sv = _signed_volume(nodes[p1], nodes[p2], nodes[p3], nodes[p4])
        if abs(sv) < 1e-15:
            n_zero += 1
            continue
        seen.add(key)
        cleaned.append(elem)
    return cleaned, n_dup, n_degen, n_zero


def _remove_overlapping_tets(volume_elements, nodes):
    """Detect faces shared by >2 tets. For each, keep the 2 with largest
    absolute volume and remove the rest. Returns cleaned list and count."""
    # Build face → tet index mapping
    face_to_tets = defaultdict(list)
    for ti, (_, _, p1, p2, p3, p4) in enumerate(volume_elements):
        for face in [tuple(sorted([p2, p3, p4])), tuple(sorted([p1, p3, p4])),
                     tuple(sorted([p1, p2, p4])), tuple(sorted([p1, p2, p3]))]:
            face_to_tets[face].append(ti)

    # Find tets to remove
    tets_to_remove = set()
    for face, tet_indices in face_to_tets.items():
        if len(tet_indices) > 2:
            # Sort by absolute volume (descending), keep best 2
            vols = []
            for ti in tet_indices:
                _, _, p1, p2, p3, p4 = volume_elements[ti]
                sv = abs(_signed_volume(nodes[p1], nodes[p2], nodes[p3], nodes[p4]))
                vols.append((sv, ti))
            vols.sort(reverse=True)
            for _, ti in vols[2:]:
                tets_to_remove.add(ti)

    if not tets_to_remove:
        return volume_elements, 0

    cleaned = [e for i, e in enumerate(volume_elements) if i not in tets_to_remove]
    return cleaned, len(tets_to_remove)


def _detect_hanging_nodes(volume_elements, nodes):
    """Detect boundary nodes that lie on a face of a non-incident tet (T-junctions).
    Returns count of suspected hanging nodes."""
    # Build face → tet mapping
    face_to_tets = defaultdict(list)
    node_to_tets = defaultdict(set)
    for ti, (_, _, p1, p2, p3, p4) in enumerate(volume_elements):
        for v in [p1, p2, p3, p4]:
            node_to_tets[v].add(ti)
        for face in [tuple(sorted([p2, p3, p4])), tuple(sorted([p1, p3, p4])),
                     tuple(sorted([p1, p2, p4])), tuple(sorted([p1, p2, p3]))]:
            face_to_tets[face].append(ti)

    # Boundary faces (shared by exactly 1 tet)
    bdy_faces = [f for f, ts in face_to_tets.items() if len(ts) == 1]
    bdy_nodes = set()
    for f in bdy_faces:
        bdy_nodes.update(f)

    # Check if any boundary node lies on a boundary face it doesn't belong to
    hanging = 0
    for face in bdy_faces:
        fa, fb, fc = face
        pa, pb, pc = nodes[fa], nodes[fb], nodes[fc]
        # Face normal
        ab = [pb[i] - pa[i] for i in range(3)]
        ac = [pc[i] - pa[i] for i in range(3)]
        nx = ab[1]*ac[2] - ab[2]*ac[1]
        ny = ab[2]*ac[0] - ab[0]*ac[2]
        nz = ab[0]*ac[1] - ab[1]*ac[0]
        nmag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nmag < 1e-15:
            continue
        # Check nearby boundary nodes not in this face
        for v in bdy_nodes:
            if v in face:
                continue
            pv = nodes[v]
            # Distance to face plane
            dist = abs(nx*(pv[0]-pa[0]) + ny*(pv[1]-pa[1]) + nz*(pv[2]-pa[2])) / nmag
            if dist < 1e-8:
                hanging += 1
    return hanging


# ──────────────── Surface Classification Repair ────────────────────

def _orient_outward(face, opp, nodes):
    """Order face (a,b,c) so its normal points away from the opposite vertex."""
    a, b, c = face
    pa, pb, pc, pd = nodes[a], nodes[b], nodes[c], nodes[opp]
    nx = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1])
    ny = (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2])
    nz = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
    dx, dy, dz = pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2]
    if nx * dx + ny * dy + nz * dz > 0:   # normal points toward opp → flip
        return (a, c, b)
    return (a, b, c)


def _repair_surface_classification(volume_elements, surface_elements, nodes,
                                   geo_surfaces=None):
    """Reconcile the surface-element list with the volume boundary.

    The display surface of a tetrahedral mesh is exactly the set of faces
    shared by a single tet.  A .msh produced upstream — or a volume list pruned
    by the cleaning steps above — can leave interior faces in the surface list
    (internal nodes/tets drawn as surface elements) or omit boundary faces newly
    exposed by tet removal.  This drops every surface element that is not a true
    boundary face and adds any boundary face missing from the list, so the .vol
    surface display segment matches the volume exactly.

    Added boundary faces are tagged by geometric face classification when
    ``geo_surfaces`` is supplied; only when geometry is unavailable or gives no
    unique match does the added face fall back to the majority kept tag (logged).

    Returns (repaired_surface_elements, n_removed, n_added).
    """
    face_owner = {}
    face_count = defaultdict(int)
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
    n_added_geo = 0
    n_added_fallback = 0
    for key in boundary:
        if key in existing:
            phys, ent, verts = existing[key]
            repaired.append((phys, ent, verts[0], verts[1], verts[2]))
        else:
            face, opp = face_owner[key]
            v = _orient_outward(face, opp, nodes)
            sid = None
            if geo_surfaces:
                sid = _geo_face_surface(nodes[v[0]], nodes[v[1]], nodes[v[2]],
                                        geo_surfaces)
            if sid is not None:
                n_added_geo += 1
            else:
                sid = default_tag
                n_added_fallback += 1
            repaired.append((sid, 1, v[0], v[1], v[2]))
            n_added += 1
    if n_added_fallback:
        print(f"  [Converter] Surface repair: {n_added_fallback} added face(s) "
              f"tagged by majority fallback (tag {default_tag}); "
              f"{n_added_geo} tagged from .geo geometry")
    return repaired, n_removed, n_added


# ──────────────── Surface Reclassification ─────────────────────────

def _reclassify_surfaces(surface_elements, nodes, geo_path):
    """Re-classify surface elements using analytical geometry from .geo file
    (geometric face-level: all three vertices on the surface plus normal alignment).
    A face with no unique geometric match keeps its incoming tag."""
    surfaces = _parse_geo_surfaces(geo_path)
    if not surfaces:
        print("  [Converter] WARNING: No surfaces parsed from .geo file")
        return surface_elements

    reclassified = 0
    unmatched = 0
    result = []
    for phys_id, entity_id, p1, p2, p3 in surface_elements:
        sid = _geo_face_surface(nodes[p1], nodes[p2], nodes[p3], surfaces)
        if sid is None:
            unmatched += 1
            new_id = phys_id
        else:
            if sid != phys_id:
                reclassified += 1
            new_id = sid
        result.append((new_id, entity_id, p1, p2, p3))
    msg = f"  [Converter] Reclassified {reclassified}/{len(surface_elements)} surface elements from .geo"
    if unmatched:
        msg += f" ({unmatched} kept incoming tag; no unique geometric match)"
    print(msg)
    return result


# ──────────────── Edge Segment Reconstruction ──────────────────────

def _reconstruct_edge_segments(surface_elements, fd_index_map):
    """Reconstruct edgesegmentsgi2 from boundary topology.
    Ridge edges = boundary edges shared by exactly 2 faces with different fd_index."""
    # Build edge → [fd_index, ...] mapping
    edge_to_fds = defaultdict(set)
    edge_to_tris = defaultdict(list)
    for phys_id, entity_id, p1, p2, p3 in surface_elements:
        fd_idx = fd_index_map.get(phys_id, 1)
        for edge in [(min(p1,p2), max(p1,p2)), (min(p1,p3), max(p1,p3)),
                     (min(p2,p3), max(p2,p3))]:
            edge_to_fds[edge].add(fd_idx)
            edge_to_tris[edge].append(fd_idx)

    # Ridge edges: exactly 2 different fd_indices
    ridge_edges = [(e, fds) for e, fds in edge_to_fds.items() if len(fds) == 2]
    if not ridge_edges:
        return []

    # Group connected ridge edges into numbered ridges (ednr)
    # Build adjacency for ridge edges
    node_to_ridge_edges = defaultdict(list)
    for i, (edge, _) in enumerate(ridge_edges):
        node_to_ridge_edges[edge[0]].append(i)
        node_to_ridge_edges[edge[1]].append(i)

    visited = [False] * len(ridge_edges)
    ednr = 0
    edge_to_ednr = {}
    for i in range(len(ridge_edges)):
        if visited[i]:
            continue
        ednr += 1
        # BFS to find connected component
        stack = [i]
        while stack:
            curr = stack.pop()
            if visited[curr]:
                continue
            visited[curr] = True
            edge_to_ednr[curr] = ednr
            edge = ridge_edges[curr][0]
            for v in edge:
                for neighbor in node_to_ridge_edges[v]:
                    if not visited[neighbor]:
                        # Same ridge if same pair of fd_indices
                        if ridge_edges[neighbor][1] == ridge_edges[curr][1]:
                            stack.append(neighbor)

    # Build output segments
    segments = []
    for i, (edge, fds) in enumerate(ridge_edges):
        if i not in edge_to_ednr:
            continue
        fd_list = sorted(fds)
        seg = {
            'p1': edge[0], 'p2': edge[1],
            'surfnr1': fd_list[0], 'surfnr2': fd_list[1],
            'ednr': edge_to_ednr[i]
        }
        segments.append(seg)

    return segments


# ──────────────── Main Conversion Function ─────────────────────────

def convert_msh_to_vol(msh_path, vol_path, geo_path=None,
                       with_edges=False, verbose=False):
    """
    Convert Gmsh .msh (2.2) to Netgen .vol format with full validation.

    Parameters
    ----------
    msh_path : str       Input .msh file path.
    vol_path : str       Output .vol file path.
    geo_path : str       Optional .geo file for surface reclassification.
    with_edges : bool    Generate edgesegmentsgi2 section.
    verbose : bool       Print detailed diagnostics.
    """
    physical_names = {}
    nodes = {}          # gmsh_node_id → (x, y, z)
    node_id_map = {}    # gmsh_node_id → 1-based contiguous index

    surface_elements = []
    volume_elements = []

    # ── Parse .msh file ───────────────────────────────────────────
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
                nid = int(parts[0])
                coords = (float(parts[1]), float(parts[2]), float(parts[3]))
                nodes[nid] = coords
                node_id_map[nid] = len(node_id_map) + 1
            elif section == '$Elements':
                parts = line.split()
                if len(parts) == 1:
                    continue
                elm_type = int(parts[1])
                num_tags = int(parts[2])
                tags = [int(x) for x in parts[3:3 + num_tags]]
                node_ids = [int(x) for x in parts[3 + num_tags:]]
                phys_id = tags[0] if num_tags >= 1 else 1
                entity_id = tags[1] if num_tags >= 2 else 1

                if elm_type == 2:  # 3-node triangle
                    surface_elements.append(
                        (phys_id, entity_id, node_ids[0], node_ids[1], node_ids[2]))
                elif elm_type == 4:  # 4-node tet
                    volume_elements.append(
                        (phys_id, entity_id,
                         node_ids[0], node_ids[1], node_ids[2], node_ids[3]))

    # ── Map node IDs to 1-based contiguous indices ────────────────
    mapped_nodes = {}
    for nid, coords in nodes.items():
        mapped_nodes[node_id_map[nid]] = coords

    mapped_surface = []
    for phys_id, entity_id, p1, p2, p3 in surface_elements:
        mapped_surface.append(
            (phys_id, entity_id,
             node_id_map[p1], node_id_map[p2], node_id_map[p3]))

    mapped_volume = []
    for phys_id, entity_id, p1, p2, p3, p4 in volume_elements:
        mapped_volume.append(
            (phys_id, entity_id,
             node_id_map[p1], node_id_map[p2], node_id_map[p3], node_id_map[p4]))

    n_original_tets = len(mapped_volume)
    n_original_tris = len(mapped_surface)

    # ── 2.1: Remove duplicates & degenerates (log + remove) ──────
    mapped_volume, n_dup, n_degen, n_zero = \
        _remove_duplicates_and_degenerates(mapped_volume, mapped_nodes)
    if n_dup or n_degen or n_zero:
        print(f"  [Converter] Removed: {n_dup} duplicate, {n_degen} degenerate, "
              f"{n_zero} zero-volume tets")

    # ── 2.2: Remove overlapping tets (log + remove) ──────────────
    mapped_volume, n_overlap = _remove_overlapping_tets(mapped_volume, mapped_nodes)
    if n_overlap:
        print(f"  [Converter] Removed: {n_overlap} overlapping tets")

    # ── 2.2b: Remove floating internal cavities ───────────────────
    # Iterate: removing cavity tets can expose new boundary faces
    total_cavity_tets = 0
    total_cavity_comps = 0
    for _cavity_pass in range(50):  # safety cap
        face_to_ti = defaultdict(list)
        for ti, (_, _, p1, p2, p3, p4) in enumerate(mapped_volume):
            for face in [tuple(sorted([p2, p3, p4])), tuple(sorted([p1, p3, p4])),
                         tuple(sorted([p1, p2, p4])), tuple(sorted([p1, p2, p3]))]:
                face_to_ti[face].append(ti)

        bdy_f = [f for f, ts in face_to_ti.items() if len(ts) == 1]
        if not bdy_f:
            break

        etf = defaultdict(list)
        for fi, face in enumerate(bdy_f):
            for e in [(min(face[0], face[1]), max(face[0], face[1])),
                      (min(face[0], face[2]), max(face[0], face[2])),
                      (min(face[1], face[2]), max(face[1], face[2]))]:
                etf[e].append(fi)
        fadj = defaultdict(set)
        for e, fl in etf.items():
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    fadj[fl[i]].add(fl[j])
                    fadj[fl[j]].add(fl[i])
        vis = set()
        comps = []
        for s in range(len(bdy_f)):
            if s in vis:
                continue
            stk = [s]
            comp = []
            while stk:
                c = stk.pop()
                if c in vis:
                    continue
                vis.add(c)
                comp.append(c)
                for nb in fadj.get(c, []):
                    if nb not in vis:
                        stk.append(nb)
            comps.append(comp)

        if len(comps) <= 1:
            break  # Only main surface remains

        comps.sort(key=len, reverse=True)
        cavity_tets = set()
        for comp in comps[1:]:
            for fi in comp:
                face = bdy_f[fi]
                cavity_tets.update(face_to_ti[tuple(sorted(face))])
        if not cavity_tets:
            break
        mapped_volume = [e for i, e in enumerate(mapped_volume)
                         if i not in cavity_tets]
        total_cavity_tets += len(cavity_tets)
        total_cavity_comps += len(comps) - 1

    if total_cavity_tets:
        print(f"  [Converter] Removed: {total_cavity_tets} cavity tets "
              f"from {total_cavity_comps} floating components")

    # ── 2.2c: Repair surface classification against the volume boundary ──
    # Drops interior faces wrongly listed as surface (internal nodes/tets in
    # the display segment) and adds any boundary face exposed by tet removal.
    geo_surfaces = _parse_geo_surfaces(geo_path) if geo_path else None
    mapped_surface, n_surf_removed, n_surf_added = \
        _repair_surface_classification(mapped_volume, mapped_surface, mapped_nodes,
                                       geo_surfaces)
    if n_surf_removed or n_surf_added:
        print(f"  [Converter] Surface repair: removed {n_surf_removed} interior "
              f"face(s), added {n_surf_added} missing boundary face(s)")

    # ── 2.3: Surface reclassification / repair ────────────────────
    unique_phys = set(e[0] for e in mapped_surface)
    if len(unique_phys) <= 1 and geo_path:
        print(f"  [Converter] All surface tags uniform ({unique_phys}), "
              f"reclassifying from {geo_path}")
        mapped_surface = _reclassify_surfaces(mapped_surface, mapped_nodes, geo_path)
    elif geo_surfaces:
        # Repair mode: retag only faces whose unique geometric surface (geometric face classification)
        # contradicts the incoming tag. Ambiguous / no-match faces are left
        # untouched, so valid incoming tags survive unless geometry is sure.
        n_retag = 0
        repaired = []
        for phys_id, entity_id, p1, p2, p3 in mapped_surface:
            sid = _geo_face_surface(mapped_nodes[p1], mapped_nodes[p2],
                                    mapped_nodes[p3], geo_surfaces)
            if sid is not None and sid != phys_id:
                repaired.append((sid, entity_id, p1, p2, p3))
                n_retag += 1
            else:
                repaired.append((phys_id, entity_id, p1, p2, p3))
        mapped_surface = repaired
        if n_retag:
            print(f"  [Converter] Retagged {n_retag} faces from .geo")
    elif len(unique_phys) <= 1 and not geo_path:
        print("  [Converter] WARNING: All surface tags uniform and no --geo provided. "
              "Face descriptors will be incomplete.")

    # ── 2.5: Hanging node detection (diagnostic) ─────────────────
    if verbose:
        n_hanging = _detect_hanging_nodes(mapped_volume, mapped_nodes)
        if n_hanging:
            print(f"  [Converter] WARNING: {n_hanging} potential hanging node detections "
                  f"(T-junctions)")

    # ── Guard: reject non-positive physical ids ──────────────────
    # A pid <= 0 would emit a negative geometry surfnr (pid - 1) into the face
    # descriptors — the exact converter-side defect this phase closes. With
    # --geo the retag pass above should already have fixed such faces; if any
    # survive, geometry could not decide, so fail loudly rather than write junk.
    bad_ids = sorted(pid for pid in set(e[0] for e in mapped_surface) if pid <= 0)
    if bad_ids:
        raise ValueError(
            f"Non-positive surface physical id(s) {bad_ids} would produce a "
            f"negative face-descriptor surfnr. Provide --geo to retag from "
            f"geometry, or correct the input .msh tags.")

    # ── Build face descriptor mapping ─────────────────────────────
    # unique_surf_phys_ids: sorted list of distinct physical IDs from surface elements
    unique_surf_phys_ids = sorted(set(e[0] for e in mapped_surface))
    unique_vol_phys_ids = sorted(set(e[0] for e in mapped_volume))

    # Map physical_id → face descriptor index (1-based)
    surf_phys_to_fd_idx = {pid: idx + 1 for idx, pid in enumerate(unique_surf_phys_ids)}

    # ── 2.4: Determine geometry surface numbers for face descriptors ──
    # The face descriptor surfnr field is the geometry surface number.
    # If we have .geo surfaces, map phys_id → geo surf number.
    # Otherwise, use phys_id - 1 as geometry surface number (0-indexed).
    geo_surfnr_map = {}
    if geo_path:
        geo_surfaces = _parse_geo_surfaces(geo_path)
        for pid in unique_surf_phys_ids:
            # phys_id should match geo surface id if reclassification was done
            matching = [s for s in geo_surfaces if s['id'] == pid]
            if matching:
                geo_surfnr_map[pid] = pid - 1  # 0-indexed geometry surface
            else:
                geo_surfnr_map[pid] = pid - 1
    else:
        for pid in unique_surf_phys_ids:
            geo_surfnr_map[pid] = pid - 1

    # ── 2.6: Edge segment reconstruction ──────────────────────────
    edge_segments = []
    if with_edges:
        edge_segments = _reconstruct_edge_segments(mapped_surface, surf_phys_to_fd_idx)
        print(f"  [Converter] Reconstructed {len(edge_segments)} edge segments "
              f"across {max((s['ednr'] for s in edge_segments), default=0)} ridges")

    # ── Write .vol file ───────────────────────────────────────────
    with open(vol_path, 'w') as f:
        f.write("# Generated by custom Python converter\n\n")
        f.write("mesh3d\ndimension\n3\ngeomtype\n0\n\n")

        # Face descriptors
        f.write("# surfnr\tdomin\tdomout\ttlosurf\tbcprop\n")
        f.write("facedescriptors\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for pid in unique_surf_phys_ids:
            geo_sn = geo_surfnr_map[pid]
            f.write(f" {geo_sn} 1 0 0 1\n")
        f.write("\n")

        # Surface elements
        f.write("# surfnr    bcnr   domin  domout      np      p1      p2      p3\n")
        f.write("surfaceelements\n")
        f.write(f"{len(mapped_surface)}\n")
        for phys_id, entity_id, p1, p2, p3 in mapped_surface:
            fd_idx = surf_phys_to_fd_idx[phys_id]
            f.write(f" {fd_idx} 1 1 0 3 {p1} {p2} {p3}\n")
        f.write("\n")

        # Volume elements
        f.write("#  matnr      np      p1      p2      p3      p4\n")
        f.write("volumeelements\n")
        f.write(f"{len(mapped_volume)}\n")
        for phys_id, entity_id, p1, p2, p3, p4 in mapped_volume:
            mat_nr = 1  # single domain
            f.write(f"{mat_nr} 4 {p1} {p2} {p3} {p4}\n")
        f.write("\n")

        # Edge segments (optional)
        if with_edges and edge_segments:
            f.write("# surfid  0   p1   p2   trignum1    trignum2   "
                    "domin/surfnr1    domout/surfnr2   ednr1   dist1   ednr2   dist2\n")
            f.write("edgesegmentsgi2\n")
            f.write(f"{len(edge_segments)}\n")
            for seg in edge_segments:
                f.write(f"       {seg['surfnr1']}       0"
                        f"       {seg['p1']}     {seg['p2']}"
                        f"       -1       -1"
                        f"       {seg['surfnr1']}        {seg['surfnr2']}"
                        f"        {seg['ednr']}            0"
                        f"       -1            0\n")
            f.write("\n")

        # Points
        f.write("#          X             Y             Z\n")
        f.write("points\n")
        n_pts = max(mapped_nodes.keys()) if mapped_nodes else 0
        f.write(f"{n_pts}\n")
        for i in range(1, n_pts + 1):
            x, y, z = mapped_nodes[i]
            f.write(f"    {x:.16f}      {y:.16f}      {z:.16f}\n")
        f.write("\n")

        # Point elements
        f.write("#          pnum             index\n")
        f.write("pointelements\n0\n\n")

        # Materials
        f.write("materials\n")
        f.write(f"{len(unique_vol_phys_ids)}\n")
        for pid in unique_vol_phys_ids:
            name = physical_names.get((3, pid), f"vol{pid}")
            f.write(f"{pid} {name}\n")
        f.write("\n")

        # BC names
        f.write("bcnames\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for pid in unique_surf_phys_ids:
            name = physical_names.get((2, pid), f"surface_{pid}")
            f.write(f"{surf_phys_to_fd_idx[pid]}\t{name}\n")
        f.write("\n")

        # Face colours
        f.write("#   Surfnr     Red     Green     Blue\n")
        f.write("face_colours\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for pid in unique_surf_phys_ids:
            fd_idx = surf_phys_to_fd_idx[pid]
            f.write(f"       {fd_idx}   0.00000000   1.00000000   0.00000000\n")

        # Face transparencies
        f.write("face_transparencies\n")
        f.write(f"{len(unique_surf_phys_ids)}\n")
        for pid in unique_surf_phys_ids:
            fd_idx = surf_phys_to_fd_idx[pid]
            f.write(f"       {fd_idx}   1.00000000\n")
        f.write("\n")

        f.write("endmesh\n")

    # ── Summary ───────────────────────────────────────────────────
    print(f"  [Converter] Input:  {n_original_tris} surface tris, {n_original_tets} tets")
    print(f"  [Converter] Output: {len(mapped_surface)} surface tris, "
          f"{len(mapped_volume)} tets, {len(unique_surf_phys_ids)} face descriptors")


# ──────────────── CLI Entry Point ──────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an input Gmsh .msh (2.2) mesh to an output Netgen .vol mesh.",
        epilog="Example: python tools/convert_mesh.py cases/cube_cross_coarse/output/bridge_0.msh --geo cases/cube_cross_coarse/input/geometry/cube.geo --output-dir cases/cube_cross_coarse/output/converted")
    parser.add_argument("input_msh", help="Input .msh file")
    parser.add_argument("output_vol", nargs='?', default=None,
                        help="Output .vol file name under --output-dir (default: input stem + .vol)")
    parser.add_argument("--output-dir", help="Output directory (default: case output/ for case input meshes, otherwise the mesh directory; never inside input/)")
    parser.add_argument("--geo", default=None,
                        help="Path to .geo file for surface reclassification")
    parser.add_argument("--with-edges", action="store_true",
                        help="Generate edgesegmentsgi2 section")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed diagnostics")

    args = parser.parse_args()

    msh_file = args.input_msh
    mesh_path = Path(msh_file).expanduser().resolve()
    input_dir = next((part for part in mesh_path.parents
                      if part.name == "input" and part.parent.parent.name == "cases"), None)
    default_dir = input_dir.parent / "output" if input_dir else mesh_path.parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_dir
    if "input" in output_dir.parts:
        parser.error(f"refused path inside input/: {output_dir}")
    output_name = args.output_vol or Path(msh_file).with_suffix(".vol").name
    vol_file = (output_dir / output_name).expanduser().resolve()
    if not vol_file.is_relative_to(output_dir):
        parser.error(f"output file must be inside --output-dir: {vol_file}")
    if "input" in vol_file.parts:
        parser.error(f"refused path inside input/: {vol_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {msh_file} to {vol_file}...")
    try:
        convert_msh_to_vol(msh_file, vol_file, geo_path=args.geo,
                           with_edges=args.with_edges, verbose=args.verbose)
        print("Conversion finished successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
