"""Build tetrahedron face adjacency from exact packed face keys.

Each face maps to its neighboring tet or a boundary sentinel. A single pass
over connectivity avoids linear scans and hash-bucket collisions; packed keys
support node indices below 2^20.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def pack_face(u: int, v: int, w: int) -> int:
    """
    Pack three sorted face-vertex indices into a single 64-bit int.

    Pre-condition: u < v < w  and all < 2^20 (1,048,576).

    Returns
    -------
    int
        Unique 60-bit key  (u << 40) | (v << 20) | w.
    """
    a, b, c = sorted((int(u), int(v), int(w)))
    return (a << 40) | (b << 20) | c


def unpack_face(key: int) -> Tuple[int, int, int]:
    """
    Inverse of :func:`pack_face`: recover the three sorted vertex indices.

    Returns
    -------
    tuple[int, int, int]
        ``(u, v, w)`` with ``u < v < w`` — identical to
        ``tuple(sorted((u, v, w)))`` for the triple that produced ``key``.
    """
    _MASK = (1 << 20) - 1
    return (key >> 40, (key >> 20) & _MASK, key & _MASK)


def build_adjacency(tetrahedra: List[int], n_points: int) -> List[int]:
    """
    Build the face-adjacency array for a tetrahedral mesh.

    Parameters
    ----------
    tetrahedra : list[int]
        Flat connectivity array [a0,b0,c0,d0, a1,...], length 4·Nt.
    n_points : int
        Total number of mesh points (used only for validation).

    Returns
    -------
    list[int]
        Flat array of length 4·Nt where ``adjacents[tetIdx*4+faceIdx]``
        is the index of the neighbouring tetrahedron sharing that face,
        or -1 if the face is on the boundary.

    Notes
    -----
    Each tetrahedron has four faces, each opposite one vertex:

        face 0: (b, c, d)   opposite vertex a
        face 1: (a, c, d)   opposite vertex b
        face 2: (a, b, d)   opposite vertex c
        face 3: (a, b, c)   opposite vertex d

    A temporary ``dict[int, (int,int)]`` maps packed face keys to the
    first tet that registered the face.  When a second tet encounters
    the same key, the bi-directional adjacency is recorded and the
    entry is deleted (each interior face is shared by exactly two tets).
    """
    n_tets = len(tetrahedra) // 4
    adjacents: List[int] = [-1] * (n_tets * 4)

    # Temporary face → (tetIdx, faceIdx) map
    face_map: Dict[int, Tuple[int, int]] = {}

    for tet_idx in range(n_tets):
        base = 4 * tet_idx
        a = tetrahedra[base]
        b = tetrahedra[base + 1]
        c = tetrahedra[base + 2]
        d = tetrahedra[base + 3]

        # Four faces opposite each vertex
        faces = [
            (b, c, d),  # face 0 — opposite a
            (a, c, d),  # face 1 — opposite b
            (a, b, d),  # face 2 — opposite c
            (a, b, c),  # face 3 — opposite d
        ]

        for face_idx, (fu, fv, fw) in enumerate(faces):
            key = pack_face(fu, fv, fw)

            if key in face_map:
                other_tet, other_face = face_map.pop(key)
                adjacents[tet_idx * 4 + face_idx] = other_tet
                adjacents[other_tet * 4 + other_face] = tet_idx
            else:
                face_map[key] = (tet_idx, face_idx)

    # After processing, face_map contains only boundary faces (unpaired).
    # They remain at the default -1 in the adjacents array.
    return adjacents
