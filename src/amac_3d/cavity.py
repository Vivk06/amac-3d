"""Query topological neighborhoods of a tetrahedral mesh.

Operators use edge shells, node balls, unique edges and faces, and boundary
faces from this module. Queries read connectivity and skip dead tets; they do
not use coordinates or metrics. Cavity framing follows Loseille & Löhner (2010).
"""

from __future__ import annotations

from typing import List, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid


def fetch_faces(tetrahedra: List[int], tet_idx: int) -> List[Tuple[int, ...]]:
    """
    Return the four face triplets of tet *tet_idx*.
    Face i is opposite vertex i in the local ordering.
    """
    base = tet_idx * 4
    a, b, c, d = (tetrahedra[base], tetrahedra[base + 1],
                  tetrahedra[base + 2], tetrahedra[base + 3])
    return [
        (b, c, d),   # face 0, opposite a
        (a, c, d),   # face 1, opposite b
        (a, b, d),   # face 2, opposite c
        (a, b, c),   # face 3, opposite d
    ]


def fetch_edges(tetrahedra: List[int], tet_idx: int) -> List[Tuple[int, int]]:
    """
    Return the six edges of tet *tet_idx*.
    """
    base = tet_idx * 4
    a, b, c, d = (tetrahedra[base], tetrahedra[base + 1],
                  tetrahedra[base + 2], tetrahedra[base + 3])
    return [
        (a, b), (a, c), (a, d),
        (b, c), (b, d), (c, d),
    ]


def shell_of_edge(grid: 'Grid', a: int, b: int) -> List[int]:
    """
    Return all live tet indices that share edge (a, b).

    This is the set-intersection of ballOfNode[a] and ballOfNode[b],
    filtered to exclude dead tets (marked with -1).

    Raises
    ------
    ValueError
        If the result is empty (no edge exists between *a* and *b*).
    """
    ball_a = grid.ball_of(a)
    ball_b = grid.ball_of(b)
    tets = grid.tetrahedra
    # Filter out dead tets from the intersection
    shell = [t for t in set(ball_a) & set(ball_b)
             if tets[4 * t] != -1]

    if not shell:
        raise ValueError(
            f"shell_of_edge({a}, {b}) is empty — no edge connects these nodes"
        )
    return shell


def collect_edges(grid: 'Grid') -> List[Tuple[int, int]]:
    """
    Collect all unique edges in the mesh.

    Returns a list of ``(min_idx, max_idx)`` tuples.  Uses a set for
    O(1) duplicate detection through a set of edge pairs.
    """
    tets = grid.tetrahedra
    n_tets = len(tets) // 4
    seen: Set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int]] = []

    for t in range(n_tets):
        base = 4 * t
        verts = [tets[base], tets[base + 1], tets[base + 2], tets[base + 3]]
        # Skip dead tets
        if verts[0] == -1:
            continue
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = verts[i], verts[j]
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

    return edges


def collect_faces(grid: 'Grid') -> List[Tuple[int, int, int]]:
    """
    Collect all unique faces in the mesh.

    Returns a list of sorted ``(u, v, w)`` tuples.
    """
    tets = grid.tetrahedra
    n_tets = len(tets) // 4
    seen: Set[Tuple[int, int, int]] = set()
    faces: List[Tuple[int, int, int]] = []

    face_local = [
        (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
    ]

    for t in range(n_tets):
        base = 4 * t
        verts = [tets[base + k] for k in range(4)]
        for a, b, c in face_local:
            tri = tuple(sorted((verts[a], verts[b], verts[c])))
            if tri not in seen:
                seen.add(tri)
                faces.append(tri)

    return faces


def nodes_around_node(grid: 'Grid', node: int) -> List[int]:
    """
    Return all vertices that share an edge with *node*.
    """
    result: Set[int] = set()
    tets = grid.tetrahedra

    for tet_idx in grid.ball_of(node):
        for k in range(4):
            v = tets[4 * tet_idx + k]
            if v != node:
                result.add(v)

    return list(result)


def edges_around_node(edges: List[Tuple[int, int]],
                      node: int) -> List[int]:
    """
    Return indices into *edges* list for edges incident to *node*.
    """
    return [i for i, (u, v) in enumerate(edges) if u == node or v == node]


def boundary_faces_of_node(
    grid: 'Grid', node: int,
) -> List[Tuple[int, ...]]:
    """
    Return boundary faces (shared by exactly one tet) incident to *node*.

    A face is on the boundary if it appears in exactly one tetrahedron.
    This is determined by counting face occurrences across all tets in
    the ball of the given node.

    Returns a list of sorted face tuples.
    """
    from collections import Counter
    face_count: Counter = Counter()
    for tet_idx in grid.ball_of(node):
        for face in fetch_faces(grid.tetrahedra, tet_idx):
            face_count[tuple(sorted(face))] += 1
    return [f for f, c in face_count.items() if c == 1 and node in f]

