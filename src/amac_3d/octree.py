"""Index mesh nodes in a recursive point octree for nearest-node queries.

The tree shares Grid's flat coordinate array and stores point indices. Leaves
subdivide at capacity, with a depth cap for coincident points. An empty tree
returns -1 from searchNearest.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid



class PointOctree:
    """
    3D point octree for efficient nearest-neighbour and ray queries.

    Each leaf holds up to *capacity* point indices.  When full the
    node subdivides into eight octant children.

    Attributes
    ----------
    bounds : tuple
        (minX, minY, minZ, maxX, maxY, maxZ).
    capacity : int
        Max points per leaf (default 8).
    globalCoordinates : list[float]
        Flat coordinate array [x0,y0,z0, ...] — shared, not copied.
    points : list[int]
        Point indices in this leaf.
    divided : bool
        True if this node has children.
    father : PointOctree | None
        Parent node.
    """

    __slots__ = (
        'bounds', 'capacity', 'depth', 'globalCoordinates', 'points', 'divided',
        'topNorthEast', 'topNorthWest', 'topSouthEast', 'topSouthWest',
        'bottomNorthEast', 'bottomNorthWest', 'bottomSouthEast', 'bottomSouthWest',
        'father',
    )

    MAX_DEPTH = 40   # safety cap against coincident-point recursion

    def __init__(
        self,
        bounds: Tuple[float, float, float, float, float, float],
        global_coordinates: List[float],
        capacity: int = 8,
        depth: int = 0,
    ) -> None:
        self.bounds = bounds
        self.capacity = capacity
        self.depth = depth
        self.globalCoordinates = global_coordinates
        self.points: List[int] = []
        self.divided = False
        self.topNorthEast: Optional[PointOctree] = None
        self.topNorthWest: Optional[PointOctree] = None
        self.topSouthEast: Optional[PointOctree] = None
        self.topSouthWest: Optional[PointOctree] = None
        self.bottomNorthEast: Optional[PointOctree] = None
        self.bottomNorthWest: Optional[PointOctree] = None
        self.bottomSouthEast: Optional[PointOctree] = None
        self.bottomSouthWest: Optional[PointOctree] = None
        self.father: Optional[PointOctree] = None

    # ──────────────────── Containment ──────────────────────────────

    def _contains(self, x: float, y: float, z: float) -> bool:
        minX, minY, minZ, maxX, maxY, maxZ = self.bounds
        return minX <= x <= maxX and minY <= y <= maxY and minZ <= z <= maxZ

    # ──────────────────── Insertion ────────────────────────────────

    def insert(self, point: int) -> bool:
        """Insert point index; subdivide if leaf overflows capacity."""
        x = self.globalCoordinates[point * 3]
        y = self.globalCoordinates[point * 3 + 1]
        z = self.globalCoordinates[point * 3 + 2]

        if not self._contains(x, y, z):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        # Safety cap: if we've subdivided too deep (coincident points),
        # just store the point here to avoid infinite recursion.
        if self.depth >= self.MAX_DEPTH:
            self.points.append(point)
            return True

        if not self.divided:
            self._subdivide()

        for child in self._children():
            if child.insert(point):
                return True

        # Fallback: point sits exactly on an octant boundary after
        # subdivision.  Store it in this node rather than dropping it.
        self.points.append(point)
        return True

    # ──────────────────── Subdivision ──────────────────────────────

    def _subdivide(self) -> None:
        minX, minY, minZ, maxX, maxY, maxZ = self.bounds
        midX = 0.5 * (minX + maxX)
        midY = 0.5 * (minY + maxY)
        midZ = 0.5 * (minZ + maxZ)

        octants = [
            (minX, minY, minZ, midX, midY, midZ),   # bottomSouthWest
            (midX, minY, minZ, maxX, midY, midZ),   # bottomSouthEast
            (minX, midY, minZ, midX, maxY, midZ),   # bottomNorthWest
            (midX, midY, minZ, maxX, maxY, midZ),   # bottomNorthEast
            (minX, minY, midZ, midX, midY, maxZ),   # topSouthWest
            (midX, minY, midZ, maxX, midY, maxZ),   # topSouthEast
            (minX, midY, midZ, midX, maxY, maxZ),   # topNorthWest
            (midX, midY, midZ, maxX, maxY, maxZ),   # topNorthEast
        ]

        d1 = self.depth + 1
        self.bottomSouthWest = PointOctree(octants[0], self.globalCoordinates, self.capacity, d1)
        self.bottomSouthEast = PointOctree(octants[1], self.globalCoordinates, self.capacity, d1)
        self.bottomNorthWest = PointOctree(octants[2], self.globalCoordinates, self.capacity, d1)
        self.bottomNorthEast = PointOctree(octants[3], self.globalCoordinates, self.capacity, d1)
        self.topSouthWest = PointOctree(octants[4], self.globalCoordinates, self.capacity, d1)
        self.topSouthEast = PointOctree(octants[5], self.globalCoordinates, self.capacity, d1)
        self.topNorthWest = PointOctree(octants[6], self.globalCoordinates, self.capacity, d1)
        self.topNorthEast = PointOctree(octants[7], self.globalCoordinates, self.capacity, d1)

        for child in self._children():
            child.father = self

        self.divided = True

        for p in self.points:
            for child in self._children():
                if child.insert(p):
                    break
        self.points = []

    # ──────────────────── Children accessor ────────────────────────

    def _children(self) -> List['PointOctree']:
        return [
            self.bottomSouthWest, self.bottomSouthEast,
            self.bottomNorthWest, self.bottomNorthEast,
            self.topSouthWest, self.topSouthEast,
            self.topNorthWest, self.topNorthEast,
        ]

    # ──────────────────── Leaf finder ──────────────────────────────

    def _findLeafNode(self, x: float, y: float, z: float) -> Optional['PointOctree']:
        """Recursively find the leaf node containing (x,y,z)."""
        if not self.divided:
            return self
        for child in self._children():
            if child._contains(x, y, z):
                result = child._findLeafNode(x, y, z)
                if result:
                    return result
        return None

    # ──────────────────── Nearest neighbour ────────────────────────

    def _distToBounds(self, x: float, y: float, z: float) -> float:
        """Squared distance from point to AABB."""
        minX, minY, minZ, maxX, maxY, maxZ = self.bounds
        dx = max(minX - x, 0.0, x - maxX)
        dy = max(minY - y, 0.0, y - maxY)
        dz = max(minZ - z, 0.0, z - maxZ)
        return dx * dx + dy * dy + dz * dz

    def _nearestPoint(self, x: float, y: float, z: float,
                      node: 'PointOctree',
                      min_dist: float,
                      nearest: int) -> Tuple[float, int]:
        if not node:
            return min_dist, nearest

        if node._distToBounds(x, y, z) > min_dist:
            return min_dist, nearest

        for p in node.points:
            px = node.globalCoordinates[p * 3]
            py = node.globalCoordinates[p * 3 + 1]
            pz = node.globalCoordinates[p * 3 + 2]
            d = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
            if d < min_dist:
                min_dist = d
                nearest = p

        if node.divided:
            children = [
                (child, child._distToBounds(x, y, z))
                for child in node._children()
            ]
            children.sort(key=lambda c: c[1])
            for child, _ in children:
                min_dist, nearest = self._nearestPoint(x, y, z, child, min_dist, nearest)

        return min_dist, nearest

    def searchNearest(self, x: float, y: float, z: float) -> int:
        """
        Find the nearest point index to (x, y, z).

        Returns an integer index, or -1 if the tree is empty.
        """
        min_dist = float('inf')
        nearest = -1
        min_dist, nearest = self._nearestPoint(x, y, z, self, min_dist, nearest)
        return nearest

    def children_points(self, out: List[int]) -> None:
        """Recursively collect all point indices from leaves."""
        if self.divided:
            for child in self._children():
                child.children_points(out)
        else:
            out.extend(self.points)


def build_point_octree(grid: 'Grid') -> PointOctree:
    """
    Build a PointOctree from *grid* using its bounding box + padding.
    """
    pad = 0.01
    bounds = (
        grid.xMin - pad, grid.yMin - pad, grid.zMin - pad,
        grid.xMax + pad, grid.yMax + pad, grid.zMax + pad,
    )
    ot = PointOctree(bounds, grid.coordinates, capacity=8)
    n_pts = len(grid.coordinates) // 3
    for i in range(n_pts):
        ot.insert(i)
    return ot
