import abc
import functools
import sys
from typing import Generic, Iterable, Self, TypeVar
import logging
import networkx as nx

logger = logging.getLogger(__name__)

VoxelT = TypeVar("VoxelT", bound=tuple)

type Voxel3D = tuple[int, int, int]
"""An (x, y, z) coordinate representing a 1x1x1 unit cube of a piece."""

type Voxel2D = tuple[int, int]
"""An (x, y) coordinate representing a 1x1 unit square of a piece."""

type Voxels = set[Voxel3D]
"""A set of voxels that represent a solid puzzle piece."""

VoxelPieceT = TypeVar("VoxelPieceT", bound="AbstractVoxelPiece")
"""Type Var for AbstractVoxelPiece."""

class AbstractPiece(abc.ABC):
    """Abstract base class for a puzzle piece."""
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Length of the piece."""
        
    @abc.abstractmethod
    def __hash__(self) -> int:
        """Hash of the piece."""
        
    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Checks if two pieces are exactly equal."""

class AbstractVoxelPiece(abc.ABC):
    """Abstract base class for an arbitrary voxel set puzzle piece."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of voxels in the piece."""

    @property
    @abc.abstractmethod
    def voxels(self) -> Voxels:
        """The voxels that make up this piece."""

    @abc.abstractmethod
    def translate(self, *args: int) -> Self:
        """Return a new Piece shifted along its coordinates by some given values."""

    @abc.abstractmethod
    def canonicalize(self) -> Self:
        """Shift coords so that minimal x,y,z are all zero or non-negative."""


class TransformationPolicy(abc.ABC, Generic[VoxelPieceT]):
    """An abstract class that represents a policy for how to transform a piece.

    A policy is designed to be given to a solver so that the solver can generate the valid orientations of a given piece within the constraints of a specific puzzle.
    """

    # @abc.abstractmethod
    # def allowed_orientations(self, piece: VoxelPieceT) -> list[VoxelPieceT]:
    #     """Return a list of all orientations for a piece."""

    @abc.abstractmethod
    def allowed_unique_orientations(self, piece: VoxelPieceT) -> list[VoxelPieceT]:
        """Return a list of all unique orientations for a piece."""


class Piece(AbstractVoxelPiece):
    def __init__(
        self,
        voxels: Iterable[Voxel3D],
        name: str = "Piece",
        canonicalize: bool = True,
        validate: bool = True,
    ):
        """
        Create a piece.

        Args:
            voxels: an iterable of (x, y, z) tuples indicating which unit voxels the piece occupies.
            name: name of the piece.
            canonicalize: Canonicalize the piece so that the piece's voxels are shifted so that the minimal x, y, z are 0.
            validate: Check that the piece is contiguous.
        """
        self.name = name
        if canonicalize:
            voxels = self._canonicalize(voxels)
        self._voxels: Voxels = set(voxels)

        if validate and not self.is_contiguous():
            raise ValueError("Piece is not contiguous.")

    @classmethod
    def from_2d(cls, voxels_2d: Iterable[tuple[int, int]], z: int = 0, **kwargs):
        """Create a Piece from 2D coordinates."""
        voxels_3d = {(x, y, z) for x, y in voxels_2d}
        return cls(voxels_3d, **kwargs)

    @property
    def voxels(self) -> Voxels:
        return self._voxels

    def __len__(self):
        """Return the number of voxels in the piece."""
        return len(self.voxels)

    def is_contiguous(self):
        """Returns True of a piece is fully connected (6-connectivity), False otherwise."""
        G = nx.Graph()
        G.add_nodes_from(self.voxels)

        # 6-connectivity
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        # Draw edges between connected voxels.
        for x, y, z in self.voxels:
            for dx, dy, dz in directions:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in self.voxels:
                    G.add_edge((x, y, z), neighbor)

        return nx.is_connected(G)

    def _canonicalize(self, voxels: Iterable[Voxel3D]) -> set[Voxel3D]:
        """Shift coords so that minimal x,y,z are all zero or non-negative."""
        min_x = min(x for (x, _, _) in voxels)
        min_y = min(y for (_, y, _) in voxels)
        min_z = min(z for (_, _, z) in voxels)
        return {(x - min_x, y - min_y, z - min_z) for (x, y, z) in voxels}

    def canonicalize(self) -> Self:
        """Shift coords so that minimal x,y,z are all zero or non-negative."""
        return self.__class__(self._voxels, canonicalize=True, name=self.name)

    def translate(self, *args: int) -> Self:
        """Return a new Piece shifted by (dx, dy, dz)."""
        try:
            dx, dy, dz = args
        except ValueError as e:
            raise ValueError(
                f"Expected 3 arguments (dx, dy, dz), got {len(args)}"
            ) from e
        new_coords = {(x + dx, y + dy, z + dz) for (x, y, z) in self.voxels}
        return self.__class__(new_coords, canonicalize=False, name=self.name)

    def rotate_x(self, canonicalize: bool = True) -> Self:
        """
        Rotate 90° about the X-axis. That is, (x,y,z) -> (x, -z, y)
        (assuming right-hand rule, etc.). After rotation canonicalize.
        """
        new_coords = {(x, -z, y) for (x, y, z) in self.voxels}
        return self.__class__(new_coords, canonicalize=canonicalize, name=self.name)

    def rotate_y(self, canonicalize: bool = True) -> Self:
        """Rotate 90° about Y axis: (x,y,z) -> (z, y, -x)"""
        new_coords = {(z, y, -x) for (x, y, z) in self.voxels}
        return self.__class__(new_coords, canonicalize=canonicalize, name=self.name)

    def rotate_z(self, canonicalize: bool = True) -> Self:
        """Rotate 90° about Z axis: (x,y,z) -> (-y, x, z)"""
        new_coords = {(-y, x, z) for (x, y, z) in self.voxels}
        return self.__class__(new_coords, canonicalize=canonicalize, name=self.name)

    @functools.cache
    def unique_rotations(self, canonicalize: bool = True) -> set["Piece"]:
        """
        Precompute all distinct rotations for this piece.
        Eliminates duplicates due to symmetry.
        """
        unique_rotations: set[Piece] = set()
        to_explore = [self]

        while to_explore:
            p = to_explore.pop()
            if p in unique_rotations:
                continue
            unique_rotations.add(p)
            # generate next rotations
            for rp in (
                p.rotate_x(canonicalize=canonicalize),
                p.rotate_y(canonicalize=canonicalize),
                p.rotate_z(canonicalize=canonicalize),
            ):
                if rp not in unique_rotations:
                    to_explore.append(rp)

        return unique_rotations

    def bounding_box(self) -> tuple[Voxel3D, Voxel3D]:
        """
        Returns ((min_x, min_y, min_z), (max_x, max_y, max_z)) for the piece.
        """
        xs = [x for (x, _, _) in self.voxels]
        ys = [y for (_, y, _) in self.voxels]
        zs = [z for (_, _, z) in self.voxels]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def bounding_box_cuboid(self) -> "Piece":
        """Create a cuboid piece of the bounding box of this piece."""
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = self.bounding_box()
        voxels = {
            (x, y, z)
            for x in range(min_x, max_x + 1)
            for y in range(min_y, max_y + 1)
            for z in range(min_z, max_z + 1)
        }
        return Piece(voxels, canonicalize=False)

    def __eq__(self, other: object) -> bool:
        """Checks if pieces are exactly equal."""
        if not isinstance(other, self.__class__):
            return False
        return self.voxels == other.voxels

    def is_congruent(self, other: Self) -> bool:
        """
        Check if this piece is congruent to another piece,
        meaning they are equal up to rotation and translation.
        """
        if not isinstance(other, Piece):
            return TypeError("Cannot compare congruence between non-pieces.")

        # Canonicalize the pieces (translation-invariant)
        self_canonical = Piece(self.voxels, canonicalize=True)
        other_canonical = Piece(other.voxels, canonicalize=True)

        # If self (any rotation) matches other, return True
        return other_canonical in self_canonical.unique_rotations()

    def __hash__(self) -> int:
        # Hash based on frozenset of coords
        return hash(frozenset(self.voxels))

    def __str__(self) -> str:
        return f"{self.name}({sorted(self.voxels)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({sorted(self.voxels)})"


class PolicyFull3DRotations(TransformationPolicy[Piece]):
    """Allows all possible rotations of a 3D piece."""

    def allowed_orientations(self, piece: Piece) -> list[Piece]:
        """
        Return a list of unique orientations (rotations) of this piece
        under the group generated by 90° rotations about X, Y, Z axes.
        """

        return list(piece.unique_rotations())


class Policy2DRotationsNoFlip(TransformationPolicy[Piece]):
    """Allows four rotations of a 2D piece. No flips allowed."""

    def allowed_unique_orientations(self, piece: Piece) -> list[Piece]:
        """Returns a list of unique rotations of a 2D piece with no flips."""

        return list(
            {
                piece,
                piece.rotate_z(),
                piece.rotate_z().rotate_z(),
                piece.rotate_z().rotate_z().rotate_z(),
            }
        )


class Space:
    def __init__(self, allowed_voxels: Voxels):
        """
        Create a Space for valid solutions.

        Can represent any shape of space, not just a cuboid.

        Args:
            allowed_voxels: Set of (x, y, z) tuples where pieces are allowed.
            constraint: Set of Voxels that act as a container / constraint. Often used for packing puzzles.
        """
        self.allowed_voxels = allowed_voxels
        self.pieces: set[AbstractVoxelPiece] = set()

    @classmethod
    def cuboid(cls, size_x: int, size_y: int, size_z: int) -> "Space":
        """Create a rectangular space of size_x * size_y * size_z"""
        voxels = {
            (x, y, z)
            for x in range(size_x)
            for y in range(size_y)
            for z in range(size_z)
        }
        return cls(voxels)

    @property
    def occupied_voxels(self) -> Voxels:
        """Return all voxels that are occupied by pieces."""
        voxels: Voxels = set()
        # Union all pieces
        for piece in self.pieces:
            voxels.update(piece.voxels)
        return voxels

    @property
    def min_x(self) -> int:
        """Return the minimum x coordinate of all allowed voxels."""
        return min(x for x, _, _ in self.allowed_voxels)

    @property
    def max_x(self) -> int:
        """Return the maximum x coordinate of all allowed voxels."""
        return max(x for x, _, _ in self.allowed_voxels)

    @property
    def min_y(self) -> int:
        """Return the minimum y coordinate of all allowed voxels."""
        return min(y for _, y, _ in self.allowed_voxels)

    @property
    def max_y(self) -> int:
        """Return the maximum y coordinate of all allowed voxels."""
        return max(y for _, y, _ in self.allowed_voxels)

    @property
    def min_z(self) -> int:
        """Return the minimum z coordinate of all allowed voxels."""
        return min(z for _, _, z in self.allowed_voxels)

    @property
    def max_z(self) -> int:
        """Return the maximum z coordinate of all allowed voxels."""
        return max(z for _, _, z in self.allowed_voxels)

    def can_place(self, piece: Piece) -> bool:
        """Check if piece in its current position and orientation can fit into the space and does not overlap occupied voxels."""
        # This is done by ensuring that the voxel set does not overlap.

        # Check if the voxel pieces is a subset of the allowed voxels.
        can_fit_in_allowed_voxels = piece.voxels <= self.allowed_voxels

        # Check if the voxel pieces is not overlapping with the occupied voxels.
        does_not_overlap_occupied_voxels = piece.voxels.isdisjoint(self.occupied_voxels)

        return can_fit_in_allowed_voxels and does_not_overlap_occupied_voxels

    def place(self, piece: Piece, validate: bool = False):
        """Place a piece (mark voxels as occupied). Assumes can_place was True."""
        if validate:
            if not self.can_place(piece):
                raise ValueError("Cannot place piece in space.")
        self.pieces.add(piece)

    def remove(self, piece: Piece):
        """Remove a piece from space (for backtracking)."""
        self.pieces.remove(piece)

    def valid_placements(self, piece: Piece) -> list[Piece]:
        """Given a piece of a certain orientation, return all translations of that piece that are valid placements in the space."""
        piece = piece.canonicalize()
        # Get bounds of allowed translations.
        ((_, _, _), (max_x, max_y, max_z)) = piece.bounding_box()
        space_min_x, space_max_x = self.min_x, self.max_x
        space_min_y, space_max_y = self.min_y, self.max_y
        space_min_z, space_max_z = self.min_z, self.max_z

        # Generate all translations.
        translations: list[Piece] = []
        for x in range(space_min_x, space_max_x - max_x + 1):
            for y in range(space_min_y, space_max_y - max_y + 1):
                for z in range(space_min_z, space_max_z - max_z + 1):
                    translated_piece = piece.translate(x, y, z)
                    if self.can_place(translated_piece):
                        translations.append(translated_piece)

        return translations


class AbstractSolver(abc.ABC):
    """Abstract class for a puzzle solver."""

    @abc.abstractmethod
    def solve(self) -> list[list[Piece]]:
        """Solve a puzzle and return a list of solutions."""


class Solver(AbstractSolver):
    def __init__(self, space: Space, pieces: list[Piece]):
        self.space = space
        self.pieces = sorted(
            pieces, key=lambda x: len(x.unique_rotations()), reverse=True
        )
        self.solutions: list[list[Piece]] = []

    def legal_translations(self, piece: Piece) -> list[Piece]:
        """Generate all legal translations of a piece inside the space."""
        (min_px, min_py, min_pz), (max_px, max_py, max_pz) = piece.bounding_box()
        xs = [x for x, _, _ in self.space.allowed_voxels]
        ys = [y for _, y, _ in self.space.allowed_voxels]
        zs = [z for _, _, z in self.space.allowed_voxels]

        min_dx, max_dx = min(xs) - min_px, max(xs) - max_px
        min_dy, max_dy = min(ys) - min_py, max(ys) - max_py
        min_dz, max_dz = min(zs) - min_pz, max(zs) - max_pz

        translations = []
        for dx in range(min_dx, max_dx + 1):
            for dy in range(min_dy, max_dy + 1):
                for dz in range(min_dz, max_dz + 1):
                    candidate = piece.translate(dx, dy, dz)
                    if self.space.can_place(candidate):
                        translations.append(candidate)
        return translations

    def solve(self) -> list[list[Piece]]:
        if not self.pieces:
            # If given no pieces, return no solutions.
            return []

        # Lock the first piece rotation to reduce global rotational duplicates
        first_piece = self.pieces[0]
        remaining_pieces = self.pieces[1:]

        fixed_rotation = first_piece  # or first_piece.canonical_rotation()
        for candidate in self.legal_translations(fixed_rotation):
            # For the first piece, try placing all legal translations, and then continue recursive solve.
            if self.space.can_place(candidate):
                self.space.place(candidate)
                sol = self._solve_recursive(remaining_pieces, [candidate])
                if sol:
                    return self.solutions
                self.space.remove(candidate)

        return self.solutions

    def _solve_recursive(
        self, remaining_pieces: list[Piece], placed_pieces: list[Piece]
    ):
        # visualize.plot_solutions([list(placed_pieces)])
        if not remaining_pieces:
            # Solution found
            self.solutions.append(list(placed_pieces))
            return True

        piece = remaining_pieces[0]
        next_remaining = remaining_pieces[1:]

        for rot in piece.unique_rotations(canonicalize=False):
            for candidate in self.legal_translations(rot):
                if self.space.can_place(candidate):
                    self.space.place(candidate)
                    placed_pieces.append(candidate)
                    sol = self._solve_recursive(next_remaining, placed_pieces)
                    if sol:
                        return True
                    placed_pieces.pop()
                    self.space.remove(candidate)


def cuboid(size_x: int, size_y: int, size_z: int) -> Voxels:
    """Create a rectangular set of Voxels of shape (size_x, size_y, size_z)."""
    voxels = {
        (x, y, z) for x in range(size_x) for y in range(size_y) for z in range(size_z)
    }
    return voxels


def pieces_disconnected(piece_a: Piece, piece_b: Piece) -> bool:
    """Checks if two pieces can be trivially infinitely separated.

    Two pieces can be considered disconnected if there is no intersection of their bounding boxes.
    """
    return piece_a.bounding_box_cuboid().voxels.isdisjoint(
        piece_b.bounding_box_cuboid().voxels
    )
