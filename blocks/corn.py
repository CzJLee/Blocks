import blocks
import visualize
from typing import Iterable

Voxel = blocks.Voxel
Voxels = blocks.Voxels
Piece = blocks.Piece

type Kern = tuple[int, int]
"""An (x, y) coordinate representing a 1x1 unit kernel of a piece."""


class Kernel(blocks.Piece):
    circumference: int = 15

    def __init__(
        self,
        voxels: Iterable[Kern],
        name: str = "Kernel",
        canonicalize: bool = True,
        validate: bool = False,
    ):
        """
        voxels: an iterable of (x, y) tuples indicating which unit voxels the piece occupies.
        We canonicalize so that the piece's voxels are shifted so that the minimal x, y, z are 0.
        """
        self.name = name
        self.voxels: Voxels = {(x, y, 0) for (x, y, *_) in voxels}
        if canonicalize:
            self.canonicalize()

        if validate and not self.is_contiguous():
            raise ValueError("Piece is not contiguous.")

    def translate(self, dx: int, dy: int, dz: int) -> "Kernel":
        """Return a new Piece shifted by (dx, dy)."""
        del dz  # Unused
        new_coords = {
            ((x + dx) % self.circumference, (y + dy)) for (x, y, _) in self.voxels
        }
        return Kernel(new_coords, canonicalize=False, name=self.name)

    def flip(self) -> "Kernel":
        return Kernel(self.rotate_z().rotate_z().voxels)

    def unique_rotations(self, canonicalize: bool = True) -> set["Kernel"]:
        return {self, self.flip()}


class Space(blocks.Space):
    def can_place(self, piece: Kernel) -> bool:
        """Check if piece in its current position and orientation can fit into the space and does not overlap occupied voxels."""
        # This is done by ensuring that the voxel set does not overlap.

        # Check if the voxel pieces is a subset of the allowed voxels.
        can_fit_in_allowed_voxels = piece.voxels <= self.allowed_voxels

        # Check if the voxel pieces is not overlapping with the occupied voxels.
        does_not_overlap_occupied_voxels = piece.voxels.isdisjoint(self.occupied_voxels)

        # Check if no gaps below set piece.
        to_be_occupied = self.occupied_voxels | piece.voxels
        for x, y, z in piece.voxels:
            for j in range(0, y):
                if (x, j, z) not in to_be_occupied:
                    return False

        return can_fit_in_allowed_voxels and does_not_overlap_occupied_voxels


class Solver(blocks.Solver):
    def legal_translations(self, piece: Kernel) -> list[Piece]:
        """Generate all legal translations of a piece inside the space."""
        # Check all x positions and increment y by 1 until we find a legal space.
        translations = []
        (min_px, min_py, min_pz), (max_px, max_py, max_pz) = piece.bounding_box()
        for i in range(15):
            for j in range(5 - max_py):
                candidate = piece.translate(i, j, 0)
                if self.space.can_place(candidate):
                    translations.append(candidate)
                    break

        return translations

    def solve(self) -> list[list[Kernel]]:
        if not self.pieces:
            # If given no pieces, return no solutions.
            return []

        for i in range(len(self.pieces)):
            first_piece = self.pieces[i]
            remaining_pieces = self.pieces[:i] + self.pieces[i + 1 :]
            print("Running next major loop.")

            for candidate in first_piece.unique_rotations():
                # for candidate in self.legal_translations(rot):
                print("Checking new rotations of major loop.")
                # For the first piece, try placing all legal translations, and then continue recursive solve.
                if self.space.can_place(candidate):
                    self.space.place(candidate)
                    sol = self._solve_recursive(remaining_pieces, [candidate])
                    if sol:
                        return self.solutions
                    self.space.remove(candidate)

        return self.solutions
    
    def _solve_recursive(
        self, remaining_pieces: list[Kernel], placed_pieces: list[Kernel]
    ):
        # visualize.plot_solutions([list(placed_pieces)])
        if not remaining_pieces:
            # Solution found
            self.solutions.append(list(placed_pieces))
            return True

        for i, piece in enumerate(remaining_pieces):
            next_remaining = remaining_pieces[:i] + remaining_pieces[i + 1 :]

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
                        
        return False


if __name__ == "__main__":
    pieces = [
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)},
        {(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (0, 3)},
        {(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (1, 3)},
        {(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)},
    ]
    pieces = [
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},

    ]
    pieces = [Kernel(p) for p in pieces]
    # for piece in pieces:
    #     visualize.plot_solution_pieces([piece.translate(-1, 0, 0)])

    space = Space.cuboid(15, 5, 1)
    solver = Solver(space, pieces)

    solutions = solver.solve()

    print(f"Found {len(solutions)} solutions.")
    visualize.plot_solutions(solutions)
