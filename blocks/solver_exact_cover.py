"""Implementation of an Exact Cover solver using Knuth's Algorithm X with Dancing Links (DLX)."""

from . import blocks
from typing import Iterator, TypeVar, Self, Iterable, Collection
import logging
import time

logger = logging.getLogger(__name__)


class DLXNode:
    """A node in the Dancing Links structure."""
    
    __slots__ = ("L", "R", "U", "D", "C")

    def __init__(self, column: "ColumnNode"):
        # Links to Left, Right, Up, Down nodes.
        self.L: Self = self
        self.R: Self = self
        self.U: DLXNode | ColumnNode = self
        self.D: DLXNode | ColumnNode = self

        self.C = column
        """A link to the column header. Used to quickly get the column size or select columns to cover/uncover."""


class ColumnNode(DLXNode):
    """Column header node."""
    
    __slots__ = ("name", "size")

    def __init__(self, name: str):
        super().__init__(self)
        self.name = name

        self.size = 0
        """Number of nodes in this column."""


T_DLXNode = TypeVar("T_DLXNode", bound=DLXNode)
"""Type Var for DLXNode."""


class DLXSolver:
    """Implementation of a Knuth's Algorithm X with Dancing Links (DLX) solver."""

    def __init__(self, matrix: list[list[int]], column_names: list[str]):
        self.matrix = matrix
        self.column_names = column_names
        self.root = self.build_dlx_matrix()

    def build_dlx_matrix(self) -> ColumnNode:
        """
        Convert a 0/1 matrix into a DLX linked structure.

        The DLX linked structure is a "toroidal doubly-linked list", or a 4-way
        circular doubly-linked structure.

        Returns:
            The root node of the DLX linked structure.
        """
        # Create header root.
        # The root is a sentinel node that links the beginning and end of the
        # column header chain.
        # So, `root.R` will point to the first column header, and
        # `root.L` will point to the last column header.
        root = ColumnNode("root")

        # Create a column header node for each column name, and create a
        # left-right circular loop.
        columns = []
        prev = root
        for name in self.column_names:
            col = ColumnNode(name)
            columns.append(col)
            prev.R = col
            col.L = prev
            prev = col
        # Connect the ends of the loop.
        prev.R = root
        root.L = prev

        # Add matrix rows to the structure
        for row in self.matrix:
            first_node = None
            for col_idx, cell in enumerate(row):
                if cell:
                    col = columns[col_idx]

                    # Create new node.
                    node = DLXNode(col)

                    # Insert into column (vertical).
                    # The "bottom" of the column is `col.U`, so we can insert the
                    # new node between `col.U` and `col`.
                    node.D = col
                    node.U = col.U
                    col.U.D = node
                    col.U = node
                    col.size += 1

                    # Insert into row (horizontal)
                    if first_node is None:
                        # If we only consider one node in the row, it is already self linking.
                        first_node = node
                    else:
                        # When we start to add more nodes, similar to before,
                        # we can insert the new node between `first_node.L` and
                        # `first_node`.
                        node.R = first_node
                        node.L = first_node.L
                        first_node.L.R = node
                        first_node.L = node
        return root

    def cover(self, column: ColumnNode):
        """Cover a column header and all rows connected to it.

        First, the specified column header node is removed from the column
        header chain.

        Then, for each node in that column chain, remove all the other nodes in
        that entire row that that node is connected to.
        """
        column.R.L = column.L
        column.L.R = column.R
        for row in self.iterate_down(column):
            for node in self.iterate_right(row):
                node.D.U = node.U
                node.U.D = node.D
                node.C.size -= 1

    def uncover(self, column: ColumnNode):
        """Undo cover (reverse order)."""
        for row in self.iterate_up(column):
            for node in self.iterate_left(row):
                node.C.size += 1
                node.D.U = node
                node.U.D = node
        column.R.L = column
        column.L.R = column

    def iterate_right(self, node: T_DLXNode) -> Iterator[T_DLXNode]:
        """Yield nodes to the right until looping back to the original node.

        Does not yield the original node.
        """
        current = node.R
        while current != node:
            yield current
            current = current.R

    def iterate_left(self, node: T_DLXNode) -> Iterator[T_DLXNode]:
        """Yield nodes to the left until looping back to the original node.

        Does not yield the original node.
        """
        current = node.L
        while current != node:
            yield current
            current = current.L

    def iterate_down(self, node: DLXNode) -> Iterator[DLXNode | ColumnNode]:
        """Yield nodes downward until looping back to the original node.

        Does not yield the original node.
        """
        current = node.D
        while current != node:
            yield current
            current = current.D

    def iterate_up(self, node: DLXNode) -> Iterator[DLXNode | ColumnNode]:
        """Yield nodes upward until looping back to the original node.

        Does not yield the original node.
        """
        current = node.U
        while current != node:
            yield current
            current = current.U

    def get_smallest_column(self, root: ColumnNode) -> ColumnNode:
        """Return the column header node with the smallest size."""
        smallest_col = root
        smallest_size = float("inf")

        for col in self.iterate_right(root):
            if col.size < smallest_size:
                smallest_col = col
                smallest_size = col.size

        return smallest_col

    def search(
        self,
        current_solution: list[DLXNode],
        root: ColumnNode,
        valid_solutions: list[list[DLXNode]],
        first_call=True,
    ) -> None:
        """Algorithm X recursive search.

        This algorithm:
        1. Choose a column (a constraint). e.g. An empty cell in the space.
        2. Choose a row in that column (a possible assignment). e.g. Some possible piece and orientation that fills this empty cell.
        3. Cover all other columns satisfied by that row (enforcing its constraints). e.g. Remove all other piece possibilities that also fill this empty cell.
        4. Recurse.
        5. Backtrack (uncover everything you covered).
        """
        if root.R == root:
            # All columns covered => success
            valid_solutions.append(current_solution.copy())
            return

        # Step 1.
        # Choose column with smallest size for the most optimal approach.
        col = self.get_smallest_column(root)

        if col.size == 0:
            # Early exit. If column is empty, this branch is impossible.
            return

        # Cover this column.
        self.cover(col)

        if first_call:
            logger.debug("Column %s has %d rows.", col.name, col.size)

        for i, row in enumerate(self.iterate_down(col), start=1):
            if first_call:
                logger.debug("Searching row %d / %d.", i, col.size)
            # Step 2.
            current_solution.append(row)

            # Step 3.
            for node in self.iterate_right(row):
                self.cover(node.C)

            # Step 4.
            self.search(current_solution, root, valid_solutions, first_call=False)

            # Step 5.
            for node in self.iterate_left(row):
                self.uncover(node.C)

            current_solution.pop()

        self.uncover(col)


class SolverExactCover2D(blocks.AbstractSolver):
    """A solver that implements Knuth's Algorithm X with Dancing Links (DLX).

    DLX is an algorithm that can be used to solve a puzzle where pieces are placed onto a board such that:
    - No piece overlaps
    - All space cells are occupied
    - All pieces are used

    This will not work if a puzzle has a solution with holes in unknown positions.
    """

    def __init__(
        self,
        space: blocks.Space,
        pieces: Collection[blocks.Piece],
        policy: blocks.TransformationPolicy,
    ):
        self.space = space
        self.pieces = list(pieces)
        self.policy = policy

        self.cells = sorted(self.space.allowed_voxels)
        """Sorted list of all allowed voxels in the space."""

        self.num_pieces = len(self.pieces)
        self.column_names = self.generate_column_names()
        self.num_columns = len(self.column_names)

    def generate_column_names(self) -> list[str]:
        """Get column names for the piece usage and board cells."""
        column_names = []

        for piece in self.pieces:
            column_names.append(f"piece_{piece.name}")

        for cell in self.cells:
            column_names.append(f"cell_{cell}")

        return column_names

    def get_cell_index(self, cell: blocks.Voxel) -> int:
        """Returns the index of a cell in the list of space cells."""
        return self.cells.index(cell)

    def generate_dlx_matrix(self) -> list[list[int]]:
        """Generate the DLX matrix.

        Returns:
            Returns a DLX matrix. The number of columns is equal to the number
            of pieces + the number of allowed voxels in the space. The number of
            rows is the total number of all valid piece placements for all valid
            orientations.
        """
        matrix: list[list[int]] = []
        # Iterate over each piece,
        # for each valid orientation,
        # for each valid space placement.
        for i, piece in enumerate(self.pieces):
            unique_orientations = self.policy.allowed_unique_orientations(piece)
            for orientation in unique_orientations:
                valid_placements = self.space.valid_placements(orientation)
                for placed_piece in valid_placements:
                    # Create new empty row to fill.
                    row = [0] * self.num_columns
                    # Mark which piece is placed.
                    row[i] = 1
                    # Mark all cells occupied by the placed piece.
                    for voxel in placed_piece.voxels:
                        row[self.num_pieces + self.get_cell_index(voxel)] = 1
                    matrix.append(row)

        logger.info(
            "Generated DLX matrix with %d rows and %d columns.",
            len(matrix),
            self.num_columns,
        )
        return matrix

    def solve(self) -> list[list[blocks.Piece]]:
        """Solve an Exact Cover 2D problem using Knuth's Algorithm X with Dancing Links (DLX)."""
        start_time = time.perf_counter()

        dlx_matrix = self.generate_dlx_matrix()

        dlx_solver = DLXSolver(matrix=dlx_matrix, column_names=self.column_names)
        valid_solutions = []
        dlx_solver.search([], dlx_solver.root, valid_solutions)

        logger.info(
            "Found %d valid solutions in %f seconds.",
            len(valid_solutions),
            round(time.perf_counter() - start_time, 3),
        )
        return valid_solutions
