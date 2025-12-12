"""
Implementation of an Exact Cover solver using Knuth's Algorithm X with Dancing
Links (DLX).

This solver implements a DLX solver that is accelerated using a NumPy
array-based data structure with Numba JIT compilation tp accelerate performance.
"""

import logging
import time
from typing import Collection, ClassVar

import numba
import numpy as np

from . import blocks

logger = logging.getLogger(__name__)

type DLXArray = np.typing.NDArray[np.int32]
"""Array used to store DLX data structure, where the node at `i` is linked to `A[i]`."""


# Numba njit optimized methods can not be class methods.
@numba.njit()
def _cover_column(
    c: int,
    L: np.ndarray,
    R: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
) -> None:
    """Cover a column.

    This is the heart of the "dancing links" technique. When we select a row
    to include in our solution, we must:
    1. Remove the column from consideration (it's now satisfied)
    2. Remove all other rows that could satisfy this column (they conflict)

    The beauty of dancing links: we only update neighbor pointers, not the
    removed nodes themselves. The removed nodes still "remember" where they
    were, enabling O(1) restoration during backtracking.

    Steps:
    1. Unlink column header.
    2. Iterate through each row in this column.
        i. Except for the nodes in this column, unlink all other nodes in
            each row.
        ii. Update each column's size counter.

    Args:
        c: Column header index to cover.
        L: Left neighbors in a circular horizontal list.
        R: Right neighbors in a circular horizontal list.
        U: Up neighbors in a circular vertical list.
        D: Down neighbors in a circular vertical list.
        C: Column header index for each node.
        S: Count of 1's remaining in each column.
    """
    # Remove column header from horizontal header list
    # Before: L[c] <-> c <-> R[c]
    # After:  L[c] <-------> R[c]
    L[R[c]] = L[c]
    R[L[c]] = R[c]

    # For each row in this column (traverse down from header).
    i = D[c]
    while i != c:  # Stop when we circle back to header.
        # Except for the node in this column, unlink all other nodes in this row.
        j = R[i]
        while j != i:  # Stop when we circle back to starting node.
            # Remove this node from its column's vertical list.
            U[D[j]] = U[j]
            D[U[j]] = D[j]
            # Update column size counter.
            S[C[j]] -= 1
            j = R[j]
        i = D[i]


@numba.njit()
def _uncover_column(
    c: int,
    L: np.ndarray,
    R: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
) -> None:
    """Uncover a column.

    This performs the exact reverse of `cover_column`.

    Because removed nodes still remember their old neighbors, they can be
    "uncovered" by simply reversing the pointer updates.

    Args:
        c: Column header index to cover.
        L: Left neighbors in a circular horizontal list.
        R: Right neighbors in a circular horizontal list.
        U: Up neighbors in a circular vertical list.
        D: Down neighbors in a circular vertical list.
        C: Column header index for each node.
        S: Count of 1's remaining in each column.
    """
    # For each row in column c (traverse UP - reverse order).
    i = U[c]
    while i != c:
        # For each node in this row (traverse LEFT - reverse order).
        j = L[i]
        while j != i:
            # Restore this node to its column's vertical list.
            D[U[j]] = j
            U[D[j]] = j
            # Update column size counter.
            S[C[j]] += 1
            j = L[j]
        i = U[i]

    # Restore column header to horizontal list.
    # Before: L[c] <-------> R[c]
    # After:  L[c] <-> c <-> R[c]
    R[L[c]] = c
    L[R[c]] = c


@numba.njit()
def _get_smallest_column(root: int, R: np.ndarray, S: np.ndarray) -> int:
    """Return the index of the column header with least amount of nodes.

    Args:
        root: The root node index.
        R: Right neighbors in a circular horizontal list.
        S: Count of 1's remaining in each column.

    Returns:
        Index of chosen column, or -1 if no columns remain.
    """
    min_size = np.int32(2147483647)  # Max int32 value
    chosen_col = -1

    # Scan all columns searching for the column with the least amount of nodes.
    c = R[root]
    while c != root:
        if S[c] < min_size:
            min_size = S[c]
            chosen_col = c
            # Early exit: can't do better than 0 or 1.
            if min_size <= 1:
                break
        c = R[c]

    return int(chosen_col)


@numba.njit()
def _search(
    L: np.ndarray,
    R: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    ROW: np.ndarray,
    S: np.ndarray,
    root: int,
    max_solutions: int,
    max_depth: int,
) -> np.ndarray:
    """
    Find solutions using iterative DLX search with explicit stack.

    This algorithm:
    1. Choose a column (a constraint to fulfill). e.g. An empty cell in the
        puzzle space.
    2. Choose a row in that column (an option that satisfies this
        constraint). e.g. Some possible piece and orientation that fills
        this empty cell.
    3. Cover all other columns satisfied by that row (enforcing its
        constraints). e.g. Remove all other piece possibilities that also
        fill this empty cell.
    4. Recurse.
    5. Backtrack (uncover everything you covered).

    Since this algorithm uses an iterative approach instead of recursion, we
    must track which step we are focused on, which we call the "phase".

    We simulate recursion using three "phases" at each stack level:

    - Phase 0 (ENTER): Step 1. Equivalent to the beginning of a new
        recursive function call.
    - Phase 1 (TRY_ROW): Step 2 and 3. Iterate through the rows of the
        selected column, trying each one as a potential solution (push to
        stack).
    - Phase 2 (BACKTRACK): Step 5. Either just found a solution or hit a
        dead end. Backtrack to the previous step.

    Args:
        L: Left neighbors in a circular horizontal list.
        R: Right neighbors in a circular horizontal list.
        U: Up neighbors in a circular vertical list.
        D: Down neighbors in a circular vertical list.
        C: Column header index for each node.
        ROW: Stores the original row index from the input matrix (or -1 for headers).
        S: Count of 1's remaining in each column.
        root: The root node index.
        max_solutions: Maximum number of solutions to find. If None, find
            all solutions.
        max_depth: This allocates the size of the stack that handles the
            search. In practice, the depth will only be as high as the
            number of pieces in the solution, so `max_depth` just needs to
            be set to be larger than the piece count.

    Returns:
        2D array of shape (num_solutions_found, max_depth) where each row is
        a solution. Unused entries in each row are filled with -1. Returns
        empty array with shape (0, max_depth) if no solution exists.
    """
    # Explicit stack arrays for simulating recursion.
    # Numba is more efficient if explicit arrays are used to track the
    # stack rather than mutable lists.
    # Column being processed at stack depth `d`. 
    stack_col = np.zeros(max_depth, dtype=np.int32)
    # Row being tried at stack depth `d`.
    stack_row = np.zeros(max_depth, dtype=np.int32)
    # Current phase at stack depth `d`. (0=ENTER, 1=TRY_ROW, 2=BACKTRACK).
    stack_phase = np.zeros(max_depth, dtype=np.int32)

    # Current partial solution and storage for found solutions.
    solution = np.zeros(max_depth, dtype=np.int32)
    solutions = np.ones((max_solutions, max_depth), dtype=np.int32) * -1

    # Keeps track of the current stack depth. This is equivalent to the
    # number of pieces placed in the current solution.
    stack_depth = 0
    num_solutions_found = 0

    # Main search loop - exit when we've backtracked past the root (depth < 0).
    while stack_depth >= 0 and num_solutions_found < max_solutions:
        # For this stack depth, what phase are we in?
        phase = stack_phase[stack_depth]

        if phase == 0:  # ENTER: Starting a new search level.
            # Check if solved (all columns covered).
            if R[root] == root:
                # Found a solution - save it.
                for i in range(stack_depth):
                    solutions[num_solutions_found, i] = solution[i]
                num_solutions_found += 1
                stack_depth -= 1  # Backtrack to find more solutions.
                continue

            # Choose column with smallest number of nodes.
            c = _get_smallest_column(root=root, R=R, S=S)

            # If selected column is empty, there are no options to fulfill
            # this condition.
            # Dead end branch, backtrack.
            if S[c] == 0:
                stack_depth -= 1
                continue

            # Cover the chosen column and prepare to try its rows.
            _cover_column(c=c, L=L, R=R, U=U, D=D, C=C, S=S)
            stack_col[stack_depth] = c
            stack_row[stack_depth] = D[c]  # First row in column.
            stack_phase[stack_depth] = 1

        elif phase == 1:  # TRY_ROW: Try the next row in this column.
            # Get the current state (column and row of the DLX matrix we
            # are focused on).
            c = stack_col[stack_depth]
            r = stack_row[stack_depth]

            # If we've tried all rows (circled back to header), backtrack.
            if r == c:
                _uncover_column(c=c, L=L, R=R, U=U, D=D, C=C, S=S)
                stack_depth -= 1
                continue

            # Try this row. Add this row to the partial solution.
            solution[stack_depth] = ROW[r]

            # Cover all other columns in this row.
            j = R[r]
            while j != r:
                _cover_column(c=int(C[j]), L=L, R=R, U=U, D=D, C=C, S=S)
                j = R[j]

            # Set up for backtracking, then recurse.
            # Set phase to 2, so when this stack depth is returned to, we
            # know to backtrack.
            stack_phase[stack_depth] = 2
            stack_row[stack_depth] = r
            # Continue recursion by entering deeper level.
            stack_depth += 1
            stack_phase[stack_depth] = 0

        else:  # phase == 2: BACKTRACK: Returned from deeper level.
            # Get the current state (column and row of the DLX matrix we
            # are focused on).
            c = stack_col[stack_depth]
            r = stack_row[stack_depth]

            # Uncover columns that were covered when we selected this row.
            j = L[r]
            while j != r:
                _uncover_column(c=int(C[j]), L=L, R=R, U=U, D=D, C=C, S=S)
                j = L[j]

            # Back track and try the next row in the selected column.
            stack_row[stack_depth] = D[r]
            stack_phase[stack_depth] = 1

    # Return all solutions found.
    return solutions[:num_solutions_found]


class DLXSolver:
    """Implementation of a Knuth's Algorithm X with Dancing Links (DLX) solver.

    All lists represented by DLXArrays are circular.
    For column headers: R[last_col] = root, L[root] = last_col.
    For vertical lists: D[last_node] = header, U[header] = last_node.
    """

    L: DLXArray
    """Left neighbors in a circular horizontal list."""
    R: DLXArray
    """Right neighbors in a circular horizontal list."""
    U: DLXArray
    """Up neighbors in a circular vertical list."""
    D: DLXArray
    """Down neighbors in a circular vertical list."""
    C: DLXArray
    """Column header index for each node."""
    ROW: DLXArray
    """Stores the original row index from the input matrix (or -1 for headers)."""
    S: DLXArray
    """Count of 1's remaining in each column. Used to find the minimal columns
    This array is updated during each cover / uncover operation."""
    ROOT: ClassVar[int] = 0
    """The root node is defined to be node 0."""

    def __init__(self, matrix: np.typing.ArrayLike, max_depth: int = 100):
        """Initialize the solver given a DLX binary constraint matrix.

        Args:
            matrix: A binary DLX matrix to build the DLX data structure from.
            max_depth: Maximum depth of the DLX search tree. In practice, the
                depth will only be as high as the number of pieces in the
                solution, so `max_depth` just needs to be set to be larger than
                the piece count.
        """
        self.matrix = np.array(matrix, dtype=np.uint8)
        self.max_depth = max_depth
        self._build_dlx_matrix(self.matrix)

    def _build_dlx_matrix(self, matrix: np.ndarray) -> None:
        """
        Convert a binary DLX matrix into array-based dancing links structure.

        The DLX linked structure is a "toroidal doubly-linked list", or a 4-way
        circular doubly-linked structure.

        Each DLX array, (L, R, U, D, C, ROW) contain pointers to other nodes
        (indices) in the structure. Node 0 is the root node. Nodes 1 to num_cols
        are column headers. Nodes num_cols+1 onwards are data nodes
        (representing 1s in the original matrix).

        Args:
            matrix: A binary DLX matrix to build the DLX data structure from.
        """
        num_rows, num_cols = matrix.shape
        logger.info(
            "Building DLX matrix with %d rows and %d columns...", num_rows, num_cols
        )

        # Since the DLX arrays will contain every node, we need to know how many
        # total nodes we need to represent in our data structure.
        # The total nodes is: root + column headers + all 1s in the matrix.
        num_nodes = 1 + num_cols + np.sum(matrix)

        # See class attributes for what each array represents.
        # pylint: disable=invalid-name
        self.L = np.zeros(num_nodes, dtype=np.int32)
        self.R = np.zeros(num_nodes, dtype=np.int32)
        self.U = np.zeros(num_nodes, dtype=np.int32)
        self.D = np.zeros(num_nodes, dtype=np.int32)
        self.C = np.zeros(num_nodes, dtype=np.int32)
        self.ROW = np.zeros(num_nodes, dtype=np.int32)
        self.S = np.zeros(num_cols + 1, dtype=np.int32)
        # pylint: enable=invalid-name

        # Create root node and column headers (nodes 0 to `num_cols`).
        for j in range(num_cols + 1):
            self.L[j] = (j - 1) % (num_cols + 1)
            self.R[j] = (j + 1) % (num_cols + 1)
            self.U[j] = j  # Points to itself (no data nodes yet)
            self.D[j] = j  # Points to itself
            self.C[j] = j  # Column header's column is itself
            self.ROW[j] = -1  # -1 indicates this is a header, not a data row

        # Fix circular links for root
        self.L[0] = num_cols
        self.R[num_cols] = 0

        # Prepare to add data nodes.
        # Keep track of the bottom node in each column to vertically link.
        # At this state, The last node in each column is the header itself.
        last_node_in_col = np.arange(num_cols + 1, dtype=np.int32)

        # Start the node index after the column headers.
        node_index = num_cols + 1

        # Iterate through each row and column of the DLX matrix, and create the
        # data nodes for true values (1's).
        for row_i in range(num_rows):
            row_start = -1  # First node in this row (for circular linking)
            row_prev = -1  # Previous node in this row

            for col_j in range(num_cols):
                if matrix[row_i, col_j] == 1:
                    # New node to be added.
                    node = node_index
                    node_index += 1

                    # Find column header that this node belongs to. (1-indexed).
                    col_header = col_j + 1
                    # Set node metadata
                    self.C[node] = col_header
                    self.ROW[node] = row_i
                    self.S[col_header] += 1

                    # Vertical linking: insert new node between bottom node and
                    # column header.
                    # fmt: off
                    self.U[node] = last_node_in_col[col_header]  # Place new node below the bottom node.
                    self.D[node] = col_header  # New node will always be right above column header.
                    self.D[last_node_in_col[col_header]] = node
                    self.U[col_header] = node
                    last_node_in_col[col_header] = node  # Update the bottom node in the column.
                    # fmt: on

                    # Horizontal linking.
                    if row_start == -1:
                        # First node in row - points to itself initially.
                        row_start = node
                        row_prev = node
                        self.L[node] = node
                        self.R[node] = node
                    else:
                        # Insert after row_prev, before row_start (circular).
                        self.L[node] = row_prev
                        self.R[node] = row_start
                        self.R[row_prev] = node
                        self.L[row_start] = node
                        row_prev = node  # Update the right most node in the row.

    def solve(self, max_solutions: int = 100) -> list[list[int]]:
        """Find solutions for the DLX matrix.

        Args:
            max_solutions: Maximum number of solutions to find.
        """

        start_time = time.perf_counter()

        solutions_array = _search(
            L=self.L,
            R=self.R,
            U=self.U,
            D=self.D,
            C=self.C,
            ROW=self.ROW,
            S=self.S,
            root=self.ROOT,
            max_solutions=max_solutions,
            max_depth=self.max_depth,
        )

        finish_time = time.perf_counter() - start_time

        # Format solutions array as a list of solutions, removing -1 padding.
        solutions = []
        for solution in solutions_array:
            solution = solution[solution != -1]
            solutions.append(solution.tolist())

        logger.info(
            "Found %d valid solutions in %.3f seconds.",
            len(solutions),
            finish_time,
        )

        return solutions


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
        """Initialize a solver for Exact Cover 2D puzzles.

        Args:
            space: The space to solve.
            pieces: The pieces to use.
            policy: The transformation policy to use.
        """
        self.space = space
        self.pieces = list(pieces)
        self.policy = policy

        self.cells = sorted(self.space.allowed_voxels)
        """Sorted list of all allowed voxels in the space."""

        self.num_pieces = len(self.pieces)
        self.num_columns = self.num_pieces + len(self.cells)

    def get_cell_index(self, cell: blocks.Voxel) -> int:
        """Returns the index of a cell in the list of space cells."""
        return self.cells.index(cell)

    def generate_dlx_matrix(self) -> tuple[list[list[int]], list[blocks.Piece]]:
        """Generate the DLX matrix.

        Returns:
            Tuple of (matrix, piece_at_row_index).
            The number of columns in the DLX matrix is equal to the number of
            pieces + the number of allowed voxels in the space. The number of
            rows is the total number of all valid piece placements for all valid
            orientations. The piece_at_row_index maps the index of the each row
            in the DLX matrix to the corresponding piece.
        """
        matrix: list[list[int]] = []
        # Track the piece at each row index in the DLX matrix to map solutions
        # to pieces.
        piece_at_row_index: list[blocks.Piece] = []
        # Iterate over each piece,
        # for each valid orientation,
        # for each valid space placement.
        for piece_index, piece in enumerate(self.pieces):
            unique_orientations = self.policy.allowed_unique_orientations(piece)
            for orientation in unique_orientations:
                valid_placements = self.space.valid_placements(orientation)
                for placed_piece in valid_placements:
                    # Create new empty row to fill.
                    row = [0] * self.num_columns
                    # Mark which piece is placed.
                    row[piece_index] = 1
                    # Mark all cells occupied by the placed piece.
                    for voxel in placed_piece.voxels:
                        row[self.num_pieces + self.get_cell_index(voxel)] = 1
                    piece_at_row_index.append(placed_piece)
                    matrix.append(row)

        logger.info(
            "Generated DLX matrix with %d rows and %d columns.",
            len(matrix),
            self.num_columns,
        )
        return matrix, piece_at_row_index

    def solve(self, max_solutions: int = 1000) -> list[list[blocks.Piece]]:
        """Solve an Exact Cover 2D problem using Knuth's Algorithm X with Dancing Links (DLX).

        Returns:
            A list of found solutions, where each solution is a list of pieces.

        Args:
            max_solutions: Maximum number of solutions to find. If None, find
                all solutions.
        """
        dlx_matrix, piece_at_row_index = self.generate_dlx_matrix()

        dlx_solver = DLXSolver(matrix=dlx_matrix, max_depth=self.num_pieces + 1)
        solutions = dlx_solver.solve(max_solutions=max_solutions)

        piece_solutions = [
            [piece_at_row_index[piece_index] for piece_index in solution]
            for solution in solutions
        ]

        return piece_solutions
