"""Blocks for puzzles."""

import blocks
import visualize

chicken_basket = [
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0), (0, 1, 1), (0, 1, 2)},
        name="L_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (0, 0, 1), (0, 1, 1)},
        name="U_shape",
    ),
    blocks.Piece(
        {
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (2, 2, 1),
            (2, 2, 2),
        },
        name="overhang_shape",
    ),
]

cake_basket = [
    blocks.Piece(
        {(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0), (2, 2, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)},
        name="stair_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0), (2, 2, 0), (0, 0, 1), (1, 1, 1), (2, 2, 1)},
        name="w_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0), (0, 0, 1), (2, 2, 1)},
        name="z_shape",
    ),
]


vegetable_basket = [
    blocks.Piece(
        {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 0, 2)},
        name="L_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (0, 1, 0), (0, 2, 0), (0, 2, 1)},
        name="V_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (2, 1, 0), (2, 2, 0)},
        name="J_shape",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2)}, name="S_shape"
    ),
]


fruit_basket = [
    blocks.Piece(
        {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 1, 0), (2, 2, 0), (1, 0, 1), (2, 2, 1)},
        name="eight_piece",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 2, 0), (0, 0, 1), (1, 2, 1)},
        name="S_piece",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (0, 0, 1)},
        name="Z_piece",
    ),
    blocks.Piece(
        {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0), (2, 0, 1)}, 
        name="L_piece",
    ),
]

for piece in vegetable_basket:
    visualize.plot_solution_pieces([piece])

solver = blocks.Solver(
    space=blocks.Space.cuboid(3, 3, 3),
    pieces=vegetable_basket,
)
solutions = solver.solve()

print(f"Found {len(solutions)} solutions.")
for sol in solutions:
    print(sol)

visualize.plot_solutions(solutions)
