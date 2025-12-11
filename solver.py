from blocks import blocks
from blocks import puzzles
from blocks import visualize
from blocks import solver_exact_cover
from blocks import solver_exact_cover_claude

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # for piece in puzzles.cat_puzzle:
    #     visualize.plot_solution_pieces([piece])

    # pieces = puzzles.cat_puzzle
    # space = blocks.Space.cuboid(8, 8, 1)

    # solver = solver_exact_cover.SolverExactCover2D(
    #     space=space,
    #     pieces=pieces,
    #     policy=blocks.Policy2DRotationsNoFlip(),
    # )

    # solutions = solver.solve()
    # print(len(solutions))

    # solutions = solver.solve()

    # print(f"Found {len(solutions)} solutions.")
    # for sol in solutions:
    #     print(sol)

    # visualize.plot_solutions(solutions)

    # solver = blocks.Solver(
    #     space=blocks.Space.cuboid(8, 8, 1),
    #     pieces=puzzles.cat_puzzle,
    # )
    # solutions = solver.solve()

    # print(f"Found {len(solutions)} solutions.")
    # for sol in solutions:
    #     print(sol)

    # visualize.plot_solutions(solutions)
    
    solver_exact_cover_claude.main()
