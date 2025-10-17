from blocks import blocks
from blocks import puzzles
from blocks import visualize

if __name__ == "__main__":
    for piece in [puzzles.basket_constraint]:
        visualize.plot_solution_pieces([piece])

    # solver = blocks.SolverAssembly(
    #     space=blocks.Space.cuboid(3, 3, 3),
    #     pieces=puzzles.vegetable_basket,
    # )
    # solutions = solver.solve()

    # print(f"Found {len(solutions)} solutions.")
    # for sol in solutions:
    #     print(sol)

    # visualize.plot_solutions(solutions)
