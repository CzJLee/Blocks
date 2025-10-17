from blocks import blocks
from blocks import puzzles
from blocks import visualize

if __name__ == "__main__":
    basket_base = blocks.Piece(blocks.cuboid(5, 5, 4))
    basket_diff = blocks.Piece(blocks.cuboid(3, 3, 3)).translate(1, 1, 1)
    basket_handle = blocks.Piece({(2, 0, 4), (2, 0, 5), (2, 1, 5), (2, 2, 5), (2, 3, 5), (2, 4, 5), (2, 4, 4)})
    basket = blocks.Piece(basket_base.voxels.difference(basket_diff.voxels).union(basket_handle.voxels))
    print(sorted(basket.voxels))
    for piece in [basket]:
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
