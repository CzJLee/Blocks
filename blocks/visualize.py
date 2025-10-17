# import numpy as np
# import matplotlib.pyplot as plt

# # Define your shapes
# J_coords = [
#     (0, 0, 0),
#     (0, 0, 1),
#     (0, 0, 2),
#     (0, 1, 2),
#     (0, 2, 1),
#     (0, 2, 2),
#     (1, 2, 2),
# ]

# L_coords = [
#     (1, 0, 0),
#     (1, 0, 1),
#     (1, 0, 2),
#     (1, 1, 0),
#     (2, 0, 0),
#     (2, 1, 0),
# ]

# # Combine coords so we can build a grid that fits both
# all_coords = J_coords + L_coords

# # Determine bounding box (min & max per axis)
# xs = [x for x, y, z in all_coords]
# ys = [y for x, y, z in all_coords]
# zs = [z for x, y, z in all_coords]

# min_x, max_x = min(xs), max(xs)
# min_y, max_y = min(ys), max(ys)
# min_z, max_z = min(zs), max(zs)

# # Size of grid
# size_x = max_x - min_x + 1
# size_y = max_y - min_y + 1
# size_z = max_z - min_z + 1

# # Create empty voxel grid
# grid = np.zeros((size_x, size_y, size_z), dtype=bool)

# # Create an array of colors; one per voxel position
# # We can use dtype=object so each entry can be a color string
# colors = np.empty(grid.shape, dtype=object)

# # Fill in the J_shape voxels, mark them with one color
# for (x, y, z) in J_coords:
#     # Shift by min_x etc in case bounding box isn't starting at 0
#     xi, yi, zi = x - min_x, y - min_y, z - min_z
#     grid[xi, yi, zi] = True
#     colors[xi, yi, zi] = 'cyan'  # choose color for J

# # Fill in the L_shape voxels with another color
# for (x, y, z) in L_coords:
#     xi, yi, zi = x - min_x, y - min_y, z - min_z
#     grid[xi, yi, zi] = True
#     colors[xi, yi, zi] = 'orange'  # choose color for L

# # Optionally, for positions not occupied, colors array entries don't matter

# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.voxels(grid, facecolors=colors, edgecolor='black')

# # Make sure cubes look cubic
# try:
#     ax.set_box_aspect((size_x, size_y, size_z))
# except AttributeError:
#     # fallback if set_box_aspect is not supported
#     padding = 0.1
#     ax.set_xlim(0 - padding, size_x + padding)
#     ax.set_ylim(0 - padding, size_y + padding)
#     ax.set_zlim(0 - padding, size_z + padding)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_solution_pieces(solution: List["Piece"], title: str = "Solution", show: bool = True):
    """
    Plot a single solution: a list of Piece objects, each drawn in a different color.
    Each Piece.voxels gives the set of (x, y, z) integer tuples.

    Args:
        solution: list of Piece objects.
        title: title string for the plot.
        show: if True, call plt.show() at end.
    """

    # Collect all voxel coords to compute bounding box
    all_coords = []
    for piece in solution:
        all_coords.extend(piece.voxels)

    if not all_coords:
        raise ValueError("Solution has no voxels to plot.")

    xs = [x for x, y, z in all_coords]
    ys = [y for x, y, z in all_coords]
    zs = [z for x, y, z in all_coords]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    size_x = max_x - min_x + 1
    size_y = max_y - min_y + 1
    size_z = max_z - min_z + 1

    # Create empty grid
    grid = np.zeros((size_x, size_y, size_z), dtype=bool)
    # Create color array; dtype=object or something that holds color strings or RGBA
    colors = np.empty((size_x, size_y, size_z), dtype=object)

    # Predefine a palette of colors; expand as needed
    # You can modify this or generate automatically
    color_palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # Plot each piece
    for idx, piece in enumerate(solution):
        color = color_palette[idx % len(color_palette)]
        for (x, y, z) in piece.voxels:
            xi = x - min_x
            yi = y - min_y
            zi = z - min_z
            grid[xi, yi, zi] = True
            colors[xi, yi, zi] = color

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Use facecolors = colors; edgecolor optional
    ax.voxels(grid, facecolors=colors, edgecolor="k")

    # Set equal aspect so cubes look like cubes
    try:
        # Matplotlib >= ~3.3 supports set_box_aspect
        ax.set_box_aspect((size_x, size_y, size_z))
    except AttributeError:
        # Fallback: manually equalize axes limits
        padding = 0.1
        ax.set_xlim(min_x - padding, max_x + 1 + padding)
        ax.set_ylim(min_y - padding, max_y + 1 + padding)
        ax.set_zlim(min_z - padding, max_z + 1 + padding)

    ax.set_title(title)

    if show:
        plt.show()


def plot_solutions(solutions: List[List["Piece"]], max_plots: int = 4):
    """
    Plot up to max_plots solutions, each in its own subplot.

    Args:
        solutions: list of solutions, where each solution is a list of Piece.
        max_plots: number of solutions to plot (side by side).
    """
    if not solutions:
        print("No solutions to plot.")
        return

    num = min(len(solutions), max_plots)
    # setup subplots
    fig = plt.figure(figsize=(5 * num, 5))
    axes = []
    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1, projection='3d')
        axes.append((ax, solutions[i]))

    for ax, solution in axes:
        # same logic as above
        all_coords = []
        for piece in solution:
            all_coords.extend(piece.voxels)
        xs = [x for x, y, z in all_coords]
        ys = [y for x, y, z in all_coords]
        zs = [z for x, y, z in all_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        size_x = max_x - min_x + 1
        size_y = max_y - min_y + 1
        size_z = max_z - min_z + 1

        grid = np.zeros((size_x, size_y, size_z), dtype=bool)
        colors = np.empty((size_x, size_y, size_z), dtype=object)

        color_palette = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        for idx, piece in enumerate(solution):
            color = color_palette[idx % len(color_palette)]
            for (x, y, z) in piece.voxels:
                xi = x - min_x
                yi = y - min_y
                zi = z - min_z
                grid[xi, yi, zi] = True
                colors[xi, yi, zi] = color

        ax.voxels(grid, facecolors=colors, edgecolor="k")

        try:
            ax.set_box_aspect((size_x, size_y, size_z))
        except AttributeError:
            padding = 0.1
            ax.set_xlim(min_x - padding, max_x + 1 + padding)
            ax.set_ylim(min_y - padding, max_y + 1 + padding)
            ax.set_zlim(min_z - padding, max_z + 1 + padding)

        ax.set_title(f"Solution {axes.index((ax, solution))+1}")

    plt.tight_layout()
    plt.show()