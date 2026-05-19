"""
generate_maze_figures.py — Generate 3 maze visualizations for the thesis:
  1. Empty maze with plane_doors (all doors closed)
  2. Empty maze with plane_doors_individual (one door open)
  3. Maze after init run (16k exploration) with PCs and topology

Usage:
    python generate_maze_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plotting.plotHelper import add_environment, TUM_colors
from system.helper import compute_axis_limits


OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


# Match the typography used by the live cognitive map plot (see
# plotting/plotResults.py:648–656): serif (Charter), size 16. Keeps the
# thesis figures consistent with the navigation-run cognitive map images.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Charter", "Bitstream Charter", "Charter BT", "URW Bookman",
    "DejaVu Serif", "Times New Roman", "Times",
]
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


def plot_empty_maze(doors_option, filename):
    """Plot an empty maze (no PCs, no trajectory) with the given door configuration."""
    fig, ax = plt.subplots(figsize=(8, 7))

    add_environment(ax, "linear_sunburst", door_positions=doors_option)

    # Start and goal markers
    ax.scatter(5.5, 0.55, s=100, c='blue', marker='o', linewidths=2, zorder=10, label='Start')
    ax.scatter(1.5, 10, s=100, c='red', marker='x', linewidths=2, zorder=10, label='Goal')

    # Match the live-cognitive-map plot axes: tight x range hugging the
    # arena, tick step 2.5 on x and 2 on y, no axis labels (matches the
    # reference style used in the thesis figures).
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 13.5, 2.5))
    ax.set_yticks(np.arange(0, 11, 2))

    title = "plane_doors" if doors_option == "plane_doors" else "plane_doors_individual"
    ax.set_title(f"Maze Layout — {title}")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / filename}")
    plt.close(fig)


def plot_after_init(filename, doors_option="plane_doors"):
    """Plot the maze after init run with PCs and topology from saved data."""
    # Check that data files exist
    data_files = [
        "data/pc_model/env_coordinates.npy",
        "data/cognitive_map/topology_cells.npy",
        "data/cognitive_map/reward_cells.npy",
    ]
    for f in data_files:
        if not Path(f).exists():
            print(f"ERROR: {f} not found. Run the init exploration first (e.g. benchmark_ai.py, benchmark_a2c.py, or main.py with ~14k steps).")
            return

    # Load PC coordinates
    env_coords = np.load("data/pc_model/env_coordinates.npy", allow_pickle=True)
    topology = np.load("data/cognitive_map/topology_cells.npy")
    reward_cells = np.load("data/cognitive_map/reward_cells.npy")

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw environment
    add_environment(ax, "linear_sunburst", door_positions=doors_option)

    # Draw topology connections
    n_pcs = len(env_coords)
    for i in range(n_pcs):
        for j in range(i + 1, n_pcs):
            if j < topology.shape[1] and topology[i][j] == 1:
                x_vals = [env_coords[i][0], env_coords[j][0]]
                y_vals = [env_coords[i][1], env_coords[j][1]]
                ax.plot(x_vals, y_vals, color='k', alpha=0.2, linewidth=1)

    # Draw place cells
    for i in range(n_pcs):
        reward = reward_cells[i] if i < len(reward_cells) else 0
        circle = plt.Circle((env_coords[i][0], env_coords[i][1]), 0.3,
                             fc='r', alpha=reward**2 * 0.6, ec='k', linewidth=0.5)
        ax.add_artist(circle)
        circle_border = plt.Circle((env_coords[i][0], env_coords[i][1]), 0.3,
                                    alpha=0.2, ec='k', fill=False, linewidth=0.5)
        ax.add_artist(circle_border)

    # Start and goal markers
    ax.scatter(5.5, 0.55, s=100, c='blue', marker='o', linewidths=2, zorder=10, label='Start')
    ax.scatter(1.5, 10, s=100, c='red', marker='x', linewidths=2, zorder=10, label='Goal')

    # Match the live-cognitive-map plot axes: tight x range hugging the
    # arena, tick step 2.5 on x and 2 on y, no axis labels (matches the
    # reference style used in the thesis figures).
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 13.5, 2.5))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_title(f"Cognitive Map after Exploration ({n_pcs} Place Cells)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / filename}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating maze figures...\n")

    # 1. Empty maze — no doors
    plot_empty_maze("plane", "maze_no_doors.png")

    # 2. Empty maze — plane_doors
    plot_empty_maze("plane_doors", "maze_plane_doors.png")

    # 3. Empty maze — plane_doors_individual
    plot_empty_maze("plane_doors_individual", "maze_plane_doors_individual.png")

    # 4. After init run with plane_doors (PCs + topology)
    plot_after_init("maze_after_init.png", doors_option="plane_doors")

    # 5. After init run with plane_doors_individual (PCs + topology)
    plot_after_init("maze_after_init_individual.png", doors_option="plane_doors_individual")

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
