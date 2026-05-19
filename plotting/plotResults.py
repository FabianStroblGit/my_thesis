import matplotlib.colors as mcolors

import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import rc
mpl.rcParams['animation.ffmpeg_path'] = "ffmpeg/ffmpeg"

from system.helper import compute_theta, compute_axis_limits
from plotting.plotHelper import *
from plotting.plotThesis import add_cognitive_map


colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Avenir'}

cmap = plt.cm.get_cmap("tab20")  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.9, .9, .9, 0.8)
# create the new map
cmap20 = mcolors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

TUM_colors = {
                'TUMBlue': '#0065BD',
                'TUMSecondaryBlue': '#005293',
                'TUMSecondaryBlue2': '#003359',
                'TUMBlack': '#000000',
                'TUMWhite': '#FFFFFF',
                'TUMDarkGray': '#333333',
                'TUMGray': '#808080',
                'TUMLightGray': '#CCCCC6',
                'TUMAccentGray': '#DAD7CB',
                'TUMAccentOrange': '#E37222',
                'TUMAccentGreen': '#A2AD00',
                'TUMAccentLightBlue': '#98C6EA',
                'TUMAccentBlue': '#64A0C8'
}

cmap_binary = mcolors.ListedColormap([TUM_colors['TUMWhite'], TUM_colors['TUMGray']])

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(256/256, 0/256, N)
vals[:, 1] = np.linspace(256/256, 101/256, N)
vals[:, 2] = np.linspace(256/256, 189/256, N)
tum_blue_map = mcolors.ListedColormap(vals)

vals2 = np.ones((N, 4))
vals2[:, 0] = np.linspace(256/256, 128/256, N)
vals2[:, 1] = np.linspace(256/256, 128/256, N)
vals2[:, 2] = np.linspace(256/256, 128/256, N)
tum_grey_map = mcolors.ListedColormap(vals2)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plotTrajectory(xy_coordinates, orientation_angle):
    x, y = zip(*xy_coordinates)
    plt.figure(1)
    plt.scatter(x, y, s=0.2)

    nr_labels = 10
    step_label = int(len(xy_coordinates)/nr_labels)
    for i in range(len(x)):
        if i % step_label == 0:
            xi = x[i]
            yi = y[i]
            label = str(int(i / step_label))

            plt.annotate(label,  # this is the text
                         (xi, yi),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 0.1),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.axis('equal')
    plt.legend(['Trajectory'])
    plt.show()

def plotCurrentAndTarget(gc_modules, virtual=False):

    fig = plt.figure()

    for m, gc in enumerate(gc_modules):
        if virtual:
            s = np.reshape(gc.s_virtual, (gc.n, gc.n))
        else:
            s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))
        fig.add_subplot(1, len(gc_modules), m + 1)
        plt.imshow(s, origin="lower")
        plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

    plt.show()


def plotCurrentAndTargetMatched(gc_modules, matches_array, vectors_array):

    fig = plt.figure()

    for m, gc in enumerate(gc_modules):
        s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))

        fig.add_subplot(1, len(gc_modules), m + 1)
        plt.imshow(s, origin="lower")
        plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

        matches = matches_array[m]
        vectors = vectors_array[m]

        if len(matches) != 0 and len(vectors) != 0:
            s_max = list(matches.keys())
            s_max_x, s_max_y = zip(*s_max)
            t_max = list(matches.values())
            t_max_x, t_max_y = zip(*t_max)

            origin_x, origin_y = zip(*list(vectors.keys()))
            vectors_x, vectors_y = zip(*list(vectors.values()))

            plt.scatter(s_max_x, s_max_y, color="blue", s=1)
            plt.scatter(t_max_x, t_max_y, color="red", s=1)

            plt.quiver(origin_x, origin_y, vectors_x, vectors_y, color='w', width=0.01, scale=1, scale_units='xy')

    plt.show()


def plot_angles(real_trajectory, target, vec_array1=None, vec_array2=None, vec_array3=None):
    fig = plt.figure()
    legend = []

    start = 400
    if len(real_trajectory) < start:
        start = 0
    stop = len(real_trajectory)
    num = int((stop-start))
    x = np.linspace(start, stop, num=num)

    if vec_array1 is not None and len(vec_array1) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array1):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Path Integration')

    if vec_array2 is not None and len(vec_array2) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array2):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Spike detection')

    if vec_array3 is not None and len(vec_array3) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array3):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Phase Offset Detector')

    angle_array_real = []
    for i, xy in enumerate(real_trajectory):
        if i >= start:
            vec = np.array(target) - np.array(xy)
            angle = compute_theta(vec)
            angle_array_real.append(angle)
    plt.plot(x, angle_array_real)
    legend.append('Real Angle')

    plt.legend(legend)
    plt.show()


def plot_current_state(env, gc_modules, f_gc, f_t, f_mon,
                       matches_array=None, vectors_array=None, pc_active_array=None,
                       pc_network=None, cognitive_map=None, exploration_phase=False, goal_vector=None):

    xy_coordinates = env.xy_coordinates

    # Trajectory plot
    f_t.clear()
    limits_t = compute_axis_limits(env.arena_size, environment=env.env_model)

    if env.env_model == "linear_sunburst":
        f_t.axis('square')
        f_t.set_xlim(-0.5, 11.5)
        f_t.set_ylim(-0.5, 11.5)
    else:
        f_t.set_xlim(limits_t[0], limits_t[1])
        f_t.set_ylim(limits_t[2], limits_t[3])

    if pc_network is not None and cognitive_map is not None:
        ax = f_t
        add_cognitive_map(ax, pc_network, cognitive_map)

    x, y = zip(*xy_coordinates)
    f_t.scatter(x[0], y[0], color=TUM_colors['TUMGray'], s=1)

    # Plot obstacles
    add_environment(f_t, env.env_model, getattr(env, "door_positions", None))
    add_robot(f_t, env)


    # Grid Cell Modules plot
    for m, gc in enumerate(gc_modules):
        if m < 4:
            f_gc[m].clear()
            s = np.reshape(gc.s, (gc.n, gc.n))
            t = np.reshape(gc.t, (gc.n, gc.n))
            f_gc[m].imshow(s, origin="lower", cmap=tum_blue_map)
            f_gc[m].imshow(t, alpha=0.5, cmap=tum_grey_map, origin="lower")

            if matches_array is not None and len(matches_array[m]) != 0:
                matches = matches_array[m]

                s_max = list(matches.keys())
                s_max_x, s_max_y = zip(*s_max)
                t_max = list(matches.values())
                t_max_x, t_max_y = zip(*t_max)

                f_gc[m].scatter(s_max_x, s_max_y, color=TUM_colors['TUMBlue'], s=1)
                f_gc[m].scatter(t_max_x, t_max_y, color=TUM_colors['TUMGray'], s=1)

            if vectors_array is not None and len(vectors_array[m]) != 0:
                vectors = vectors_array[m]
                origin_x, origin_y = zip(*list(vectors.keys()))
                vectors_x, vectors_y = zip(*list(vectors.values()))

                f_gc[m].quiver(origin_x, origin_y, vectors_x, vectors_y, color=TUM_colors['TUMDarkGray'], width=0.01, scale=1, scale_units='xy')

    # Description Plot
    f_mon.clear()
    f_mon.axis("off")
    if exploration_phase:
        description_string = r"Currently in exploration phase"
    else:
        description_string = r"Currently in navigation phase"
    f_mon.annotate(description_string, xy=(0, 0.8), fontweight='bold')

    if goal_vector is not None and not exploration_phase:
        goal_vector_string = r"Computed vector: [" + "{:.2f}".format(goal_vector[0]) + ", " + "{:.2f}".format(goal_vector[1]) + "]"
        f_mon.annotate(goal_vector_string, xy=(0, 0.6))

        actual_vector = env.goal_location - xy_coordinates[-1]
        goal_vector_string = r"Actual vector:        [" + "{:.2f}".format(actual_vector[0]) + ", " + "{:.2f}".format(
            actual_vector[1]) + "]"
        f_mon.annotate(goal_vector_string, xy=(0, 0.5))

        error_vector = actual_vector - goal_vector
        error_string = r"Error: " + "{:.2f}".format(np.linalg.norm(error_vector))
        f_mon.annotate(error_string, xy=(0, 0.4))




def layout_video():
    fig = plt.figure(constrained_layout=False)
    fig.suptitle(r'Biologically inspired navigation', fontsize=12, x=0.08, y=0.91, ha='left', fontweight='semibold')
    logo = plt.imread('plotting/tum_logo.png')
    fig.figimage(logo, 530, 395, zorder=1)

    gs0 = fig.add_gridspec(2, 1)

    gs01 = gs0[0].subgridspec(nrows=1, ncols=4, wspace=0.3)

    f_gc1 = fig.add_subplot(gs01[0:1, 0:1])
    f_gc2 = fig.add_subplot(gs01[0:1, 1:2])
    f_gc3 = fig.add_subplot(gs01[0:1, 2:3])
    f_gc4 = fig.add_subplot(gs01[0:1, 3:4])
    f_gc = [f_gc1, f_gc2, f_gc3, f_gc4]

    gs02 = gs0[1].subgridspec(nrows=2, ncols=4)

    f_t = fig.add_subplot(gs02[0:2, 0:2])
    f_mon = fig.add_subplot(gs02[0:2, 2:4])
    f_mon.axis('off')

    return [fig, f_gc, f_t, f_mon]


def error_plot(error_array):
    plt.hist(error_array, 50, density=True)
    plt.show()


class LiveCognitiveMapPlot:
    """Live-updating cognitive map visualization that doesn't block."""
    
    def __init__(self, environment=None, door_positions=None, update_interval=50):
        """Initialize the live plot window.
        
        Args:
            environment: str, environment name for plotting obstacles
            door_positions: list, door positions for plotting
            update_interval: int, only redraw every N calls to update() for performance
        """
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.environment = environment
        self.door_positions = door_positions
        self.update_interval = update_interval
        self.update_counter = 0

        # Store plot elements for efficient updates
        self.trajectory_lines = []
        self.trajectory_starts = []
        self.trajectory_ends = []
        self.pc_circles = []
        self.connection_lines = []
        
        # Add environment once
        add_environment(self.ax, environment, door_positions)
        
        # Set axis limits
        limits = compute_axis_limits(11, environment=environment)
        self.ax.set_xlim(limits[0], limits[1])
        self.ax.set_ylim(limits[2], limits[3])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update(self, pc_network, cognitive_map, xy_coordinates=None, step=None):
        """Update the live plot with current state.

        Args:
            pc_network: PlaceCellNetwork
            cognitive_map: CognitiveMapNetwork
            xy_coordinates: agent trajectory
            step: current simulation step (optional, for title)
        """
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return  # Skip this update for performance

        self.ax.clear()

        # Re-add environment
        add_environment(self.ax, self.environment, self.door_positions)

        # Set axis limits
        limits = compute_axis_limits(11, environment=self.environment)
        self.ax.set_xlim(limits[0], limits[1])
        self.ax.set_ylim(limits[2], limits[3])

        # Cap the rendered trajectory at the last `tail` points so that
        # very long runs (>50k steps) don't make every plot update
        # O(N) in trajectory length — that was producing seconds-per-update
        # rendering after ~100k steps and grinding the live demo to a halt.
        # The start marker (X) is plotted from the absolute first point so
        # the user can still see where the agent began.
        tail = getattr(self, "trajectory_tail", 5000)

        if xy_coordinates is not None and len(xy_coordinates) > 0:
            tail_slice = xy_coordinates[-tail:] if len(xy_coordinates) > tail else xy_coordinates
            x, y = zip(*tail_slice)
            # Plot trajectory tail line + current position marker
            self.ax.plot(x, y, c='red', alpha=0.6, linewidth=1.5, label='Agent')
            x0, y0 = xy_coordinates[0]
            self.ax.scatter(x0, y0, s=80, c='red', marker='x', linewidths=2)
            self.ax.scatter(x[-1], y[-1], s=150, c='red', marker='o',
                           edgecolors='black', linewidths=2, zorder=10)
            self.ax.legend(loc='upper right', fontsize=8)
        
        # Plot place cells
        for i, pc in enumerate(pc_network.place_cells):
            circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                                fc='r', alpha=cognitive_map.reward_cells[i]**2 * 0.6, ec='k')
            self.ax.add_artist(circle)
            circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                                       alpha=0.2, ec='k', fill=False)
            self.ax.add_artist(circle_border)
            
            # Plot connections
            for j, connection in enumerate(cognitive_map.topology_cells[i]):
                if connection == 1 and i != j:
                    x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                    y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                    self.ax.plot(x_values, y_values, color='k', alpha=0.2)
        
        # Title with step count
        if step is not None:
            self.ax.set_title(f'Cognitive Map - Step {step}', fontsize=12)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the live plot window."""
        plt.ioff()
        plt.close(self.fig)


def cognitive_map_plot(pc_network, cognitive_map, xy_coordinates=None, pc_active_array=None, environment=None,
                       door_positions=None, save_path=None):

    # Axis tick labels: use a serif family at a size that, when the resulting
    # PNG is embedded into the LaTeX document at the usual subfigure width
    # (~0.44\linewidth), lands near body-text size. The thesis uses the AIR
    # Charter font; matplotlib falls back to whatever serif font is locally
    # available if Charter is not installed for the Python environment.
    _saved_rc = {
        "font.family": plt.rcParams["font.family"],
        "font.serif": list(plt.rcParams["font.serif"]),
        "font.size": plt.rcParams["font.size"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
    }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Charter", "Bitstream Charter", "Charter BT", "URW Bookman",
        "DejaVu Serif", "Times New Roman", "Times",
    ]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    plt.figure()

    if xy_coordinates is not None:
        x, y = zip(*xy_coordinates)
        if pc_active_array is not None:
            idx_pc_active = np.array(pc_active_array)[:, 0] + 5
            # Threshold clearly differentiates place cells from each each other
            spiking_value = np.where(np.array(pc_active_array)[:, 1] > 0.75, 1, 0)
            idx_pc_active = np.multiply(idx_pc_active, spiking_value)
            plt.scatter(x, y, s=3, c=idx_pc_active, cmap=cmap20)
        else:
            plt.scatter(x, y, s=3, c=cmaplist[0])

    ax = plt.gca()
    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                            fc='b', alpha=cognitive_map.reward_cells[i]**2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    # Plot obstacles
    add_environment(ax, environment, door_positions)

    limits_t = compute_axis_limits(11, environment=environment)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Restore the matplotlib rcParams we changed so this function doesn't
    # leak its font configuration into any subsequent plot the caller draws.
    for _k, _v in _saved_rc.items():
        plt.rcParams[_k] = _v


def plot_linear_lookahead(f_gc, f_t, f_mon, frame, gc_network, xy_coordinates=None, reward_array=None, goal_found=None):
    for m, gc in enumerate(gc_network.gc_modules):
        if m > 1:
            m = m - 2
            f_gc[m].clear()
            s = np.reshape(gc.s_video_array[frame], (gc.n, gc.n))
            t = np.reshape(gc.t, (gc.n, gc.n))
            f_gc[m].imshow(s, origin="lower")
            f_gc[m].imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

    f_mon.clear()
    f_mon.axis("off")
    if reward_array is not None:
        reward_string = "Reward is: " + str(reward_array[frame])
        f_mon.annotate(reward_string, xy=(0, 0.6))
    if goal_found is not None:
        goal_string = "Found at:  " + str(goal_found) + " | Currently at: " + str(frame)
        f_mon.annotate(goal_string, xy=(0, 0.8))

    if xy_coordinates is not None:
        # Trajectory plot
        environment = "linear_sunburst"
        f_t.clear()
        limits_t = compute_axis_limits(11, environment=environment)
        f_t.set_xlim(limits_t[0], limits_t[1])
        f_t.set_ylim(limits_t[2], limits_t[3])
        if len(xy_coordinates) > 1:
            x, y = zip(*xy_coordinates)
            f_t.scatter(x, y, color="grey", s=0.3)
            f_t.scatter(x[0], y[0], color="red", s=1)

            size = 0.05
            heading = np.array([x[-1] - x[-10], y[-1] - y[-10]])
            heading = 10 ** -5 * heading / np.linalg.norm(heading)
            f_t.quiver(x[-1], y[-1], heading[0], heading[1], scale_units="dots", width=size, color="k",
                       headwidth=8, headlength=10, headaxislength=10)

        ax = plt.gca()
        add_environment(ax, environment)


def export_linear_lookahead_video(gc_network, filename, xy_coordinates=None, reward_array=None, goal_found=None):

    [fig, f_gc, f_t, f_mon] = layout_video()
    fps = 5
    length = len(gc_network.gc_modules[0].s_video_array)
    step = int((1 / fps) / gc_network.dt)
    frames = np.arange(0, length, step)

    def animation_frame(frame):
        plot_linear_lookahead(f_gc, f_t, f_mon, frame, gc_network, xy_coordinates=xy_coordinates,
                              reward_array=reward_array, goal_found=goal_found)

    anim = animation.FuncAnimation(fig, func=animation_frame, frames=frames, interval=1 / fps, blit=False)

    # Finished simulation
    f = filename
    video_writer = animation.FFMpegWriter(fps=fps)
    anim.save(f, writer=video_writer)
    plt.close()


def plot_sub_goal_localization(env, cognitive_map, pc_network, goal_spiking,
                               goal_vector, chosen_idx):

    xy_coordinates = env.xy_coordinates
    fig = plt.figure()

    # Trajectory plot
    maze = True if env.env_model == "linear_sunburst" else False
    limits_t = compute_axis_limits(11, environment=env.env_model)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])

    # Plot obstacles
    ax = plt.gca()
    add_environment(ax, env.env_model, getattr(env, "door_positions", None))

    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                            fc='r', alpha=cognitive_map.reward_cells[i] ** 2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    x, y = zip(*xy_coordinates)
    plt.scatter(x[0], y[0], color="red", s=1)


    # Plot robot
    add_robot(ax, env)

    plt.quiver(x[-1], y[-1], goal_vector[0], goal_vector[1], color='grey', angles='xy', scale_units='xy', scale=1)

    for idx, angle in enumerate(goal_spiking):

        color = "b"

        if idx == chosen_idx:
            color = "r"
        if goal_spiking[angle]["reward"] == -1:
            color = "grey"
        if goal_spiking[angle]["blocked"]:
            color = "gainsboro"

        vector = np.array([np.cos(angle), np.sin(angle)])
        plt.quiver(x[-1], y[-1], vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    plt.show()


def plot_sub_goal_localization_pod(env, cognitive_map, pc_network, sub_goal_dict,
                                   goal_vector, chosen_idx):

    xy_coordinates = env.xy_coordinates
    fig = plt.figure()

    # Trajectory plot
    maze = True if env.env_model == "linear_sunburst" else False
    limits_t = compute_axis_limits(11, environment=env.env_model)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])

    # Plot obstacles
    ax = plt.gca()
    add_environment(ax, env.env_model, getattr(env, "door_positions", None))

    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                            fc='r', alpha=cognitive_map.reward_cells[i] ** 2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    x, y = zip(*xy_coordinates)
    plt.scatter(x[0], y[0], color="red", s=1)


    # Plot robot
    add_robot(ax, env)

    plt.quiver(x[-1], y[-1], goal_vector[0], goal_vector[1], angles='xy', scale_units='xy', scale=1,)

    for idx, pc_idx in enumerate(sub_goal_dict):

        color = "b"

        if idx == chosen_idx:
            color = "r"
        if sub_goal_dict[idx]["blocked"]:
            color = "gainsboro"

        vector = sub_goal_dict[idx]["goal_vector"]
        plt.quiver(x[-1], y[-1], vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    plt.show()

