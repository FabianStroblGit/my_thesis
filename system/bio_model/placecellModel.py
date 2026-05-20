from plotting.plotThesis import *
import os


class PlaceCell:
    """Class to keep track of an individual Place Cell"""
    def __init__(self, gc_connections):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = None  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        firing = np.average(modules_firing)  # compute overall pc firing by summing averaging over modules
        return firing

    def compute_firing_2x(self, s_vectors, axis, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        new_dim = int(np.sqrt(len(s_vectors[0])))  # n

        s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # mute weak connections, transform to binary vector

        proj_s_vectors = np.empty((len(s_vectors[:, 0]), new_dim))
        for i, s in enumerate(s_vectors):
            s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_s_vectors[i] = np.sum(s, axis=axis)  # sum over column/row

        proj_gc_connections = np.empty_like(proj_s_vectors)
        for i, gc_vector in enumerate(gc_connections):
            gc_vector = np.reshape(gc_vector, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_gc_connections[i] = np.sum(gc_vector, axis=axis)  # sum over column/row

        filtered = np.multiply(proj_gc_connections, proj_s_vectors)  # filter projected firing, by projected connections

        norm = np.sum(np.multiply(proj_s_vectors, proj_s_vectors), axis=1)  # compute unnormed firing at optimal case

        firing = 0
        modules_firing = 0
        for idx, filtered_vector in enumerate(filtered):
            # We have to distinguish between modules tuned for x direction and modules tuned for y direction
            if np.amin(filtered_vector) == 0:
                # If tuned for right direction there will be clearly distinguishable spikes
                firing = firing + np.sum(filtered_vector) / norm[idx]  # normalize firing and add to firing
                modules_firing = modules_firing + 1

        if modules_firing == 0:
            # No module contributed: return zero firing to avoid division by zero.
            return 0.0
        firing = firing/modules_firing  # divide by modules that we considered to get overall firing

        # Plotting options, used for linear lookahead debugging
        if plot:
            for idx, s_vector in enumerate(s_vectors):
                plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx)
            plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis)

        if firing > 0.97 and not self.plotted_found[axis]:
            for idx, s_vector in enumerate(s_vectors):
                plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx, found=True)
            plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis, found=True)
            self.plotted_found[axis] = True

        return firing


def compute_weights(s_vectors):
    weights = np.array(s_vectors)  # decided to not change anything here, but do it when computing firing
    return weights


class PlaceCellNetwork:
    """A PlaceCellNetwork holds information about all Place Cells"""
    def __init__(self, from_data=False):
        self.place_cells = []  # array of place cells

        if from_data:
            gc_connections = np.load("data/pc_model/gc_connections.npy")
            env_coordinates = np.load("data/pc_model/env_coordinates.npy")

            for idx, gc_connection in enumerate(gc_connections):
                pc = PlaceCell(gc_connection)
                pc.env_coordinates = env_coordinates[idx]
                self.place_cells.append(pc)

    def create_new_pc(self, gc_modules):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        for m, gc in enumerate(gc_modules):
            s_vectors[m] = gc.s
        weights = compute_weights(s_vectors)
        pc = PlaceCell(weights)
        self.place_cells.append(pc)

    def track_movement(self, gc_modules, reward_first_found):
        """Keeps track of current grid cell firing."""

        firing_values = self.compute_firing_values(gc_modules)

        created_new_pc = False
        if len(firing_values) == 0 or np.max(firing_values) < 0.86 or reward_first_found:
            # No place cell shows significant excitement based on GC state - create a new one.
            self.create_new_pc(gc_modules)
            firing_values.append(1)
            creation_str = "Created new place cell with idx: " + str(len(self.place_cells) - 1) \
                           + " | Highest alternative spiking value: " + str(np.max(firing_values[:-1]) if len(firing_values) > 1 else 0)
            # print(creation_str)
            created_new_pc = True

        return [firing_values, created_new_pc]

    def compute_firing_values(self, gc_modules, virtual=False, axis=None, plot=False):
        """Computes firing values for all place cells."""

        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        # Consolidate grid cell spiking vectors that we want to consider
        for m, gc in enumerate(gc_modules):
            if virtual:
                s_vectors[m] = gc.s_virtual
            else:
                s_vectors[m] = gc.s

        firing_values = []
        for i, pc in enumerate(self.place_cells):
            if axis is not None:
                plot = plot if i == 0 else False  # linear lookahead debugging plotting
                firing = pc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
            else:
                firing = pc.compute_firing(s_vectors)  # overall firing
            firing_values.append(firing)
        return firing_values

    def save_pc_network(self, filename=""):
        gc_connections = []
        env_coordinates = []
        for pc in self.place_cells:
            gc_connections.append(pc.gc_connections)
            env_coordinates.append(pc.env_coordinates)

        directory = "data/pc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/pc_model/gc_connections" + filename + ".npy", gc_connections)
        np.save("data/pc_model/env_coordinates" + filename + ".npy", env_coordinates)



