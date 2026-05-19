
from plotting.plotThesis import plot_grid_cell_modules, plot_3D_sheets
import numpy as np
import os


# Grid Cell model is based on Edvardsen 2015. Please refer to the thesis or the paper for detailed explanations


def rec_d(d):
    """Recurrent connectivity profile used for calculating connection weight between neurons"""
    lam = 15  # determines periodicity of pattern (lambda ~ #neurons between)
    beta = 3 / (lam ** 2)
    gamma = 1.05 * beta

    weight = np.exp(-gamma * (d ** 2)) - np.exp(-beta * (d ** 2))
    return weight


# Computes distance between
def compute_ds(x1, x2):
    """Calculates distance in along one axis between x1 and x2
    x1: vector of x coordinates (eg. [0, 0, 0, 1, 1, 1, 2, 2, 2]) of size n^2
    x2: vector of other x coordinates (eg. [1, 1, 1, 2, 1, 0, 3, 1, 2]) of size n^2
    returns dx from each neuron to each neuron, matrix of size n^2 x n^2
    """
    n = int(np.sqrt(len(x1)))  # size of grid cell sheet (width and height)
    x1 = np.tile(x1, (n**2, 1))  # tile vector to fit size n^2 x n^2
    x2 = np.transpose(np.tile(x2, (n**2, 1)))  # tile vector to fit size n^2 x n^2, but transpose
    dx1 = np.abs(x2 - x1)  # calculate distance from each neuron to each neuron
    dx2 = n - dx1  # as edges of grid cell sheet are connected dx < n -> calculate other way
    dx = np.min([dx1, dx2], axis=0)  # choose shortest path (left or right)
    return dx  # return matrix of size n^2 x n^2, representing shortest distance along 1 axis


def compute_gm(m, M, gmin, gmax=None):
    """Calculates velocity gain factor g_m for all modules according to formula in thesis"""

    if gmax is None:
        # If only 1 boundary is provided, we let the gain factor increase linearly (Edvardsen 2017)
        gm = gmin * 1.5 ** m
    else:
        # Otherwise we make sure the gain factors are properly spaced between the boundaries (Edvardsen 2015)
        if M != 1:
            R = np.power(gmax / gmin, 1 / (M - 1))
        else:
            R = 1
        gm = gmin * np.power(R, m)

    return gm


def implicit_euler(s0, w, b, tau, dt):
    """Solve the grid cell spiking equation with implicit euler for one time step of size dt"""
    f = np.maximum(0, np.tensordot(s0, w, axes=1) + b)
    s = (s0 + f * dt / tau) / (1 + dt / tau)
    return s


# Not used, but defined grid cell spiking equation to be solved with built in numeric solver
def ds_dt(t, s, w, b, tau):
    f = np.maximum(0, np.tensordot(s, w, axes=1) + b)
    return (f - s) / tau


class GridCellModule:
    """One GridCellModule holds the information of a sheet of n x n neurons"""
    def __init__(self, n, gm, dt, data=None):

        self.n = n  # Grid Cell sheet size (height and width)
        self.gm = gm  # velocity gain factor

        array_length = n**2

        # connection weight matrix from each to each neuron
        self.w = np.random.random_sample((array_length, array_length))

        # self.h = np.random.random_sample((array_length, 2))
        self.s = np.random.rand(array_length) * 10**-4  # firing vector of size (n^2 x 1); random firing at beginning
        self.t = self.s  # target grid cell firing (of goal or home-base)
        self.s_virtual = self.s  # used for linear lookahead to preplay trajectories, without actually moving
        self.dt = dt  # time step size

        self.s_video_array = []

        # If we are not loading grid cell data we have to calculate grid cell sheet weights
        if data is None:
            # Refer to thesis for concept of grid cell sheet and how weights are computed

            headings = [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]

            grid = np.indices((n, n))  # grid function to create x and y vectors
            x = np.concatenate(grid[1])  # x vector of form eg. [0, 0, 0, 1, 1, 1, 2, 2, 2]
            y = np.concatenate(grid[0])  # y vector of form eg. [0, 1, 2, 0, 1, 2, 0, 1, 2]
            index = 2 * np.mod(y, 2) + np.mod(x, 2)  # refer to thesis for explanation of formula
            index.astype(int)
            self.h = np.take(headings, index, axis=0)  # pick preferred heading direction for each neuron

            x_tuned = np.subtract(x, self.h[:, 0])  # tune x vector according to preferred heading direction
            y_tuned = np.subtract(y, self.h[:, 1])  # tune y vector according to preferred heading direction

            dx = compute_ds(x_tuned, x)  # compute shortest x distance between each pair of neurons (i - e_i, j)
            dy = compute_ds(y_tuned, y)  # compute shortest y distance between each pair of neurons (i - e_i, j)
            d = np.linalg.norm([dx, dy], axis=0)  # compute shortest overall distance between each pair of neurons
            self.w = rec_d(d)  # apply recurrent connectivity profile to get weights
        else:
            self.w = data["w"]
            self.h = data["h"]

    def update_s(self, v, virtual=False, dt_alternative=None):
        """Updates grid cell spiking from one to next time step"""

        tau = 1e-1  # defined by model
        alpha = 0.10315  # defined by model

        g = self.gm

        s0 = self.s_virtual if virtual else self.s  # virtual or actual mode
        dt = self.dt if dt_alternative is None else dt_alternative  # determine wanted time step size

        b = 1 + g * alpha * np.tensordot(self.h, v, axes=1)  # calculate b according to formula

        s = self.s
        if dt_alternative is None:
            # apply implicit euler once to update spiking
            s = implicit_euler(s0, self.w, b, tau, dt)
        else:
            # Alternative approach to use built in solver to calculate bigger time steps at once, large computation time
            # Because Implicit euler is unstable for large dt
            # sol = solve_ivp(ds_dt, (0, dt_alternative), s0, t_eval=[dt_alternative], args=(self.w, b, tau))
            # s = sol.y[:, 0]

            # It is faster to just apply the implicit euler several times until targeted time step is reached
            for n in range(int(dt_alternative/self.dt)):
                s0 = implicit_euler(s0, self.w, b, tau, self.dt)
            s = s0

        if virtual:
            self.s_virtual = s  # updates spiking value
            self.s_video_array.append(self.s_virtual)  # save for lookahead video
        else:
            s = implicit_euler(s0, self.w, b, tau, dt)  # this step might actually not be necessary, pls investigate
            self.s = s  # updates spiking value


class GridCellNetwork:
    """GridCellNetwork holds all Grid Cell Modules"""
    def __init__(self, n, M, dt, gmin, gmax=None, from_data=False):

        self.gc_modules = []  # array holding objects GridCellModule
        self.dt = dt

        if not from_data:
            # Create new GridCellModules
            for m in range(M):
                gm = compute_gm(m, M, gmin, gmax)
                gc = GridCellModule(n, gm, dt)
                self.gc_modules.append(gc)
                print("Created GC module with gm", gc.gm)
            self.save_gc_model()
            nr_steps_init = 1000
            self.initialize_network(nr_steps_init, "s_vectors_initialized.npy")
        else:
            # Load previous data
            w_vectors = np.load("data/gc_model/w_vectors.npy")
            h_vectors = np.load("data/gc_model/h_vectors.npy")
            gm_values = np.load("data/gc_model/gm_values.npy")

            n = int(np.sqrt(len(w_vectors[0][0])))
            for m, gm in enumerate(gm_values):
                gc = GridCellModule(n, gm, dt, {"w": w_vectors[m], "h": h_vectors[m]})
                self.gc_modules.append(gc)
                print("Loaded GC module with gm", gc.gm)

            self.load_initialized_network("s_vectors_initialized.npy")

        self.set_current_as_target_state()  # by default home-base is set as goal vector

    def track_movement(self, xy_speed, virtual=False, dt_alternative=None):
        """For each grid cell module update spiking"""
        for gc in self.gc_modules:
            gc.update_s(xy_speed, virtual=virtual, dt_alternative=dt_alternative)

    def initialize_network(self, nr_steps, filename):
        """For each grid cell module initialize spiking"""
        xy_speed = [0, 0]
        for i in range(nr_steps):
            if np.random.random() > 0.95:
                # Apply a small velocity vector in some cases to ensure that peaks form
                xy_speed = np.random.rand(2) * 0.2
            self.track_movement(xy_speed)
            if i % 50 == 0:
                print("Currently at Timestep:", i)
                # plot_grid_cell_modules(self.gc_modules, i)
                # plot_3D_sheets(self.gc_modules, i)
        print("Finished Initialization of nr_steps:", nr_steps)
        plot_grid_cell_modules(self.gc_modules, nr_steps)
        plot_3D_sheets(self.gc_modules, nr_steps)

        self.save_gc_spiking(filename)

    def load_initialized_network(self, filename):
        s_vectors = np.load("data/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.s = s_vectors[m]
        # plot_grid_cell_modules(self.gc_modules, "final")
        # plot_3D_sheets(self.gc_modules, "final")

    def save_gc_model(self):
        w_vectors = []
        h_vectors = []
        gm_values = []
        for gc in self.gc_modules:
            w_vectors.append(gc.w)
            h_vectors.append(gc.h)
            gm_values.append(gc.gm)

        directory = "data/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/gc_model/w_vectors.npy", w_vectors)
        np.save("data/gc_model/h_vectors.npy", h_vectors)
        np.save("data/gc_model/gm_values.npy", gm_values)

    def consolidate_gc_spiking(self, virtual=False):
        """Consolidate spiking in one matrix for saving"""
        s_vectors = np.zeros((len(self.gc_modules), len(self.gc_modules[0].s)))
        for idx, gc in enumerate(self.gc_modules):
            s = gc.s if not virtual else gc.s_virtual
            s_vectors[idx] = s
        return s_vectors

    def save_gc_spiking(self, filename):
        s_vectors = self.consolidate_gc_spiking()

        directory = "data/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/gc_model/"+filename, s_vectors)

    def set_current_as_target_state(self):
        for m, gc in enumerate(self.gc_modules):
            gc.t = np.copy(gc.s)

    def set_as_target_state(self, gc_connections):
        for m, gc in enumerate(self.gc_modules):
            gc.t = gc_connections[m]
        print("Set new target state")

    def reset_s_virtual(self):
        for m, gc in enumerate(self.gc_modules):
            gc.s_virtual = np.copy(gc.s)
            gc.s_video_array.clear()

    def set_filename_as_target_state(self, filename):
        t_vectors = np.load("data/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.t = t_vectors[m]
        print("Set loaded data as new target state:", filename)
