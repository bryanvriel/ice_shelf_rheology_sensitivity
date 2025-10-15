
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import scipy.ndimage as ndimage
from tqdm import tqdm
import h5py

import firedrake


def init_function_spaces(mesh, degree=2):
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=degree)
    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=degree)
    return Q, V

def const_to_R_fn(value, R):
    out = firedrake.Function(R)
    out.assign(value)
    return out 

def coordinate_values(Q):
    """Return the coordinates associated with a function space.
    """
    mesh = Q.mesh()
    element = Q.ufl_element()
    V = firedrake.VectorFunctionSpace(mesh, element)
    out = firedrake.assemble(firedrake.interpolate(mesh.coordinates, V))
    return out.dat.data_ro


class NumpyConverter:

    def __init__(self, shelf_file=None):

        # Load coordinate array
        with h5py.File('aux/coordinate_data.h5', 'r') as fid:
            self.X = fid['X'][()]
            self.Y = fid['Y'][()]
            Ny, Nx = self.X.shape
            self.xarr = self.X[0, :]
            self.yarr = self.Y[:, 0]

        # Load shelf coordinates
        if shelf_file is None:
            shelf_file = 'aux/coords_rutford_simplified_shelf.txt'
        sx, sy = np.loadtxt(shelf_file, unpack=True)
        path = Path(np.column_stack((sx, sy)))
        points = np.column_stack((self.X.ravel(), self.Y.ravel()))
        self.in_mask = path.contains_points(points, radius=-1.0)

        # Mask points
        self.X_pts = self.X.ravel()
        self.Y_pts = self.Y.ravel()
        self.grid_points = []
        for pt in points[self.in_mask]:
            self.grid_points.append(tuple(pt))

        # Store shape parameters
        self.shape = self.out_shape = (Ny, Nx)
        self.x_out = np.linspace(self.xarr[0], self.xarr[-1], Nx)
        self.y_out = np.linspace(self.yarr[0], self.yarr[-1], Ny)

    def __call__(self, function):
        """
        Evaluate function at grid points and convert to 2D.
        """
        # Evaluate function
        values = np.array(function(self.grid_points))

        # Unpack and oversample
        if values.ndim == 2:
            z1 = self.to2d(values[:, 0])
            z2 = self.to2d(values[:, 1])
            return z1, z2

        else:
            return self.to2d(values)

    def to2d(self, array):
        out = np.full(self.X.shape, np.nan, dtype=array.dtype)
        out.ravel()[self.in_mask] = array
        return out


# end of file
