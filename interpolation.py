from __future__ import annotations

from typing import Sequence

import numpy as np

import firedrake
from cryogen import interpolate as cryogen_interpolate


def _interpolated_coordinate_values(function_space) -> np.ndarray:
    mesh = function_space.mesh()
    element = function_space.ufl_element()
    coord_space = firedrake.VectorFunctionSpace(mesh, element)
    coords = firedrake.assemble(firedrake.interpolate(mesh.coordinates, coord_space))
    coord_values = np.asarray(coords.dat.data_ro, dtype=float)
    if coord_values.ndim == 1:
        coord_values = coord_values[:, None]
    return coord_values


def function_space_coordinates(function_space) -> np.ndarray:
    """Return nodal coordinates as an array with shape ``(n_nodes, geom_dim)``."""
    value_shape = tuple(function_space.value_shape)
    if len(value_shape) == 0:
        scalar_space = function_space
    elif len(value_shape) == 1:
        scalar_space = function_space.sub(0)
    else:
        raise ValueError(
            "Only scalar and vector function spaces are supported. "
            f"Got value shape {value_shape}."
        )

    return _interpolated_coordinate_values(scalar_space)


def _coordinate_tuple(
    coords: np.ndarray | Sequence[np.ndarray],
    *,
    label: str,
) -> tuple[np.ndarray, ...]:
    if isinstance(coords, tuple):
        out = tuple(np.asarray(axis, dtype=float) for axis in coords)
    elif isinstance(coords, list):
        out = tuple(np.asarray(axis, dtype=float) for axis in coords)
    else:
        coord_array = np.asarray(coords, dtype=float)
        if coord_array.ndim == 1:
            out = (coord_array,)
        elif coord_array.ndim == 2:
            out = tuple(coord_array[:, axis] for axis in range(coord_array.shape[1]))
        else:
            raise ValueError(
                f"{label} must be a coordinate matrix or a sequence of coordinate arrays; "
                f"got shape {coord_array.shape}."
            )

    if not out:
        raise ValueError(f"{label} must contain at least one coordinate axis.")
    return out


def interpolate_grid_to_mesh_array(
    grid: np.ndarray,
    grid_coords: Sequence[np.ndarray],
    mesh_coords: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    """Interpolate structured-grid values onto mesh-ordered nodal data."""
    grid = np.asarray(grid)
    grid_coords = _coordinate_tuple(grid_coords, label="grid_coords")
    mesh_coords = _coordinate_tuple(mesh_coords, label="mesh_coords")

    spatial_ndim = len(grid_coords)
    if len(mesh_coords) != spatial_ndim:
        raise ValueError(
            "Grid and mesh coordinates must have the same spatial dimension; "
            f"got {spatial_ndim} and {len(mesh_coords)}."
        )

    if grid.ndim == spatial_ndim:
        return cryogen_interpolate.interp(grid, grid_coords, mesh_coords)

    if grid.ndim == spatial_ndim + 1:
        return np.column_stack(
            [
                cryogen_interpolate.interp(grid[i], grid_coords, mesh_coords)
                for i in range(grid.shape[0])
            ]
        )

    raise ValueError(
        f"Unsupported grid shape {grid.shape}. Expected a {spatial_ndim}-D scalar grid "
        f"or a {spatial_ndim + 1}-D component-first grid."
    )


def interpolate_grid_to_function(
    grid: np.ndarray,
    grid_coords: Sequence[np.ndarray],
    *,
    function=None,
    function_space=None,
    mesh_coords: np.ndarray | Sequence[np.ndarray] | None = None,
    name: str | None = None,
):
    """Interpolate structured-grid data into a Firedrake Function."""
    if function is None and function_space is None:
        raise ValueError("Provide either `function` or `function_space`.")

    if function is None:
        function = firedrake.Function(function_space, name=name)
    else:
        if function_space is not None and function.function_space() != function_space:
            raise ValueError("`function` and `function_space` refer to different spaces.")
        function_space = function.function_space()

    if mesh_coords is None:
        mesh_coords = function_space_coordinates(function_space)

    values = interpolate_grid_to_mesh_array(grid, grid_coords, mesh_coords)
    function.dat.data[:] = np.asarray(values).reshape(function.dat.data.shape)
    return function


# Backward-friendly aliases for callers that prefer shorter names.
grid_to_mesh_array = interpolate_grid_to_mesh_array
grid_to_function = interpolate_grid_to_function
