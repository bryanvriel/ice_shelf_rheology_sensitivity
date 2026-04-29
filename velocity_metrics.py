from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

import firedrake

from . import config_utilities as cu
from . import file_utilities as fu
from .interpolation import function_space_coordinates, interpolate_grid_to_mesh_array
from .problems import make_mesh


_PREDICTION_KEY_PATTERN = re.compile(r"^u(\d+)$")


def _load_cfg_like(cfg: Any):
    if cfg is None:
        return None, None
    if isinstance(cfg, (str, Path)):
        cfg_path = Path(cfg).expanduser().resolve()
        return cu.load_config(cfg_path), cfg_path.parent
    return cu.resolve_config_aliases(cfg), None


def _resolve_meshfile(meshfile: str | Path | None, cfg, cfg_dir: Path | None) -> str | None:
    if meshfile is None and cfg is not None:
        meshfile = cfg.inverse.meshfile
    if meshfile is None:
        return None

    mesh_path = Path(meshfile).expanduser()
    if not mesh_path.is_absolute() and cfg_dir is not None:
        mesh_path = (cfg_dir / mesh_path).resolve()
    else:
        mesh_path = mesh_path.resolve()
    return str(mesh_path)


def _resolve_stage_group(
    result_file: str | Path,
    *,
    stage: int | str | None = None,
    group: str | None = None,
) -> str:
    if stage is not None and group is not None:
        raise ValueError("Provide only one of `stage` or `group`.")
    if group is not None:
        return str(group)
    if stage is None:
        return fu.get_latest_stage(str(result_file))
    if isinstance(stage, str):
        return stage if stage.startswith("stage") else f"stage{stage}"
    return f"stage{int(stage)}"


def _normalize_dirichlet_ids(
    dirichlet_ids: Sequence[int] | str | int | None,
) -> Sequence[int] | str | None:
    if dirichlet_ids is None or isinstance(dirichlet_ids, str):
        return dirichlet_ids
    if np.isscalar(dirichlet_ids):
        return [int(dirichlet_ids)]
    return [int(idx) for idx in dirichlet_ids]


def _load_prediction_grids(
    result_file: str | Path,
    group_name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[int]]:
    with h5py.File(result_file, "r") as fid:
        try:
            stage_group = fid[group_name]
        except KeyError as exc:
            raise ValueError(
                f"Group `{group_name}` not found in results file `{result_file}`."
            ) from exc

        try:
            mean_group = stage_group["mean"]
        except KeyError as exc:
            raise ValueError(
                f"Group `{group_name}` does not contain a `mean` subgroup."
            ) from exc

        epoch_keys = []
        for key in mean_group.keys():
            match = _PREDICTION_KEY_PATTERN.fullmatch(key)
            if match is not None:
                epoch_keys.append((int(match.group(1)), key))

        if not epoch_keys:
            raise ValueError(
                f"No predicted velocity datasets `u0`, `u1`, ... found in "
                f"`{group_name}/mean`."
            )

        epoch_keys.sort()
        epoch_numbers = [idx for idx, _ in epoch_keys]
        grids = np.stack([mean_group[key][()] for _, key in epoch_keys], axis=0)
        x = stage_group["x"][()]
        y = stage_group["y"][()]

    return grids, (x, y), epoch_numbers


def _load_observation_grids(
    data_file: str | Path,
    *,
    epoch_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[int]]:
    with h5py.File(data_file, "r") as fid:
        try:
            grids = fid["u"][()]
        except KeyError as exc:
            raise ValueError(f"Dataset `u` not found in observation file `{data_file}`.") from exc
        x = fid["x"][()]
        y = fid["y"][()]

    if grids.ndim != 4:
        raise ValueError(
            f"Observation dataset `u` must have shape (Nt, n_comp, Ny, Nx); got {grids.shape}."
        )

    if epoch_indices is None:
        indices = list(range(grids.shape[0]))
    else:
        indices = [int(idx) for idx in epoch_indices]
        grids = grids[indices]

    return grids, (x, y), indices


def _zero_dirichlet_value(V):
    value_shape = getattr(V, "value_shape", ())
    if len(value_shape) == 0:
        return 0.0
    if len(value_shape) == 1:
        return firedrake.Constant((0.0,) * V.value_size)
    raise ValueError(f"Unsupported function-space value shape {value_shape}.")


def _build_node_mask(
    *,
    mesh=None,
    meshfile: str | None = None,
    dirichlet_ids: Sequence[int] | str | None = None,
    mesh_degree: int = 2,
    exclude_boundary: bool = True,
):
    if mesh is None:
        if meshfile is None:
            raise ValueError("Need either `mesh` or `meshfile` to build node coordinates.")
        mesh = make_mesh(meshfile=meshfile)

    Q = firedrake.FunctionSpace(mesh, family="CG", degree=mesh_degree)
    V = firedrake.VectorFunctionSpace(mesh, family="CG", degree=mesh_degree)
    coords = function_space_coordinates(Q)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Expected a 2-D mesh with planar coordinates; got coordinate shape {coords.shape}."
        )

    mask = np.ones(coords.shape[0], dtype=bool)
    boundary_nodes = np.array([], dtype=int)

    if exclude_boundary:
        if dirichlet_ids is None:
            raise ValueError(
                "Boundary exclusion requires `dirichlet_ids`, `dirichlet_ids='on_boundary'`, "
                "or a config that provides `inverse.dirichlet_ids`."
            )
        if isinstance(dirichlet_ids, str):
            if dirichlet_ids != "on_boundary":
                raise ValueError(
                    "String-valued `dirichlet_ids` must be `'on_boundary'`."
                )
            bc = firedrake.DirichletBC(V, _zero_dirichlet_value(V), dirichlet_ids)
            boundary_nodes = np.asarray(bc.nodes, dtype=int)
        else:
            dirichlet_ids = _normalize_dirichlet_ids(dirichlet_ids)
            if dirichlet_ids:
                bc = firedrake.DirichletBC(
                    V,
                    _zero_dirichlet_value(V),
                    dirichlet_ids,
                )
                boundary_nodes = np.asarray(bc.nodes, dtype=int)
        if boundary_nodes.size:
            mask[boundary_nodes] = False

    return mesh, coords, boundary_nodes, mask


def _scalar_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    obs = np.asarray(obs, dtype=float).reshape(-1)
    if pred.shape != obs.shape:
        raise ValueError(f"Mismatched scalar shapes {pred.shape} and {obs.shape}.")
    if pred.size == 0:
        raise ValueError("Cannot compute error metrics on an empty array.")

    diff = pred - obs
    obs_norm = np.linalg.norm(obs)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / obs_norm) if obs_norm > 0.0 else np.nan,
    }


def _vector_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"Mismatched vector shapes {pred.shape} and {obs.shape}.")
    if pred.ndim != 2:
        raise ValueError(
            f"Expected vector arrays with shape (n_samples, n_components); got {pred.shape}."
        )
    if pred.shape[0] == 0:
        raise ValueError("Cannot compute error metrics on an empty array.")

    diff = pred - obs
    diff_norm = np.linalg.norm(diff, axis=1)
    obs_norm = np.linalg.norm(obs.reshape(-1))
    mean_error = diff.mean(axis=0)

    return {
        "rmse": float(np.sqrt(np.mean(diff_norm**2))),
        "mae": float(np.mean(diff_norm)),
        "max_abs": float(np.max(diff_norm)),
        "relative_l2": float(np.linalg.norm(diff.reshape(-1)) / obs_norm)
        if obs_norm > 0.0
        else np.nan,
        "mean_error": mean_error.tolist(),
    }


def _error_metrics(
    errors: np.ndarray,
    reference: np.ndarray,
    *,
    signed: bool,
) -> dict[str, float]:
    errors = np.asarray(errors, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if errors.shape != reference.shape:
        raise ValueError(
            f"Mismatched error/reference shapes {errors.shape} and {reference.shape}."
        )

    if errors.size == 0:
        center_key = "bias" if signed else "mean"
        return {
            "rmse": np.nan,
            "mae": np.nan,
            center_key: np.nan,
            "max_abs": np.nan,
            "relative_l2": np.nan,
        }

    abs_errors = np.abs(errors)
    reference_norm = np.linalg.norm(reference)
    center_key = "bias" if signed else "mean"
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_errors)),
        center_key: float(np.mean(errors)),
        "max_abs": float(np.max(abs_errors)),
        "relative_l2": float(np.linalg.norm(errors) / reference_norm)
        if reference_norm > 0.0
        else np.nan,
    }


def _flow_direction_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    *,
    speed_floor: float = 0.0,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"Mismatched vector shapes {pred.shape} and {obs.shape}.")
    if pred.ndim != 2:
        raise ValueError(
            f"Expected vector arrays with shape (n_samples, n_components); got {pred.shape}."
        )
    if pred.shape[0] == 0:
        raise ValueError("Cannot compute error metrics on an empty array.")
    if speed_floor < 0.0:
        raise ValueError("speed_floor must be non-negative.")

    diff = pred - obs
    speed_obs = np.linalg.norm(obs, axis=1)
    flow_mask = speed_obs > speed_floor
    obs_flow = obs[flow_mask]
    diff_flow = diff[flow_mask]
    speed_ref = speed_obs[flow_mask]

    if speed_ref.size == 0:
        empty = np.array([], dtype=float)
        return {
            "n_valid_flow_nodes": 0,
            "along_flow": _error_metrics(empty, empty, signed=True),
            "cross_flow": _error_metrics(empty, empty, signed=False),
        }

    unit_obs = obs_flow / speed_ref[:, None]
    along_flow = np.einsum("ij,ij->i", diff_flow, unit_obs)
    cross_flow_vec = diff_flow - along_flow[:, None] * unit_obs
    cross_flow = np.linalg.norm(cross_flow_vec, axis=1)

    return {
        "n_valid_flow_nodes": int(speed_ref.size),
        "along_flow": _error_metrics(along_flow, speed_ref, signed=True),
        "cross_flow": _error_metrics(cross_flow, speed_ref, signed=False),
    }


def _component_names(n_components: int) -> list[str]:
    if n_components == 1:
        return ["u"]
    if n_components == 2:
        return ["u", "v"]
    return [f"component_{i}" for i in range(n_components)]


def _normalize_epoch_velocity_input(values: Any, n_nodes: int) -> np.ndarray:
    if hasattr(values, "dat") and hasattr(values.dat, "data_ro"):
        values = np.asarray(values.dat.data_ro)
    else:
        values = np.asarray(values)

    if values.ndim == 1:
        if values.shape[0] != n_nodes:
            raise ValueError(
                f"Expected {n_nodes} nodal values, got shape {values.shape}."
            )
        values = values[:, None]
    elif values.ndim == 2:
        if values.shape[0] == n_nodes:
            pass
        elif values.shape[1] == n_nodes:
            values = values.T
        else:
            raise ValueError(
                f"Expected one axis to match the number of nodes ({n_nodes}); "
                f"got shape {values.shape}."
            )
    else:
        raise ValueError(
            f"Each epoch must be a Firedrake Function or a 1-D/2-D array; got {values.shape}."
        )

    return np.asarray(values, dtype=float)


def _normalize_mesh_velocity_epochs(values: Any, n_nodes: int) -> np.ndarray:
    if hasattr(values, "dat") and hasattr(values.dat, "data_ro"):
        return _normalize_epoch_velocity_input(values, n_nodes)[None, ...]

    if isinstance(values, (list, tuple)):
        epochs = [_normalize_epoch_velocity_input(epoch, n_nodes) for epoch in values]
        if not epochs:
            raise ValueError("Velocity input must contain at least one epoch.")
        return np.stack(epochs, axis=0)

    values = np.asarray(values)

    if values.ndim <= 2:
        return _normalize_epoch_velocity_input(values, n_nodes)[None, ...]

    if values.ndim != 3:
        raise ValueError(
            f"Velocity input must be a list of epochs or a 2-D/3-D array; got shape {values.shape}."
        )

    if values.shape[1] == n_nodes:
        out = values
    elif values.shape[2] == n_nodes:
        out = np.swapaxes(values, 1, 2)
    else:
        raise ValueError(
            f"Expected one array axis to match the number of nodes ({n_nodes}); "
            f"got shape {values.shape}."
        )

    return np.asarray(out, dtype=float)


def _compute_metrics_from_mesh_epochs(
    pred_epochs: np.ndarray,
    obs_epochs: np.ndarray,
    *,
    coords: np.ndarray,
    boundary_nodes: np.ndarray,
    interior_mask: np.ndarray,
    prediction_epoch_ids: Sequence[int] | None = None,
    observation_epoch_ids: Sequence[int] | None = None,
    group: str | None = None,
    meshfile: str | None = None,
    mesh_degree: int = 2,
    exclude_boundary: bool = True,
    dirichlet_ids: Sequence[int] | str | None = None,
) -> dict[str, Any]:
    pred_epochs = np.asarray(pred_epochs, dtype=float)
    obs_epochs = np.asarray(obs_epochs, dtype=float)

    if pred_epochs.shape != obs_epochs.shape:
        raise ValueError(
            f"Prediction and observation arrays must match exactly; got "
            f"{pred_epochs.shape} and {obs_epochs.shape}."
        )
    if pred_epochs.ndim != 3:
        raise ValueError(
            f"Expected normalized epoch arrays with shape (Nt, n_nodes, n_comp); "
            f"got {pred_epochs.shape}."
        )

    n_epochs, n_nodes_total, n_components = pred_epochs.shape
    if n_nodes_total != coords.shape[0]:
        raise ValueError(
            f"Velocity arrays have {n_nodes_total} nodes but the mesh mask uses "
            f"{coords.shape[0]} nodes."
        )

    if prediction_epoch_ids is None:
        prediction_epoch_ids = list(range(n_epochs))
    else:
        prediction_epoch_ids = [int(idx) for idx in prediction_epoch_ids]

    if observation_epoch_ids is None:
        observation_epoch_ids = list(range(n_epochs))
    else:
        observation_epoch_ids = [int(idx) for idx in observation_epoch_ids]

    if len(prediction_epoch_ids) != n_epochs or len(observation_epoch_ids) != n_epochs:
        raise ValueError(
            "Epoch id lists must have the same length as the number of epochs."
        )

    component_names = _component_names(n_components)
    aggregate_pred = []
    aggregate_obs = []
    per_epoch = []

    for k in range(n_epochs):
        pred_interior = pred_epochs[k][interior_mask]
        obs_interior = obs_epochs[k][interior_mask]

        finite_mask = np.isfinite(pred_interior).all(axis=1) & np.isfinite(obs_interior).all(axis=1)
        pred_valid = pred_interior[finite_mask]
        obs_valid = obs_interior[finite_mask]
        if pred_valid.shape[0] == 0:
            raise ValueError(
                f"No finite interior-node comparisons are available for epoch "
                f"{observation_epoch_ids[k]}."
            )

        aggregate_pred.append(pred_valid)
        aggregate_obs.append(obs_valid)

        epoch_components = {
            name: _scalar_metrics(pred_valid[:, i], obs_valid[:, i])
            for i, name in enumerate(component_names)
        }
        speed_pred = np.linalg.norm(pred_valid, axis=1)
        speed_obs = np.linalg.norm(obs_valid, axis=1)
        epoch_flow_metrics = _flow_direction_metrics(pred_valid, obs_valid)
        per_epoch.append(
            {
                "prediction_epoch": prediction_epoch_ids[k],
                "observation_epoch": observation_epoch_ids[k],
                "n_valid_nodes": int(pred_valid.shape[0]),
                "vector": _vector_metrics(pred_valid, obs_valid),
                "speed": _scalar_metrics(speed_pred, speed_obs),
                "along_flow": epoch_flow_metrics["along_flow"],
                "cross_flow": epoch_flow_metrics["cross_flow"],
                "n_valid_flow_nodes": epoch_flow_metrics["n_valid_flow_nodes"],
                "components": epoch_components,
            }
        )

    aggregate_pred = np.concatenate(aggregate_pred, axis=0)
    aggregate_obs = np.concatenate(aggregate_obs, axis=0)
    aggregate_components = {
        name: _scalar_metrics(aggregate_pred[:, i], aggregate_obs[:, i])
        for i, name in enumerate(component_names)
    }
    aggregate_speed_pred = np.linalg.norm(aggregate_pred, axis=1)
    aggregate_speed_obs = np.linalg.norm(aggregate_obs, axis=1)
    aggregate_flow_metrics = _flow_direction_metrics(aggregate_pred, aggregate_obs)

    return {
        "group": group,
        "meshfile": meshfile,
        "mesh_degree": int(mesh_degree),
        "exclude_boundary": bool(exclude_boundary),
        "dirichlet_ids": dirichlet_ids,
        "n_epochs": int(n_epochs),
        "n_nodes_total": int(coords.shape[0]),
        "n_nodes_excluded": int(boundary_nodes.size) if exclude_boundary else 0,
        "n_nodes_interior": int(interior_mask.sum()),
        "n_valid_node_samples": int(aggregate_pred.shape[0]),
        "aggregate": {
            "vector": _vector_metrics(aggregate_pred, aggregate_obs),
            "speed": _scalar_metrics(aggregate_speed_pred, aggregate_speed_obs),
            "along_flow": aggregate_flow_metrics["along_flow"],
            "cross_flow": aggregate_flow_metrics["cross_flow"],
            "n_valid_flow_nodes": aggregate_flow_metrics["n_valid_flow_nodes"],
            "components": aggregate_components,
        },
        "per_epoch": per_epoch,
    }


class VelocityErrorMetricsEvaluator:
    """Cache mesh-dependent data for repeated velocity error computations."""

    def __init__(
        self,
        cfg: Any = None,
        *,
        mesh=None,
        meshfile: str | Path | None = None,
        dirichlet_ids: Sequence[int] | str | None = None,
        mesh_degree: int = 2,
        exclude_boundary: bool = True,
    ):
        cfg, cfg_dir = _load_cfg_like(cfg)
        meshfile = _resolve_meshfile(meshfile, cfg, cfg_dir)
        if dirichlet_ids is None and cfg is not None:
            dirichlet_ids = getattr(cfg.inverse, "dirichlet_ids", [])
        dirichlet_ids = _normalize_dirichlet_ids(dirichlet_ids)

        self.mesh_degree = int(mesh_degree)
        self.exclude_boundary = bool(exclude_boundary)
        self.meshfile = meshfile
        self.dirichlet_ids = dirichlet_ids

        self.mesh, self.coords, self.boundary_nodes, self.interior_mask = _build_node_mask(
            mesh=mesh,
            meshfile=meshfile,
            dirichlet_ids=dirichlet_ids,
            mesh_degree=self.mesh_degree,
            exclude_boundary=self.exclude_boundary,
        )
    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    def compute_on_mesh(
        self,
        pred_velocity: Any,
        obs_velocity: Any,
        *,
        prediction_epoch_ids: Sequence[int] | None = None,
        observation_epoch_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        pred_epochs = _normalize_mesh_velocity_epochs(pred_velocity, self.n_nodes)
        obs_epochs = _normalize_mesh_velocity_epochs(obs_velocity, self.n_nodes)

        return _compute_metrics_from_mesh_epochs(
            pred_epochs,
            obs_epochs,
            coords=self.coords,
            boundary_nodes=self.boundary_nodes,
            interior_mask=self.interior_mask,
            prediction_epoch_ids=prediction_epoch_ids,
            observation_epoch_ids=observation_epoch_ids,
            group=None,
            meshfile=self.meshfile,
            mesh_degree=self.mesh_degree,
            exclude_boundary=self.exclude_boundary,
            dirichlet_ids=self.dirichlet_ids,
        )

    def compute_from_grids(
        self,
        pred_grids: np.ndarray,
        pred_grid_coords: tuple[np.ndarray, np.ndarray],
        obs_grids: np.ndarray,
        obs_grid_coords: tuple[np.ndarray, np.ndarray],
        *,
        prediction_epoch_ids: Sequence[int] | None = None,
        observation_epoch_ids: Sequence[int] | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        pred_grids = np.asarray(pred_grids)
        obs_grids = np.asarray(obs_grids)

        if pred_grids.ndim != 4:
            raise ValueError(
                f"Predicted velocity datasets must stack to shape (Nt, n_comp, Ny, Nx); "
                f"got {pred_grids.shape}."
            )
        if obs_grids.ndim != 4:
            raise ValueError(
                f"Observation velocity datasets must stack to shape (Nt, n_comp, Ny, Nx); "
                f"got {obs_grids.shape}."
            )
        if pred_grids.shape[0] != obs_grids.shape[0]:
            raise ValueError(
                "Prediction and observation files contain different numbers of epochs: "
                f"{pred_grids.shape[0]} predicted vs {obs_grids.shape[0]} observed. "
                "Use `epoch_indices` to choose matching observation epochs."
            )
        if pred_grids.shape[1] != obs_grids.shape[1]:
            raise ValueError(
                "Prediction and observation component counts differ: "
                f"{pred_grids.shape[1]} vs {obs_grids.shape[1]}."
            )

        pred_epochs = []
        obs_epochs = []
        for local_epoch in range(pred_grids.shape[0]):
            pred_epochs.append(
                interpolate_grid_to_mesh_array(
                    pred_grids[local_epoch],
                    pred_grid_coords,
                    self.coords,
                )
            )
            obs_epochs.append(
                interpolate_grid_to_mesh_array(
                    obs_grids[local_epoch],
                    obs_grid_coords,
                    self.coords,
                )
            )

        return _compute_metrics_from_mesh_epochs(
            np.stack(pred_epochs, axis=0),
            np.stack(obs_epochs, axis=0),
            coords=self.coords,
            boundary_nodes=self.boundary_nodes,
            interior_mask=self.interior_mask,
            prediction_epoch_ids=prediction_epoch_ids,
            observation_epoch_ids=observation_epoch_ids,
            group=group,
            meshfile=self.meshfile,
            mesh_degree=self.mesh_degree,
            exclude_boundary=self.exclude_boundary,
            dirichlet_ids=self.dirichlet_ids,
        )

    def compute_from_files(
        self,
        result_file: str | Path,
        data_file: str | Path,
        *,
        stage: int | str | None = None,
        group: str | None = None,
        epoch_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        group_name = _resolve_stage_group(result_file, stage=stage, group=group)
        pred_grids, pred_grid_coords, pred_epoch_ids = _load_prediction_grids(
            result_file,
            group_name,
        )
        obs_grids, obs_grid_coords, obs_epoch_ids = _load_observation_grids(
            data_file,
            epoch_indices=epoch_indices,
        )

        return self.compute_from_grids(
            pred_grids,
            pred_grid_coords,
            obs_grids,
            obs_grid_coords,
            prediction_epoch_ids=pred_epoch_ids,
            observation_epoch_ids=obs_epoch_ids,
            group=group_name,
        )


def compute_velocity_error_metrics_on_mesh(
    pred_velocity: Any,
    obs_velocity: Any,
    cfg: Any = None,
    *,
    mesh=None,
    meshfile: str | Path | None = None,
    dirichlet_ids: Sequence[int] | str | None = None,
    mesh_degree: int = 2,
    exclude_boundary: bool = True,
    prediction_epoch_ids: Sequence[int] | None = None,
    observation_epoch_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Compute the same velocity metrics when both fields are already on mesh nodes.

    Parameters
    ----------
    pred_velocity, obs_velocity
        Velocity values already evaluated on the inversion mesh nodes. Supported
        forms are:
        - list/tuple of Firedrake `Function`s
        - list/tuple of arrays shaped `(n_nodes, n_comp)` or `(n_comp, n_nodes)`
        - array shaped `(n_epochs, n_nodes, n_comp)` or `(n_epochs, n_comp, n_nodes)`
        - single-epoch arrays/functions, which are treated as one epoch
    cfg, mesh, meshfile, dirichlet_ids, mesh_degree, exclude_boundary
        Same boundary-mask arguments as `compute_velocity_error_metrics(...)`.
    prediction_epoch_ids, observation_epoch_ids
        Optional labels to carry into the returned `per_epoch` entries.

    Notes
    -----
    For repeated evaluations on the same mesh, prefer constructing
    `VelocityErrorMetricsEvaluator(...)` once and reusing `compute_on_mesh(...)`.
    """
    evaluator = VelocityErrorMetricsEvaluator(
        cfg,
        mesh=mesh,
        meshfile=meshfile,
        dirichlet_ids=dirichlet_ids,
        mesh_degree=mesh_degree,
        exclude_boundary=exclude_boundary,
    )
    return evaluator.compute_on_mesh(
        pred_velocity,
        obs_velocity,
        prediction_epoch_ids=prediction_epoch_ids,
        observation_epoch_ids=observation_epoch_ids,
    )


def compute_velocity_error_metrics(
    result_file: str | Path,
    data_file: str | Path,
    cfg: Any = None,
    *,
    mesh=None,
    meshfile: str | Path | None = None,
    dirichlet_ids: Sequence[int] | str | None = None,
    stage: int | str | None = None,
    group: str | None = None,
    epoch_indices: Sequence[int] | None = None,
    mesh_degree: int = 2,
    exclude_boundary: bool = True,
) -> dict[str, Any]:
    """
    Compare saved MAP velocity predictions against observations on inversion nodes.

    Parameters
    ----------
    result_file
        HDF5 results file written by `scripts/optimize_MAP_joint.py`.
    data_file
        Observation HDF5 file containing `u`, `x`, and `y`.
    cfg
        Config object or config path. When provided, the utility reads
        `[inverse].meshfile` and `[inverse].dirichlet_ids` from it.
    mesh
        Optional Firedrake mesh. Useful for testing or notebook workflows.
    meshfile
        Optional mesh path override. Used when `mesh` is not supplied.
    dirichlet_ids
        Optional Dirichlet boundary IDs override. Use `'on_boundary'` to exclude
        all boundary nodes.
    stage, group
        Select which saved stage to load. If neither is provided, the latest
        stage in `result_file` is used.
    epoch_indices
        Optional indices into the observation-file `u` dataset. This is useful
        if the inversion used only a subset of epochs.
    mesh_degree
        Firedrake CG degree used for the inversion mesh nodes. MAP workflows use
        degree 2 by default.
    exclude_boundary
        If True, remove Dirichlet-boundary nodes before computing metrics.

    Returns
    -------
    dict
        Nested metrics dictionary with aggregate and per-epoch errors.

    Notes
    -----
    For repeated evaluations on the same mesh, prefer constructing
    `VelocityErrorMetricsEvaluator(...)` once and reusing `compute_from_files(...)`.
    """
    evaluator = VelocityErrorMetricsEvaluator(
        cfg,
        mesh=mesh,
        meshfile=meshfile,
        dirichlet_ids=dirichlet_ids,
        mesh_degree=mesh_degree,
        exclude_boundary=exclude_boundary,
    )
    return evaluator.compute_from_files(
        result_file,
        data_file,
        stage=stage,
        group=group,
        epoch_indices=epoch_indices,
    )


__all__ = [
    "VelocityErrorMetricsEvaluator",
    "compute_velocity_error_metrics",
    "compute_velocity_error_metrics_on_mesh",
]
