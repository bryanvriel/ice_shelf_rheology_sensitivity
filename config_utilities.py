# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any
import tomllib

from jbao.utils import DictToStruct, update_cfg_with_extra


class ConfigValidationError(ValueError):
    pass


class ConfigInheritanceError(ValueError):
    pass


_MISSING = object()
_SUPPORTED_NORM_TYPES = {"log", "tanh", "arctan", "cdf", "sp"}
_SUPPORTED_B_NORM_TYPES = _SUPPORTED_NORM_TYPES | {"viscosity"}
_SUPPORTED_HYPERPARAM_LENGTHS = {0, 2, 3}


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(base_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_cfg_dict(path: Path, chain: tuple[Path, ...]) -> dict:
    path = path.expanduser().resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigInheritanceError(f"Config extends cycle detected: {cycle}")

    try:
        with path.open("rb") as fid:
            data = tomllib.load(fid)
    except FileNotFoundError as exc:
        raise ConfigInheritanceError(f"Config file not found: {path}") from exc

    extends = data.pop("extends", None)
    if extends is None:
        return data
    if not isinstance(extends, str):
        raise ConfigInheritanceError(f"`extends` in {path} must be a string path.")

    base_path = path.parent / extends
    base_cfg = _load_cfg_dict(base_path, (*chain, path))
    return _merge_dicts(base_cfg, data)


def _to_dict(cfg: DictToStruct | dict) -> dict:
    if isinstance(cfg, DictToStruct):
        return cfg.todict()
    return deepcopy(cfg)


def _get_path(data: dict, path: str, default=_MISSING):
    ref = data
    for key in path.split("."):
        if not isinstance(ref, dict) or key not in ref:
            return default
        ref = ref[key]
    return ref


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def primary_penalty(penalty):
    if _is_sequence(penalty):
        if len(penalty) == 0:
            raise ConfigValidationError("Penalty list cannot be empty.")
        return penalty[0]
    return penalty


def load_cfg_file(path: str | Path) -> DictToStruct:
    return DictToStruct(_load_cfg_dict(Path(path), ()))


def apply_calibration_overrides(
    cfg: DictToStruct,
    *,
    cal: bool = False,
    disc: bool = False,
    calibration_file: str | Path = "calibration.toml",
) -> DictToStruct:
    if not (cal or disc):
        return cfg

    cal_cfg = load_cfg_file(calibration_file).todict()[cfg.output_dir]
    key = "calibration_hyper_params" if cal else "discovery_hyper_params"
    return cfg.update("inverse.hyper_params", cal_cfg[key])


def load_config(
    path: str | Path,
    *,
    extra_args=None,
    cal: bool = False,
    disc: bool = False,
    calibration_file: str | Path = "calibration.toml",
) -> DictToStruct:
    cfg = load_cfg_file(path)
    if extra_args:
        cfg = update_cfg_with_extra(cfg, extra_args)
    return apply_calibration_overrides(
        cfg,
        cal=cal,
        disc=disc,
        calibration_file=calibration_file,
    )


def validate_cfg(cfg: DictToStruct | dict, *, require_vi: bool = False) -> DictToStruct:
    data = _to_dict(cfg)
    errors = []

    def require(path: str):
        if _get_path(data, path) is _MISSING:
            errors.append(f"Missing required config key `{path}`.")

    for path in (
        "output_dir",
        "shelf_coords_file",
        "data.u_sigma",
        "inverse.meshfile",
        "inverse.joint",
        "inverse.nll_scale",
        "inverse.precondition",
        "inverse.objective_scale",
        "inverse.penalties_B",
        "inverse.penalties_n",
        "inverse.norm_scale",
        "inverse.B_norm_type",
        "inverse.n_norm_type",
        "inverse.dirichlet_ids",
        "inverse.lr_schedule",
        "inverse.smooth_boundary",
        "inverse.opt_type",
        "inverse.n_iter",
        "inverse.opt_init_seed",
        "inverse.opt_init_amp",
        "inverse.hyper_params",
        "model.B_min",
        "model.B_max",
        "model.n_min",
        "model.n_max",
        "matrix.use_ref",
        "matrix.mask",
        "matrix.spectral",
    ):
        require(path)

    if require_vi:
        require("vi.hess_reg")

    opt_type = _get_path(data, "inverse.opt_type", None)
    if opt_type == "bfgs":
        require("inverse.bfgs_factr")
        require("inverse.bfgs_maxcor")
    else:
        require("inverse.lr")

    if _get_path(data, "inverse.joint", True) is False:
        require("inverse.fixed_n")

    if errors:
        raise ConfigValidationError("\n".join(errors))

    for path in ("inverse.penalties_B", "inverse.penalties_n"):
        penalty = _get_path(data, path)
        if _is_sequence(penalty):
            if len(penalty) == 0:
                errors.append(f"`{path}` must not be empty.")
            elif not all(isinstance(v, Number) for v in penalty):
                errors.append(f"`{path}` must contain numeric values.")
        elif not isinstance(penalty, Number):
            errors.append(f"`{path}` must be a number or a numeric list.")

    hyper_params = _get_path(data, "inverse.hyper_params")
    if not _is_sequence(hyper_params):
        errors.append("`inverse.hyper_params` must be a list.")
    elif len(hyper_params) not in _SUPPORTED_HYPERPARAM_LENGTHS:
        errors.append("`inverse.hyper_params` must have length 0, 2, or 3.")

    if len(hyper_params) == 0:
        for path in ("inverse.penalties_B", "inverse.penalties_n"):
            penalty = primary_penalty(_get_path(data, path))
            if not isinstance(penalty, Number):
                errors.append(
                    f"`{path}` must resolve to a numeric primary penalty when "
                    "`inverse.hyper_params` is empty."
                )
            elif penalty <= 0:
                errors.append(
                    f"`{path}` must be positive when `inverse.hyper_params` is empty."
                )

    B_norm_type = _get_path(data, "inverse.B_norm_type")
    if B_norm_type not in _SUPPORTED_B_NORM_TYPES:
        errors.append(
            "`inverse.B_norm_type` must be one of "
            f"{sorted(_SUPPORTED_B_NORM_TYPES)}."
        )

    n_norm_type = _get_path(data, "inverse.n_norm_type")
    if n_norm_type not in _SUPPORTED_NORM_TYPES:
        errors.append(
            "`inverse.n_norm_type` must be one of "
            f"{sorted(_SUPPORTED_NORM_TYPES)}."
        )

    dirichlet_ids = _get_path(data, "inverse.dirichlet_ids")
    if not _is_sequence(dirichlet_ids) or len(dirichlet_ids) == 0:
        errors.append("`inverse.dirichlet_ids` must be a non-empty list.")

    infer_boundary_velocity = _get_path(data, "inverse.infer_boundary_velocity", False)
    if infer_boundary_velocity:
        for path in (
            "inverse.boundary_spline_degree",
            "inverse.boundary_sigma_alpha",
            "inverse.boundary_corr_length",
        ):
            if _get_path(data, path, _MISSING) is _MISSING:
                errors.append(
                    f"Missing required config key `{path}` when "
                    "`inverse.infer_boundary_velocity = true`."
                )

        n_basis = _get_path(data, "inverse.boundary_n_basis", _MISSING)
        basis_spacing = _get_path(data, "inverse.boundary_basis_spacing", _MISSING)
        degree = _get_path(data, "inverse.boundary_spline_degree", None)
        sigma_alpha = _get_path(data, "inverse.boundary_sigma_alpha", None)
        corr_length = _get_path(data, "inverse.boundary_corr_length", None)

        if n_basis is _MISSING and basis_spacing is _MISSING:
            errors.append(
                "Boundary inference requires either `inverse.boundary_basis_spacing` "
                "or `inverse.boundary_n_basis`."
            )
        if n_basis is not _MISSING and basis_spacing is not _MISSING:
            errors.append(
                "Specify only one of `inverse.boundary_basis_spacing` and "
                "`inverse.boundary_n_basis`."
            )

        if n_basis is not _MISSING and (
            not isinstance(n_basis, Number) or int(n_basis) != n_basis or n_basis < 2
        ):
            errors.append("`inverse.boundary_n_basis` must be an integer >= 2.")
        if basis_spacing is not _MISSING and (
            not isinstance(basis_spacing, Number) or basis_spacing <= 0
        ):
            errors.append("`inverse.boundary_basis_spacing` must be positive.")
        if not isinstance(degree, Number) or int(degree) != degree or degree < 0:
            errors.append("`inverse.boundary_spline_degree` must be a non-negative integer.")
        if (
            n_basis is not _MISSING
            and isinstance(n_basis, Number)
            and isinstance(degree, Number)
            and int(n_basis) == n_basis
            and int(degree) == degree
            and n_basis < degree + 1
        ):
            errors.append(
                "`inverse.boundary_n_basis` must be at least "
                "`inverse.boundary_spline_degree + 1`."
            )
        if not isinstance(sigma_alpha, Number) or sigma_alpha <= 0:
            errors.append("`inverse.boundary_sigma_alpha` must be positive.")
        if not isinstance(corr_length, Number) or corr_length <= 0:
            errors.append("`inverse.boundary_corr_length` must be positive.")

    lr_schedule = _get_path(data, "inverse.lr_schedule", [])
    if not _is_sequence(lr_schedule):
        errors.append("`inverse.lr_schedule` must be a list of [step, lr] pairs.")
    elif len(lr_schedule) > 0:
        first = lr_schedule[0]
        if not _is_sequence(first) or len(first) != 2 or first[0] != 0:
            errors.append(
                "`inverse.lr_schedule` must start with a [0, learning_rate] entry."
            )

    if _get_path(data, "matrix.spectral", False):
        errors.append(
            "`matrix.spectral = true` is not supported in the current golden path; "
            "the spectral matrix helper is still unresolved."
        )

    if errors:
        raise ConfigValidationError("\n".join(errors))

    return DictToStruct(data)


# end of file
