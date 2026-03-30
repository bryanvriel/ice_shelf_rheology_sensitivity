# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any
import tomllib

from jbao.utils import DictToStruct


class ConfigValidationError(ValueError):
    pass


class ConfigInheritanceError(ValueError):
    pass


_MISSING = object()
_SUPPORTED_NORM_TYPES = {"log", "tanh", "arctan", "cdf", "sp"}
_SUPPORTED_B_NORM_TYPES = _SUPPORTED_NORM_TYPES | {"viscosity"}
_SUPPORTED_HYPERPARAM_LENGTHS = {0, 2, 3}

_NORMALIZED_DEFAULTS = {
    "data": {
        "v_sigma": 10.0,
    },
    "inverse": {
        "lr": 5.0e-4,
        "lr_schedule": [],
        "dirichlet_ids": [1],
        "fixed_n": 3.0,
        "smooth_boundary": False,
        "bfgs_maxcor": 50,
        "opt_init_seed": 744,
        "opt_init_amp": 0.05,
        "hyper_params": [],
    },
    "matrix": {
        "spectral": False,
        "spectral_L_min": 10.0,
        "use_ref": False,
        "mask": "none",
    },
    "prep": {},
    "vi": {
        "hess_reg": 1.0,
    },
}


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


def load_cfg_file(path: str | Path) -> DictToStruct:
    return DictToStruct(_load_cfg_dict(Path(path), ()))


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


def _set_path(data: dict, path: str, value) -> None:
    ref = data
    keys = path.split(".")
    for key in keys[:-1]:
        ref = ref.setdefault(key, {})
    ref[keys[-1]] = value


def _apply_defaults(data: dict, defaults: dict, prefix: str = "") -> None:
    for key, value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        existing = _get_path(data, path)
        if isinstance(value, dict):
            if existing is _MISSING:
                _set_path(data, path, {})
            _apply_defaults(data, value, path)
        elif existing is _MISSING:
            _set_path(data, path, deepcopy(value))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def primary_penalty(penalty):
    if _is_sequence(penalty):
        if len(penalty) == 0:
            raise ConfigValidationError("Penalty list cannot be empty.")
        return penalty[0]
    return penalty


def normalize_cfg(cfg: DictToStruct | dict) -> DictToStruct:
    data = _to_dict(cfg)

    for section in ("data", "inverse", "model", "matrix", "prep", "vi"):
        data.setdefault(section, {})

    # Accept older configs that stored the inverse mesh under [model].
    inverse_meshfile = _get_path(data, "inverse.meshfile")
    model_meshfile = _get_path(data, "model.meshfile")
    if inverse_meshfile is _MISSING and model_meshfile is not _MISSING:
        _set_path(data, "inverse.meshfile", model_meshfile)

    # Prep workflows often run on a different mesh than the inverse path.
    # Prefer an explicit prep mesh when present, otherwise fall back to the
    # legacy model mesh and finally the inverse mesh.
    prep_meshfile = _get_path(data, "prep.meshfile")
    if prep_meshfile is _MISSING:
        if model_meshfile is not _MISSING:
            _set_path(data, "prep.meshfile", model_meshfile)
        elif inverse_meshfile is not _MISSING:
            _set_path(data, "prep.meshfile", inverse_meshfile)

    # Canonicalize penalties onto the plural keys used in configs.
    for suffix in ("B", "n"):
        singular_path = f"inverse.penalty_{suffix}"
        plural_path = f"inverse.penalties_{suffix}"
        singular = _get_path(data, singular_path)
        plural = _get_path(data, plural_path)

        if plural is _MISSING and singular is not _MISSING:
            _set_path(data, plural_path, deepcopy(singular))
            plural = singular

        if plural is not _MISSING:
            _set_path(data, singular_path, primary_penalty(plural))

    _apply_defaults(data, _NORMALIZED_DEFAULTS)
    return DictToStruct(data)


def validate_cfg(cfg: DictToStruct | dict, *, require_vi: bool = False) -> DictToStruct:
    cfg = normalize_cfg(cfg)
    data = cfg.todict()
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
        "inverse.smooth_boundary",
        "inverse.opt_type",
        "inverse.n_iter",
        "inverse.opt_init_seed",
        "inverse.opt_init_amp",
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

    inverse_meshfile = _get_path(data, "inverse.meshfile")
    model_meshfile = _get_path(data, "model.meshfile", inverse_meshfile)
    if model_meshfile is not _MISSING and inverse_meshfile != model_meshfile:
        errors.append(
            "`model.meshfile` and `inverse.meshfile` disagree. Use "
            "`inverse.meshfile` as the canonical golden-path key."
        )

    hyper_params = _get_path(data, "inverse.hyper_params", [])
    if not _is_sequence(hyper_params):
        errors.append("`inverse.hyper_params` must be a list.")
    elif len(hyper_params) not in _SUPPORTED_HYPERPARAM_LENGTHS:
        errors.append("`inverse.hyper_params` must have length 0, 2, or 3.")

    if len(hyper_params) == 0:
        for path in ("inverse.penalty_B", "inverse.penalty_n"):
            penalty = _get_path(data, path)
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

    return cfg


# end of file
