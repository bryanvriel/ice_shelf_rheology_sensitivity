# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any

import jbao

import schurflow.config_utilities as cu
import schurflow.file_utilities as fu
import schurflow.train_utilities as tu


@dataclass
class WorkflowContext:
    cfg: Any
    data_file: str
    outfile: str
    field: Any
    problem: Any
    objective: Any
    converter: Any


def load_config(args, extra_args, default="default_config.toml"):
    cfg = jbao.utils.parse_toml(args.cfg, default=default)
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    return cu.normalize_cfg(cfg)


def apply_calibration_overrides(cfg, args, calibration_file="calibration.toml"):
    if not (getattr(args, "cal", False) or getattr(args, "disc", False)):
        return cfg

    cal = jbao.utils.parse_toml(calibration_file).todict()[cfg.output_dir]
    key = "calibration_hyper_params" if args.cal else "discovery_hyper_params"
    return cfg.update("inverse.hyper_params", cal[key])


def resolve_output_file(cfg, args, *, data_file=None, prefix="r", ext="h5"):
    if getattr(args, "output", None):
        return args.output

    if data_file is None:
        data_file = fu.data_filename(cfg, args)

    return fu.make_results_filename(
        cfg,
        prefix=prefix,
        ext=ext,
        data_file=data_file,
    )


def initialize_workflow(
    cfg,
    args,
    *,
    comm=None,
    data_obj=False,
    time_ind=None,
    mesh_degree=2,
    outfile_prefix="r",
    outfile_ext="h5",
    require_vi=False,
):
    cfg = apply_calibration_overrides(cfg, args)
    cfg = cu.validate_cfg(cfg, require_vi=require_vi)
    data_file = fu.data_filename(cfg, args)
    outfile = resolve_output_file(
        cfg,
        args,
        data_file=data_file,
        prefix=outfile_prefix,
        ext=outfile_ext,
    )
    field, problem, objective, converter = tu.initialize_objects(
        cfg,
        data_file,
        comm=comm,
        data_obj=data_obj,
        time_ind=time_ind,
        mesh_degree=mesh_degree,
    )
    return WorkflowContext(
        cfg=cfg,
        data_file=data_file,
        outfile=outfile,
        field=field,
        problem=problem,
        objective=objective,
        converter=converter,
    )


def resolve_restore_stage(outfile, restore_stage=None, *, default_to_latest=False):
    if restore_stage is not None:
        return restore_stage, f"stage{restore_stage}"

    if not default_to_latest:
        raise ValueError("No restore stage specified.")

    stage_name = fu.get_latest_stage(outfile)
    return fu.get_stage_number(stage_name), stage_name


# end of file
