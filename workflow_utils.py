# -*- coding: utf-8 -*-

from dataclasses import dataclass
import os
from typing import Any

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


def _expected_output_suffix(ext: str) -> str:
    return os.path.splitext(f"placeholder.{ext.lstrip('.')}")[1].lower()


def _matches_output_extension(filepath: str, ext: str) -> bool:
    return os.path.splitext(filepath)[1].lower() == _expected_output_suffix(ext)


def resolve_output_file(
    cfg,
    args,
    *,
    data_file=None,
    prefix="r",
    ext="h5",
    match_output_ext=False,
):
    forced_output = getattr(args, "output", None)
    if forced_output and (
        not match_output_ext or _matches_output_extension(forced_output, ext)
    ):
        return forced_output

    if data_file is None:
        data_file = fu.data_filename(cfg, args)

    return fu.make_results_filename(
        cfg,
        prefix=prefix,
        ext=ext,
        data_file=data_file,
    )


def resolve_map_file(cfg, args, *, data_file=None):
    if getattr(args, "mapfile", None):
        return args.mapfile

    return resolve_output_file(
        cfg,
        args,
        data_file=data_file,
        prefix="r",
        ext="h5",
        match_output_ext=True,
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
    match_output_ext=False,
    require_vi=False,
):
    cfg = cu.validate_cfg(cfg, require_vi=require_vi)
    data_file = fu.data_filename(cfg, args)
    outfile = resolve_output_file(
        cfg,
        args,
        data_file=data_file,
        prefix=outfile_prefix,
        ext=outfile_ext,
        match_output_ext=match_output_ext,
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
