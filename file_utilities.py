# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
import argparse
import datetime
import hashlib, unicodedata
import h5py
import json
import sys
import jax
import re
import os

from jbao.utils import DictToStruct
from typing import Any, Dict, Tuple


def common_parse() -> argparse.ArgumentParser:
    """
    Command line arguments that are common to many of the scripts.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("cfg", type=str, help="Input configuration file.")
    parser.add_argument(
        "--amp",
        action="store",
        type=str,
        default="high",
        help="Amplitude from (low, mid, high). Default: high.",
    )
    parser.add_argument(
        "--loc",
        action="store",
        type=str,
        default="GL",
        help="Perturbation location from (terminus, GL). Default: GL.",
    )
    parser.add_argument(
        "--cal",
        action="store_true",
        help="Use calibration file to set calibration hyperparameters.",
    )
    parser.add_argument(
        "--disc",
        action="store_true",
        help="Use calibration file to set discovery hyperparameters.",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        dest="restore",
        help="Restore from previous solution.",
    )
    parser.add_argument(
        "-rs",
        action="store",
        type=int,
        default=None,
        dest="restore_stage",
        help="Integer stage to restore from. Default: None.",
    )
    parser.add_argument(
        "--output",
        action="store",
        type=str,
        default=None,
        help="Force output filename. Default: None.",
    )
    parser.add_argument(
        "--datafile",
        action="store",
        type=str,
        default=None,
        help="Override the default observation data filename. Default: None.",
    )
    return parser

def data_filename(cfg, args):
    outdir = cfg.output_dir
    if args is None:
        return os.path.join(outdir, "data_inverse.h5")
    datafile = getattr(args, "datafile", None)
    if datafile is not None:
        return datafile
    elif args.amp == "None":
        return os.path.join(outdir, "data_inverse.h5")
    else:
        loc = f"{args.loc}"
        amp = f"amp-{args.amp}"
        seed = f"seed{cfg.data.seed:03d}"
        return os.path.join(
            outdir,
            f"data_inverse_perturbed_{loc}_{amp}_{seed}.h5",
        )


def read_raster(filepath):
    """
    Reads a raster file and returns the raster array and 1D coordinate arrays.

    Parameters
    ----------
    filepath : str
        Path to the GeoTIFF file.

    Returns
    -------
    data : np.ndarray
        2D raster array (single band).
    x_coords : np.ndarray
        1D array of x (easting or longitude) coordinates.
    y_coords : np.ndarray
        1D array of y (northing or latitude) coordinates.
    """
    import rasterio

    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform

        height, width = src.height, src.width

        # Generate 1D coordinate arrays
        x_coords = transform * (np.arange(width), np.zeros(width))
        y_coords = transform * (np.zeros(height), np.arange(height))

        x_coords = x_coords[0]  # x values only
        y_coords = y_coords[1]  # y values only

    return data, x_coords, y_coords


def save_raster(data, x_coords, y_coords, filename, mid=False):

    import rasterio
    from rasterio.transform import from_origin

    # Compute resolution
    xres = (x_coords[-1] - x_coords[0]) / (data.shape[1] - 1)
    yres = (y_coords[0] - y_coords[-1]) / (data.shape[0] - 1)

    # Create transform
    if mid:
        transform = from_origin(
            x_coords[0] - xres / 2, y_coords[0] + yres / 2, xres, yres
        )
    else:
        transform = from_origin(x_coords[0], y_coords[0], xres, yres)

    # Define CRS (example: UTM zone 33N)
    crs = "EPSG:3031"

    # Save to GeoTIFF
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def write_results_to_file(filename, preds, group=None, mode="a", extra_data={}):

    meta_keys = ("x", "y", "z", "final_loss")
    with h5py.File(filename, mode) as fid:

        if group is not None:
            try:
                del fid[group]
            except KeyError:
                pass
            group = fid.create_group(group)
        else:
            group = fid

        for key in meta_keys + ("mean",):
            try:
                del group[key]
            except KeyError:
                pass

        mean_group = group.create_group("mean")
        for key in preds.keys():
            if key not in meta_keys:
                mean_group[key] = preds[key]

        for key in meta_keys:
            group[key] = preds[key]

        for key, value in extra_data.items():
            try:
                del group[key]
            except KeyError:
                pass
            group[key] = value

        # Backwards compatibility
        # group["B"] = preds["B"]
        # group["n"] = preds["n"]


def save_state_to_group(
    optimizer_state, opt_type, filepath: str, group_name: str, verbose=True
):
    """Saves the optimizer state to a specific group in an HDF5 file."""
    # Use 'a' mode to create file if it doesn't exist or open for appending
    with h5py.File(filepath, "a") as f:
        # If the group already exists, delete it for a clean save
        if group_name in f:
            del f[group_name]

        # Create the main group for the optimizer state
        group = f.create_group(group_name)

        # Manually save the optimizer name
        group["name"] = opt_type

        # A function to save each leaf array in the pytree within the group
        def save_leaf(path, leaf_data):
            # The path within the group is derived from the pytree path
            leaf_path = "/".join(map(str, path))
            group.create_dataset(leaf_path, data=np.asarray(leaf_data))

        # Use tree_map_with_path to apply the saving function to each leaf
        jax.tree_util.tree_map_with_path(save_leaf, optimizer_state)

    if verbose:
        print(f"✅ Optimizer state saved to group '{group_name}' in {filepath}")


def load_state_from_group(template_state, filepath: str, group_name: str, verbose=True):
    """Loads the optimizer state from a specific group in an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group '{group_name}' not found in {filepath}")

        # Get the group containing the optimizer state
        group = f[group_name]

        # Get the name
        opt_type = group["name"][()]

        # A function to load each leaf array from the HDF5 group
        def load_leaf(path, _):  # We only need the path, not the template data
            leaf_path = "/".join(map(str, path))
            if leaf_path in group:
                return jnp.asarray(group[leaf_path])
            raise ValueError(f"Path '{leaf_path}' not found in group '{group_name}'.")

        # Use tree_map_with_path to reconstruct the state from the group
        loaded_state = jax.tree_util.tree_map_with_path(load_leaf, template_state)

    if verbose:
        print(f"✅ Optimizer state loaded from group '{group_name}' in {filepath}")
    return loaded_state, opt_type


_STAGE_KEY_PATTERN = re.compile(r"^stage(\d+)$")


def get_stage_number(stage_name: str) -> int:
    match = _STAGE_KEY_PATTERN.fullmatch(stage_name)
    if match is None:
        raise ValueError(f"Invalid stage name '{stage_name}'. Expected format 'stage<N>'.")
    return int(match.group(1))


def get_latest_stage(filepath):

    with h5py.File(filepath, "r") as fid:
        stage_keys = [key for key in fid.keys() if _STAGE_KEY_PATTERN.fullmatch(key)]

    if not stage_keys:
        raise ValueError(f"No stage groups found in {filepath}")
    return max(stage_keys, key=get_stage_number)


def print_file_last_modified(filepath):
    """
    Prints the date and time that a file was last modified.

    Args:
        filepath (str): The path to the file.
    """
    try:
        timestamp = os.path.getmtime(filepath)
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        print(f"The file '{filepath}' was last modified on: {dt_object}")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# -------------------------
# Core utilities for filename creation and parsing
# -------------------------


def _canonicalize(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serializable, sorted, stable form."""
    try:
        import numpy as np

        np_types = (np.generic, np.ndarray)
    except Exception:
        np_types = tuple()

    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    elif np_types and isinstance(obj, np_types):
        # Convert numpy scalars/arrays to py types
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback: stable repr
        return repr(obj)


def _short_hash(payload: Dict[str, Any], digest_bytes: int = 8) -> str:
    """Stable short hash (hex) of a payload."""
    canon = _canonicalize(payload)
    s = json.dumps(canon, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=digest_bytes).hexdigest()


def _is_simple_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _fmt_simple(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"  # compact but precise enough
    if isinstance(v, bool):
        return "t" if v else "f"
    if v is None:
        return "none"
    return str(v)


def _sanitize_filename(s: str) -> str:
    """ASCII-only, safe characters; collapse repeats; trim edges."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9._=+-]+", "-", s)  # allow = _ . + -
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _resolve_data_source(data_file=None, data_label=None) -> str | None:
    if data_label is not None:
        return data_label
    if data_file is None:
        return None
    return os.path.splitext(os.path.basename(data_file))[0]


# -------------------------
# Public API
# -------------------------


def make_data_digest(
    cfg: DictToStruct,
    ignore=("v_sigma",),
    *,
    data_file=None,
    data_label=None,
) -> str:
    data_cfg = cfg.data.todict()
    for key in ignore:
        data_cfg.pop(key, None)
    data_source = _resolve_data_source(data_file=data_file, data_label=data_label)
    if data_source is not None:
        data_cfg["source"] = data_source
    digest = _short_hash({"data": data_cfg}, digest_bytes=4)
    return digest


def make_inverse_digest(
    cfg: DictToStruct, ignore=("n_iter", "lr_schedule", "lr", "objective_scale"),
) -> str:
    inverse_cfg = cfg.inverse.todict()
    for key in ignore:
        inverse_cfg.pop(key, None)

    digest = _short_hash(
        {
            "inverse": inverse_cfg,
            "model": cfg.model.todict(),
            "matrix": {"mask": cfg.matrix.mask},
        }
    )
    return digest


def make_geom_digest(cfg: DictToStruct, ignore=("use_ref",)) -> str:
    mat_cfg = cfg.matrix.todict()
    for key in ignore:
        mat_cfg.pop(key, None)
    mat_cfg['mesh'] = cfg.inverse.meshfile # matrix tied to mesh as well
    digest = _short_hash({"matrix": mat_cfg}, digest_bytes=4)
    return digest


def make_matrix_filename(cfg: DictToStruct) -> str:
    digest = make_geom_digest(cfg)
    if cfg.matrix.use_ref:
        mdir = "ref_data"
    else:
        mdir = cfg.output_dir
    return os.path.join(mdir, f"matrix_data_{digest}.h5")


def make_results_filename(
    cfg: DictToStruct,
    *,
    prefix: str = "r",
    ext: str = "h5",
    max_slug_chars: int = 140,
    data_file=None,
    data_label=None,
) -> str:
    """
    Create a unique, human-readable results filename from {data,inverse} configs.
    Only simple top-level scalars are exposed in the slug; everything else is covered
    by the hash.
    """
    # Unpack dictionaries
    data_cfg = cfg.data.todict()
    inverse_cfg = cfg.inverse.todict()
    outdir = cfg.output_dir
    data_source = _resolve_data_source(data_file=data_file, data_label=data_label)

    # Build human-readable tokens from simple top-level scalars (sorted by key)
    items = {
        "d": ["u_sigma"],
        "inv": [
            "B_norm_type",
        ],
    }
    tokens = []
    for sect_tag, _cfg in (("d", data_cfg), ("inv", inverse_cfg)):
        for k in items[sect_tag]:
            v = _cfg[k]
            if _is_simple_scalar(v):
                tokens.append(f"{sect_tag}.{k}={_fmt_simple(v)}")
    if data_source is not None:
        tokens.append(f"d.source={data_source}")

    # Add a token for spectral matrix
    if cfg.matrix.spectral:
        tokens.append("spectral")

    # Keep it short but useful
    slug = "__".join(tokens)
    if len(slug) > max_slug_chars:
        slug = slug[:max_slug_chars].rstrip("_-")

    # Stable short hashes over the full canonicalized configs
    data_digest = make_data_digest(
        cfg,
        data_file=data_file,
        data_label=data_label,
    )
    geom_digest = make_geom_digest(cfg)
    inv_digest = make_inverse_digest(cfg)

    # Assemble + sanitize
    pieces = [prefix]
    if slug:
        pieces.append(slug)
    pieces.append(data_digest)
    pieces.append(geom_digest)
    pieces.append(inv_digest)
    base = "__".join(pieces)
    fname = _sanitize_filename(base) + f".{ext.lstrip('.')}"
    return os.path.join(outdir, fname)


# end of file
