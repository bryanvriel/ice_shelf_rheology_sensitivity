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
    return parser


def generate_filename(cfg, loc, amp):

    # Step 1: Short codes for bools and optimizer
    pre = "p" if cfg.inverse.precondition else "np"
    joint = "j" if cfg.inverse.joint else "nj"

    # Step 2: Compact penalty string
    penalties = cfg.inverse.penalties_n
    penalties_str = "-".join(
        f"{int(p) if p.is_integer() else p:.2g}".replace(".0", "") for p in penalties
    )

    # Step 3: Compact nll scale string
    scale_str = str(int(cfg.inverse.nll_scale))

    # Location and amplitude of perturbation
    loc_dict = {"terminus": "t", "GL": "gl"}
    loc_amp = f"l{loc_dict[loc]}_a{amp[0]}"

    # Step 4: type of function
    if cfg.matrix.spectral:
        ftype = "spec"
    else:
        ftype = "wav"

    # Final: Combine all into filename
    filename = f"r_{ftype}_{loc_amp}_{pre}_{joint}_p{penalties_str}_s{scale_str}.h5"
    return os.path.join(cfg.output_dir, filename)


def data_filename(cfg, args):
    outdir = cfg.output_dir
    if args is None:
        return os.path.join(outdir, "data_inverse.h5")
    else:
        loc = f"{args.loc}"
        amp = f"amp-{args.amp}"
        return os.path.join(
            outdir,
            f"data_inverse_perturbed_{loc}_{amp}.h5",
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


def get_latest_stage(filepath):

    with h5py.File(filepath, "r") as fid:
        stage_keys = []
        for key in fid.keys():
            if key.startswith("stage"):
                stage_keys.append(key)

    stage_keys = sorted(stage_keys)
    return stage_keys[-1]


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


# -------------------------
# Public API
# -------------------------


def make_data_digest(
    cfg: DictToStruct, ignore=["v_sigma", "amp", "loc"],
) -> str:
    data_cfg = cfg.data.todict()
    for key in ignore:
        del data_cfg[key]
    digest = _short_hash({"data": data_cfg}, digest_bytes=4)
    return digest


def make_inverse_digest(
    cfg: DictToStruct, ignore=["n_iter", "lr_schedule", "lr", "objective_scale"],
) -> str:
    inverse_cfg = cfg.inverse.todict()
    for key in ignore:
        del inverse_cfg[key]

    digest = _short_hash(
        {"inverse": inverse_cfg, "model": cfg.model.todict()}
    )
    return digest


def make_geom_digest(cfg: DictToStruct) -> str:
    digest = _short_hash({"matrix": cfg.matrix.todict()}, digest_bytes=4)
    return digest


def make_matrix_filename(cfg: DictToStruct) -> str:
    if cfg.matrix.use_ref:
        return "ref_data/matrix_data.h5"
    else:
        outdir = cfg.output_dir
        digest = make_geom_digest(cfg)
        return os.path.join(outdir, f"matrix_data_{digest}.h5")


def make_results_filename(
    cfg: DictToStruct,
    *,
    prefix: str = "r",
    ext: str = "h5",
    max_slug_chars: int = 140,
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

    # Build human-readable tokens from simple top-level scalars (sorted by key)
    items = {
        "d": ["u_sigma", "amp", "loc"],
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

    # Add a token for spectral matrix
    if cfg.matrix.spectral:
        tokens.append("spectral")

    # Keep it short but useful
    slug = "__".join(tokens)
    if len(slug) > max_slug_chars:
        slug = slug[:max_slug_chars].rstrip("_-")

    # Stable short hashes over the full canonicalized configs
    data_digest = make_data_digest(cfg)
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
