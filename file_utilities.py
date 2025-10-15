#-*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
import argparse
import datetime
import h5py
import sys
import jax
import os


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
        default="mid",
        help="Amplitude from (low, mid, high). Default: mid.",
    )
    parser.add_argument(
        "--loc",
        action="store",
        type=str,
        default="terminus",
        help="Perturbation location from (terminus, GL). Default: terminus.",
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
        transform = from_origin(x_coords[0] - xres / 2, y_coords[0] + yres / 2, xres, yres)
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

    meta_keys = ("x", "y", "z")
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


def save_state_to_group(optimizer_state, opt_type, filepath: str, group_name: str):
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
    print(f"✅ Optimizer state saved to group '{group_name}' in {filepath}")


def load_state_from_group(template_state, filepath: str, group_name: str):
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


# end of file
