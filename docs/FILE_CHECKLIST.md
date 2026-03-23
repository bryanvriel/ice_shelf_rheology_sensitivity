# File-By-File Checklist

This checklist is meant to be used alongside `docs/REFRACTOR_PLAN.md`.

It is organized by file and roughly grouped by phase. The intent is to make implementation easy to track without re-deriving the plan from scratch.

## Phase 1: Golden Path

### New helper module

- [ ] Add a shared workflow helper module, likely `workflow_utils.py` or `script_utils.py`
- [ ] Centralize:
  - [ ] config parsing
  - [ ] calibration/discovery override handling
  - [ ] `data_file` resolution
  - [ ] output filename resolution
  - [ ] restore-stage resolution
  - [ ] object construction through `train_utilities.initialize_objects`

### `scripts/optimize_MAP_joint.py`

- [ ] Remove duplicated setup logic that should live in the shared helper
- [ ] Keep only MAP-specific logic in the script
- [ ] Make restore behavior explicit and consistent with the shared helper
- [ ] Keep callback logic small and focused on periodic save/checkpoint behavior

### `scripts/compute_hessian.py`

- [ ] Fix missing imports and script bootstrap drift
- [ ] Fix the broken `__main__` call
- [ ] Replace duplicated setup logic with the shared helper
- [ ] Reuse shared restore-stage and output-path logic
- [ ] Keep only Hessian-specific behavior in the script

### `file_utilities.py`

- [ ] Choose one results filename convention
- [ ] Deprecate or remove `generate_filename()` if `make_results_filename()` is the standard
- [ ] Keep restore-stage and results-path helpers aligned with the chosen convention

## Phase 2: Config Contract

### New config helper module

- [ ] Add `validate_cfg(cfg)` or equivalent
- [ ] Add readable error messages for missing required keys
- [ ] Validate the fields needed by:
  - [ ] `train_utilities.py`
  - [ ] `fields.py`
  - [ ] `problems.py`
  - [ ] the golden-path scripts

### `default_config.toml`

- [ ] Update to reflect actual required core keys
- [ ] Remove or mark keys that are legacy or currently unused
- [ ] Document the minimum viable config for the core workflow

### `config_sinusoidal.toml`

- [ ] Bring naming into alignment with the validated config schema
- [ ] Check that it still exercises a real supported path

### `fields.py`

- [ ] Resolve `penalty_B` vs `penalties_B`
- [ ] Resolve `penalty_n` vs `penalties_n`
- [ ] Make the expected config fields obvious in one place

### `train_utilities.py`

- [ ] Make required config fields explicit in `initialize_objects()`
- [ ] Avoid hidden assumptions about optional config entries

## Phase 3: Core Interfaces

### `problems.py`

- [ ] Replace dynamic rebinding of `self.forward` with explicit methods or small subclasses
- [ ] Replace dynamic rebinding of `self.data_nll` with explicit methods or small subclasses
- [ ] Fix `make_mesh()` so it does not depend on hidden `cfg`
- [ ] Reset observation lists at the start of `load_observations()`
- [ ] Replace mutable default arguments
- [ ] Review unused imports and dead code paths

### `objectives.py`

- [ ] Simplify the control API
- [ ] Replace implicit array/control conversions with explicit methods
- [ ] Remove or fix `PosteriorObjective.update_controls()`
- [ ] Decide whether `PosteriorObjectiveMesh` is still active
- [ ] Remove dead or inactive objective code from the main path if not used

### `models.py`

- [ ] Replace mutable default arguments in `create_solver()`
- [ ] Keep only solver variants that are still useful for active workflows
- [ ] Confirm that normalization helpers still match the active inverse path

## Phase 4: Field Layer And Hessian Path

### `fields.py`

- [ ] Fix or remove `precondition_matrix()`
- [ ] Fix or remove the unresolved `threshold_spectral_matrix()` dependency
- [ ] Decide whether `RheologyFieldMatrix` is still needed
- [ ] Keep one primary joint-field implementation for active workflows
- [ ] Remove stale branches that are not used by the golden path

### `train_utilities.py`

- [ ] Move non-training workflow helpers into a more clearly named module if useful
- [ ] Keep `generate_predictions()` as the one shared export path
- [ ] Remove unused helpers if they are no longer part of active workflows

### `scripts/compute_hessian.py`

- [ ] Reuse `train_utilities.generate_predictions()` where possible
- [ ] Remove the unused dense covariance inverse if it is not needed
- [ ] Precompute sampling factors once instead of inside the loop
- [ ] Keep the Hessian workflow readable:
  - [ ] load MAP
  - [ ] compute or read Hessian
  - [ ] sample
  - [ ] export

## Phase 5: Advanced Posterior Code

### `vi.py`

- [ ] Decide whether this is the primary Gaussian VI path
- [ ] Keep only one fit/update pattern
- [ ] Reduce duplicated MPI and sampling logic if shared with other VI modules

### `vi_precision.py`

- [ ] Fix broken references such as undefined `tril_bij`
- [ ] Decide whether this module stays active or is retired
- [ ] If retired, move it out of the main path

### `hvi.py`

- [ ] Keep only the parts needed for hierarchical VI
- [ ] Avoid duplicating generic Gaussian VI logic already present elsewhere
- [ ] Reuse shared save/load helpers instead of duplicating them in scripts

### `old_hvi.py`

- [ ] Move to an `experimental/` area or remove if obsolete

### `schur_claude.py`

- [ ] Move to an `experimental/` area or remove if obsolete

## Additional Cleanup Candidates

### `adjoint_utils.py`

- [ ] Review whether it is still part of active workflows
- [ ] Move to `experimental/` if it is only used for one-off derivative checks

### `adjoint_utils_mesh.py`

- [ ] Review whether it belongs in the active path or in diagnostics/experimental tooling

### `bayesopt.py`

- [ ] Check whether it still uses current config and field conventions
- [ ] Update or quarantine if it depends on deprecated naming

### `diagnostics.py`

- [ ] Keep if it remains part of active posterior diagnostics
- [ ] Consider adding a short module-level note about intended use

### `schur.py`

- [ ] Keep if it remains part of the main uncertainty toolkit
- [ ] Consider isolating matrix-free utilities from higher-level analysis helpers

### `schur_dense.py`

- [ ] Keep if the dense Schur workflow is actively used
- [ ] Otherwise move to a clearly secondary path

### `noise_utils.py`

- [ ] Fix or remove if it is stale
- [ ] Review for correctness before keeping it as active utility code

### `strain.py`

- [ ] Keep if it is still used for preprocessing or diagnostics
- [ ] Otherwise mark as supporting utility rather than core inverse code

## Final Pass

- [ ] Update `README.md` after the refactor settles
- [ ] Add a short `docs/ARCHITECTURE.md` later if the final structure warrants it
- [ ] Check that the golden-path scripts are the easiest scripts in the repo to read
