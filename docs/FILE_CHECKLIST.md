# File-By-File Checklist

This checklist is meant to be used alongside `docs/REFRACTOR_PLAN.md`.

It is organized by file and roughly grouped by phase. The intent is to make implementation easy to track without re-deriving the plan from scratch.

## Phase 1: Golden Path

### New helper module

- [x] Add a shared workflow helper module, `workflow_utils.py`
- [x] Centralize:
  - [x] config parsing
  - [x] calibration/discovery override handling
  - [x] `data_file` resolution
  - [x] output filename resolution
  - [x] restore-stage resolution
  - [x] object construction through `train_utilities.initialize_objects`

### `scripts/optimize_MAP_joint.py`

- [x] Remove duplicated setup logic that should live in the shared helper
- [x] Keep only MAP-specific logic in the script
- [x] Make restore behavior explicit and consistent with the shared helper
- [x] Keep callback logic small and focused on periodic save/checkpoint behavior

### `scripts/compute_hessian.py`

- [x] Fix missing imports and script bootstrap drift
- [x] Fix the broken `__main__` call
- [x] Replace duplicated setup logic with the shared helper
- [x] Reuse shared restore-stage and output-path logic
- [x] Keep only Hessian-specific behavior in the script

### `file_utilities.py`

- [x] Choose one results filename convention
- [x] Deprecate or remove `generate_filename()` if `make_results_filename()` is the standard
- [x] Keep restore-stage and results-path helpers aligned with the chosen convention

## Phase 2: Config Contract

### New config helper module

- [x] Add `validate_cfg(cfg)` or equivalent
- [x] Add readable error messages for missing required keys
- [x] Validate the fields needed by:
  - [x] `train_utilities.py`
  - [x] `fields.py`
  - [x] `problems.py`
  - [x] the golden-path scripts

### `config/default_config.toml`

- [x] Keep this as the explicit base config users can inspect for defaults
- [x] Update to reflect actual required core keys
- [x] Remove or mark keys that are legacy or currently unused
- [x] Document the minimum viable config for the core workflow

### `config/config_sinusoidal.toml`

- [x] Declare its base config explicitly with `extends = "default_config.toml"`
- [x] Bring naming into alignment with the validated config schema
- [x] Check that it still exercises a real supported path

### `fields.py`

- [x] Resolve `penalty_B` vs `penalties_B`
- [x] Resolve `penalty_n` vs `penalties_n`
- [x] Make the expected config fields obvious in one place

### `train_utilities.py`

- [x] Make required config fields explicit in `initialize_objects()`
- [x] Avoid hidden assumptions about optional config entries

## Phase 3: Core Interfaces

### `problems.py`

- [x] Replace dynamic rebinding of `self.forward` with explicit methods or small subclasses
- [x] Replace dynamic rebinding of `self.data_nll` with explicit methods or small subclasses
- [x] Fix `make_mesh()` so it does not depend on hidden `cfg`
- [x] Reset observation lists at the start of `load_observations()`
- [x] Replace mutable default arguments
- [ ] Review unused imports and dead code paths

### `objectives.py`

- [x] Simplify the control API
- [x] Replace implicit array/control conversions with explicit methods
- [x] Remove or fix `PosteriorObjective.update_controls()`
- [x] Decide whether `PosteriorObjectiveMesh` is still active
- [x] Remove dead or inactive objective code from the main path if not used

### `models.py`

- [x] Replace mutable default arguments in `create_solver()`
- [ ] Keep only solver variants that are still useful for active workflows
- [ ] Confirm that normalization helpers still match the active inverse path

## Phase 4: Field Layer And Hessian Path

### `fields.py`

- [x] Fix or remove `precondition_matrix()`
- [x] Fix or remove the unresolved `threshold_spectral_matrix()` dependency
- [x] Decide whether `RheologyFieldMatrix` is still needed
- [x] Keep one primary joint-field implementation for active workflows
- [x] Remove stale branches that are not used by the golden path

### `train_utilities.py`

- [x] Move non-training workflow helpers into a more clearly named module if useful
- [x] Keep `generate_predictions()` as the one shared export path
- [x] Remove unused helpers if they are no longer part of active workflows

### `scripts/compute_hessian.py`

- [x] Reuse `train_utilities.generate_predictions()` where possible
- [ ] Remove the unused dense covariance inverse if it is not needed
- [x] Precompute sampling factors once instead of inside the loop
- [x] Keep the Hessian workflow readable:
  - [x] load MAP
  - [x] compute or read Hessian
  - [x] sample
  - [x] export

## Phase 5: Advanced Posterior Code

### `vi.py`

- [x] Keep this as the primary Gaussian VI path
- [x] Keep only one fit/update pattern
- [x] Reduce duplicated MPI and sampling logic if shared with other VI modules

### `hvi.py`

- [x] Keep only the parts needed for hierarchical VI
- [x] Avoid duplicating generic Gaussian VI logic already present elsewhere
- [x] Reuse shared save/load helpers instead of duplicating them in scripts

### Retired precision path

- [x] Remove `vi_precision.py`
- [x] Remove precision-only driver scripts

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

- [x] Update `README.md` after the refactor settles
- [ ] Add a short `docs/ARCHITECTURE.md` later if the final structure warrants it
- [ ] Check that the golden-path scripts are the easiest scripts in the repo to read
