# Refactor Plan

This document captures a concrete refactor plan for the core `schurflow` codebase.

The main goal is not only to make the code faster. The primary goal is to make it easier to return to after 1-2 months away and quickly recover the main workflow, assumptions, and extension points.

Two scripts define the "golden path" for this refactor:

- `scripts/optimize_MAP_joint.py`
- `scripts/compute_hessian.py`

After each phase, those two workflows should still be understandable and runnable.

## Principles

- Prefer one obvious way to do something over several partially overlapping ways.
- Make hidden contracts explicit, especially config requirements and file naming.
- Keep the forward-model path easy to read: `fields -> problem -> objective -> workflow`.
- Move experimental code out of the main path when it adds cognitive load.
- Treat maintainability improvements as at least as important as runtime improvements.

## Phase 1: Stabilize The Golden Path

Goal: make the MAP and Hessian workflows consistent and remove obvious script drift.

Scope:

- create one shared bootstrap/helper path for:
  - parsing config
  - applying calibration/discovery overrides
  - resolving `data_file`
  - resolving output filename
  - resolving restore stage
  - constructing `(field, problem, objective, converter)`
- update `scripts/optimize_MAP_joint.py` to use that shared helper
- update `scripts/compute_hessian.py` to use that shared helper
- fix script correctness issues in `scripts/compute_hessian.py`
- standardize on one results filename convention

Expected outcomes:

- less duplicated setup logic across scripts
- fewer chances for scripts to drift apart
- one obvious pattern for future scripts

## Phase 2: Make The Config Contract Explicit

Goal: stop relying on implicit config knowledge scattered across modules.

Scope:

- add a small config validation layer
- check required keys before any expensive initialization
- resolve mismatches such as:
  - `penalty_B` vs `penalties_B`
  - `penalty_n` vs `penalties_n`
- expand or clean up the default config so it reflects the actual required core inputs

Expected outcomes:

- faster failure when config is incomplete or inconsistent
- easier re-entry into the project
- easier scripting because required inputs are visible in one place

## Phase 3: Simplify Core Interfaces

Goal: make the main inversion path readable without tracing dynamic behavior.

Scope:

- simplify `problems.py` so key behavior is explicit rather than assigned dynamically
- make `make_mesh()` explicit about its required inputs
- simplify the objective/control interface in `objectives.py`
- remove or quarantine inactive code paths if they are not part of the current workflow

Expected outcomes:

- easier to understand the forward/inverse stack
- fewer implicit side effects
- clearer extension points for future experiments

## Phase 4: Reduce Field-Layer And Hessian Complexity

Goal: reduce cognitive load in the matrix/field layer and keep Hessian sampling straightforward.

Scope:

- fix or remove stale/broken code in `fields.py`
- decide which field implementation is the primary one and remove ambiguity
- simplify the Hessian sampling/export path
- reuse shared prediction-generation logic instead of reimplementing it in scripts

Expected outcomes:

- fewer dead paths
- less duplicated logic
- simpler Hessian workflow with fewer moving parts

## Phase 5: Consolidate Advanced Posterior Code

Goal: reduce the number of overlapping VI implementations in active use.

Scope:

- choose one primary Gaussian VI implementation
- keep one hierarchical extension if needed
- move stale or experimental variants out of the main path
- reduce duplicated MPI, fitting, and sampling logic

Expected outcomes:

- one obvious variational path
- less drift between advanced modules
- lower maintenance cost

## Suggested Order Of Work

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

This order is intentional. It fixes the workflow entry point first, then the config contract, then the deeper internal abstractions.

## Definition Of Done

The refactor is successful when the following are true:

- a new script can follow one shared bootstrap pattern
- config requirements are validated early and documented clearly
- the core workflow can be understood by reading:
  - `fields.py`
  - `problems.py`
  - `objectives.py`
  - the script helper layer
- `scripts/optimize_MAP_joint.py` and `scripts/compute_hessian.py` no longer carry large amounts of duplicated setup logic
- there is one obvious filename convention for MAP and downstream products
- inactive or experimental variants are clearly separated from the main path
