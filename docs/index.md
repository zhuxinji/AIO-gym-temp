# AIO-Gym documentation

This is the starting point for AIO-Gym's user and contributor documentation.
The pages are grouped by audience so that current usage guidance is not mixed
with generated reference material or historical implementation records.

## Installed-package access

Package metadata publishes Homepage, Documentation, Repository, and Issues
links. The canonical online entry point is this
[documentation index](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/index.md).
Wheel installations intentionally contain runtime code and package data rather
than a duplicate `docs/` tree; source distributions include the documentation
for offline inspection. Historical plans and audits in a source distribution
retain the snapshot limitations stated below.

## Start here

- [Repository README](../README.md) — installation, quick start, benchmark
  commands, training, and validation.
- [Concepts](concepts.md) — how scenario, task, objective, controller, action
  mode, suite, and artifact fit together.
- [Architecture](architecture.md) — source layout, layer ownership, runtime data
  flow, and extension points.
- [Capability matrix](capabilities.md) — current built-in scenarios, formal
  tasks, objective declarations, action modes, and controller entry points.
- [Public API guide](public_api.md) — supported Python and command-line entry
  points for environments and benchmarks.

## Models and experiments

- [Physical-model infrastructure](model_infrastructure.md) — model contracts,
  parameter provenance, numerical readiness, and the acceptance gate for new
  physical models.
- [Task-profile reference](../aiogym/models/tasks/builtin/README.md) — versioned
  experiment declarations, objective resolution, and task-owned operation
  semantics.

The `docs/scenarios/` directory is the single documentation entry point for
all built-in process models:

- [Heated-tank cascade](scenarios/cascade.md)
- [Recirculating heated-tank cascade](scenarios/cascade_recirculating.md)
- [Johansson quadruple-tank process](scenarios/quadruple.md)
- [Exothermic CSTR](scenarios/cstr.md)
- [Two-zone HVAC](scenarios/hvac.md)
- [Multistage extraction column](scenarios/extraction.md)
- [Fired heater](scenarios/heater.md)
- [Batch crystallization](scenarios/crystallization.md)

## Validation and research records

The files under `docs/reports/` record model-specific validation evidence.
Their header determines whether a report is current guidance or a historical
snapshot. Snapshot conclusions and hashes apply only to the date, Git baseline,
and working-tree state stated in that report.

- [Open-cascade accuracy and authenticity snapshot](reports/cascade-model-accuracy-authenticity.md)
- [Recirculating-cascade accuracy and authenticity snapshot](reports/cascade-recirculating-accuracy-authenticity.md)

The adjacent JSON files are machine-readable evidence from the same snapshots;
they are not a live catalog of current tasks. Historical implementation plans
are retained in Git history rather than published as current documentation.
