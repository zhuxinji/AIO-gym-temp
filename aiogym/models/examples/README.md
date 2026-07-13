# AIO-Gym Model Examples

These examples are small templates for extending the process-model layer without
editing built-in model source.

Run them from the repository root after installing the backend:

```bash
pip install -e ./aiogym
python aiogym/models/examples/declarative_model.py
python aiogym/models/examples/custom_model.py
```

`declarative_model.py` is the recommended custom-scenario path: define states,
actions, parameters, dynamics formulas, and outputs in one dictionary. AIO-Gym
generates the model contract and oracle-compatible dynamics automatically.

`custom_model.py` shows the lower-level class API for cases that need custom
Python methods, disturbances, constraints, or process-specific helpers.
