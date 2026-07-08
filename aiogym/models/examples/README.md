# AIO-Gym Model Examples

These examples are small templates for extending the process-model layer without
editing built-in model source.

Run them from the repository root after installing the backend:

```bash
pip install -e ./aiogym
python aiogym/models/examples/custom_model.py
```

`custom_model.py` shows how to define a process model with metadata,
disturbances, constraints, and a reusable `register_model(...)` call.
