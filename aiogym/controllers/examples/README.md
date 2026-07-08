# AIO-Gym Controller Examples

These examples are small templates for extending the controller layer without
editing built-in controller source.

Run them from the repository root after installing the backend:

```bash
pip install -e ./aiogym
python aiogym/controllers/examples/custom_controller.py
```

`custom_controller.py` shows how to implement and register a controller that
uses the unified `ControllerContext` and can be evaluated through the same
benchmark protocol as PID, MPC, oracle, and SB3 policies.
