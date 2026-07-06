"""Control algorithm implementations for AIO-Gym.

Import concrete controllers from their modules, for example
``aiogym.control.baselines`` or ``aiogym.control.oracle``. Keeping this package
initializer lightweight avoids circular imports with the env.
"""

__all__ = ["baselines", "mpc_grid", "oracle"]
