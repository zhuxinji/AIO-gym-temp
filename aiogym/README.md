# aiogym - native Gymnasium env for AIO-Gym

A fast, synchronous, **Gymnasium-first** re-implementation of the AIO-Gym plant
dynamics in numpy, built for RL **training** (the browser sim is for the
arena/demo and is too slow + too loosely-coupled to train against).

It is **provably consistent** with the browser physics: the ODE right-hand
sides, constants and RK4 integrator are a line-for-line port of
`frontend/js/sim/{models,kernel}.js`, and `tests/test_parity.py` checks the
native trajectories against golden checkpoints generated from the browser engine
(max delta <= 1e-15, i.e. bit-for-bit). So a policy trained here drops straight into
the browser AIO-Gym (export to ONNX, load in RL mode) and competes with PID/MPC.

## Install & verify
```bash
pip install -e .                                # core Gymnasium backend
pip install -e ".[oracle]"                      # add NMPC oracle support
pip install -e ".[train]"                       # add SB3/Torch training support
node scripts/generate_golden.mjs --check         # verify golden was generated from JS source
python aiogym/tests/test_parity.py               # prove consistency with the JS sim
```

## Use
```python
import gymnasium as gym, aiogym                   # registers the ids on import
env = gym.make("AIOGym/CSTR-v0")                  # Cascade / Quadruple / CSTR / HVAC
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step(env.action_space.sample())
```
Single env runs roughly 2k steps/s (pure python); wrap in `gymnasium.vector` for more.

## Contract (matches the browser RL contract; ONNX policies are interchangeable)
- `obs = [levels(n), temps(n), t_sp(n), h_sp(controlled k), t_cold, t_amb]`
- `action = [pumps..., valves..., heaters...] in [0,1]`  (direct-actuator mode)
- `reward_mode="track"` uses pure setpoint tracking: `reward = -tracking_error`.
  `reward_mode="kpi"` uses the composite tracking + excess-energy + safety score;
  `reward_mode="economic"` uses value minus energy and soft constraint penalties.
  `control_dt`, `episode_steps`, `randomize`, and `randomize_setpoints` are
  constructor args.

## Train
Stable backend entry points live under `aiogym.cli` and are also exposed as
console commands when installed:

```bash
aiogym-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-suite --suite core --episodes 3
aiogym-train-sb3 --scenario cstr --algo sac --steps 10000 --onnx
```

For the **offline-to-online** goal (RLPD / Cal-QL): generate a historian dataset
from this env, seed the replay buffer, keep exploring online. RLPD = SAC +
offline data in buffer + critic LayerNorm + critic ensemble + symmetric
sampling (official JAX impl: ikostrikov/rlpd).

## Roadmap (next, on top of this consistent core)
- **Supervisory setpoint mode** (`action = setpoints`, inner PID regulates): the
  RL-over-PID architecture; a thin wrapper, dynamics unchanged.
- **Realism layer** port (sensor deadtime/noise, actuator stiction) for sim-to-real.
- **Plant drift / aging** axis: the "RL re-training vs MPC re-identification"
  maintenance experiment.
- **Safety filter / predictive shield** wrapper for true online exploration.
