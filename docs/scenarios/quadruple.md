# Johansson quadruple-tank process

The `quadruple` scenario implements the nonlinear physical model introduced by
Karl H. Johansson. It has four liquid-level states, two pump-voltage inputs, and
the two lower-tank levels as controlled outputs. The previous temperature and
heater extension has been removed.

## Equations

With levels $h_i$, tank areas $A_i$, outlet areas $a_i$, pump gains $k_i$,
physical pump voltages $v_i$, valve splits $\gamma_i$, and gravity $g$:

$$
\dot h_1 = \frac{-a_1\sqrt{2gh_1}+a_3\sqrt{2gh_3}+\gamma_1k_1v_1}{A_1}
$$

$$
\dot h_2 = \frac{-a_2\sqrt{2gh_2}+a_4\sqrt{2gh_4}+\gamma_2k_2v_2}{A_2}
$$

$$
\dot h_3 = \frac{-a_3\sqrt{2gh_3}+(1-\gamma_2)k_2v_2}{A_3},\qquad
\dot h_4 = \frac{-a_4\sqrt{2gh_4}+(1-\gamma_1)k_1v_1}{A_4}.
$$

The implementation keeps the reference units: cm, cm², cm³/s, V, and s.
AIO-Gym actions $u_i\in[0,1]$ map to voltage through $v_i=10u_i$.

The minimum-phase condition is $\gamma_1+\gamma_2>1$. The P− task uses
$(\gamma_1,\gamma_2)=(0.70,0.60)$, while the P+ task uses
$(0.43,0.34)$ and has a right-half-plane transmission zero. The latter
creates inverse-response limitations and a much slower achievable closed-loop
response; it is not merely a different controller tuning.

## Reference parameters

| Quantity | Values | Unit |
|---|---:|---|
| $A_1,A_2,A_3,A_4$ | 28, 32, 28, 32 | cm² |
| $a_1,a_2,a_3,a_4$ | 0.071, 0.057, 0.071, 0.057 | cm² |
| $k_1,k_2$ | 3.33, 3.35 | cm³/(V·s) |
| $\gamma_1,\gamma_2$ | 0.70, 0.60 | – |
| $g$ | 981 | cm/s² |

The paper reports the P− operating point as
$(12.4,12.7,1.8,1.4)$ cm at $(3,3)$ V. Because the reported parameters and
levels are rounded experimental values, they do not form an exact equilibrium of
the nonlinear equations. The simulator starts from the model-consistent
equilibrium $(12.26297,12.78316,1.63394,1.40904)$ cm while retaining the
reported point in the parameter profile.

For P+, the paper reports $(12.6,13.0,4.8,4.9)$ cm at $(3.15,3.15)$ V.
The corresponding model-consistent equilibrium used by AIO-Gym is
$(12.44186,13.16681,4.73026,4.98633)$ cm.

Primary reference: K. H. Johansson, “The Quadruple-Tank Process: A
Multivariable Laboratory Process with an Adjustable Zero,” IEEE Transactions on
Control Systems Technology, 8(3), 456–465, 2000,
[doi:10.1109/87.845876](https://doi.org/10.1109/87.845876).

## Minimum-phase task

Run the deterministic minimum-phase tracking task with:

```python
import aiogym

env = aiogym.make_env(
    "quadruple",
    objective="tracking",
    task="minimum-phase",
)
```

The task uses 1 s control intervals for 600 steps. It starts at the exact 3 V
equilibrium, applies an opposed 1 cm setpoint move at step 120, and reverses to
an asymmetric target at step 360.

The paper-reference PI profile uses Johansson's minimum-phase tuning, converted
from physical sensor and voltage units to AIO-Gym's cm outputs and normalized
actions. The general `quadruple` benchmark uses a separate family-tuned PID
profile, so improving the baseline does not alter the paper reproduction.
The successive-linearization MPC uses a short task-validated prediction horizon
for the minimum-phase response; the longer horizon previously used here made its
single-move approximation unnecessarily sluggish.

Run the complete PID/MPC/NMPC comparison with:

```bash
aiogym benchmark suite --suite quadruple-classic --episodes 1
```

## Nonminimum-phase task

`nonminimum-phase` changes the valve splits, pump gains, nominal pump
voltages, exact equilibrium, setpoint experiment, and horizon as one coherent
task. It runs for 1800 s because the P+ response reported by Johansson is much
slower. The paper-reference PID profile uses the paper's P+ decentralized PI
settings; the benchmark-tuned PID uses cross pairing for this nonminimum-phase
plant because it performed substantially better on the P+ benchmark. The
MPC profile combines a longer prediction horizon with a model-derived
steady-state pump target so the initial inverse response does not send the pump
allocation in the wrong long-term direction.

```python
env = aiogym.make_env(
    "quadruple",
    objective="tracking",
    task="nonminimum-phase",
)
```

Run it alone or compare both physical configurations:

```bash
aiogym benchmark suite --suite quadruple-nonminimum --episodes 1
aiogym benchmark suite --suite quadruple-phase-comparison --episodes 1
```

The comparison suite produces separate leaderboards for the two tasks. A lower
tracking error cost in P− and a lower tracking error cost in P+ are two distinct ranking
claims; the tool does not rank them against each other.

Run all four formal quadruple-tank tasks with:

```bash
aiogym benchmark suite --suite quadruple --episodes 1
```

The `quadruple` suite runs PID, MPC, and NMPC Oracle on every formal task.
Rankings remain separate for each task and objective. Every tracking case also records the first
evaluation seed as a rollout and writes one state/setpoint/actuator control chart
per task under the run's `figures/tracking_control_*.svg` files. The tracking
comparison table includes separate Scenario and Task columns.

Tracking leaderboards use cumulative raw squared output error in cm²
(`tracking_error_cost`) as the primary metric. The regularized `tracking_cost`
uses physical pump-voltage differences in V² for its move/steady-input
components. These metrics are summed over all control steps and channels; they
are not divided by the episode length, number of outputs, output range, or
actuator range. The components remain secondary diagnostics, so a
controller cannot win the tracking ranking merely by moving less while tracking
the controlled outputs poorly.

The NMPC Oracle uses task-family profiles rather than one generic tuning:
minimum phase, nonminimum phase, zero-boundary stress, and disturbance rejection
have separate prediction horizons and solve frequencies. To keep
the benchmark computationally practical, they currently re-solve every 1, 10, 1, and 2
control steps, respectively (every 1, 10, 1, and 2 s). A setpoint or explicit
disturbance change triggers an
immediate re-solve even between those periodic updates. The actions between solves
are replayed from the optimized plan instead of holding only its first action. For
tracking cases, raw output error is both the internal objective and the reported
primary metric. Minimum-phase and zero-boundary Oracle profiles use no move,
steady-input, or terminal tracking regularization. The nonminimum-phase MPC and
Oracle profiles impose no move-rate limit. Both use the model's exact nonlinear
steady-state inverse only to initialize the first optimization after reset or a
setpoint change; it is not part of the final objective. At the 1 s control
interval, the nonminimum-phase MPC and Oracle horizons are 60 and 180 steps
(60 s and 180 s), respectively. Minimum-phase and zero-boundary Oracle horizons
are 4 steps; the disturbance-rejection horizon is 3 steps. These horizons are tuned
profile values, not framework limits; controller construction accepts any
positive horizon. Feasibility slack remains a constrained-solver safeguard.
Oracle and the environment both integrate every 1 s control interval using
10 RK4 substeps of at most 0.1 s.
Setpoint preview is disabled in the standard benchmark. For an explicitly
noncausal upper-bound experiment, Oracle accepts `preview_setpoints=true` and
uses deterministic task setpoint events that enter its prediction horizon.
State/input constraints remain optimization safeguards outside the reported
error cost. The disturbance-rejection case retains a
robustness-specific objective because its primary metric is `normalized_score`,
not `tracking_error_cost`. These profiles are cost-bounded benchmark baselines, not the
paper-reference PI parameters.

PID follows the same separation: the main suite uses four fixed benchmark
profiles (minimum phase, nonminimum phase, zero boundary, and disturbance
rejection), while `quadruple-paper-reference` retains the two converted paper PI
profiles. The reproducible search entrypoint is:

```bash
python -m aiogym.controllers.tuning.tune_quadruple_pid \
  --family minimum-phase --out /tmp/quadruple_pid_minimum-phase.json
```

## Paper-reference steps

The default task schedules above are useful AIO-Gym benchmarks, but they are not
literal copies of the paper figures. The reference suite reuses the two phase
tasks with explicit custom timing and setpoint overrides:

- P− Fig. 10: `minimum-phase`, 360 s;
- P+ Fig. 11: `nonminimum-phase`, 3600 s.

Both apply the plotted 1 V step in `r1` at time zero. With the paper's
`kc=0.5 V/cm` sensor gain this is represented as a 2 cm `h1` step. The initial
states are exact nonlinear equilibria for the rounded paper parameters; the
reported experimental operating points remain in the archived reference
configuration files.

```bash
aiogym benchmark suite --suite quadruple-paper-reference --episodes 1
```

This suite contains only the two custom paper-reference runs and the paper's
decentralized PI controller. Its control charts show the two controlled lower
tank levels and two pump voltages, matching the four simulated signals in each
paper figure rather than adding the two unmeasured upper-tank states.

## Zero-boundary stress

`zero-boundary-stress` uses `gamma1=0.55` and `gamma2=0.45`. The adjustable
zero is at the origin and the steady-state input/output map is singular. An
independent 0.5 cm `h1` demand therefore tests offset, integral windup, slow
response, and constraint handling close to the fundamental limit.

```bash
aiogym benchmark suite --suite quadruple-zero-boundary --episodes 1
```

## Deterministic disturbance rejection

Task disturbance events are executable runtime conditions, not descriptive
metadata. `disturbance-rejection` applies a 20% common pump-capacity loss,
restores it, then applies and restores a 20% outlet restriction. Controllers
observe the changed disturbance values after they occur, and the exact schedule
is recorded in benchmark artifacts.

```bash
aiogym benchmark suite --suite quadruple-disturbance-rejection --episodes 1
```

## Acceptance tests

The scenario-specific test suite verifies:

- exact reference parameters, both operating points, and phase classification;
- left- versus right-half-plane transmission-zero classification;
- nonlinear equilibrium residual below $10^{-12}$;
- total four-tank volumetric balance below $10^{-12}$ cm³/s;
- correct diagonal pump routing;
- RK4 0.1 s versus 0.05 s transient agreement;
- deterministic task scheduling and task-specific PI bias smoke tests;
- executable task disturbance schedules and recorded provenance;
- zero-at-origin classification and equilibrium consistency.
