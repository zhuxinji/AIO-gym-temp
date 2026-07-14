# Johansson quadruple-tank process

The `quadruple` scenario implements the nonlinear physical model introduced by
Karl H. Johansson. It has four liquid-level states, two pump-voltage inputs, and
the two lower-tank levels as controlled outputs. The previous temperature and
heater extension has been removed.

## Equations

With levels (h_i), tank areas (A_i), outlet areas (a_i), pump gains (k_i),
physical pump voltages (v_i), valve splits \(\gamma_i\), and gravity (g):

\[
\dot h_1 = \frac{-a_1\sqrt{2gh_1}+a_3\sqrt{2gh_3}+\gamma_1k_1v_1}{A_1}
\]

\[
\dot h_2 = \frac{-a_2\sqrt{2gh_2}+a_4\sqrt{2gh_4}+\gamma_2k_2v_2}{A_2}
\]

\[
\dot h_3 = \frac{-a_3\sqrt{2gh_3}+(1-\gamma_2)k_2v_2}{A_3},\qquad
\dot h_4 = \frac{-a_4\sqrt{2gh_4}+(1-\gamma_1)k_1v_1}{A_4}.
\]

The implementation keeps the reference units: cm, cm², cm³/s, V, and s.
AIO-Gym actions (u_i\in[0,1]) map to voltage through (v_i=10u_i).

The minimum-phase condition is \(\gamma_1+\gamma_2>1\). The P− task uses
\((\gamma_1,\gamma_2)=(0.70,0.60)\), while the P+ task uses
\((0.43,0.34)\) and has a right-half-plane transmission zero. The latter
creates inverse-response limitations and a much slower achievable closed-loop
response; it is not merely a different controller tuning.

## Reference parameters

| Quantity | Values | Unit |
|---|---:|---|
| (A_1,A_2,A_3,A_4) | 28, 32, 28, 32 | cm² |
| (a_1,a_2,a_3,a_4) | 0.071, 0.057, 0.071, 0.057 | cm² |
| (k_1,k_2) | 3.33, 3.35 | cm³/(V·s) |
| \(\gamma_1,\gamma_2\) | 0.70, 0.60 | – |
| (g) | 981 | cm/s² |

The paper reports the P− operating point as
\((12.4,12.7,1.8,1.4)\) cm at \((3,3)\) V. Because the reported parameters and
levels are rounded experimental values, they do not form an exact equilibrium of
the nonlinear equations. The simulator starts from the model-consistent
equilibrium \((12.26297,12.78316,1.63394,1.40904)\) cm while retaining the
reported point in the parameter profile.

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
    task="minimum-phase-classic",
)
```

The task uses 1 s control intervals for 600 steps. It starts at the exact 3 V
equilibrium, applies an opposed 1 cm setpoint move at step 120, and reverses to
an asymmetric target at step 360.

The bundled PI controller uses Johansson's minimum-phase tuning, converted from
physical sensor and voltage units to AIO-Gym's cm outputs and normalized actions.

Run the complete PID/MPC/NMPC comparison with:

```bash
aiogym-suite-benchmark --suite quadruple-classic --episodes 1
```

## Nonminimum-phase task

`nonminimum-phase-classic` changes the valve splits, pump gains, nominal pump
voltages, exact equilibrium, setpoint experiment, and horizon as one coherent
task. It runs for 1800 s because the P+ response reported by Johansson is much
slower. The bundled PID profile uses the paper's P+ decentralized PI settings;
the MPC profile uses a longer prediction horizon.

```python
env = aiogym.make_env(
    "quadruple",
    objective="tracking",
    task="nonminimum-phase-classic",
)
```

Run it alone or compare both physical configurations:

```bash
aiogym-suite-benchmark --suite quadruple-nonminimum --episodes 1
aiogym-suite-benchmark --suite quadruple-phase-comparison --episodes 1
```

The comparison suite produces separate leaderboards for the two tasks. A lower
tracking cost in P− and a lower tracking cost in P+ are two distinct ranking
claims; the tool does not rank them against each other.

## Paper-reference steps

The `classic` tasks above are useful AIO-Gym benchmarks, but their setpoint
schedules are not literal copies of the paper figures. The strict reference
suite adds two separate tasks:

- `pminus-reference-step`: the Fig. 10 P− experiment, 360 s;
- `pplus-reference-step`: the Fig. 11 P+ experiment, 3600 s.

Both apply the plotted 1 V step in `r1` at time zero. With the paper's
`kc=0.5 V/cm` sensor gain this is represented as a 2 cm `h1` step. The initial
states are exact nonlinear equilibria for the rounded paper parameters; the
reported experimental operating points remain in task metadata.

```bash
aiogym-suite-benchmark --suite quadruple-paper-reference --episodes 1
```

## Zero-boundary stress

`zero-boundary-stress` uses `gamma1=0.55` and `gamma2=0.45`. The adjustable
zero is at the origin and the steady-state input/output map is singular. An
independent 0.5 cm `h1` demand therefore tests offset, integral windup, slow
response, and constraint handling close to the fundamental limit.

```bash
aiogym-suite-benchmark --suite quadruple-zero-boundary --episodes 1
```

## Deterministic disturbance rejection

Task disturbance events are executable runtime conditions, not descriptive
metadata. `disturbance-rejection` applies a 20% common pump-capacity loss,
restores it, then applies and restores a 20% outlet restriction. Controllers
observe the changed disturbance values after they occur, and the exact schedule
is recorded in benchmark artifacts.

```bash
aiogym-suite-benchmark --suite quadruple-disturbance-rejection --episodes 1
```

## Acceptance tests

The scenario-specific test suite verifies:

- exact reference parameters, both operating points, and phase classification;
- left- versus right-half-plane transmission-zero classification;
- nonlinear equilibrium residual below (10^{-12});
- total four-tank volumetric balance below (10^{-12}) cm³/s;
- correct diagonal pump routing;
- RK4 0.1 s versus 0.05 s transient agreement;
- deterministic task scheduling and task-specific PI bias smoke tests;
- executable task disturbance schedules and recorded provenance;
- zero-at-origin classification and equilibrium consistency.
