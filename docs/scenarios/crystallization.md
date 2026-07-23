# Batch Crystallization (`crystallization`)

Single-actuator crystallization benchmark with moment dynamics and Ln/CV quality targets.

## Recommended Use

PC-Gym-style crystallization tracking and single-actuator quality control

## Dynamics And Assumptions

Moment dynamics and concentration balance map cooling-temperature action to crystal-size quality outputs such as Ln and CV.

- The actuator is a normalized cooling-temperature command.
- Growth, nucleation, and solubility terms are benchmark-scale simplifications.
- Quality metrics are exposed through process info and tracking outputs.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## Physical Metadata Status

- Parameter status: `legacy-unverified`
- Fidelity: `legacy-benchmark`
- Time unit: `s`
- Solver: `rk4` with maximum step `0.02`

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| mu0 | moment | [0, 10000] |
| mu1 | moment | [0, 1e+06] |
| mu2 | moment | [0, 1e+08] |
| mu3 | moment | [0, 1e+10] |
| c | kg/L | [0, 2] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| cooling_temperature_fraction | heater | 0 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| coefficient_variation | dimensionless | [0, 5] |
| mean_crystal_size | um | [0, 50] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| coefficient_variation | coefficient_variation | dimensionless | [0, 5] |
| mean_crystal_size | mean_crystal_size | um | [0, 50] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| growth_factor | growth_factor_step | multiplier | [0.7, 1.3] | 1 |
| nucleation_factor | nucleation_factor_step | multiplier | [0.7, 1.3] | 1 |
| solubility_bias | solubility_bias_step | concentration | [-10, 10] | 0 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| B0_max | 0.05 | 1/s | [0, 1] |
| CV_sp | 1 | dimensionless | [0, 5] |
| Ginf_max | 0.0002 | um/s | [0, 0.01] |
| Ginf_scale | 2e-06 | dimensionless | [0, 0.001] |
| Ln_sp | 15 | um | [0, 50] |
| Tc_max | 40 | degC | [0, 100] |
| Tc_min | 30 | degC | [0, 100] |
| Tc_nom | 35 | degC | [0, 100] |
| a | 0.51 | dimensionless | [0, 2] |
| alpha | 7.5 | dimensionless | [0, 20] |
| b | 7.3 | dimensionless | [0, 20] |
| eps | 1e-09 | dimensionless | [1e-12, 0.001] |
| k1 | -4900 | K | [-20000, 0] |
| k2 | 1.9 | dimensionless | [0, 5] |
| ka | 0.92 | 1/s | [0, 10] |
| kb | -6800 | K | [-20000, 0] |
| kc | 0.92 | dimensionless | [0, 5] |
| kd | 1.3 | dimensionless | [0, 5] |
| kg | 48 | 1/s | [0, 200] |
| rho | 2.7 | kg/L | [0, 10] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| moment_nonnegative | mu0, mu1, mu2, mu3, c | [0, none] |
| cooling_temperature_bounds | Tc | [30, 40] |
| quality_targets | Ln, CV | [none, none] |

## Benchmark Semantics

- State vector length: 5
- Action vector length: 1
- Controlled output vector length: 2
- Setpoint vector length: 2
- Dynamics disturbances: [growth_factor, nucleation_factor, solubility_bias]
- Micro integration step: 0.02 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| Ginf_scale | [0.7, 1.3] |
| ka | [0.7, 1.3] |
| kg | [0.7, 1.3] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [[none, none]] |
| temp_band | [[none, none]] |
| value | none |
| w_energy | 0 |
| w_value | 0 |
| w_viol | 0 |
