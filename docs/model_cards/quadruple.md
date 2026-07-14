# Johansson quadruple-tank process (`quadruple`)

Four-state, two-input nonlinear level process with adjustable minimum/nonminimum-phase behavior.

## Recommended Use

classic two-input/two-output MIMO level control, interaction analysis, and minimum/nonminimum-phase studies

## Dynamics And Assumptions

Johansson's four nonlinear liquid-level balances route two voltage-driven pump streams through fixed three-way valve splits; upper tanks drain into the two lower tanks.

- Liquid is incompressible and each outlet follows Torricelli flow.
- Tank cross-sections, outlet areas, pump gains, and valve split fractions are constant within one run.
- Pipe, sensor, and pump dynamics are neglected as in the reference nonlinear physical model.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## Physical Metadata Status

- Parameter status: `reference-parameterized`
- Fidelity: `first-principles-laboratory-benchmark`
- Time unit: `s`
- Solver: `rk4` with maximum step `0.1`

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| h1 | cm | [0, 20] |
| h2 | cm | [0, 20] |
| h3 | cm | [0, 20] |
| h4 | cm | [0, 20] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| pump_1_voltage | pump | 0 | [0, 1] |
| pump_2_voltage | pump | 1 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| lower_tank_1_level | cm | [0, 20] |
| lower_tank_2_level | cm | [0, 20] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| h1_sp | lower_tank_1_level | cm | [0, 20] |
| h2_sp | lower_tank_2_level | cm | [0, 20] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| pump_flow_factor | pump_capacity_shift | fraction | [0.7, 1.3] | 1 |
| outlet_area_factor | outlet_restriction_shift | fraction | [0.7, 1.3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| gamma | [0.7, 0.6] | fraction | [0.01, 0.99] |
| gravity | 981 | cm/s2 | [900, 1100] |
| max_level | 20 | cm | [1, 100] |
| max_voltage | 10 | V | [0.1, 20] |
| nominal_voltage | [3, 3] | V | [0, 20] |
| outlet_area | [0.071, 0.057, 0.071, 0.057] | cm2 | [0.0001, 10] |
| pump_gain | [3.33, 3.35] | cm3/(V*s) | [0.0001, 100] |
| tank_area | [28, 32, 28, 32] | cm2 | [1, 1000] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| tank_level_bounds | h1, h2, h3, h4 | [0, 20] |

## Benchmark Semantics

- State vector length: 4
- Action vector length: 2
- Controlled output vector length: 2
- Setpoint vector length: 2
- Dynamics disturbances: [pump_flow_factor, outlet_area_factor]
- Micro integration step: 0.1 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| outlet_area | [0.9, 1.1] |
| pump_gain | [0.9, 1.1] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [[0, 20], [0, 20]] |
| temp_band | [] |
| value | none |
| w_energy | 0 |
| w_value | 0 |
| w_viol | 30 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [y_sp, 0, 5, 18] |
| 1 | [y_sp, 1, 5, 18] |
