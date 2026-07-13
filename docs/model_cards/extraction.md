# Multistage Extraction Column (`extraction`)

Five-stage counter-current liquid-gas extraction column from the PC-Gym benchmark.

## Recommended Use

multi-stage concentration tracking and PC-Gym-style controller comparisons

## Dynamics And Assumptions

A five-stage counter-current extraction column updates liquid and gas concentrations using simplified stage-transfer balances.

- The column follows the PC-Gym-style benchmark abstraction rather than a detailed tray model.
- Concentrations are bounded to a normalized operating interval.
- Flow actions and mass-transfer coefficients are scaled benchmark parameters.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| CX1 | concentration | [0, 1] |
| CY1 | concentration | [0, 1] |
| CX2 | concentration | [0, 1] |
| CY2 | concentration | [0, 1] |
| CX3 | concentration | [0, 1] |
| CY3 | concentration | [0, 1] |
| CX4 | concentration | [0, 1] |
| CY4 | concentration | [0, 1] |
| CX5 | concentration | [0, 1] |
| CY5 | concentration | [0, 1] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| liquid_flow_L | pump | 0 | [0, 1] |
| gas_flow_G | pump | 1 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| stage_1_liquid_concentration | concentration | [0, 1] |
| stage_2_liquid_concentration | concentration | [0, 1] |
| stage_3_liquid_concentration | concentration | [0, 1] |
| stage_4_liquid_concentration | concentration | [0, 1] |
| stage_5_liquid_concentration | concentration | [0, 1] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| stage_1_liquid_concentration | stage_1_liquid_concentration | concentration | [0, 1] |
| stage_2_liquid_concentration | stage_2_liquid_concentration | concentration | [0, 1] |
| stage_3_liquid_concentration | stage_3_liquid_concentration | concentration | [0, 1] |
| stage_4_liquid_concentration | stage_4_liquid_concentration | concentration | [0, 1] |
| stage_5_liquid_concentration | stage_5_liquid_concentration | concentration | [0, 1] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| CX0 | feed_liquid_concentration_step | concentration | [0.3, 0.9] | 0.6 |
| CY6 | gas_feed_concentration_step | concentration | [0, 0.12] | 0.05 |
| Kla | mass_transfer_shift | 1/s | [2, 8] | 5 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| CX0 | 0.6 | concentration | [0, 1] |
| CY6 | 0.05 | concentration | [0, 1] |
| Gmax | 1000 | flow | [0, 2000] |
| Gmin | 500 | flow | [0, 2000] |
| Kla | 5 | 1/s | [0.1, 20] |
| Lmax | 10 | flow | [0, 100] |
| Lmin | 5 | flow | [0, 100] |
| Vg | 5 | volume | [1, 20] |
| Vl | 5 | volume | [1, 20] |
| c_max | 1 | concentration | [0, 10] |
| e | 2 | dimensionless | [0.1, 5] |
| m | 1 | dimensionless | [0.01, 20] |
| pump_power_max | [1000, 1000] | W | [0, 10000] |
| t_amb | 0 | degC | [0, 45] |
| t_cold | 0 | degC | [0, 40] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| concentration_bounds | CX1, CY1, CX2, CY2, CX3, CY3, CX4, CY4, CX5, CY5 | [0, 1] |
| cx5_quality_band | CX5 | [0.0035, 0.0048] |

## Benchmark Semantics

- State vector length: 10
- Action vector length: 2
- Controlled output vector length: 5
- Setpoint vector length: 5
- Dynamics disturbances: [CX0, CY6, Kla]
- Micro integration step: 0.01 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| CX0 | [0.75, 1.25] |
| CY6 | [0.75, 1.25] |
| Kla | [0.6, 1.5] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [] |
| temp_band | [[none, none], [none, none], [none, none], [none, none], [0.0035, 0.0048]] |
| value | none |
| w_energy | 0 |
| w_value | 0 |
| w_viol | 25 |
