# Johansson quadruple tank (`quadruple`)

Four-tank interacting level process with a thermal extension.

## Recommended Use

interacting level control, MIMO setpoint tracking, and robustness tests

## Dynamics And Assumptions

Four interacting tank balances route two pump streams through upper and lower tanks, with a thermal state attached to each tank.

- Hydraulic coupling is simplified to benchmark-scale orifice flow.
- Thermal dynamics share lumped heater and heat-loss terms.
- The model is intended for control comparison rather than hardware sizing.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| h0 | m | [0, 0.8] |
| T0 | degC | [0, 120] |
| h1 | m | [0, 0.8] |
| T1 | degC | [0, 120] |
| h2 | m | [0, 0.8] |
| T2 | degC | [0, 120] |
| h3 | m | [0, 0.8] |
| T3 | degC | [0, 120] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| pump_0 | pump | 0 | [0, 1] |
| pump_1 | pump | 1 | [0, 1] |
| heater_0 | heater | 2 | [0, 1] |
| heater_1 | heater | 3 | [0, 1] |
| heater_2 | heater | 4 | [0, 1] |
| heater_3 | heater | 5 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| lower_tank_0_level | m | [0, 0.8] |
| lower_tank_1_level | m | [0, 0.8] |
| tank_0_temperature | degC | [25, 72] |
| tank_1_temperature | degC | [25, 72] |
| tank_2_temperature | degC | [20, 58] |
| tank_3_temperature | degC | [20, 58] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| lower_tank_0_level | lower_tank_0_level | m | [0, 0.8] |
| lower_tank_1_level | lower_tank_1_level | m | [0, 0.8] |
| tank_0_temperature | tank_0_temperature | degC | [25, 72] |
| tank_1_temperature | tank_1_temperature | degC | [25, 72] |
| tank_2_temperature | tank_2_temperature | degC | [20, 58] |
| tank_3_temperature | tank_3_temperature | degC | [20, 58] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_cold | cold_inlet_step | degC | [0, 40] | 15 |
| t_amb | ambient_step | degC | [0, 45] | 20 |
| extra_outflow | demand_surge | m3/s | [0, 0.002] | 0 |
| pump_flow_factor | pump_capacity_shift | fraction | [0.4, 1.4] | 1 |
| heater_efficiency | heater_efficiency_shift | fraction | [0.4, 1.2] | 1 |
| heat_loss_factor | heat_loss_shift | fraction | [0.3, 3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| a_out | [0.00022, 0.00022, 0.0001, 0.0001] | m2 | [0, 0.01] |
| area | 0.06 | m2 | [0.01, 2] |
| gamma1 | 0.7 | fraction | [0.05, 0.95] |
| gamma2 | 0.7 | fraction | [0.05, 0.95] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| heater_max | [90000, 90000, 30000, 30000] | W | [0, 500000] |
| height_max | 0.8 | m | [0.1, 5] |
| pump_flow_max | 0.0013 | m3/s | [0, 0.02] |
| pump_power_max | 1200 | W | [0, 10000] |
| t_amb | 20 | degC | [0, 45] |
| t_cold | 15 | degC | [0, 40] |
| ua_loss | 40 | W/K | [0, 1000] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| level_bounds | h0, h1, h2, h3 | [0, 0.8] |
| temperature_trip | T0, T1, T2, T3 | [none, 92] |

## Benchmark Semantics

- State vector length: 8
- Action vector length: 6
- Controlled output vector length: 6
- Setpoint vector length: 6
- Dynamics disturbances: [t_cold, t_amb, extra_outflow, pump_flow_factor, heater_efficiency, heat_loss_factor]
- Micro integration step: 0.02 s
- Energy is scored: True

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| a_out | [0.8, 1.25] |
| heater_max | [0.6, 1.15] |
| pump_flow_max | [0.7, 1.3] |
| ua_loss | [0.4, 2.6] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [[0.32, 0.56], [0.32, 0.56]] |
| temp_band | [[46, 58], [46, 58], [32, 46], [32, 46]] |
| value | none |
| w_energy | 0.7 |
| w_value | 0 |
| w_viol | 29 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [y_sp, 2, 25, 72] |
| 1 | [y_sp, 3, 25, 72] |
| 2 | [y_sp, 4, 20, 58] |
| 3 | [y_sp, 5, 20, 58] |
