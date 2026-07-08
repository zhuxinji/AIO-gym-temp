# Heated-tank cascade (`cascade`)

Three interlinked heated tanks with level and temperature dynamics.

## Recommended Use

multi-loop level and temperature tracking, actuator allocation, and disturbance-rejection smoke tests

## Dynamics And Assumptions

Cascaded material balances move liquid through three heated tanks; energy balances mix inlet, interstage, ambient-loss, and heater terms.

- Well-mixed tanks with lumped heat loss and actuator gains.
- Levels are clamped to the declared tank range after integration.
- Heat-transfer and flow coefficients are benchmark parameters, not plant-identification claims.

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

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| feed_pump | pump | 0 | [0, 1] |
| outlet_valve_0 | valve | 0 | [0, 1] |
| outlet_valve_1 | valve | 1 | [0, 1] |
| outlet_valve_2 | valve | 2 | [0, 1] |
| heater_0 | heater | 0 | [0, 1] |
| heater_1 | heater | 1 | [0, 1] |
| heater_2 | heater | 2 | [0, 1] |

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
| area | 0.15 | m2 | [0.01, 2] |
| cv_out | 0.0026 | m2.5/s | [0, 0.02] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| heater_max | 90000 | W | [0, 500000] |
| height_max | 0.8 | m | [0.1, 5] |
| pump_flow_max | 0.0016 | m3/s | [0, 0.02] |
| pump_power_max | 1500 | W | [0, 10000] |
| t_amb | 20 | degC | [0, 45] |
| t_cold | 15 | degC | [0, 40] |
| ua_loss | 40 | W/K | [0, 1000] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| level_bounds | h0, h1, h2 | [0, 0.8] |
| temperature_trip | T0, T1, T2 | [none, 92] |

## Benchmark Semantics

- State vector length: 6
- Action vector length: 7
- Dynamics disturbances: [t_cold, t_amb, extra_outflow, pump_flow_factor, heater_efficiency, heat_loss_factor]
- Micro integration step: 0.02 s
- Energy is scored: True

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| cv_out | [0.7, 1.4] |
| heater_max | [0.6, 1.15] |
| pump_flow_max | [0.7, 1.3] |
| ua_loss | [0.4, 2.6] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [[0.32, 0.58], [0.32, 0.58], [0.32, 0.58]] |
| temp_band | [[34, 44], [48, 58], [60, 72]] |
| value | none |
| w_energy | 0.7 |
| w_value | 0 |
| w_viol | 29 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [t_sp, 0, 25, 80] |
| 1 | [t_sp, 1, 30, 82] |
| 2 | [t_sp, 2, 35, 85] |
