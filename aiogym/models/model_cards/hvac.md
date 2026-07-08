# Two-zone HVAC (`hvac`)

Two coupled thermal zones with bidirectional heat-pump actuation.

## Recommended Use

comfort tracking, energy-aware control, and disturbance-adaptation tests

## Dynamics And Assumptions

Two coupled thermal-zone balances combine outdoor exchange, inter-zone coupling, internal heat load, and bidirectional HVAC actuation.

- Thermal capacitance, envelope conductance, and actuator efficiency are lumped.
- Outdoor temperature and heat-load disturbances are benchmark schedules.
- Comfort and operating bounds are safety/evaluation constraints, not a building-code model.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| T0 | degC | [-20, 60] |
| T1 | degC | [-20, 60] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| hvac_zone_0 | heater | 0 | [0, 1] |
| hvac_zone_1 | heater | 1 | [0, 1] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_amb | ambient_step | degC | [-30, 50] | 5 |
| heat_load | internal_heat_load_step | W | [-1000, 2000] | [0, 0] |
| hvac_efficiency | hvac_efficiency_shift | fraction | [0.4, 1.3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| C | 6000 | J/K | [100, 100000] |
| Kc | 35 | W/K | [0, 1000] |
| Ko | 45 | W/K | [0, 1000] |
| Pmax | 1800 | W | [0, 20000] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| t_amb | 5 | degC | [-30, 50] |
| t_cold | 5 | degC | [-30, 50] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| comfort_band | T0, T1 | [20, 24] |
| temperature_operating_bounds | T0, T1 | [-20, 60] |

## Benchmark Semantics

- State vector length: 2
- Action vector length: 2
- Dynamics disturbances: [t_amb, heat_load[0], heat_load[1], hvac_efficiency]
- Micro integration step: 0.02 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| C | [0.7, 1.4] |
| Kc | [0.5, 1.7] |
| Ko | [0.5, 1.9] |
| Pmax | [0.7, 1.2] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [] |
| temp_band | [[20, 24], [20, 24]] |
| value | none |
| w_energy | 0.7 |
| w_value | 0 |
| w_viol | 8.2 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [t_sp, 0, 18, 26] |
| 1 | [t_sp, 1, 18, 26] |
