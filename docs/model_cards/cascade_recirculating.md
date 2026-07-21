# Recirculating heated-tank cascade (`cascade_recirculating`)

Three non-identical tanks in a closed P101-Tank 1-V12-Tank 2-V23-Tank 3 loop with one 2 kW heater.

## Recommended Use

closed-loop three-tank control development and staged software validation against the retrofit design intent

## Dynamics And Assumptions

P101 removes liquid and enthalpy from Tank 3 and returns it to Tank 1; V12 and V23 close the gravity-fed loop, while only Tank 1 receives heater power.

- The normal process is a closed liquid inventory with no product or makeup flow.
- The provisional geometry interprets the PDF ranges as 16 L Tank 1, 16 L Tank 2, and 60 L Tank 3 inventory at the 0.40 m nominal level; these are range-derived design values, not as-built measurements.
- The PDF specifies one 2 kW Tank 1 heater and gives design ranges for tank height, effective volume, and pump power; valve capacity, flow calibration, heat loss, and interlock setpoints remain unverified.
- Pipes, sensors, pump, valves, and heater have no independent dynamics in the foundational model.
- Tank 1 and Tank 2 passive standpipes return mass and enthalpy to Tank 3 using provisional overflow levels and coefficients.
- P101 low level and H1 L2/L3/L4 protection disable equipment without projecting the physical state; only hard physical boundaries terminate simulation.
- Tracking, KPI, robustness, and safety are supported; production economics is intentionally unsupported because the closed loop has no declared product stream or production-value basis.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## Physical Metadata Status

- Parameter status: `design-provisional`
- Fidelity: `first-principles-closed-loop-structure-with-partial-design-source`
- Time unit: `s`
- Solver: `rk4` with maximum step `0.02`

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| h1 | m | [0, 0.5] |
| T1 | degC | [0, 100] |
| h2 | m | [0, 0.5] |
| T2 | degC | [0, 100] |
| h3 | m | [0, 0.5] |
| T3 | degC | [0, 100] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| pump_P101 | pump | 0 | [0, 1] |
| valve_V12 | valve | 1 | [0, 1] |
| valve_V23 | valve | 2 | [0, 1] |
| heater_H1 | heater | 3 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| tank_1_level | m | [0, 0.5] |
| tank_2_level | m | [0, 0.5] |
| tank_3_level | m | [0, 0.5] |
| tank_1_temperature | degC | [0, 100] |
| tank_2_temperature | degC | [0, 100] |
| tank_3_temperature | degC | [0, 100] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| tank_1_level | tank_1_level | m | [0, 0.5] |
| tank_2_level | tank_2_level | m | [0, 0.5] |
| tank_3_level | tank_3_level | m | [0, 0.5] |
| tank_1_temperature | tank_1_temperature | degC | [20, 80] |
| tank_2_temperature | tank_2_temperature | degC | [20, 80] |
| tank_3_temperature | tank_3_temperature | degC | [20, 80] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_amb | ambient_step | degC | [0, 45] | 20 |
| pump_flow_factor | pump_capacity_shift | fraction | [0.4, 1.4] | 1 |
| heater_efficiency | heater_efficiency_shift | fraction | [0.4, 1] | 1 |
| heat_loss_factor | heat_loss_shift | fraction | [0.3, 3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| area | [0.04, 0.04, 0.15] | m2 | [0.01, 5] |
| cv_interstage | [0.0026, 0.0026] | m2.5/s | [0, 0.02] |
| cv_overflow | [0.0052, 0.0052] | m2.5/s | [0, 0.05] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| heater_min_level | 0.05 | m | [0, 1] |
| heater_power | 2000 | W | [0, 5000] |
| height_max | [0.5, 0.5, 0.5] | m | [0.1, 5] |
| high_level_trip | [0.45, 0.45] | m | [0.05, 5] |
| overflow_head_floor | 1e-09 | m | [1e-12, 0.001] |
| overflow_level | [0.48, 0.48] | m | [0.05, 5] |
| pump_flow_max | 0.0016 | m3/s | [0, 0.02] |
| pump_min_level | 0.05 | m | [0, 1] |
| pump_power_max | 500 | W | [0, 5000] |
| t_amb | 20 | degC | [0, 45] |
| temperature_hard_limit | 100 | degC | [60, 120] |
| temperature_trip | 92 | degC | [40, 100] |
| ua_loss | [40, 40, 60] | W/K | [0, 2000] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| level_bounds | h1, h2, h3 | [0, 0.5] |
| P101_low_level_interlock | h3 | [0.05, none] |
| passive_overflow_onset | h1, h2 | [none, 0.48] |
| L3_high_level_trip | h1, h2 | [none, 0.45] |
| H1_low_level_interlock | h1 | [0.05, none] |
| H1_temperature_trip | T1 | [none, 92] |
| temperature_hard_limit | T1, T2, T3 | [none, 100] |

## Benchmark Semantics

- State vector length: 6
- Action vector length: 4
- Controlled output vector length: 6
- Setpoint vector length: 6
- Dynamics disturbances: [t_amb, pump_flow_factor, heater_efficiency, heat_loss_factor]
- Micro integration step: 0.02 s
- Energy is scored: True

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| cv_interstage | [0.7, 1.3] |
| heater_power | [0.9, 1] |
| pump_flow_max | [0.7, 1.3] |
| ua_loss | [0.5, 2] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [] |
| temp_band | [] |
| value | none |
| w_energy | 0 |
| w_value | 0 |
| w_viol | 0 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [y_sp, 3, 20, 80] |
