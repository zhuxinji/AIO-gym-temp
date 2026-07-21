# Heated-tank cascade (`cascade`)

Three interlinked heated tanks with level and temperature dynamics.

## Recommended Use

multi-loop level and temperature tracking, actuator allocation, safety-boundary tests, and task-declared continuous-production benchmarks

## Dynamics And Assumptions

Cascaded material balances move liquid through three heated tanks; energy balances mix inlet, interstage, ambient-loss, and heater terms.

- Constant-area, perfectly mixed tanks use constant liquid density and heat capacity.
- Each outlet is a one-way free drain with no downstream backpressure or reverse flow.
- Pipes have no volume, transport delay, pressure dynamics, or heat loss.
- Tank walls, heaters, pumps, valves, and sensors have no independent dynamics; boiling and other phase changes are excluded.
- Negative levels and overflow are hard termination events; levels are not silently clamped.
- Low liquid level or a 92 degC tank temperature disables that tank's heater before the 120 degC hard termination boundary.
- Direct operation is batch; continuous throughput economics belongs to the continuous-benchmark task.
- Heat-transfer and flow coefficients are benchmark parameters, not plant-identification claims.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## Physical Metadata Status

- Parameter status: `legacy-unverified`
- Fidelity: `first-principles-structure-with-unverified-benchmark-parameters`
- Time unit: `s`
- Solver: `rk4` with maximum step `0.02`

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
| outlet_valve_0 | valve | 1 | [0, 1] |
| outlet_valve_1 | valve | 2 | [0, 1] |
| outlet_valve_2 | valve | 3 | [0, 1] |
| heater_0 | heater | 4 | [0, 1] |
| heater_1 | heater | 5 | [0, 1] |
| heater_2 | heater | 6 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| tank_0_level | m | [0, 0.8] |
| tank_1_level | m | [0, 0.8] |
| tank_2_level | m | [0, 0.8] |
| tank_0_temperature | degC | [0, 120] |
| tank_1_temperature | degC | [0, 120] |
| tank_2_temperature | degC | [0, 120] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| tank_0_level | tank_0_level | m | [0, 0.8] |
| tank_1_level | tank_1_level | m | [0, 0.8] |
| tank_2_level | tank_2_level | m | [0, 0.8] |
| tank_0_temperature | tank_0_temperature | degC | [25, 80] |
| tank_1_temperature | tank_1_temperature | degC | [30, 82] |
| tank_2_temperature | tank_2_temperature | degC | [35, 85] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_cold | cold_inlet_step | degC | [0, 40] | 15 |
| t_amb | ambient_step | degC | [0, 45] | 20 |
| extra_outflow | demand_surge | m3/s | [0, 0.002] | 0 |
| pump_flow_factor | pump_capacity_shift | fraction | [0.4, 1.4] | 1 |
| heater_efficiency | heater_efficiency_shift | fraction | [0.4, 1] | 1 |
| heat_loss_factor | heat_loss_shift | fraction | [0.3, 3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| area | 0.15 | m2 | [0.01, 2] |
| cv_out | 0.0026 | m2.5/s | [0, 0.02] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| heater_max | 90000 | W | [0, 500000] |
| heater_min_level | 0.05 | m | [0, 0.8] |
| height_max | 0.8 | m | [0.1, 5] |
| pump_flow_max | 0.0016 | m3/s | [0, 0.02] |
| pump_power_max | 1500 | W | [0, 10000] |
| t_amb | 20 | degC | [0, 45] |
| t_cold | 15 | degC | [0, 40] |
| temperature_hard_limit | 120 | degC | [92, 150] |
| temperature_trip | 92 | degC | [40, 120] |
| ua_loss | 40 | W/K | [0, 1000] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| level_bounds | h0, h1, h2 | [0, 0.8] |
| heater_min_level | h0, h1, h2 | [0.05, none] |
| temperature_trip | T0, T1, T2 | [none, 92] |
| temperature_hard_limit | T0, T1, T2 | [none, 120] |

## Benchmark Semantics

- State vector length: 6
- Action vector length: 7
- Controlled output vector length: 6
- Setpoint vector length: 6
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
| value | production |
| w_energy | 0.7 |
| w_product_shortfall | 100 |
| w_value | 100000 |
| w_viol | 29 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [y_sp, 3, 25, 80] |
| 1 | [y_sp, 4, 30, 82] |
| 2 | [y_sp, 5, 35, 85] |
