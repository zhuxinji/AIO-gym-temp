# Fired Heater (`heater`)

Refinery fired heater with outlet-temperature and flue-O2 dynamics.

## Recommended Use

supervisory setpoint control, safety-aware temperature control, and energy tradeoff studies

## Dynamics And Assumptions

A fired-heater surrogate couples firebox temperature, outlet temperature, and flue oxygen through fuel, air, heat-transfer, and load terms.

- Combustion and heat-transfer behavior is represented by lumped benchmark equations.
- Oxygen is treated as a controlled quality/safety variable, not a full emissions model.
- Economic scoring favors useful outlet heat while penalizing fuel and unsafe operation.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| T_firebox | degC | [20, 1400] |
| T_out | degC | [20, 650] |
| O2 | % | [0, 20.9] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| air_damper | valve | 0 | [0, 1] |
| fuel_valve | heater | 0 | [0, 1] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_cold | cold_inlet_step | degC | [240, 330] | 280 |
| t_amb | ambient_step | degC | [-10, 45] | 20 |
| extra_outflow | demand_surge | throughput fraction | [0, 0.002] | 0 |
| lhv_factor | fuel_lhv_shift | fraction | [0.5, 1.3] | 1 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| Amax | 40 | kg/s | [0, 200] |
| Cc | 7e+06 | J/K | [100000, 1e+08] |
| Cfb | 3.5e+06 | J/K | [100000, 1e+08] |
| Fmax | 1 | kg/s | [0, 10] |
| Fp0 | 88 | kg/s | [1, 300] |
| UA | 42000 | W/K | [0, 500000] |
| cp_g | 1400 | J/kg/K | [100, 5000] |
| cp_p | 2300 | J/kg/K | [100, 5000] |
| h_floor | 0.001 | fraction | [1e-06, 0.1] |
| lhv | 4.6e+07 | J/kg | [1e+06, 8e+07] |
| stoich | 17.2 | kg_air/kg_fuel | [1, 40] |
| t_amb | 20 | degC | [-30, 60] |
| t_cold | 280 | degC | [0, 500] |
| tau_o2 | 20 | s | [1, 200] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| outlet_temperature_band | T_out | [362, 378] |
| tube_skin_temperature_trip | T_out | [none, 415] |
| flue_o2_band | O2 | [1.6, 5.5] |
| flue_o2_trip | O2 | [1.2, none] |

## Benchmark Semantics

- State vector length: 3
- Action vector length: 2
- Dynamics disturbances: [t_cold, t_amb, extra_outflow, lhv_factor]
- Micro integration step: 0.05 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| Fp0 | [0.8, 1.25] |
| UA | [0.75, 1.2] |
| lhv | [0.82, 1.15] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [[1.6, 5.5]] |
| level_scale | 1 |
| temp_band | [[362, 378]] |
| value | none |
| w_energy | 0.35 |
| w_value | 0 |
| w_viol | 2100 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [t_sp, 0, 364, 372] |
| 1 | [h_sp, 0, 1.8, 5] |
