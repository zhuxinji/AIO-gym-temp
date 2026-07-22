# CascadeModel 修改实施计划

> 文档状态：历史实施计划快照；不作为当前配置或待办清单
> 计划日期：2026-07-17
> 适用基线：`3fc9412`（`main`，当时工作区另含未提交修改）
> 依据：外部审查文档 `cascade_model_codex_modification_report.md`（未随仓库发布）
> 范围：`cascade` 模型、必要的运行时/任务/KPI 接口、测试与文档
> 本文件仅制定计划，不包含模型代码修改。

> Snapshot 提示：本文保留 2026-07-17 的实施上下文，包括当时存在的
> `legacy-default` 路径。它不描述当前任务注册表；查看当前可用任务时应以
> `aiogym list`/公开注册 API（若当前版本提供）或 `aiogym/evaluation/tasks/`
> 的实际内容为准。

## 1. 目标与不变项

目标是在不改变公开模型拓扑的前提下，修复物理边界、安全联锁、数值/CasADi 一致性以及连续生产经济指标口径。

必须保持：

- 类名与场景名：`CascadeModel`、`cascade`；
- 状态顺序：`(h0, T0, h1, T1, h2, T2)`；
- 动作顺序：泵、3 个阀、3 个加热器；
- 受控输出顺序：3 个液位后接 3 个温度；
- 单向自由排水拓扑：冷水 → 0 号罐 → 1 号罐 → 2 号罐 → 排出；
- RK4 积分器与当前 0.02 s 最大微步长。

本轮不做：

- Euler/RK4 对比；
- 双向液压耦合或下游背压；
- 温度分层、罐壁温度、加热棒热惯性；
- 沸腾/相变；
- 每个水箱独立参数；
- 真实溢流动力学。

## 2. 当前代码基线

实施前应以工作区实时内容为准，不能直接按修改报告中的示例代码覆盖。当前已确认：

- `AIOGymNativeEnv._split()` 会把环境动作裁剪到 `[0,1]`；直接调用 `CascadeModel.dynamics()` 时没有同等防御。
- `ProcessModelContract.action_vector()` 只校验维度和数值转换，不校验动作上下界。
- `cascade` 没有 `clamp_state()`；液位可变负或超过 0.8 m。
- `state_bounds`/`safety_constraints` 主要是元数据和评分输入；是否终止仍受 `terminate_on_runaway` 控制，默认值为 `False`。
- numeric `display_outputs()` 会把负液位显示为 0，CasADi 路径返回原始值；`controlled_output()` 复用了这一差异。
- `runtime_env()` 可生成默认环境，但直接传 `env={}` 给 `dynamics()` 仍会在 `_dynamics()` 中缺少 `t_cold`/`t_amb`。
- `ideal_energy_kw()` 使用当前泵动作作为基准流量，没有效率修正和泵功率。
- `cascade` 目前只有 `legacy-default` 任务；经济套件却把它列为可用经济场景。
- 参数档案仍是 `legacy-unverified`，没有真实装置参数来源。
- `core.py`、`env.py`、`objectives.py` 等相关文件目前存在用户未提交修改，实施时必须逐文件保存并复核现有差异。

## 3. 实施前决策门

以下 6 项最好由设备资料确认。若实施启动时仍无实测值，使用表中的显式 benchmark 假设，并在参数档案中标记为 `assumed-benchmark`，不得写成实测参数。

| 决策 | 待确认值 | 临时 benchmark 方案 |
| --- | --- | --- |
| 最低安全浸没液位 | `heater_min_level` | 0.05 m，仅作安全测试假设 |
| 连续生产目标流量 | `product_flow_sp` | 4.0e-4 m³/s，仅用于新连续任务 |
| 满罐处理 | 溢流或终止 | 本轮不建溢流模型，超过 0.8 m 立即终止 |
| 92 °C 处理 | 单加热器或全系统 | 关闭对应加热器；120 °C 作为硬终止上限 |
| `heater_efficiency` 语义 | 效率或容量倍率 | 保留“热效率”，范围 `(0,1]` |
| 额外排水能力 | 最大值和液位关系 | 保留 0–0.002 m³/s；空罐时强制为 0 |

兼容策略：现有 `legacy-default` 保持批式/兼容语义；新增正式的连续生产任务，不把测试流量偷偷写进所有直接环境调用。

## 4. 总体架构选择

### 4.1 各层职责

| 层 | 职责 |
| --- | --- |
| `CascadeModel` | 有效动作、环境解析、流量、加热联锁、连续方程、稳态需求和模型诊断 |
| `ProcessModelContract`/ops | 仅提供必要的后端中性算子和可选安全扩展点，不硬编码 `cascade` |
| `Integrator` | 继续只做 RK4；不静默裁剪满罐或负液位 |
| `AIOGymNativeEnv` | 应用动作、推进状态、汇总结构化安全原因并返回终止信息 |
| `evaluation.objectives` | 纯函数式评分、硬终止语义和经济约束，不保存模型状态 |
| 任务/协议层 | 声明 `batch`/`continuous`、吞吐目标，并确保环境和控制器模型得到同一配置 |
| 测试 | 分别验证物理、后端、运行时、KPI 和回归兼容 |

### 4.2 边界处理选择

- 空罐时阀门流量和额外排水均为 0。
- 如果固定步长仍从正液位跨到负液位，不做无质量去向的静默投影；记录 `negative_level` 并终止该 episode。
- 超过 `height_max` 不裁剪，记录 `overflow` 并终止。
- 92 °C 触发对应加热器关闭；达到温度硬上限才终止。
- 新增“硬终止原因”扩展点，使这类物理边界不依赖用户是否设置 `terminate_on_runaway=True`。

### 4.3 动作处理选择

- 环境继续在 plant 入口裁剪，并保留评价层现有的“裁剪前动作越界”指标。
- `CascadeModel._dynamics()` 再做一次后端中性的防御性 `[0,1]` 限幅，覆盖直接模型调用和 CasADi 图。
- 不全局改变 `ProcessModelContract.action_vector()` 语义，避免影响其他 6 个模型和自定义模型。

## 5. 分阶段实施

### 阶段 0：冻结基线与建立失败用例

目标：先记录正常行为和已知缺陷，避免重构时误改无关语义。

工作项：

1. 保存相关未提交 diff，确认与本任务重叠的位置。
2. 新建 `aiogym/tests/test_cascade_physics.py`，集中放置场景专用物理测试。
3. 加入当前应失败的验收用例：负液位、满罐、干烧、温度跳车、空罐排水、后端输出不一致、理想能耗口径。
4. 保留现有名义稳态、质量衡算、能量衡算和步长收敛结果作为回归基线。

退出条件：所有目标行为均有明确测试表达；测试失败原因与修改报告一一对应。

### 阶段 1：输入、环境和后端基础能力（P0-1、P0-5、P0-6）

主要文件：

- `aiogym/models/scenarios/cascade.py`
- `aiogym/models/core.py`
- `aiogym/tests/test_cascade_physics.py`
- `aiogym/tests/test_models_api.py`

工作项：

1. 在 `cascade` 内新增 `_resolved_env()`：
   - `env={}` 使用模型参数默认值；
   - 所有数值必须有限；
   - `extra_outflow >= 0`；
   - 各倍率按声明范围统一拒绝非法值，不静默采用不同策略。
2. 新增 `_effective_action(u, ops)`，numeric 和 CasADi 均使用 `max/min` 限制到 `[0,1]`。
3. 若联锁需要符号条件，在统一 ops 层加入最小的 `if_else()`/条件门算子；不在模型里散落后端判断。
4. 给 `RHO_CP` 增加体积热容单位注释和数量级测试。
5. 验证 direct numeric、环境 plant、CasADi 三条路径对相同输入具有一致作用。

退出条件：非法环境值明确失败；越界动作不产生负或超额物理作用；CasADi 3.7.2 后端测试通过。

### 阶段 2：流量边界、加热联锁和硬终止（P0-2、P0-3、P0-4）

主要文件：

- `aiogym/models/scenarios/cascade.py`
- `aiogym/models/core.py`（仅可选安全扩展点）
- `aiogym/env.py`
- `aiogym/evaluation/objectives.py`
- `aiogym/evaluation/metrics/safety.py`
- `aiogym/tests/test_cascade_physics.py`
- `aiogym/tests/test_env_contract.py`
- `aiogym/tests/test_objectives.py`

工作项：

1. 拆分 `_flow_terms()`：
   - 使用 `sqrt(max(h,0))`；
   - `h<=0` 时阀门出流严格为 0；
   - `h2<=0` 时额外排水严格为 0；
   - 保持单向自由排水，不引入下游背压。
2. 新增参数 `heater_min_level`、`temperature_trip`、`temperature_hard_limit`。
3. 拆分 `_heater_terms()`：低液位或达到跳车温度时，传给液体的热功率为 0；numeric/CasADi 使用同一逻辑。
4. 新增模型级结构化安全诊断：
   - `low_level_interlock_active`
   - `temperature_trip_active`
   - `negative_level`
   - `overflow`
   - `temperature_hard_limit`
5. 为模型契约增加可选 `hard_termination_reasons(...)`；`stage_reward()` 无条件执行硬终止，原有 `terminate_on_runaway` 继续控制普通软 runaway 行为。
6. `info` 增加 `termination_reason`/`safety_events`；安全指标能统计事件、严重度和持续时间。
7. 不实现上限液位裁剪；无溢流模型时必须终止。

退出条件：空罐不再继续失水；干烧和高温继续加热被阻止；上下液位硬边界具有可执行终止语义和结构化原因。

### 阶段 3：输出、参数与诊断一致性（P1-1、P1-2、P1-3、P1-4、P1-7）

主要文件：

- `aiogym/models/scenarios/cascade.py`
- `aiogym/models/parameters/cascade.json`
- `aiogym/models/cards.py`
- `aiogym/tests/test_cascade_physics.py`
- `aiogym/tests/test_models_api.py`
- `aiogym/tests/test_model_infrastructure.py`

工作项：

1. `controlled_output()` 在 numeric/CasADi 均直接返回相同物理状态重排，不做 UI 裁剪。
2. `display_outputs()` 只用于展示；KPI、约束和控制器不再依赖其裁剪结果。
3. `height_max` 属性从 `self.p["height_max"]` 派生；状态边界、安全约束和参数值增加一致性测试。
4. 明确拆分：
   - `state_bounds`/`output_bounds`：物理/仿真边界；
   - `setpoint_bounds`：原 25–80、30–82、35–85 °C 操作目标范围。
5. `heater_efficiency` 范围改为 `(0,1]`，采样上限同步为 1；容量变化仍由 `heater_max` plant regime 表示。
6. 扩展 `process_info(..., action)`，返回实际进料流、两级中间流、产品流、额外排水、实际加热功率和三个联锁数组。
7. 参数档案逐项记录 `source/status/description`；未知装置参数保持 `assumed-benchmark` 或 `legacy-unverified`。

退出条件：两个后端的动力学、受控输出和联锁一致；初态不会因为“设定值范围”被误判为物理越界；诊断字段可由 rollout artifact 保存。

### 阶段 4：稳态需求、理想能耗与连续生产（P1-5、P1-6）

这一阶段单独实施，因为它会改变经济评价语义，不应与安全修复混在一次提交中。

主要文件：

- `aiogym/models/scenarios/cascade.py`
- `aiogym/evaluation/task_profiles.py`
- `aiogym/evaluation/protocols.py`
- `aiogym/evaluation/runner.py`
- `aiogym/evaluation/objectives.py`
- `aiogym/env.py`
- `aiogym/cli/suite_benchmark.py`
- `aiogym/evaluation/tasks/cascade/legacy-default.json`
- 新增 `aiogym/evaluation/tasks/cascade/continuous-benchmark.json`
- 相关 suite 配置与评价测试

工作项：

1. 新增纯函数 `steady_state_requirements(y_sp, env, product_flow_sp)`，返回：
   - 泵/阀/加热器稳态命令；
   - 热负荷、电加热功率、泵功率和总理想功率；
   - `feasible` 与逐执行器不可行原因。
2. `ideal_energy_kw()` 调用该 helper：
   - 不再使用当前泵动作确定基准吞吐量；
   - 热负荷除以效率得到电功率；
   - 加入泵功率。
3. 在任务 profile 中增加独立 `operation` 区段，而不是把 `batch/continuous` 字符串塞进物理参数 `.p`：
   - `mode: batch | continuous`
   - `product_flow_sp`
   - 可选 `min_product_flow`
4. 增加统一的“任务上下文配置模型”路径，使环境模型和 runner 创建的控制器模型得到完全相同的 operation 配置；不修改 PID/MPC/Oracle 算法本身。
5. `legacy-default` 明确为 `batch`；新连续任务使用显式 benchmark 流量并标注非装置额定值。
6. 连续模式缺失吞吐目标时在任务加载阶段失败，不推迟到 rollout 中途。
7. 经济目标增加产品流量短缺项或硬约束；零泵/零阀策略不得因低能耗成为连续模式最优。
8. suite 中的 `cascade` 经济案例必须绑定连续任务；批式任务不得伪装成连续生产经济基准。

退出条件：环境、KPI 和经济 Oracle 使用相同吞吐目标；效率下降会提高理想电功率；连续模式关泵不再占优；批式模式仍允许零吞吐量。

### 阶段 5：文档、迁移和完整验证

主要文件：

- `docs/model_cards/cascade.md`
- `aiogym/models/cards.py`
- `aiogym/models/parameters/cascade.json`
- `docs/reports/cascade-model-accuracy-authenticity.md`
- `docs/reports/cascade-model-validation-results.json`
- 必要的 README/任务说明

工作项：

1. 删除“液位已钳位”的错误表述，改为真实边界和终止语义。
2. 完整声明自由排水、完全混合、常物性、无管路延迟/壁热容/相变等剩余假设。
3. 记录 `heater_efficiency` 语义变化、连续任务新增和经济指标迁移说明。
4. 重新生成模型卡并验证生成结果与手写说明一致。
5. 重新运行准确性/真实性审计，更新旧报告中的文件哈希、边界结果、稳态功率和结论等级。
6. 运行定向测试、全部模型契约测试和全量测试。

退出条件：文档、模型卡、参数档案、任务配置、代码和测试使用同一套语义。

## 6. 测试矩阵

| 类别 | 必测内容 |
| --- | --- |
| 守恒 | 零流零热平衡、随机内部点质量/能量残差、可行稳态导数 |
| 输入 | 动作 `<0`/`>1`、空环境默认值、NaN/Inf、负额外排水 |
| 流量边界 | 空罐阀全开、空第三罐额外排水、正液位正常自由出流 |
| 热安全 | 三个罐分别测试低液位联锁、92 °C 联锁、正常液位正常加热 |
| 终止 | 负液位、超过 0.8 m、温度硬上限、结构化终止原因 |
| 后端 | numeric/CasADi 导数、受控输出、有效动作和联锁一致 |
| 元数据 | `height_max`、状态边界、安全边界、设定值边界一致 |
| 能耗 | 实际电功率、效率修正、泵功率、不可行执行器原因 |
| 运行模式 | 连续模式缺目标失败、零流量受罚、批式零流量合法 |
| 回归 | 公开向量顺序、环境构造、PID/MPC/Oracle、报告和 artifact schema |

建议命令：

```bash
cd <repository-root>
PYTHONPYCACHEPREFIX=/private/tmp/aiogym_pycache .venv/bin/python -m pytest aiogym/tests/test_cascade_physics.py -q
PYTHONPYCACHEPREFIX=/private/tmp/aiogym_pycache .venv/bin/python -m pytest aiogym/tests/test_models_api.py aiogym/tests/test_env_contract.py aiogym/tests/test_objectives.py -q
PYTHONPYCACHEPREFIX=/private/tmp/aiogym_pycache .venv/bin/python -m pytest aiogym/tests -q
```

## 7. 建议提交拆分

为便于回退和审查，建议形成 5 个独立提交：

1. `cascade: validate inputs and add backend-neutral helpers`
2. `cascade: enforce flow boundaries and heater safety interlocks`
3. `cascade: align outputs metadata and diagnostics`
4. `cascade: add task-owned throughput and steady-state energy semantics`
5. `docs/tests: validate and document cascade physical contract`

每个提交必须同时包含对应测试，不能先提交行为再留待后续补测试。

## 8. 最终验收门

只有同时满足以下条件才算完成：

- 所有 P0 测试通过；
- numeric/CasADi 一致性通过；
- 负液位和溢流不会静默继续；
- 低液位和温度跳车真实关闭液体加热功率；
- 物理输出边界与设定值边界分离；
- 效率、电功率和理想能耗口径一致；
- 连续生产目标同时进入 plant、控制器预测和 KPI；
- 公开状态/动作/输出顺序无变化；
- 所有现有测试通过，且没有覆盖用户原有未提交修改；
- 模型卡和准确性报告已更新，不再声称未实现的液位钳位。

## 9. P2 后续项

本轮完成后再单独评估：

- NMPC 在 `sqrt(h)` 和硬联锁处的平滑近似；
- 显式溢流流量与质量守恒模型；
- 三个水箱独立面积、阀系数、UA 和加热器额定功率；
- 真实装置数据辨识与验证。

P2 不应阻塞本轮 P0/P1，但不能在本轮中顺手加入，以免改变模型拓扑和验证范围。
