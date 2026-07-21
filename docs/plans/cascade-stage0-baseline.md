# Cascade 修改阶段 0 基线

> 记录时间：2026-07-17
> Git 基线：`3fc9412`，分支 `main`
> 目的：在逐步修改前记录与本任务有关的工作区状态；本文件不声明这些文件已提交。

## 相关文件快照

| 文件 | SHA-256 | 阶段 0 状态 |
| --- | --- | --- |
| `aiogym/models/scenarios/cascade.py` | `cd37dcc03ae289ee96ec0717324e825d63e979c534eef7f9f2a7dd6043d8e6eb` | 当前无已有 tracked diff |
| `aiogym/models/core.py` | `46713686bd8b012afe8f2ca1c600adfc77934b2d98ea372d4a00e342f588bc28` | 已有用户修改：新增 `physical_action_vector()` |
| `aiogym/env.py` | `c96f2550563ec322d98a513dc810fa12676c49ed09b96319a8d7f76f3dc0cbe6` | 已有用户修改：tracking steady cost 与 setpoint 事件时序 |
| `aiogym/evaluation/objectives.py` | `e7085da046af866c5eba74d7c193a97e390799697a7d074a4a56e36cba62d2ab` | 已有用户修改：tracking steady-input cost |
| `aiogym/evaluation/metrics/safety.py` | `6e46998c6be04da730f040d49df171cf9a02b18e86218f43a00f8d2cfa9cfeb7` | 当前无已有 tracked diff |

实施后续阶段时，必须在上述工作区版本上增量修改，不能用基线提交内容覆盖 `core.py`、`env.py` 或 `objectives.py`。

## 阶段 0 测试策略

新增 `aiogym/tests/test_cascade_physics.py`：

- 当前已经满足的守恒、向量顺序、稳态和常量检查使用普通测试；
- 尚未实现的 P0/P1 行为使用 `pytest.mark.xfail(strict=True)`；
- 后续某阶段实现对应行为时，必须在同一改动中删除相应 xfail 标记；
- `strict=True` 会把意外通过视为失败，防止修复完成后遗留“已知缺陷”标记。

阶段 0 不修改 `cascade` 模型、积分器、环境、KPI 或控制器行为。
