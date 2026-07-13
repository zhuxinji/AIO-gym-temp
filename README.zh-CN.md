<div align="center">

# AIO-Gym-temp

**整理后的过程控制 Python 后端：用于 benchmark、controller evaluation 和强化学习训练。**

[English](README.md) · [**简体中文**](README.zh-CN.md) · [日本語](README.ja.md)

MIT 许可

</div>

---

这个 checkout 称为 **AIO-Gym-temp**。它是从 **AIO-Gym** 派生出来的后端整理版。在本文档中，**AIO-Gym** 指整理前的旧后端快照。

本文只说明后端改动：包结构、Python API、终端命令、benchmark evaluation、controller 组织、模型 registry、RL 训练、artifacts 和生成输出策略。

更详细的后端使用手册在 [aiogym/README.md](aiogym/README.md)。

## 后端范围

`AIO-Gym-temp` 在 `aiogym/` 下提供原生 Python 后端：

- 原生 Gymnasium 环境。
- 七个注册后端场景：`cascade`、`quadruple`、`cstr`、`hvac`、`extraction`、`heater`、`crystallization`。
- PID、MPC、oracle、通用 policy 对象、SB3 policy 的统一 controller 评估。
- 命名 benchmark suite，输出 summary table、report、plot 和可复用 artifacts。
- SB3 和 RLPD 的强化学习训练入口。
- 面向 notebook、脚本和外部 Python 用户的精简公共 API。
- 面向终端和自动化流程的 console commands。

从仓库根目录安装后端：

```bash
pip install -e ./aiogym
```

默认安装已经包含 CasADi/IPOPT、SB3/Torch 和 ONNX 全套后端依赖。
`[oracle]`、`[train]`、`[export]`、`[all]` 仅作为旧安装命令的兼容别名保留，
计划在 `aiogym 0.3` 移除。

常用命令：

```bash
aiogym-suite-benchmark --suite standard-baselines --episodes 3
aiogym-single-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-suite-benchmark --suite economic-supervisory --scenarios cstr --controllers onnx --onnx-path frontend/models/rlpd_cstr.onnx --episodes 1
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-artifact-check aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-train-sb3 --scenario cstr --algo sac --steps 10000 --onnx
```

Python API：

```python
import aiogym

env = aiogym.make_env(model="cstr", objective="tracking", seed=7)
payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controller": "pid",
})
figures = aiogym.plot_results(payload["run_dir"])
```

## AIO-Gym-temp 与 AIO-Gym 对比

**AIO-Gym** 指整理前的旧后端布局。它是脚本式结构：大多数后端源码直接平铺在 `aiogym/` 下面，用户通过运行单个 Python 文件来使用。

**AIO-Gym-temp** 指当前整理后的后端。它是包式结构：源码按职责分层，用户入口清晰，生成输出和源码分离。

| 方面 | AIO-Gym | AIO-Gym-temp |
|---|---|---|
| 包配置 | 没有 `pyproject.toml`，不能作为标准包安装。 | 后端包有 `aiogym/pyproject.toml`，支持可安装包、可选依赖组、package data 和 console scripts。 |
| 用户入口 | 直接运行脚本，例如 `python aiogym/train.py`、`python aiogym/train_rlpd.py`、`train_all.sh`。 | 稳定命令，例如 `aiogym-suite-benchmark`、`aiogym-report`、`aiogym-train-sb3`、`aiogym-train-rlpd`。 |
| Python API | 主要从 `aiogym.__init__` 和内部模块直接导入。 | 精简公共 API：`aiogym.define_model`、`aiogym.register_model`、`aiogym.make_model`、`aiogym.make_env`、`aiogym.run_benchmark` 和 `aiogym.plot_results`。 |
| 后端结构 | 平铺文件：`models.py`、`kernel.py`、`baselines.py`、`oracle.py`、`rlpd.py`、`train.py`、`train_sac.py`、`train_rlpd.py`。 | 分层包结构：`models/`、`env`、`controllers/`、`evaluation/`、`rl/` 和 `cli/`，公共用户函数由 `aiogym` 顶层重新导出。 |
| 模型覆盖 | 后端 registry 覆盖 cascade、quadruple、CSTR、HVAC、fired heater。 | 后端 registry 覆盖 cascade、quadruple、CSTR、HVAC、extraction、fired heater、crystallization。 |
| 模型元数据 | model card 不是一等包能力。 | model card 和模型契约校验放在 `aiogym.models` 中，并由测试覆盖导出。 |
| 控制器 | PID、MPC、评估 helper 和 baseline 概念集中在 `baselines.py`，oracle 单独在 `oracle.py`。 | controller 接口、registry、configs、adapters、PID、MPC、oracle、tuning tools 都在 `aiogym.controllers` 下。 |
| 评估 | 评估主要靠脚本和 helper 串起来。 | `aiogym.evaluation` 负责 benchmark objective/config、rollout、metrics、reports、plots、artifact 生成和 suites。 |
| Benchmark suites | 没有标准 suite 配置包。 | 命名 suite JSON 放在 `aiogym/evaluation/suites/`，包括 `standard-baselines`、tracking/economic、robustness、RL-direct、crystallization 等。 |
| 报告和图 | 生成结果散落在 runs 或脚本输出里。 | artifact 使用标准目录：`benchmark.json`、`config/`、`metadata/`、`summary/`、`results/`、`figures/`。 |
| RL 代码 | `rlpd.py`、`train_rlpd.py`、`train_sac.py`、`train.py` 和核心后端模块放在同一层。 | RL 算法和训练流程统一放在 `aiogym.rl`；默认输出到 `aiogym/runs/rl/`。 |
| 生成输出 | 历史 run JSON 被跟踪在 `aiogym/runs/`。 | `aiogym/runs/` 只作为本地输出目录，源码控制里只保留 `.gitignore`。 |
| Parity 工具 | 已有 JS parity 测试。 | 保留 parity 测试，并把 golden 生成器放到后端测试目录：`aiogym/tests/generate_golden.mjs`。 |
| 文档 | 后端使用说明分散在脚本注释、README 文本和旧 helper 位置。 | 后端文档整合进 README，并明确包边界和命令入口。 |

## 已实现的后端优化

### 1. 可安装包和标准命令

`AIO-Gym-temp` 可以从后端包目录安装。用户命令在 `aiogym/pyproject.toml` 中定义：

```text
aiogym-single-benchmark
aiogym-suite-benchmark
aiogym-report
aiogym-artifact-check
aiogym-model-cards
aiogym-train-sb3
aiogym-train-rlpd
```

旧的兼容 wrapper 和过宽的 shell 脚本已移除，包括旧的 `train_all.sh` 流程。

### 2. 精简公共 API

顶层 `aiogym` 包重新导出普通用户会直接调用的函数：

```python
aiogym.make_env(...)
aiogym.run_benchmark(...)
aiogym.plot_results(...)
aiogym.define_model(...)
aiogym.register_model(...)
aiogym.make_model(...)
```

自定义场景里，`define_model(...)` 用声明式 spec 定义模型，
`register_model(...)` 把模型绑定到场景名，`make_model(...)` 则和
`make_env(...)` 对应，是模型层的实例化/校验入口：它从场景名、模型类、
factory 或模型实例得到一个标准 process model 实例。

更底层的 artifact 写入逻辑保留在 `aiogym.evaluation.artifacts` 内部，不再挂到顶层 API。

### 3. 清晰的命令行层

`aiogym/cli/` 现在只包含终端入口：

```text
aiogym/cli/
  single_benchmark.py
  suite_benchmark.py
  artifact_tools.py
```

CLI 负责解析参数、打印进度、设置退出行为，并调用共享实现模块。核心 evaluation 或 RL 逻辑不放在 CLI 里。

### 4. 模型从单文件拆成 package

旧的 `models.py` 和 `kernel.py` 被拆成：

```text
aiogym/models/
  core.py
  kernel.py
  registry.py
  scenarios/
```

具体场景实现放在 `models/scenarios/`，registry、校验、model-card 导出和 Gym ID helper 放在 `models/registry.py`。

### 5. Controller package 与调参溯源

原先 baseline controller 相关逻辑整理为：

```text
aiogym/controllers/
  __init__.py
  pid.py
  mpc.py
  oracle.py
  configs/
  tuning/
```

这个包负责正式 controller API、controller registry、policy/SB3 adapter、默认配置和可复现调参脚本。PID、MPC、oracle 现在是并列的 controller 实现，不再混在一个 baseline 文件里。

### 6. Evaluation、metrics、reports、plots、artifacts 集中

评估相关代码统一放到：

```text
aiogym/evaluation/
  core.py
  metrics/
  reports.py
  plots.py
  artifacts.py
  suites/
```

边界更明确：

- `core.py` 定义 benchmark objective/config 和 rollout/evaluation 逻辑。
- `metrics/` 放指标计算器。
- `reports.py` 渲染 benchmark report。
- `plots.py` 生成 SVG 图。
- `artifacts.py` 写 benchmark artifact 目录。
- `suites/` 放命名 benchmark 配置。

### 7. 标准 benchmark artifact

benchmark run 现在会写入一个可复用 artifact 目录：

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  figures/
```

这样源码和生成结果分开，report、plot、model card、summary 和完整结果 payload 都有固定位置。

### 8. RL 代码统一到 `aiogym.rl`

RL 相关源码现在集中在：

```text
aiogym/rl/
  rlpd.py
  train_rlpd.py
  train_sb3.py
```

默认训练模型输出到：

```text
aiogym/runs/rl/
  rlpd/
  sb3/
```

`aiogym.rl` 使用懒加载，所以只查看训练命令的 `--help` 不需要提前安装 Torch/SB3 这些可选依赖。

### 9. Runs 清理

`aiogym/runs/` 现在被视为本地输出目录。历史 tracked run JSON/SVG 已移除，源码控制里只保留 `.gitignore`。

### 10. 保留测试和 parity 覆盖

整理后仍保留关键验证路径：

```bash
python aiogym/tests/test_interface.py
python aiogym/tests/test_parity.py
node aiogym/tests/generate_golden.mjs --check
```

interface 测试覆盖模型契约、公共 API 入口、controller evaluation、suite configs、reports/artifacts、设定点对齐和 oracle baseline 行为。parity 测试校验原生 NumPy 动力学和 JavaScript 参考实现的 golden 轨迹一致。

## 推荐理解方式

新增后端代码时可以按这个结构判断归属：

```text
aiogym/
  cli/            # 只放终端入口
  env.py          # Gymnasium 环境和 make_env helper
  models/         # 过程模型、契约、registry、model cards
  controllers/    # controller API、内置控制器、configs、tuning
  evaluation/     # benchmark objectives/configs、metrics、reports、plots、artifacts
  rl/             # RL 算法和训练流程
  runs/           # 本地生成输出
```

关键原则是：`cli/` 是终端入口，不是核心逻辑的存放处；普通 Python 用户入口由顶层 `aiogym` 包重新导出。核心行为应该放在 `env.py`、`models/`、`controllers/`、`evaluation/` 或 `rl/` 中。
