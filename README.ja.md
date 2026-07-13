<div align="center">

# AIO-Gym-temp

**プロセス制御のベンチマーク、コントローラ評価、強化学習訓練のために整理された Python バックエンド。**

[English](README.md) · [简体中文](README.zh-CN.md) · [**日本語**](README.ja.md)

MIT License

</div>

---

この checkout は **AIO-Gym-temp** と呼びます。これは **AIO-Gym** から派生した、バックエンド整理版です。この文書では、**AIO-Gym** は再整理前の古いバックエンドスナップショットを指します。

この README はバックエンドの変更だけを説明します。対象は、パッケージ構成、Python API、コマンドライン入口、benchmark evaluation、controller 構成、model registry、RL training、artifacts、生成出力の扱いです。

詳しいバックエンド利用手順は [aiogym/README.md](aiogym/README.md) にあります。

## Backend Scope

`AIO-Gym-temp` は `aiogym/` 以下に native Python backend を提供します。

- Native Gymnasium environments.
- 7 つの登録済み backend scenarios: `cascade`, `quadruple`, `cstr`, `hvac`, `extraction`, `heater`, `crystallization`.
- PID, MPC, oracle, generic policy objects, SB3 policies の統一 controller evaluation.
- summary tables, reports, plots, reusable artifacts を出力する named benchmark suites.
- SB3 と RLPD の RL training entrypoints.
- notebooks/scripts/外部 Python 利用者向けの小さな public Python API.
- shell/automation workflow 向けの console commands.

リポジトリルートから backend をインストールします。

```bash
pip install -e ./aiogym
```

デフォルトインストールには CasADi/IPOPT、SB3/Torch、ONNX の backend 依存関係が
すべて含まれます。`[oracle]`、`[train]`、`[export]`、`[all]` は旧インストール
コマンドとの互換 alias としてのみ残され、`aiogym 0.3` で削除予定です。

よく使うコマンド:

```bash
aiogym-suite-benchmark --suite standard-baselines --episodes 3
aiogym-single-benchmark --scenario cstr --objective tracking --controllers pid,mpc
aiogym-suite-benchmark --suite economic-supervisory --scenarios cstr --controllers onnx --onnx-path frontend/models/rlpd_cstr.onnx --episodes 1
aiogym-report aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-artifact-check aiogym/runs/bench_suite_standard-baselines_artifacts
aiogym-train-sb3 --scenario cstr --algo sac --steps 10000 --onnx
```

Python API:

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

## AIO-Gym-temp vs AIO-Gym

**AIO-Gym** は再整理前の古い backend layout です。これは script-oriented な構造で、多くの backend source files が `aiogym/` 直下に置かれ、利用者は個別の Python files を直接実行していました。

**AIO-Gym-temp** は現在の reorganized backend です。source code は責務ごとに分割され、user entrypoints は明確になり、generated outputs は source files から分離されています。

| Area | AIO-Gym | AIO-Gym-temp |
|---|---|---|
| Package setup | `pyproject.toml` がなく、標準 package として install できない。 | `aiogym/pyproject.toml`、optional dependency groups、package data、console scripts を持つ installable backend package。 |
| User entrypoints | `python aiogym/train.py`, `python aiogym/train_rlpd.py`, `train_all.sh` などを直接実行。 | `aiogym-suite-benchmark`, `aiogym-report`, `aiogym-train-sb3`, `aiogym-train-rlpd` などの stable commands。 |
| Python API | `aiogym.__init__` や internal modules から直接 import する形が中心。 | `aiogym.define_model`, `aiogym.register_model`, `aiogym.make_model`, `aiogym.make_env`, `aiogym.run_benchmark`, `aiogym.plot_results` の小さな public API。 |
| Backend layout | `models.py`, `kernel.py`, `baselines.py`, `oracle.py`, `rlpd.py`, `train.py`, `train_sac.py`, `train_rlpd.py` が flat に配置。 | `models/`, `env`, `controllers/`, `evaluation/`, `rl/`, `cli/` に分割し、public user functions は top-level `aiogym` から re-export。 |
| Model coverage | cascade, quadruple, CSTR, HVAC, fired heater. | cascade, quadruple, CSTR, HVAC, extraction, fired heater, crystallization. |
| Controllers | PID, MPC, evaluation helpers, baseline concepts が `baselines.py` と `oracle.py` に集中。 | controller interface, registry, configs, adapters, PID, MPC, oracle, tuning tools が `aiogym.controllers` に集約。 |
| Evaluation | scripts/helpers が中心。 | `aiogym.evaluation` が benchmark objectives/configs, rollout collection, metrics, reports, plots, artifacts, suites を担当。 |
| RL code | `rlpd.py`, `train_rlpd.py`, `train_sac.py`, `train.py` が core backend modules と同じ階層。 | RL algorithms と training flows は `aiogym.rl` に集約。default outputs は `aiogym/runs/rl/`。 |
| Generated outputs | historical run JSON が `aiogym/runs/` に tracked されていた。 | `aiogym/runs/` は local output area。source control には `.gitignore` のみ残す。 |

## Recommended Mental Model

新しい backend code を追加するときは、次の構造に従います。

```text
aiogym/
  cli/            # terminal entrypoints only
  env.py          # Gymnasium environment and make_env helper
  models/         # process models, contracts, registry, model cards
  controllers/    # controller API, built-in controllers, configs, tuning
  evaluation/     # benchmark objectives/configs, metrics, reports, plots, artifacts
  rl/             # RL algorithms and training workflows
  runs/           # local generated outputs
```

`cli/` は terminal entrypoint であり、core logic を置く場所ではありません。通常の Python user entrypoints は top-level `aiogym` package から re-export されます。core behavior は `env.py`, `models/`, `controllers/`, `evaluation/`, `rl/` に置きます。

Custom scenario の public API は役割を分けています。`define_model(...)`
は declarative spec から model を定義し、`register_model(...)` は scenario
name に登録します。`make_model(...)` は `make_env(...)` の model-layer
counterpart で、scenario name、model class、factory、model instance から
process model instance を作成または検証します。
