"""Generated-output path defaults and overrides."""
from __future__ import annotations

from types import SimpleNamespace

from aiogym._internal.paths import RUNS_DIR_ENV, run_path, runs_dir
from aiogym.evaluation.suite import artifact_dir_for
from aiogym.evaluation import run_benchmark
from aiogym.rl.train_rlpd import output_base_for


def test_runs_dir_defaults_to_working_directory_runs(monkeypatch):
    monkeypatch.delenv(RUNS_DIR_ENV, raising=False)

    assert str(runs_dir()) == "runs"
    assert str(run_path("rl", "sb3")) == "runs/rl/sb3"
    assert artifact_dir_for("example", run_id="fixed") == (
        "runs/bench_suite_example_fixed_artifacts"
    )
    args = SimpleNamespace(out=None, scenario="cstr")
    assert output_base_for(args, "fixed") == "runs/rl/rlpd/cstr_fixed"


def test_runs_dir_environment_override_applies_only_to_defaults(monkeypatch, tmp_path):
    custom_root = tmp_path / "generated"
    monkeypatch.setenv(RUNS_DIR_ENV, str(custom_root))

    assert runs_dir() == custom_root
    assert run_path("model_metadata") == custom_root / "model_metadata"
    assert artifact_dir_for("example", run_id="fixed") == str(
        custom_root / "bench_suite_example_fixed_artifacts"
    )
    args = SimpleNamespace(out=None, scenario="heater")
    assert output_base_for(args, "fixed") == str(
        custom_root / "rl" / "rlpd" / "heater_fixed"
    )

    assert artifact_dir_for("example", artifact_dir="explicit/suite") == (
        "explicit/suite"
    )
    args.out = "explicit/rlpd"
    assert output_base_for(args, "ignored") == "explicit/rlpd"


def test_empty_runs_dir_environment_uses_default(monkeypatch):
    monkeypatch.setenv(RUNS_DIR_ENV, "")

    assert str(runs_dir()) == "runs"


def test_public_benchmark_uses_configured_default_root(monkeypatch, tmp_path):
    custom_root = tmp_path / "benchmark-output"
    monkeypatch.setenv(RUNS_DIR_ENV, str(custom_root))

    payload = run_benchmark({
        "scenario": "cstr",
        "objective": "tracking",
        "controllers": ["pid"],
        "seeds": [3],
        "episode_steps": 1,
        "include_episodes": False,
    })

    expected = custom_root / "benchmark_cstr_default_tracking"
    assert payload["run_dir"] == str(expected)
    assert (expected / "benchmark.json").is_file()
