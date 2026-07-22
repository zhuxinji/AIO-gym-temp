from __future__ import annotations

from pathlib import Path

import pytest

import aiogym.cli.main as cli
from aiogym.catalog import list_controllers, list_scenarios, list_suites, list_tasks


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["list", "scenarios"], list_scenarios()),
        (["list", "tasks"], list_tasks()),
        (["list", "suites"], list_suites()),
        (["list", "controllers"], list_controllers()),
    ],
)
def test_list_commands_print_canonical_ids(argv, expected, capsys):
    assert cli.main(argv) is None
    assert capsys.readouterr().out.splitlines() == list(expected)


def test_list_tasks_can_filter_by_scenario(capsys):
    scenarios_with_tasks = sorted({task.split("/", 1)[0] for task in list_tasks()})
    scenario = scenarios_with_tasks[0]

    cli.main(["list", "tasks", "--scenario", scenario])

    assert capsys.readouterr().out.splitlines() == list(list_tasks(scenario))


@pytest.mark.parametrize(
    ("route", "target"),
    [
        (("benchmark", "run"), "_single_benchmark"),
        (("benchmark", "suite"), "_suite_benchmark"),
        (("train", "sb3"), "_train_sb3"),
        (("train", "rlpd"), "_train_rlpd"),
        (("artifacts", "report"), "_artifact_report"),
        (("artifacts", "check"), "_artifact_check"),
        (("model-cards",), "_model_cards"),
    ],
)
def test_unified_commands_delegate_arguments_unchanged(monkeypatch, route, target):
    received = []

    def fake_handler(argv):
        received.append(argv)
        return 17

    monkeypatch.setattr(cli, target, fake_handler)

    result = cli.main([*route, "--example", "value", "--help"])

    assert result == 17
    assert received == [["--example", "value", "--help"]]


def test_group_without_leaf_prints_group_help(capsys):
    assert cli.main(["benchmark"]) == 0
    output = capsys.readouterr().out
    assert "usage: aiogym benchmark" in output
    assert "run" in output
    assert "suite" in output


def test_console_script_metadata_keeps_legacy_aliases():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text()

    expected_entries = (
        'aiogym = "aiogym.cli.main:main"',
        'aiogym-artifact-check = "aiogym.cli.artifact_tools:artifact_check_main"',
        'aiogym-model-cards = "aiogym.cli.artifact_tools:model_cards_main"',
        'aiogym-report = "aiogym.cli.artifact_tools:report_main"',
        'aiogym-single-benchmark = "aiogym.cli.single_benchmark:main"',
        'aiogym-suite-benchmark = "aiogym.cli.suite_benchmark:main"',
        'aiogym-train-rlpd = "aiogym.rl.train_rlpd:main"',
        'aiogym-train-sb3 = "aiogym.rl.train_sb3:main"',
    )
    assert all(entry in text for entry in expected_entries)
