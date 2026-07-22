#!/usr/bin/env python3
"""Command-line tools for benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aiogym._internal.paths import run_path
from aiogym.evaluation import check_benchmark_artifacts, render_benchmark_report
from aiogym.models import (
    collect_model_cards,
    export_model_card_markdown,
    export_model_cards,
    render_model_card_markdown,
)


def parse_scenarios(raw: str | None):
    if raw is None:
        return None
    scenarios = [part.strip() for part in raw.split(",") if part.strip()]
    if not scenarios:
        raise ValueError("--scenarios must contain at least one scenario name")
    return scenarios


def model_cards_main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Export or validate model-card metadata for registered scenarios."
    )
    ap.add_argument("--out-dir", default=str(run_path("model_cards")),
                    help="directory for exported model cards")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario override; defaults to all registered scenarios")
    ap.add_argument("--format", default="json", choices=["json", "markdown", "both"],
                    help="export machine-readable JSON, human-readable Markdown, or both")
    ap.add_argument("--check", action="store_true",
                    help="validate and print the manifest without writing files")
    ap.add_argument("--no-manifest", action="store_true",
                    help="skip writing manifest/index files")
    args = ap.parse_args(argv)

    scenarios = parse_scenarios(args.scenarios)
    if args.check:
        cards = collect_model_cards(scenarios)
        stale = []
        if args.format in {"markdown", "both"}:
            out = Path(args.out_dir)
            expected = {
                f"{scenario}.md": render_model_card_markdown(card)
                for scenario, card in cards.items()
            }
            index_lines = [
                "# AIO-Gym Model Cards",
                "",
                "Human-readable summaries generated from the registered process-model metadata.",
                "",
            ]
            for scenario, card in cards.items():
                index_lines.append(f"- [{card['name']} (`{scenario}`)]({scenario}.md)")
            expected["README.md"] = "\n".join(index_lines) + "\n"
            for name, content in expected.items():
                path = out / name
                if not path.exists() or path.read_text() != content:
                    stale.append(str(path))
        print(json.dumps({"count": len(cards), "format": args.format, "scenarios": list(cards)}, indent=2))
        if stale:
            raise SystemExit("stale model-card docs: " + ", ".join(stale))
        return

    if args.format == "json":
        manifest = export_model_cards(args.out_dir, scenarios=scenarios, write_manifest=not args.no_manifest)
    elif args.format == "markdown":
        manifest = export_model_card_markdown(args.out_dir, scenarios=scenarios, write_index=not args.no_manifest)
    else:
        out = Path(args.out_dir)
        manifest = {
            "json": export_model_cards(out / "json", scenarios=scenarios, write_manifest=not args.no_manifest),
            "markdown": export_model_card_markdown(out / "markdown", scenarios=scenarios,
                                                   write_index=not args.no_manifest),
        }
    print(json.dumps(manifest, indent=2))


def report_main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Render report.md from a standard benchmark artifact directory."
    )
    ap.add_argument("artifact_dir", help="standard benchmark artifact directory")
    ap.add_argument("--out", default=None, help="report path; defaults to <artifact_dir>/report.md")
    ap.add_argument("--stdout", action="store_true", help="also print the report")
    args = ap.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    out_path = Path(args.out) if args.out else artifact_dir / "report.md"
    text = render_benchmark_report(artifact_dir, out_path=out_path)
    if args.stdout:
        print(text)
    print(f"saved report {out_path}")


def artifact_check_main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Validate a standard benchmark artifact directory."
    )
    ap.add_argument("artifact_dir", help="standard benchmark artifact directory")
    ap.add_argument("--json", action="store_true", help="print structured JSON")
    args = ap.parse_args(argv)

    result = check_benchmark_artifacts(args.artifact_dir)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['artifact_dir']}")
        for row in result["checks"]:
            mark = "OK" if row["ok"] else "FAIL"
            print(f"  {mark:4s} {row['name']}: {row['message']} ({row['path']})")
    if not result["ok"]:
        raise SystemExit(1)


def main(argv=None):
    commands = {
        "model-cards": model_cards_main,
        "report": report_main,
        "check": artifact_check_main,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m aiogym.cli.artifact_tools {model-cards,report,check} ...")
        return
    command = args[0]
    if command not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"unknown artifacts command {command!r}; choose one of: {choices}")
    commands[command](args[1:])


if __name__ == "__main__":
    main()
