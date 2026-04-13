#!/usr/bin/env python3
"""
Summarize two-region maritime demo validation into markdown and JSON reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


PROMOTED_LABEL = "photonic_gravity_bathymetry_lag"
LIVE_LABEL = "live_ins"


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(statistics.median(values))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}.")
    return [dict(row) for row in payload]


def _group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["label"]), []).append(row)
    return out


def _region_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = _group_by_label(rows)
    if LIVE_LABEL not in by_label or PROMOTED_LABEL not in by_label:
        raise KeyError(
            f"Expected labels {LIVE_LABEL!r} and {PROMOTED_LABEL!r} in region {name!r}."
        )

    live_rows = by_label[LIVE_LABEL]
    promoted_rows = by_label[PROMOTED_LABEL]

    live_rmse_med = _median(live_rows, "ins_horizontal_rmse_m")
    live_cep95_med = _median(live_rows, "ins_cep95_m")
    promoted_rmse_med = _median(promoted_rows, "earth_signature_horizontal_rmse_m")
    promoted_cep95_med = _median(promoted_rows, "earth_signature_cep95_m")

    seed_checks: list[dict[str, Any]] = []
    zero_hmi_all = True
    max_worse_fraction = 0.0
    for row in promoted_rows:
        ins_rmse = float(row["ins_horizontal_rmse_m"])
        earth_rmse = float(row["earth_signature_horizontal_rmse_m"])
        hmi = float(row["earth_signature_hmi_horizontal"])
        worse_fraction = 0.0 if ins_rmse <= 0.0 else max(0.0, earth_rmse - ins_rmse) / ins_rmse
        zero_hmi_all = zero_hmi_all and (hmi == 0.0)
        max_worse_fraction = max(max_worse_fraction, worse_fraction)
        seed_checks.append(
            {
                "seed": int(row["seed"]),
                "ins_horizontal_rmse_m": ins_rmse,
                "earth_signature_horizontal_rmse_m": earth_rmse,
                "earth_signature_cep95_m": float(row["earth_signature_cep95_m"]),
                "earth_signature_hmi_horizontal": hmi,
                "worse_fraction_vs_live_ins": worse_fraction,
            }
        )

    promoted_output_mode = str(promoted_rows[0]["earth_signature_mode"])
    accepted = bool(
        promoted_output_mode == "lag_smoothed"
        and promoted_rmse_med is not None
        and promoted_cep95_med is not None
        and live_rmse_med is not None
        and live_cep95_med is not None
        and promoted_rmse_med < live_rmse_med
        and promoted_cep95_med < live_cep95_med
        and zero_hmi_all
        and max_worse_fraction <= 0.05
    )

    return {
        "region_name": name,
        "num_rows": len(rows),
        "num_seeds": len(promoted_rows),
        "promoted_label": PROMOTED_LABEL,
        "promoted_output_mode": promoted_output_mode,
        "live_ins_horizontal_rmse_m_median": live_rmse_med,
        "live_ins_cep95_m_median": live_cep95_med,
        "promoted_horizontal_rmse_m_median": promoted_rmse_med,
        "promoted_cep95_m_median": promoted_cep95_med,
        "promoted_hmi_horizontal_median": _median(
            promoted_rows,
            "earth_signature_hmi_horizontal",
        ),
        "promoted_hmi_zero_all_seeds": zero_hmi_all,
        "max_seed_worse_fraction_vs_live_ins": max_worse_fraction,
        "seed_checks": seed_checks,
        "accepted": accepted,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _write_markdown(
    path: Path,
    *,
    region_one_summary: dict[str, Any],
    region_two_summary: dict[str, Any],
    next_branch: str,
) -> None:
    lines = [
        "# Two-Region Validation Report",
        "",
        "## Decision",
        "",
        f"- first region: `{region_one_summary['region_name']}` -> `{_format_bool(bool(region_one_summary['accepted']))}` as promoted output",
        f"- second region: `{region_two_summary['region_name']}` -> `{_format_bool(bool(region_two_summary['accepted']))}` as promoted output",
        f"- next branch: `{next_branch}`",
        "",
        "## Acceptance Criteria",
        "",
        "- promoted output must be `photonic_gravity_bathymetry_lag`",
        "- promoted output must beat live INS on median horizontal RMSE and median CEP95",
        "- horizontal HMI must remain `0.0` on every acceptance seed",
        "- no seed may be worse than live INS by more than `5%` on horizontal RMSE",
        "",
        "## Region Summaries",
        "",
        "| Region | Live INS RMSE med [m] | Promoted RMSE med [m] | Live INS CEP95 med [m] | Promoted CEP95 med [m] | HMI zero all seeds | Max seed worse fraction | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for summary in (region_one_summary, region_two_summary):
        lines.append(
            f"| `{summary['region_name']}` | "
            f"{summary['live_ins_horizontal_rmse_m_median']:.3f} | "
            f"{summary['promoted_horizontal_rmse_m_median']:.3f} | "
            f"{summary['live_ins_cep95_m_median']:.3f} | "
            f"{summary['promoted_cep95_m_median']:.3f} | "
            f"`{summary['promoted_hmi_zero_all_seeds']}` | "
            f"{summary['max_seed_worse_fraction_vs_live_ins']:.3f} | "
            f"`{summary['accepted']}` |"
        )

    lines.extend(
        [
            "",
            f"## Second-Region Detail: `{region_two_summary['region_name']}`",
            "",
            "| Seed | Live INS RMSE [m] | Promoted RMSE [m] | Promoted CEP95 [m] | HMI horiz | Worse fraction vs INS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in region_two_summary["seed_checks"]:
        lines.append(
            f"| `{row['seed']}` | "
            f"{row['ins_horizontal_rmse_m']:.3f} | "
            f"{row['earth_signature_horizontal_rmse_m']:.3f} | "
            f"{row['earth_signature_cep95_m']:.3f} | "
            f"{row['earth_signature_hmi_horizontal']:.3f} | "
            f"{row['worse_fraction_vs_live_ins']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The first-region milestone is considered stable only if the rerun still satisfies the promoted-output gate.",
            "- The second region is the actual generalization gate for this branch.",
            f"- Branch promotion decision: `{next_branch}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_second_region_markdown(
    path: Path,
    *,
    region_summary: dict[str, Any],
    next_branch: str,
) -> None:
    lines = [
        "# Second-Region Validation Report",
        "",
        "## Region",
        "",
        f"- region: `{region_summary['region_name']}`",
        f"- promoted output label: `{region_summary['promoted_label']}`",
        f"- promoted output mode: `{region_summary['promoted_output_mode']}`",
        f"- accepted: `{region_summary['accepted']}`",
        f"- next branch: `{next_branch}`",
        "",
        "## Median Results",
        "",
        f"- live INS horizontal RMSE: `{region_summary['live_ins_horizontal_rmse_m_median']:.3f} m`",
        f"- promoted horizontal RMSE: `{region_summary['promoted_horizontal_rmse_m_median']:.3f} m`",
        f"- live INS CEP95: `{region_summary['live_ins_cep95_m_median']:.3f} m`",
        f"- promoted CEP95: `{region_summary['promoted_cep95_m_median']:.3f} m`",
        f"- zero horizontal HMI on every seed: `{region_summary['promoted_hmi_zero_all_seeds']}`",
        f"- max per-seed worse fraction vs live INS: `{region_summary['max_seed_worse_fraction_vs_live_ins']:.3f}`",
        "",
        "## Seed Checks",
        "",
        "| Seed | Live INS RMSE [m] | Promoted RMSE [m] | Promoted CEP95 [m] | HMI horiz | Worse fraction vs INS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in region_summary["seed_checks"]:
        lines.append(
            f"| `{row['seed']}` | "
            f"{row['ins_horizontal_rmse_m']:.3f} | "
            f"{row['earth_signature_horizontal_rmse_m']:.3f} | "
            f"{row['earth_signature_cep95_m']:.3f} | "
            f"{row['earth_signature_hmi_horizontal']:.3f} | "
            f"{row['worse_fraction_vs_live_ins']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This branch passes only if the second region also promotes the bounded-lag output under the frozen passive stack.",
            f"- Current outcome: `{_format_bool(bool(region_summary['accepted']))}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the two-region validation report.")
    parser.add_argument("--region-one-name", required=True)
    parser.add_argument("--region-one-summary", required=True)
    parser.add_argument("--region-two-name", required=True)
    parser.add_argument("--region-two-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    region_one_summary = _region_summary(
        str(args.region_one_name),
        _load_rows(Path(args.region_one_summary).expanduser().resolve()),
    )
    region_two_summary = _region_summary(
        str(args.region_two_name),
        _load_rows(Path(args.region_two_summary).expanduser().resolve()),
    )

    overall_pass = bool(region_one_summary["accepted"] and region_two_summary["accepted"])
    next_branch = (
        "feature/magnetic-passive-aiding"
        if overall_pass
        else "feature/regional-demo-pack-hardening"
    )
    payload = {
        "overall_pass": overall_pass,
        "next_branch": next_branch,
        "regions": [region_one_summary, region_two_summary],
    }
    _write_json(output_dir / "two_region_validation_summary.json", payload)
    _write_json(output_dir / "second_region_validation_summary.json", region_two_summary)
    _write_markdown(
        output_dir / "two_region_validation_report.md",
        region_one_summary=region_one_summary,
        region_two_summary=region_two_summary,
        next_branch=next_branch,
    )
    _write_second_region_markdown(
        output_dir / "second_region_validation_report.md",
        region_summary=region_two_summary,
        next_branch=next_branch,
    )
    print(f"Summary JSON: {output_dir / 'two_region_validation_summary.json'}")
    print(f"Report: {output_dir / 'two_region_validation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
