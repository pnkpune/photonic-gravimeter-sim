from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_hybrid_feedback_program.py"
    spec = importlib.util.spec_from_file_location(
        "run_hybrid_feedback_program",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_primary_metrics_uses_lag_output_for_lag_smoothed_label() -> None:
    module = _load_module()
    metrics = module.primary_metrics_from_row(
        {
            "label": "sequence_lag_smoothed",
            "horizontal_rmse_m": 90.0,
            "cep95_m": 120.0,
            "hmi_horizontal": 0.2,
            "lag_smoothed_horizontal_rmse_m": 55.0,
            "lag_smoothed_cep95_m": 80.0,
            "lag_hmi_horizontal": 0.0,
        }
    )
    assert metrics["horizontal_rmse_m"] == 55.0
    assert metrics["cep95_m"] == 80.0
    assert metrics["hmi_horizontal"] == 0.0


def test_aggregate_and_acceptance_for_primary_target() -> None:
    module = _load_module()
    target = module.BenchmarkTarget(
        name="test_region",
        scenario_path=Path("/tmp/scenario.json"),
        demo_pack_manifest=Path("/tmp/demo.json"),
        acceptance_mode="beat_live",
    )
    rows = [
        {
            "label": "live_ins",
            "horizontal_rmse_m": 100.0,
            "cep95_m": 140.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": None,
            "sequence_allowed_updates": None,
            "sequence_trust_positive_updates": None,
        },
        {
            "label": "live_ins",
            "horizontal_rmse_m": 110.0,
            "cep95_m": 150.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": None,
            "sequence_allowed_updates": None,
            "sequence_trust_positive_updates": None,
        },
        {
            "label": "sequence_replay_learned_gain",
            "horizontal_rmse_m": 80.0,
            "cep95_m": 120.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 3,
            "sequence_allowed_updates": 4,
            "sequence_trust_positive_updates": 4,
            "sequence_median_gain_alpha_applied": 0.4,
        },
        {
            "label": "sequence_replay_learned_gain",
            "horizontal_rmse_m": 82.0,
            "cep95_m": 121.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 2,
            "sequence_allowed_updates": 5,
            "sequence_trust_positive_updates": 5,
            "sequence_median_gain_alpha_applied": 0.5,
        },
        {
            "label": "sequence_replay_committee_gain",
            "horizontal_rmse_m": 75.0,
            "cep95_m": 115.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 2,
            "sequence_allowed_updates": 3,
            "sequence_trust_positive_updates": 3,
            "sequence_median_gain_alpha_applied": 0.3,
        },
        {
            "label": "sequence_replay_committee_gain",
            "horizontal_rmse_m": 78.0,
            "cep95_m": 118.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 2,
            "sequence_allowed_updates": 3,
            "sequence_trust_positive_updates": 3,
            "sequence_median_gain_alpha_applied": 0.32,
        },
    ]
    summary = module.aggregate_benchmark_rows(rows)
    decision = module.evaluate_target_acceptance(target, summary)
    assert summary["sequence_replay_committee_gain"]["total_sequence_applied_updates"] == 4
    assert summary["sequence_replay_committee_gain"]["hmi_zero_all_rows"] is True
    assert summary["sequence_replay_committee_gain"]["median_sequence_gain_alpha_applied"] == 0.31
    assert decision["beats_single_model_learned"] is True
    assert decision["beats_live"] is True
    assert decision["accepted"] is True


def test_acceptance_prefers_committee_topk_when_present() -> None:
    module = _load_module()
    target = module.BenchmarkTarget(
        name="test_region",
        scenario_path=Path("/tmp/scenario.json"),
        demo_pack_manifest=Path("/tmp/demo.json"),
        acceptance_mode="beat_live",
    )
    rows = [
        {
            "label": "live_ins",
            "horizontal_rmse_m": 100.0,
            "cep95_m": 140.0,
            "hmi_horizontal": 0.0,
        },
        {
            "label": "sequence_replay_learned_gain",
            "horizontal_rmse_m": 90.0,
            "cep95_m": 135.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 1,
        },
        {
            "label": "sequence_replay_committee_gain",
            "horizontal_rmse_m": 88.0,
            "cep95_m": 132.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 1,
        },
        {
            "label": "sequence_replay_committee_topk_gain",
            "horizontal_rmse_m": 82.0,
            "cep95_m": 128.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 1,
        },
    ]
    summary = module.aggregate_benchmark_rows(rows)
    decision = module.evaluate_target_acceptance(target, summary)
    assert decision["committee_label"] == "sequence_replay_committee_topk_gain"
    assert decision["beats_single_model_learned"] is True
    assert decision["beats_live"] is True
    assert decision["accepted"] is True


def test_nonregression_target_allows_small_regression_only() -> None:
    module = _load_module()
    target = module.BenchmarkTarget(
        name="external_region",
        scenario_path=Path("/tmp/scenario.json"),
        demo_pack_manifest=Path("/tmp/demo.json"),
        acceptance_mode="nonregress_live_5pct",
    )
    rows = [
        {
            "label": "live_ins",
            "horizontal_rmse_m": 100.0,
            "cep95_m": 140.0,
            "hmi_horizontal": 0.0,
        },
        {
            "label": "sequence_replay_learned_gain",
            "horizontal_rmse_m": 104.0,
            "cep95_m": 141.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 0,
        },
        {
            "label": "sequence_replay_committee_gain",
            "horizontal_rmse_m": 104.0,
            "cep95_m": 141.0,
            "hmi_horizontal": 0.0,
            "sequence_applied_updates": 0,
        },
    ]
    summary = module.aggregate_benchmark_rows(rows)
    decision = module.evaluate_target_acceptance(target, summary)
    assert decision["within_live_5pct"] is True
    assert decision["accepted"] is True

    rows[2]["horizontal_rmse_m"] = 106.0
    summary = module.aggregate_benchmark_rows(rows)
    decision = module.evaluate_target_acceptance(target, summary)
    assert decision["within_live_5pct"] is False
    assert decision["accepted"] is False


def test_find_benchmark_summary_file_ignores_per_run_summaries(tmp_path: Path) -> None:
    module = _load_module()
    overall = tmp_path / "scenario_summary.json"
    overall.write_text("{}\n", encoding="utf-8")
    for label in module.HYBRID_BENCHMARK_LABELS:
        (tmp_path / f"scenario_{label}_summary.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
    found = module.find_benchmark_summary_file(tmp_path)
    assert found == overall
