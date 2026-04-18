from __future__ import annotations

from pathlib import Path

from gravnav.ml.experiment_registry import (
    list_corpus_presets,
    list_region_sets,
    resolve_corpus_preset,
    resolve_region_set,
)


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_registry_lists_named_sets_and_presets() -> None:
    assert "norway4" in list_region_sets(project_root=ROOT)
    assert "all_wave1" in list_region_sets(project_root=ROOT)
    assert "dev" in list_corpus_presets(project_root=ROOT)
    assert "full" in list_corpus_presets(project_root=ROOT)


def test_resolve_norway4_region_set_has_no_missing_manifests() -> None:
    resolved = resolve_region_set("norway4", project_root=ROOT)
    assert len(resolved.manifest_paths) == 4
    assert len(resolved.missing_manifest_paths) == 0
    assert all(path.exists() for path in resolved.manifest_paths)


def test_resolve_all_wave1_region_set_tracks_missing_external_manifests() -> None:
    resolved = resolve_region_set("all_wave1", project_root=ROOT)
    assert len(resolved.manifest_paths) >= 4
    assert len(resolved.missing_manifest_paths) == 3
    assert resolved.metadata["allow_missing_manifests"] is True


def test_resolve_corpus_preset_matches_expected_fixed_stage_defaults() -> None:
    dev = resolve_corpus_preset("dev", project_root=ROOT)
    full = resolve_corpus_preset("full", project_root=ROOT)

    assert int(dev["max_examples_per_region"]) == 32
    assert int(dev["num_route_variants_per_region"]) == 2
    assert int(full["max_examples_per_region"]) == 48
    assert int(full["num_edge_biased_realizations_per_region"]) == 4
