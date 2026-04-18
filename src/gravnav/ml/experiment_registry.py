"""
Named region-set and corpus-preset helpers for real-ocean ML experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..utils.config import find_project_root


DEFAULT_EXPERIMENT_CONFIG = Path("configs/ml/real_ocean_experiments.json")


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return find_project_root()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ResolvedRegionSet:
    name: str
    manifest_paths: tuple[Path, ...]
    missing_manifest_paths: tuple[Path, ...]
    metadata: dict[str, Any]


def load_real_ocean_experiment_config(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    config_path = (
        root / DEFAULT_EXPERIMENT_CONFIG
        if path is None
        else Path(path).expanduser().resolve()
    )
    return _load_json(config_path)


def list_region_sets(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    config = load_real_ocean_experiment_config(path, project_root=project_root)
    return tuple(str(name) for name in config.get("region_sets", {}).keys())


def list_corpus_presets(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    config = load_real_ocean_experiment_config(path, project_root=project_root)
    return tuple(str(name) for name in config.get("corpus_presets", {}).keys())


def _resolve_region_set_definition(
    *,
    set_name: str,
    region_sets: dict[str, Any],
    root: Path,
    visited: set[str],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if set_name in visited:
        raise ValueError(f"Cyclic region-set dependency detected at {set_name!r}.")
    if set_name not in region_sets:
        raise KeyError(f"Unknown region set {set_name!r}.")
    visited.add(set_name)
    node = dict(region_sets[set_name])
    allow_missing = bool(node.get("allow_missing_manifests", False))

    resolved: list[Path] = []
    missing: list[Path] = []
    for child in node.get("include_sets", []):
        child_resolved, child_missing, _ = _resolve_region_set_definition(
            set_name=str(child),
            region_sets=region_sets,
            root=root,
            visited=visited,
        )
        resolved.extend(child_resolved)
        missing.extend(child_missing)
    for raw_path in node.get("manifests", []):
        manifest_path = (root / str(raw_path)).resolve()
        if manifest_path.exists():
            resolved.append(manifest_path)
        elif allow_missing:
            missing.append(manifest_path)
        else:
            raise FileNotFoundError(
                f"Region set {set_name!r} requires manifest {manifest_path}."
            )

    unique_resolved = list(dict.fromkeys(resolved))
    unique_missing = list(dict.fromkeys(missing))
    metadata = {
        "description": node.get("description"),
        "allow_missing_manifests": allow_missing,
        "selected_regions": list(node.get("selected_regions", [])),
        "candidate_metadata": dict(node.get("candidate_metadata", {})),
    }
    return unique_resolved, unique_missing, metadata


def resolve_region_set(
    set_name: str,
    *,
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ResolvedRegionSet:
    root = _project_root(project_root)
    config = load_real_ocean_experiment_config(path, project_root=root)
    resolved, missing, metadata = _resolve_region_set_definition(
        set_name=str(set_name),
        region_sets=dict(config.get("region_sets", {})),
        root=root,
        visited=set(),
    )
    return ResolvedRegionSet(
        name=str(set_name),
        manifest_paths=tuple(resolved),
        missing_manifest_paths=tuple(missing),
        metadata=metadata,
    )


def resolve_corpus_preset(
    preset_name: str,
    *,
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_real_ocean_experiment_config(path, project_root=project_root)
    presets = dict(config.get("corpus_presets", {}))
    if preset_name not in presets:
        raise KeyError(f"Unknown corpus preset {preset_name!r}.")
    return dict(presets[preset_name])


__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG",
    "ResolvedRegionSet",
    "list_corpus_presets",
    "list_region_sets",
    "load_real_ocean_experiment_config",
    "resolve_corpus_preset",
    "resolve_region_set",
]
