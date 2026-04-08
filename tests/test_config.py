from __future__ import annotations

from pathlib import Path

import pytest

from gravnav.truth.scenarios import available_scenario_names
from gravnav.utils.config import (
    ConfigPathError,
    OptionalDependencyError,
    load_scenario_spec,
)


def test_load_scenario_spec_falls_back_to_builtin_for_empty_yaml_stub(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    scenario_dir = tmp_path / "configs" / "scenarios"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "maritime_baseline.yaml").write_text("", encoding="utf-8")

    scenario = load_scenario_spec("maritime_baseline", project_root=tmp_path)

    assert scenario.name == "maritime_baseline"
    assert "maritime_baseline" in available_scenario_names()


def test_load_scenario_spec_explicit_empty_yaml_does_not_fallback(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    scenario_dir = tmp_path / "configs" / "scenarios"
    scenario_dir.mkdir(parents=True)
    scenario_path = scenario_dir / "maritime_baseline.yaml"
    scenario_path.write_text("", encoding="utf-8")

    with pytest.raises((OptionalDependencyError, KeyError)):
        load_scenario_spec(scenario_path, project_root=tmp_path)


def test_load_unknown_scenario_name_still_raises(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "configs" / "scenarios").mkdir(parents=True)

    with pytest.raises(ConfigPathError):
        load_scenario_spec("does_not_exist", project_root=tmp_path)
