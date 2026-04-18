from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "bootstrap_public_region.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bootstrap_public_region_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_loads_mid_atlantic_ridge_recipe() -> None:
    module = _load_module()
    recipe = module._resolve_recipe(
        Path("configs/ml/public_region_recipes.json").resolve(),
        "mid_atlantic_ridge",
    )
    assert recipe["region_name"] == "mid_atlantic_ridge"
    assert recipe["gravity"]["scenario_out"].endswith(
        "mid_atlantic_ridge_public_demo_maritime.json"
    )


def test_bootstrap_builds_gravity_command_for_mid_atlantic_ridge() -> None:
    module = _load_module()
    recipe = module._resolve_recipe(
        Path("configs/ml/public_region_recipes.json").resolve(),
        "mid_atlantic_ridge",
    )
    cmd = module._build_gravity_command(
        recipe=recipe,
        raw_format="scattered_xyz",
        raw_path="data/gravity_maps/raw/mid_atlantic_ridge/region_input.xyz",
    )
    rendered = " ".join(cmd)
    assert "prepare_public_gravity_region.py" in rendered
    assert "mid_atlantic_ridge_public_gravity_manifest.json" in rendered
    assert "mid_atlantic_ridge_public_demo_maritime.json" in rendered


def test_bootstrap_builds_demo_pack_and_multimodal_commands() -> None:
    module = _load_module()
    recipe = module._resolve_recipe(
        Path("configs/ml/public_region_recipes.json").resolve(),
        "mid_atlantic_ridge",
    )
    demo_cmd = module._build_demo_pack_command(recipe=recipe)
    multimodal_cmd = module._build_multimodal_command(recipe=recipe)
    demo_rendered = " ".join(demo_cmd)
    multimodal_rendered = " ".join(multimodal_cmd)

    assert "prepare_norwegian_maritime_demo.py" in demo_rendered
    assert "mid_atlantic_ridge_public_base_demo_pack.json" in demo_rendered
    assert "prepare_public_multimodal_region.py" in multimodal_rendered
    assert "mid_atlantic_ridge_public_demo_pack.json" in multimodal_rendered
