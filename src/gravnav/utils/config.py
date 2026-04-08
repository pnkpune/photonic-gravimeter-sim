"""
config.py

Configuration loading, serialization, and path-resolution utilities for the
gravity-aided navigation simulator.

This module is the thin glue layer between:
- on-disk configuration files under `configs/...`
- typed scenario objects in `gravnav.truth.scenarios`
- later simulation runners that will need a predictable way to locate, load,
  and validate scenario / sensor / Monte Carlo configs

Why this file exists
--------------------
The repository already contains configuration directories for:
- scenarios
- sensors
- Monte Carlo runs

and the truth layer now exposes:
- `ScenarioSpec`
- named built-in scenarios
- mapping-based construction via `ScenarioSpec.from_mapping(...)`

Therefore this module should do four things well:

1) Locate the project root and standard config directories.
2) Read plain mappings from YAML / JSON / TOML files.
3) Write mappings back to disk where practical.
4) Convert scenario config files into typed `ScenarioSpec` objects.

Supported formats
-----------------
Read support:
- YAML  (via optional PyYAML dependency)
- JSON  (Python standard library)
- TOML  (Python standard library `tomllib`, read-only)

Write support:
- YAML  (via optional PyYAML dependency)
- JSON  (Python standard library)

TOML note
---------
Python's standard-library `tomllib` module parses TOML but does not support
writing TOML. For that reason this module intentionally treats TOML as
read-only unless you later add an external writer.

Security note
-------------
When reading YAML, this module uses `yaml.safe_load(...)` rather than
`yaml.load(...)`, because the PyYAML documentation explicitly warns that
`yaml.load(...)` is unsafe on untrusted input and recommends `safe_load(...)`
for ordinary data files.

Primary references used here
----------------------------
1) Python documentation: `tomllib`
   https://docs.python.org/3/library/tomllib.html

   Used for:
   - standard-library TOML parsing support
   - the statement that `tomllib` reads TOML but does not provide write support

2) PyYAML documentation
   https://pyyaml.org/wiki/PyYAMLDocumentation

   Used for:
   - `yaml.safe_load(...)`
   - the warning that `yaml.load(...)` is unsafe on untrusted input

Design notes
------------
- This module is intentionally small and conservative.
- It returns plain mappings for generic configs and typed `ScenarioSpec`
  instances for scenario configs.
- It does not try to become a full schema-validation framework.
- It uses explicit, readable exceptions so later scripts can fail loudly and
  informatively when config files are missing or malformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import tomllib

from ..truth.scenarios import ScenarioSpec, available_scenario_names, get_named_scenario

ConfigDict = dict[str, Any]

PROJECT_MARKER_FILES: Final[tuple[str, ...]] = ("pyproject.toml", "README.md")
DEFAULT_SCENARIO_DIRNAME: Final[str] = "configs/scenarios"
DEFAULT_SENSOR_DIRNAME: Final[str] = "configs/sensors"
DEFAULT_MONTE_CARLO_DIRNAME: Final[str] = "configs/monte_carlo"


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class ConfigError(Exception):
    """Base class for configuration-related errors."""


class ConfigPathError(ConfigError):
    """Raised when a configuration path cannot be resolved."""


class ConfigFormatError(ConfigError):
    """Raised when a configuration file has an unsupported or malformed format."""


class OptionalDependencyError(ConfigError):
    """Raised when a needed optional dependency (for example PyYAML) is unavailable."""


# -----------------------------------------------------------------------------
# YAML backend helpers
# -----------------------------------------------------------------------------


def _import_yaml():
    """
    Import PyYAML on demand.

    Returns
    -------
    module
        Imported `yaml` module.

    Raises
    ------
    OptionalDependencyError
        If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "PyYAML is required for YAML config support but is not installed. "
            "Install it with `pip install pyyaml`, or use JSON/TOML instead."
        ) from exc
    return yaml


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------


def expand_path(path_like: str | Path) -> Path:
    """
    Expand `~` and return an absolute path.

    Parameters
    ----------
    path_like : str or pathlib.Path
        Input path.

    Returns
    -------
    pathlib.Path
        Expanded absolute path.
    """
    return Path(path_like).expanduser().resolve()


def find_project_root(start: str | Path | None = None) -> Path:
    """
    Find the project root by walking upward until a marker file is found.

    Parameters
    ----------
    start : str or pathlib.Path, optional
        Starting file or directory. If omitted, this module's file location is
        used.

    Returns
    -------
    pathlib.Path
        Project root directory.

    Raises
    ------
    ConfigPathError
        If no plausible project root can be found.

    Notes
    -----
    A directory is considered the project root if it contains at least one of:
    - `pyproject.toml`
    - `README.md`

    The repository snapshot already includes both at the top level, so this
    heuristic is appropriate here.
    """
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = expand_path(start)

    current = start_path if start_path.is_dir() else start_path.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKER_FILES):
            return candidate

    raise ConfigPathError(
        f"Could not locate project root from start path {start_path!s}. "
        f"Expected one of {PROJECT_MARKER_FILES} in this directory or a parent."
    )


def scenario_config_dir(project_root: str | Path | None = None) -> Path:
    """
    Return the standard scenario-config directory.
    """
    root = find_project_root(project_root)
    return root / DEFAULT_SCENARIO_DIRNAME


def sensor_config_dir(project_root: str | Path | None = None) -> Path:
    """
    Return the standard sensor-config directory.
    """
    root = find_project_root(project_root)
    return root / DEFAULT_SENSOR_DIRNAME


def monte_carlo_config_dir(project_root: str | Path | None = None) -> Path:
    """
    Return the standard Monte Carlo config directory.
    """
    root = find_project_root(project_root)
    return root / DEFAULT_MONTE_CARLO_DIRNAME


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Create the parent directory of a path if needed and return the resolved path.
    """
    p = expand_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def supported_config_suffixes() -> tuple[str, ...]:
    """
    Return the supported configuration filename suffixes.

    Returns
    -------
    tuple[str, ...]
        Supported suffixes in lowercase.

    Notes
    -----
    YAML is included under both `.yaml` and `.yml`.
    TOML is read-only in this module.
    """
    return (".yaml", ".yml", ".json", ".toml")


def _normalized_suffix(path: Path) -> str:
    """
    Return the lowercase filename suffix.
    """
    return path.suffix.lower()


def is_supported_config_path(path: str | Path) -> bool:
    """
    Return True if the path suffix is one of the supported config formats.
    """
    return _normalized_suffix(Path(path)) in supported_config_suffixes()


def _is_explicit_path_reference(path_or_name: str | Path) -> bool:
    """
    Return True when the caller appears to be naming a concrete path rather than
    a bare config stem.

    Examples treated as explicit:
    - `/abs/path/file.yaml`
    - `configs/scenarios/foo.json`
    - `./foo.yaml`
    - `foo.json`
    """
    raw = Path(path_or_name).expanduser()
    text = str(path_or_name)
    return (
        raw.is_absolute()
        or raw.suffix != ""
        or len(raw.parts) > 1
        or text.startswith((".", "~"))
    )


def resolve_config_path(
    path_or_name: str | Path,
    *,
    default_dir: str | Path | None = None,
    allow_extension_inference: bool = True,
    must_exist: bool = True,
) -> Path:
    """
    Resolve a config path that may be:
    - a direct path
    - a bare stem like `maritime_baseline`
    - a relative name to be searched under `default_dir`

    Parameters
    ----------
    path_or_name : str or pathlib.Path
        Input path or bare config name.
    default_dir : str or pathlib.Path, optional
        Directory to search when a bare name is provided.
    allow_extension_inference : bool, default=True
        If True, try appending supported suffixes when no suffix is present.
    must_exist : bool, default=True
        If True, require that the resolved file exists.

    Returns
    -------
    pathlib.Path
        Resolved path.

    Raises
    ------
    ConfigPathError
        If the path cannot be resolved.
    """
    raw = Path(path_or_name).expanduser()

    candidates: list[Path] = []

    # Case 1: explicit path with suffix, or a path-like string containing separators.
    if raw.suffix:
        candidates.append(raw)
        if default_dir is not None and not raw.is_absolute():
            candidates.append(Path(default_dir) / raw)
    else:
        # Bare name or suffixless relative path.
        if raw.is_absolute():
            candidates.append(raw)
        else:
            if default_dir is not None:
                base = Path(default_dir) / raw
                candidates.append(base)
                if allow_extension_inference:
                    for suffix in supported_config_suffixes():
                        candidates.append(base.with_suffix(suffix))
            candidates.append(raw)
            if allow_extension_inference:
                for suffix in supported_config_suffixes():
                    candidates.append(raw.with_suffix(suffix))

    resolved_candidates: list[Path] = []
    for candidate in candidates:
        try:
            resolved_candidates.append(candidate.expanduser().resolve())
        except OSError:
            # Keep the original candidate semantics if resolve fails on a nonexistent path.
            resolved_candidates.append(candidate.expanduser().absolute())

    if must_exist:
        for candidate in resolved_candidates:
            if candidate.exists():
                return candidate
        raise ConfigPathError(
            f"Could not resolve config path from {path_or_name!r}. "
            f"Tried: {[str(p) for p in resolved_candidates]}"
        )

    return resolved_candidates[0]


# -----------------------------------------------------------------------------
# Generic mapping validation
# -----------------------------------------------------------------------------


def _require_mapping(data: Any, *, path: Path | None = None) -> ConfigDict:
    """
    Require that parsed config data is a mapping and return it as a plain dict.
    """
    if not isinstance(data, Mapping):
        where = "" if path is None else f" in file {path!s}"
        raise ConfigFormatError(
            f"Expected a top-level mapping/dictionary{where}, got {type(data).__name__}."
        )
    return dict(data)


def _reject_unsupported_suffix(path: Path) -> None:
    """
    Raise `ConfigFormatError` for unsupported file suffixes.
    """
    suffix = _normalized_suffix(path)
    if suffix not in supported_config_suffixes():
        raise ConfigFormatError(
            f"Unsupported config format {suffix!r} for {path!s}. "
            f"Supported suffixes: {supported_config_suffixes()}."
        )


# -----------------------------------------------------------------------------
# Generic read/write functions
# -----------------------------------------------------------------------------


def load_config_mapping(path: str | Path) -> ConfigDict:
    """
    Load a configuration mapping from YAML, JSON, or TOML.

    Parameters
    ----------
    path : str or pathlib.Path
        Config file path.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.

    Raises
    ------
    ConfigPathError
        If the file does not exist.
    ConfigFormatError
        If the file format is unsupported or malformed.
    OptionalDependencyError
        If YAML is requested but PyYAML is not installed.

    Format handling
    ---------------
    - YAML: `yaml.safe_load(...)`
    - JSON: `json.load(...)`
    - TOML: `tomllib.load(...)` with a binary file object
    """
    resolved = resolve_config_path(path, must_exist=True)
    _reject_unsupported_suffix(resolved)
    suffix = _normalized_suffix(resolved)

    try:
        if suffix in (".yaml", ".yml"):
            yaml = _import_yaml()
            with resolved.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return _require_mapping(data, path=resolved)

        if suffix == ".json":
            with resolved.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return _require_mapping(data, path=resolved)

        if suffix == ".toml":
            with resolved.open("rb") as f:
                data = tomllib.load(f)
            return _require_mapping(data, path=resolved)

    except OptionalDependencyError:
        raise
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigFormatError(f"Failed to parse config file {resolved!s}: {exc}") from exc

    raise ConfigFormatError(
        f"Unhandled config suffix {_normalized_suffix(resolved)!r} for {resolved!s}."
    )


def dump_config_mapping(
    mapping: Mapping[str, Any],
    path: str | Path,
    *,
    sort_keys: bool = False,
    indent: int = 2,
) -> Path:
    """
    Write a configuration mapping to YAML or JSON.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Data to serialize.
    path : str or pathlib.Path
        Output path.
    sort_keys : bool, default=False
        Whether to sort keys during serialization.
    indent : int, default=2
        Indentation level for pretty-printed output.

    Returns
    -------
    pathlib.Path
        Resolved output path.

    Raises
    ------
    ConfigFormatError
        If the output suffix is unsupported.
    OptionalDependencyError
        If YAML output is requested but PyYAML is unavailable.

    Notes
    -----
    TOML output is intentionally not supported here because Python's standard
    `tomllib` module is parse-only.
    """
    if not isinstance(mapping, Mapping):
        raise ConfigFormatError(
            f"dump_config_mapping expected a mapping, got {type(mapping).__name__}."
        )

    resolved = ensure_parent_dir(path)
    suffix = _normalized_suffix(resolved)

    try:
        if suffix in (".yaml", ".yml"):
            yaml = _import_yaml()
            with resolved.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    dict(mapping),
                    f,
                    sort_keys=sort_keys,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            return resolved

        if suffix == ".json":
            with resolved.open("w", encoding="utf-8") as f:
                json.dump(
                    dict(mapping),
                    f,
                    indent=indent,
                    sort_keys=sort_keys,
                    ensure_ascii=False,
                )
                f.write("\n")
            return resolved

        if suffix == ".toml":
            raise ConfigFormatError(
                "TOML writing is not supported by this module because the Python "
                "standard library `tomllib` is read-only. Write YAML or JSON instead."
            )

    except OptionalDependencyError:
        raise
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigFormatError(f"Failed to write config file {resolved!s}: {exc}") from exc

    raise ConfigFormatError(
        f"Unsupported output config format {suffix!r} for {resolved!s}. "
        "Use .yaml, .yml, or .json."
    )


# -----------------------------------------------------------------------------
# Scenario-specific helpers
# -----------------------------------------------------------------------------


def list_scenario_config_files(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """
    Return all scenario config files present in the standard scenario directory.

    Parameters
    ----------
    project_root : str or pathlib.Path, optional
        Project root or any descendant path. If omitted, auto-detect.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing scenario config files, sorted by name.
    """
    base = scenario_config_dir(project_root)
    if not base.exists():
        return ()
    files = [p for p in base.iterdir() if p.is_file() and is_supported_config_path(p)]
    return tuple(sorted(files))


def scenario_config_stems(project_root: str | Path | None = None) -> tuple[str, ...]:
    """
    Return the stem names of scenario config files in the standard scenario directory.
    """
    return tuple(p.stem for p in list_scenario_config_files(project_root))


def load_scenario_mapping(
    path_or_name: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ConfigDict:
    """
    Load a raw scenario mapping from the standard scenario config directory or an
    explicit path.

    Parameters
    ----------
    path_or_name : str or pathlib.Path
        Explicit path or bare scenario filename stem.
    project_root : str or pathlib.Path, optional
        Project root used to resolve the standard scenario config directory.

    Returns
    -------
    dict[str, Any]
        Raw scenario mapping.
    """
    base = scenario_config_dir(project_root)
    path = resolve_config_path(
        path_or_name,
        default_dir=base,
        allow_extension_inference=True,
        must_exist=True,
    )
    return load_config_mapping(path)


def load_scenario_spec(
    path_or_name: str | Path,
    *,
    project_root: str | Path | None = None,
    allow_named_builtin_fallback: bool = True,
) -> ScenarioSpec:
    """
    Load a typed `ScenarioSpec` from a scenario config file or built-in name.

    Parameters
    ----------
    path_or_name : str or pathlib.Path
        Either:
        - an explicit config file path,
        - a bare stem like `maritime_baseline`,
        - or, optionally, a built-in scenario name.
    project_root : str or pathlib.Path, optional
        Project root used to find `configs/scenarios`.
    allow_named_builtin_fallback : bool, default=True
        If True and no file is found, fall back to the built-in scenario
        registry in `gravnav.truth.scenarios`.

    Returns
    -------
    ScenarioSpec
        Parsed scenario spec.

    Resolution order
    ----------------
    1. Try to resolve `path_or_name` as a file under `configs/scenarios`
       (or as an explicit path).
    2. If that fails and fallback is allowed, try the built-in named registry.

    Raises
    ------
    ConfigError
        If neither a file-based nor a built-in scenario can be resolved.
    """
    explicit_ref = _is_explicit_path_reference(path_or_name)

    try:
        mapping = load_scenario_mapping(path_or_name, project_root=project_root)
    except ConfigPathError as file_exc:
        if allow_named_builtin_fallback and not explicit_ref:
            try:
                return get_named_scenario(str(path_or_name))
            except KeyError as builtin_exc:
                raise ConfigPathError(
                    f"Could not resolve scenario {path_or_name!r} as a file in "
                    f"{scenario_config_dir(project_root)!s} or as a built-in name. "
                    f"Available built-ins: {available_scenario_names()}."
                ) from builtin_exc
        raise file_exc
    except OptionalDependencyError as file_exc:
        if allow_named_builtin_fallback and not explicit_ref:
            try:
                return get_named_scenario(str(path_or_name))
            except KeyError:
                pass
        raise file_exc

    if allow_named_builtin_fallback and not explicit_ref and mapping == {}:
        try:
            return get_named_scenario(str(path_or_name))
        except KeyError:
            pass

    try:
        return ScenarioSpec.from_mapping(mapping)
    except ConfigPathError as file_exc:
        if allow_named_builtin_fallback and not explicit_ref:
            try:
                return get_named_scenario(str(path_or_name))
            except KeyError:
                pass
        raise file_exc


def dump_scenario_spec(
    scenario: ScenarioSpec,
    path: str | Path,
    *,
    sort_keys: bool = False,
) -> Path:
    """
    Serialize a `ScenarioSpec` to YAML or JSON.

    Parameters
    ----------
    scenario : ScenarioSpec
        Scenario specification to serialize.
    path : str or pathlib.Path
        Output path.
    sort_keys : bool, default=False
        Whether to sort keys during serialization.

    Returns
    -------
    pathlib.Path
        Output path.
    """
    return dump_config_mapping(
        scenario.to_mapping(),
        path,
        sort_keys=sort_keys,
    )


def export_named_scenario(
    name: str,
    path: str | Path,
    *,
    sort_keys: bool = False,
) -> Path:
    """
    Export one built-in scenario definition to a YAML or JSON file.

    Parameters
    ----------
    name : str
        Built-in scenario name.
    path : str or pathlib.Path
        Output path.
    sort_keys : bool, default=False
        Whether to sort serialized keys.

    Returns
    -------
    pathlib.Path
        Output path.
    """
    scenario = get_named_scenario(name)
    return dump_scenario_spec(scenario, path, sort_keys=sort_keys)


def export_all_named_scenarios(
    directory: str | Path,
    *,
    suffix: str = ".yaml",
    sort_keys: bool = False,
) -> tuple[Path, ...]:
    """
    Export all built-in scenarios to one directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Output directory.
    suffix : str, default=".yaml"
        Output suffix. Must be `.yaml`, `.yml`, or `.json`.
    sort_keys : bool, default=False
        Whether to sort serialized keys.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Paths written.
    """
    suffix_norm = suffix.lower()
    if suffix_norm not in (".yaml", ".yml", ".json"):
        raise ConfigFormatError(
            f"Unsupported export suffix {suffix!r}. Use '.yaml', '.yml', or '.json'."
        )

    out_dir = expand_path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in available_scenario_names():
        out_path = out_dir / f"{name}{suffix_norm}"
        written.append(export_named_scenario(name, out_path, sort_keys=sort_keys))
    return tuple(written)


# -----------------------------------------------------------------------------
# Generic helpers for other config families
# -----------------------------------------------------------------------------


def load_sensor_config(
    path_or_name: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ConfigDict:
    """
    Load a raw sensor configuration mapping from `configs/sensors`.

    Notes
    -----
    This returns a generic mapping rather than a typed sensor spec because the
    typed sensor dataclasses will be instantiated later by the sensor modules or
    simulation runner.
    """
    base = sensor_config_dir(project_root)
    path = resolve_config_path(
        path_or_name,
        default_dir=base,
        allow_extension_inference=True,
        must_exist=True,
    )
    return load_config_mapping(path)


def load_monte_carlo_config(
    path_or_name: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ConfigDict:
    """
    Load a raw Monte Carlo configuration mapping from `configs/monte_carlo`.
    """
    base = monte_carlo_config_dir(project_root)
    path = resolve_config_path(
        path_or_name,
        default_dir=base,
        allow_extension_inference=True,
        must_exist=True,
    )
    return load_config_mapping(path)


def merge_mappings(*mappings: Mapping[str, Any]) -> ConfigDict:
    """
    Recursively merge mappings left to right.

    Parameters
    ----------
    *mappings : Mapping[str, Any]
        Input mappings. Later mappings override earlier ones.

    Returns
    -------
    dict[str, Any]
        Recursively merged result.

    Rules
    -----
    - If both old and new values are mappings, merge them recursively.
    - Otherwise the later value replaces the earlier one.

    Notes
    -----
    This is handy for:
    - default config + environment override
    - base scenario config + quick experiment overrides
    - CLI overrides materialized as mappings
    """
    def merge_two(left: Mapping[str, Any], right: Mapping[str, Any]) -> ConfigDict:
        out: ConfigDict = dict(left)
        for key, value in right.items():
            if (
                key in out
                and isinstance(out[key], Mapping)
                and isinstance(value, Mapping)
            ):
                out[key] = merge_two(dict(out[key]), dict(value))
            else:
                out[key] = value
        return out

    merged: ConfigDict = {}
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            raise ConfigFormatError(
                f"merge_mappings expected only mappings, got {type(mapping).__name__}."
            )
        merged = merge_two(merged, dict(mapping))
    return merged


def apply_overrides(
    mapping: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> ConfigDict:
    """
    Return a merged mapping with optional overrides.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Base configuration.
    overrides : Mapping[str, Any], optional
        Override values.

    Returns
    -------
    dict[str, Any]
        Merged config.
    """
    if overrides is None:
        return dict(mapping)
    return merge_mappings(mapping, overrides)


__all__ = [
    "ConfigDict",
    "ConfigError",
    "ConfigFormatError",
    "ConfigPathError",
    "OptionalDependencyError",
    "PROJECT_MARKER_FILES",
    "DEFAULT_MONTE_CARLO_DIRNAME",
    "DEFAULT_SCENARIO_DIRNAME",
    "DEFAULT_SENSOR_DIRNAME",
    "apply_overrides",
    "available_scenario_names",
    "dump_config_mapping",
    "dump_scenario_spec",
    "ensure_parent_dir",
    "expand_path",
    "export_all_named_scenarios",
    "export_named_scenario",
    "find_project_root",
    "is_supported_config_path",
    "list_scenario_config_files",
    "load_config_mapping",
    "load_monte_carlo_config",
    "load_scenario_mapping",
    "load_scenario_spec",
    "load_sensor_config",
    "merge_mappings",
    "monte_carlo_config_dir",
    "resolve_config_path",
    "scenario_config_dir",
    "scenario_config_stems",
    "sensor_config_dir",
    "supported_config_suffixes",
]
