"""
rng.py

Random-number-generation utilities for the gravity-aided navigation simulator.

This module standardizes how the repository creates, spawns, serializes, and
reconstructs NumPy random generators.

Why this file exists
--------------------
The repository's sensor classes already accept optional `np.random.Generator`
instances and otherwise fall back to `np.random.default_rng(...)`. That is the
right low-level behavior, but as the simulator grows we also need a shared place
for:
- reproducible top-level seeding
- independent child streams for Monte Carlo runs
- serializable seed / stream provenance
- checkpoint / replay support through bit-generator state snapshots

Rather than letting every simulation script invent its own seeding scheme, this
module provides one consistent RNG interface for the whole project.

Primary references used here
----------------------------
1) NumPy Random Generator documentation:
   https://numpy.org/doc/stable/reference/random/generator.html

   Used for:
   - the recommendation to use `numpy.random.default_rng(...)`
   - the fact that the default bit generator behind `default_rng` is `PCG64`
   - the accepted seed inputs for `default_rng`
   - generator spawning support via `Generator.spawn(...)`

2) NumPy SeedSequence documentation:
   https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html

   Used for:
   - the role of `SeedSequence` in reproducibly mixing entropy
   - the ability to call `spawn(n)` to create child seed sequences
   - the note that best practice for reproducible bit streams is to record the
     SeedSequence entropy that was used

3) NumPy parallel random-number-generation guide:
   https://numpy.org/doc/stable/reference/random/parallel.html

   Used for:
   - the recommendation to spawn independent child generators
   - the explanation that `SeedSequence` hashing turns nearby integer seeds into
     well-separated initial states with very high probability
   - the safe pattern of combining a root seed with worker IDs as a sequence of
     integers when needed

4) Python `secrets` module:
   https://docs.python.org/3/library/secrets.html

   Used for:
   - generating high-entropy default seed material when the caller requests a
     fresh reproducible root entropy token

Design notes
------------
- Default to NumPy's modern `Generator`, not legacy global state.
- Default to `PCG64`, matching `default_rng(...)`.
- Keep state serialization explicit and reversible.
- Make child-stream generation first-class, because Monte Carlo and multi-sensor
  simulations will need it.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np

BitGeneratorName: TypeAlias = Literal["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"]
SeedEntropy: TypeAlias = int | Sequence[int]
SeedLike: TypeAlias = (
    None
    | int
    | Sequence[int]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | np.random.RandomState
)

_DEFAULT_BIT_GENERATOR_NAME: BitGeneratorName = "PCG64"


def _bit_generator_class(name: BitGeneratorName):
    """
    Return the NumPy bit-generator class for a supported name.
    """
    mapping = {
        "PCG64": np.random.PCG64,
        "PCG64DXSM": np.random.PCG64DXSM,
        "Philox": np.random.Philox,
        "SFC64": np.random.SFC64,
        "MT19937": np.random.MT19937,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported bit-generator name {name!r}. "
            f"Supported names: {tuple(mapping.keys())}."
        ) from exc


def _validate_seed_entropy(seed: SeedEntropy | None) -> SeedEntropy | None:
    """
    Validate raw entropy used to construct a SeedSequence.

    Accepted forms
    --------------
    - None
    - non-negative Python int
    - sequence of non-negative ints

    Returns
    -------
    same type as input
        Normalized, validated entropy material.

    Raises
    ------
    ValueError
        If any supplied integer is negative.
    TypeError
        If the input is not an accepted form.
    """
    if seed is None:
        return None

    if isinstance(seed, int):
        if seed < 0:
            raise ValueError(f"seed integer must be nonnegative, got {seed}.")
        return int(seed)

    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
        values = [int(x) for x in seed]
        if any(v < 0 for v in values):
            raise ValueError(f"All integers in seed sequence must be nonnegative, got {values}.")
        return tuple(values)

    raise TypeError(
        "Seed entropy must be None, a nonnegative int, or a sequence of nonnegative ints."
    )


@dataclass(frozen=True)
class RNGSeedRecord:
    """
    Serializable description of a `SeedSequence`.

    Attributes
    ----------
    entropy : int | tuple[int, ...]
        Root entropy supplied to the SeedSequence.
    spawn_key : tuple[int, ...]
        Spawn path within the SeedSequence tree.
    pool_size : int
        Size of the entropy pool stored by SeedSequence.
    n_children_spawned : int
        Number of children already spawned from this SeedSequence.

    Notes
    -----
    NumPy's SeedSequence is the right object to log for reproducible stream
    creation, because it explicitly records both the root entropy and the spawn
    path through the child-tree mechanism.
    """

    entropy: int | tuple[int, ...]
    spawn_key: tuple[int, ...]
    pool_size: int
    n_children_spawned: int

    @classmethod
    def from_seed_sequence(cls, seed_sequence: np.random.SeedSequence) -> "RNGSeedRecord":
        """
        Create a record from a SeedSequence.
        """
        entropy_obj = seed_sequence.entropy
        if isinstance(entropy_obj, Sequence) and not isinstance(entropy_obj, (str, bytes, bytearray)):
            entropy_val: int | tuple[int, ...] = tuple(int(x) for x in entropy_obj)
        else:
            entropy_val = int(entropy_obj)

        return cls(
            entropy=entropy_val,
            spawn_key=tuple(int(x) for x in seed_sequence.spawn_key),
            pool_size=int(seed_sequence.pool_size),
            n_children_spawned=int(seed_sequence.n_children_spawned),
        )

    def to_seed_sequence(self) -> np.random.SeedSequence:
        """
        Reconstruct the SeedSequence described by this record.
        """
        return np.random.SeedSequence(
            entropy=self.entropy,
            spawn_key=self.spawn_key,
            pool_size=self.pool_size,
            n_children_spawned=self.n_children_spawned,
        )

    def to_mapping(self) -> dict[str, Any]:
        """
        Convert the record to a plain-Python mapping for JSON/YAML serialization.
        """
        entropy_serialized: int | list[int]
        if isinstance(self.entropy, tuple):
            entropy_serialized = list(self.entropy)
        else:
            entropy_serialized = int(self.entropy)

        return {
            "entropy": entropy_serialized,
            "spawn_key": list(self.spawn_key),
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RNGSeedRecord":
        """
        Reconstruct an RNGSeedRecord from a plain mapping.
        """
        entropy_raw = mapping["entropy"]
        if isinstance(entropy_raw, Sequence) and not isinstance(entropy_raw, (str, bytes, bytearray)):
            entropy: int | tuple[int, ...] = tuple(int(x) for x in entropy_raw)
        else:
            entropy = int(entropy_raw)

        return cls(
            entropy=entropy,
            spawn_key=tuple(int(x) for x in mapping.get("spawn_key", ())),
            pool_size=int(mapping.get("pool_size", 4)),
            n_children_spawned=int(mapping.get("n_children_spawned", 0)),
        )


@dataclass(frozen=True)
class RNGStateRecord:
    """
    Serializable snapshot of a NumPy Generator state.

    Attributes
    ----------
    bit_generator_name : BitGeneratorName
        Name of the underlying NumPy bit generator.
    bit_generator_state : dict[str, Any]
        Bit-generator state dictionary.
    seed_record : RNGSeedRecord | None
        Optional seed provenance for how the stream was originally created.

    Notes
    -----
    The bit-generator state is sufficient for exact replay from a checkpoint.
    The optional seed record is useful for provenance and for creating related
    child streams later.
    """

    bit_generator_name: BitGeneratorName
    bit_generator_state: dict[str, Any]
    seed_record: RNGSeedRecord | None = None

    def to_mapping(self) -> dict[str, Any]:
        """
        Convert the state record into a plain-Python mapping.
        """
        return {
            "bit_generator_name": self.bit_generator_name,
            "bit_generator_state": self.bit_generator_state,
            "seed_record": None if self.seed_record is None else self.seed_record.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RNGStateRecord":
        """
        Reconstruct the state record from a plain mapping.
        """
        seed_mapping = mapping.get("seed_record")
        return cls(
            bit_generator_name=str(mapping["bit_generator_name"]),
            bit_generator_state=dict(mapping["bit_generator_state"]),
            seed_record=None if seed_mapping is None else RNGSeedRecord.from_mapping(seed_mapping),
        )


def fresh_entropy(bits: int = 128) -> int:
    r"""
    Return a fresh nonnegative integer seed with the requested entropy size.

    Parameters
    ----------
    bits : int, default=128
        Number of entropy bits to request.

    Returns
    -------
    int
        Random nonnegative integer suitable for seeding a SeedSequence.

    Notes
    -----
    NumPy's parallel RNG guide recommends high-entropy seeds, and specifically
    points to `secrets.randbits(128)` as a good way to obtain 128 bits of
    entropy. This helper centralizes that practice.

    Reference
    ---------
    NumPy parallel RNG guide:
    https://numpy.org/doc/stable/reference/random/parallel.html
    """
    nbits = int(bits)
    if nbits <= 0:
        raise ValueError(f"bits must be positive, got {nbits}.")
    return int(secrets.randbits(nbits))


def make_seed_sequence(
    seed: SeedEntropy | np.random.SeedSequence | None = None,
) -> np.random.SeedSequence:
    """
    Construct a SeedSequence from raw entropy or return the given SeedSequence.

    Parameters
    ----------
    seed : None, int, sequence[int], or numpy.random.SeedSequence
        Seed material.

    Returns
    -------
    numpy.random.SeedSequence
        SeedSequence instance.

    Behavior
    --------
    - If `seed` is already a SeedSequence, it is returned unchanged.
    - If `seed` is None, fresh entropy is generated and wrapped in a
      SeedSequence.
    - Otherwise the validated integer or integer-sequence entropy is used.

    Notes
    -----
    This deliberately separates "entropy creation" from "generator creation",
    which makes it much easier to log reproducible seed provenance.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed

    validated = _validate_seed_entropy(seed)
    if validated is None:
        validated = fresh_entropy(128)

    return np.random.SeedSequence(validated)


def make_rng(
    seed: SeedLike = None,
    *,
    bit_generator_name: BitGeneratorName = _DEFAULT_BIT_GENERATOR_NAME,
) -> np.random.Generator:
    """
    Create a NumPy Generator using the repository's standard RNG policy.

    Parameters
    ----------
    seed : SeedLike, default=None
        Any seed form accepted by NumPy's `default_rng(...)`, plus `None`.
    bit_generator_name : BitGeneratorName, default="PCG64"
        Bit generator to use when constructing from raw seed material.

    Returns
    -------
    numpy.random.Generator
        Random generator.

    Policy
    ------
    - If `seed` is already a `Generator`, it is returned unchanged.
    - If `seed` is a `BitGenerator`, it is wrapped by `Generator`.
    - If `seed` is a legacy `RandomState`, it is coerced via `default_rng(...)`.
    - If `seed` is a `SeedSequence`, int, integer sequence, or None, a Generator
      is built using the requested bit generator.

    Notes
    -----
    NumPy documents `default_rng(...)` as the recommended constructor for modern
    random generation. This helper mirrors that recommendation while still
    allowing the simulator to explicitly choose a bit-generator family when
    checkpointing or comparing streams.

    Reference
    ---------
    NumPy Generator docs:
    https://numpy.org/doc/stable/reference/random/generator.html
    """
    if isinstance(seed, np.random.Generator):
        return seed

    if isinstance(seed, np.random.BitGenerator):
        return np.random.Generator(seed)

    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed)

    if bit_generator_name == _DEFAULT_BIT_GENERATOR_NAME:
        return np.random.default_rng(seed)

    if isinstance(seed, np.random.SeedSequence):
        seed_sequence = seed
    else:
        seed_sequence = make_seed_sequence(seed)

    bitgen_cls = _bit_generator_class(bit_generator_name)
    return np.random.Generator(bitgen_cls(seed_sequence))


def seed_record_from_seed(
    seed: SeedEntropy | np.random.SeedSequence | None = None,
) -> RNGSeedRecord:
    """
    Build a serializable seed record from raw entropy or a SeedSequence.
    """
    return RNGSeedRecord.from_seed_sequence(make_seed_sequence(seed))


def spawn_seed_sequences(
    parent: SeedEntropy | np.random.SeedSequence,
    n_children: int,
) -> list[np.random.SeedSequence]:
    r"""
    Spawn child SeedSequences from a parent seed source.

    Parameters
    ----------
    parent : int, sequence[int], or SeedSequence
        Parent seed source.
    n_children : int
        Number of child seed sequences to produce.

    Returns
    -------
    list[numpy.random.SeedSequence]
        Child SeedSequences.

    Notes
    -----
    NumPy documents `SeedSequence.spawn(n)` as the standard way to derive
    independent child seed sequences for threads, workers, or independent
    simulation streams.
    """
    n = int(n_children)
    if n < 0:
        raise ValueError(f"n_children must be nonnegative, got {n}.")
    return list(make_seed_sequence(parent).spawn(n))


def spawn_generators(
    parent: SeedLike,
    n_children: int,
    *,
    bit_generator_name: BitGeneratorName = _DEFAULT_BIT_GENERATOR_NAME,
) -> list[np.random.Generator]:
    r"""
    Spawn child generators from a parent seed source or generator.

    Parameters
    ----------
    parent : SeedLike
        Parent generator or seed source.
    n_children : int
        Number of child generators to create.
    bit_generator_name : BitGeneratorName, default="PCG64"
        Bit generator to use if the parent is not already a Generator that
        supports native spawning.

    Returns
    -------
    list[numpy.random.Generator]
        Child generators.

    Strategy
    --------
    - If `parent` is already a Generator and provides `.spawn(...)`, use it.
    - Otherwise build a parent SeedSequence and spawn child SeedSequences, then
      convert those to Generators.

    Reference
    ---------
    NumPy parallel RNG guide:
    https://numpy.org/doc/stable/reference/random/parallel.html
    """
    n = int(n_children)
    if n < 0:
        raise ValueError(f"n_children must be nonnegative, got {n}.")

    if isinstance(parent, np.random.Generator):
        return list(parent.spawn(n))

    children = spawn_seed_sequences(parent, n)
    return [make_rng(child, bit_generator_name=bit_generator_name) for child in children]


def monte_carlo_generators(
    root_seed: SeedEntropy | np.random.SeedSequence | None,
    n_runs: int,
    *,
    bit_generator_name: BitGeneratorName = _DEFAULT_BIT_GENERATOR_NAME,
) -> list[np.random.Generator]:
    """
    Create one independent RNG stream per Monte Carlo run.

    Parameters
    ----------
    root_seed : int, sequence[int], SeedSequence, or None
        Root seed source.
    n_runs : int
        Number of Monte Carlo child streams.
    bit_generator_name : BitGeneratorName, default="PCG64"
        Bit generator to use for each child stream.

    Returns
    -------
    list[numpy.random.Generator]
        Independent child generators.

    Notes
    -----
    This is just a scenario-specific name over `spawn_generators(...)`, but it
    makes later Monte Carlo code much clearer at the call site.
    """
    return spawn_generators(
        root_seed,
        n_runs,
        bit_generator_name=bit_generator_name,
    )


def indexed_generator(
    root_seed: SeedEntropy | np.random.SeedSequence,
    index: int,
    *,
    bit_generator_name: BitGeneratorName = _DEFAULT_BIT_GENERATOR_NAME,
) -> np.random.Generator:
    r"""
    Construct a reproducible generator for one deterministic stream index.

    Parameters
    ----------
    root_seed : int, sequence[int], or SeedSequence
        Root seed source.
    index : int
        Deterministic stream index, such as a worker ID or Monte Carlo run ID.
    bit_generator_name : BitGeneratorName, default="PCG64"
        Bit generator to use.

    Returns
    -------
    numpy.random.Generator
        Generator associated with the requested index.

    Method
    ------
    NumPy's parallel RNG guide explicitly documents the safe pattern of combining
    a worker ID and a root seed as a sequence of integers, such as
    `[worker_id, root_seed]`, rather than doing arithmetic like
    `worker_seed = root_seed + worker_id`.

    This helper implements that policy.
    """
    idx = int(index)
    if idx < 0:
        raise ValueError(f"index must be nonnegative, got {idx}.")

    if isinstance(root_seed, np.random.SeedSequence):
        root_record = RNGSeedRecord.from_seed_sequence(root_seed)
        entropy_obj = root_record.entropy
    else:
        entropy_obj = _validate_seed_entropy(root_seed)
        if entropy_obj is None:
            raise ValueError("root_seed must not be None for indexed_generator(...).")

    if isinstance(entropy_obj, tuple):
        seed_material: SeedEntropy = (idx, *entropy_obj)
    else:
        seed_material = (idx, int(entropy_obj))

    return make_rng(seed_material, bit_generator_name=bit_generator_name)


def generator_seed_record(rng: np.random.Generator) -> RNGSeedRecord | None:
    """
    Extract a SeedSequence-derived provenance record from a Generator if available.

    Parameters
    ----------
    rng : numpy.random.Generator
        Input generator.

    Returns
    -------
    RNGSeedRecord or None
        Seed record if the underlying bit generator exposes a `seed_seq`
        attribute, else None.

    Notes
    -----
    NumPy's common bit generators expose their originating SeedSequence through
    `bit_generator.seed_seq`. When present, recording it is useful for
    provenance. When absent, the generator can still be checkpointed by state.
    """
    seed_seq = getattr(rng.bit_generator, "seed_seq", None)
    if seed_seq is None:
        return None
    if not isinstance(seed_seq, np.random.SeedSequence):
        return None
    return RNGSeedRecord.from_seed_sequence(seed_seq)


def generator_state_record(rng: np.random.Generator) -> RNGStateRecord:
    """
    Capture a serializable checkpoint of a Generator.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator to snapshot.

    Returns
    -------
    RNGStateRecord
        Serializable state record.

    Notes
    -----
    The `bit_generator.state` mapping is enough to reconstruct the exact stream
    position for the same bit-generator family.
    """
    bitgen_name = type(rng.bit_generator).__name__
    return RNGStateRecord(
        bit_generator_name=bitgen_name,
        bit_generator_state=dict(rng.bit_generator.state),
        seed_record=generator_seed_record(rng),
    )


def restore_generator_from_state(state: RNGStateRecord | Mapping[str, Any]) -> np.random.Generator:
    """
    Reconstruct a Generator from a serialized state record.

    Parameters
    ----------
    state : RNGStateRecord or mapping
        State record produced by `generator_state_record(...)` or its serialized
        mapping form.

    Returns
    -------
    numpy.random.Generator
        Restored generator at the exact saved state.
    """
    record = state if isinstance(state, RNGStateRecord) else RNGStateRecord.from_mapping(state)
    bitgen_cls = _bit_generator_class(record.bit_generator_name)

    if record.seed_record is not None:
        bitgen = bitgen_cls(record.seed_record.to_seed_sequence())
    else:
        bitgen = bitgen_cls()

    bitgen.state = record.bit_generator_state
    return np.random.Generator(bitgen)


def random_uint64(rng: np.random.Generator) -> int:
    """
    Draw one unsigned 64-bit integer from a Generator.

    This is a lightweight utility that is occasionally useful when downstream
    code needs an integer token rather than a floating-point sample.
    """
    return int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


__all__ = [
    "BitGeneratorName",
    "RNGSeedRecord",
    "RNGStateRecord",
    "SeedEntropy",
    "SeedLike",
    "fresh_entropy",
    "generator_seed_record",
    "generator_state_record",
    "indexed_generator",
    "make_rng",
    "make_seed_sequence",
    "monte_carlo_generators",
    "random_uint64",
    "restore_generator_from_state",
    "seed_record_from_seed",
    "spawn_generators",
    "spawn_seed_sequences",
]