"""Loading of the five raw source CSVs, with fingerprints and schema checks.

Raw inputs are never modified; every load records the file's SHA-256
fingerprint so later runs can prove which data version produced which
artifacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.config import (
    NULLABLE_INT_COLUMNS,
    REQUIRED_COLUMNS,
    SOURCE_FILES,
    STRING_COLUMNS,
)


class MissingSourceError(FileNotFoundError):
    """Raised when a required raw input file is absent or unreadable."""


class SchemaValidationError(ValueError):
    """Raised when required columns are absent; lists every problem at once."""


@dataclass
class SourceData:
    name: str
    path: Path
    df: pd.DataFrame
    fingerprint: str
    size_bytes: int

    @property
    def fingerprint_short(self) -> str:
        return self.fingerprint[:8]


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtypes_for(name: str) -> dict[str, str]:
    dtypes = {col: "string" for col in STRING_COLUMNS[name]}
    dtypes.update({col: "Int64" for col in NULLABLE_INT_COLUMNS[name]})
    return dtypes


def load_source(name: str, raw_dir: Path) -> SourceData:
    """Load one raw CSV by its logical name (see SOURCE_FILES)."""
    path = raw_dir / SOURCE_FILES[name]
    if not path.is_file():
        raise MissingSourceError(
            f"Required source file '{SOURCE_FILES[name]}' for '{name}' was not "
            f"found at {path}. Place the five supplied CSVs in the raw data "
            f"directory or point FLIGHT_DELAY_RAW_DIR at it."
        )
    try:
        df = pd.read_csv(path, dtype=_dtypes_for(name))
    except Exception as exc:  # unreadable/corrupt file should name the file
        raise MissingSourceError(
            f"Source file '{SOURCE_FILES[name]}' exists at {path} but could "
            f"not be read as CSV: {exc}"
        ) from exc
    return SourceData(
        name=name,
        path=path,
        df=df,
        fingerprint=file_fingerprint(path),
        size_bytes=path.stat().st_size,
    )


def validate_schema(name: str, df: pd.DataFrame) -> list[str]:
    """Return the list of required columns missing from `df` (may be empty)."""
    required = REQUIRED_COLUMNS[name]
    return [col for col in required if col not in df.columns]


def load_all_sources(raw_dir: Path) -> dict[str, SourceData]:
    """Load and schema-check all five sources; fail with one clear error."""
    sources: dict[str, SourceData] = {}
    for name in SOURCE_FILES:
        sources[name] = load_source(name, raw_dir)

    problems = []
    for name, source in sources.items():
        missing = validate_schema(name, source.df)
        if missing:
            problems.append(
                f"{source.path.name}: missing required columns {missing}"
            )
    if problems:
        raise SchemaValidationError(
            "Schema validation failed for the supplied raw files:\n  - "
            + "\n  - ".join(problems)
        )
    return sources
