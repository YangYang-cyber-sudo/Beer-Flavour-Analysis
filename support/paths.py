"""Repository-relative paths used by every script.

The repository never stores the measurement data.  Point ``BEER_DATA_DIR`` at
a directory containing the five files downloaded from Zenodo (see
``data/README.md``) or use the default ``data/raw`` layout.

All other paths are resolved relative to the repository root, so the code can
be cloned anywhere and run from any working directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw inputs (not distributed with this repository).
DATA = Path(os.environ.get("BEER_DATA_DIR", ROOT / "data" / "raw")).expanduser().resolve()

# Frozen validation splits and analysis protocols.
SPLITS = ROOT / "validation" / "splits"
PREREG = ROOT / "analysis" / "preregistration"

# Archived derived inputs that the reanalysis scripts read.
DERIVED = ROOT / "data" / "derived"

# Outputs.  The release ships the archived outputs; reruns overwrite them.
RESULTS = Path(os.environ.get("BEER_RESULTS_DIR", ROOT / "results")).expanduser().resolve()
TABLES = RESULTS / "tables"
PREDICTIONS = RESULTS / "predictions"
REPORTS = RESULTS / "reports"

REQUIRED_DATA_FILES = ("1.csv", "4.csv", "6.csv", "8.xlsx",
                       "Raw sensory data anonymized.csv")


def ensure_output_dirs() -> None:
    for path in (TABLES, PREDICTIONS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def configure_console() -> None:
    """Make non-ASCII progress messages safe on Windows GBK consoles.

    The scientific content is UTF-8; this only prevents a ``UnicodeEncodeError``
    when a print statement contains a symbol that the active console code page
    cannot encode.  It never changes file encoding.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # pragma: no cover - platform dependent
                pass


def require_data() -> Path:
    """Return the raw-data directory, or explain exactly what is missing."""
    missing = [name for name in REQUIRED_DATA_FILES if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing raw input file(s) {missing} in {data}. "
            "Download the public Schreurs et al. (2024) deposit from "
            "https://doi.org/10.5281/zenodo.10653704 and place the files there, "
            "or set BEER_DATA_DIR to the directory that holds them. "
            "See data/README.md for the required file list.".format(
                missing=", ".join(missing), data=DATA)
        )
    return DATA
