"""Download / verify the APPA-REAL release.

APPA-REAL (Agustsson et al. 2017) requires registration at
``https://chalearnlap.cvc.uab.cat/dataset/26/`` to obtain the per-rater
archives. The dataset cannot be fetched anonymously, so this script
runs in two modes:

1. Default: print instructions for fetching the archives, then exit
   non-zero. Future contributors who locate a public mirror should
   wire it in via ``CANONICAL_URL_TEMPLATE`` below.
2. ``--skip-download``: walk ``<dest>/`` and report the SHA256 of
   each ``gt_*.csv`` file, comparing against the table in
   ``EXPECTED_SHA256``.

Usage::

    python download_appa_real.py --dest data/appa_real --skip-download
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

# TODO: fill these in once the user has downloaded a known-good copy.
# Run `sha256sum data/appa_real/gt_*.csv` and paste here.
EXPECTED_SHA256: dict[str, str | None] = {
    "gt_train.csv": None,
    "gt_valid.csv": None,
    "gt_test.csv": None,
    "gt_avg_train.csv": None,
    "gt_avg_valid.csv": None,
    "gt_avg_test.csv": None,
}

_INSTRUCTIONS = (
    "APPA-REAL requires registration at https://chalearnlap.cvc.uab.cat/dataset/26/.\n"
    "Once you have the archives, place them at data/appa_real/{train,valid,test}/\n"
    "plus the gt_<split>.csv files at data/appa_real/, and re-run with\n"
    "    python download_appa_real.py --skip-download\n"
    "to verify hashes."
)


def _sha256(path: pathlib.Path) -> str:
    """Return the SHA256 hex digest of the file at ``path``."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(dest: pathlib.Path) -> int:
    """Walk ``dest`` and report per-file SHA256 against ``EXPECTED_SHA256``.

    Args:
        dest: Root directory containing the per-rater CSVs.

    Returns:
        Exit code: 0 if all known hashes match (or all entries are
        unset), 1 if any expected hash mismatches.
    """
    if not dest.is_dir():
        print(f"`{dest}` does not exist; nothing to verify.", file=sys.stderr)
        return 1
    mismatches: list[str] = []
    for name, expected in EXPECTED_SHA256.items():
        path = dest / name
        if not path.is_file():
            print(f"missing: {path}")
            mismatches.append(name)
            continue
        digest = _sha256(path)
        if expected is None:
            print(
                f"no expected sha256 for {name} yet -- got {digest}; "
                f"please paste this into download_appa_real.py."
            )
        elif digest != expected:
            print(f"sha256 mismatch for {name}: expected {expected}, got {digest}")
            mismatches.append(name)
        else:
            print(f"ok: {name} ({digest})")
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path("data/appa_real"),
        help="Destination directory containing the gt_<split>.csv files and image folders.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and only verify hashes of files already present at --dest.",
    )
    args = parser.parse_args(argv)
    dest = args.dest.expanduser().resolve()

    if args.skip_download:
        return verify(dest)

    print(_INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
