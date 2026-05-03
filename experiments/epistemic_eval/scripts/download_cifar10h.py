"""Download the CIFAR-10H human-annotation archive.

Fetches ``master.zip`` from ``github.com/jcpeterson/cifar-10h`` and
extracts it under ``<dest>/cifar-10h-master/`` so the layout matches
what :class:`probly.datasets.torch.CIFAR10H` expects, namely::

    <dest>/cifar-10h-master/data/cifar10h-counts.npy

After extraction the script verifies the SHA256 of the counts file
against ``EXPECTED_COUNTS_SHA256``. Replace the ``None`` placeholder
once a known-good copy is available on this machine -- run
``sha256sum <dest>/cifar-10h-master/data/cifar10h-counts.npy`` and
paste the result into the constant below.

Usage::

    python download_cifar10h.py --dest data/cifar10h
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tempfile
from urllib import request
import zipfile

_ARCHIVE_URL = "https://github.com/jcpeterson/cifar-10h/archive/refs/heads/master.zip"
_COUNTS_RELATIVE_PATH = pathlib.Path("cifar-10h-master") / "data" / "cifar10h-counts.npy"

# TODO: fill this in once a known-good copy has been downloaded on this
# machine. Run `sha256sum <dest>/cifar-10h-master/data/cifar10h-counts.npy`
# and paste the hex digest here.
EXPECTED_COUNTS_SHA256: str | None = None


def _sha256(path: pathlib.Path) -> str:
    """Return the SHA256 hex digest of the file at ``path``."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_extract(dest: pathlib.Path) -> pathlib.Path:
    """Download the master archive and extract it under ``dest``.

    Args:
        dest: Destination directory. Will be created if missing.

    Returns:
        The path of the extracted ``cifar10h-counts.npy`` file.

    Raises:
        RuntimeError: If the download or extraction fails, or if the
            expected counts file is not present after extraction.
    """
    dest.mkdir(parents=True, exist_ok=True)
    counts_path = dest / _COUNTS_RELATIVE_PATH
    if counts_path.is_file():
        print(f"Skipping download: {counts_path} already exists.")
        return counts_path

    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        tmp_path = pathlib.Path(tmp)
        archive_path = tmp_path / "master.zip"
        print(f"Downloading {_ARCHIVE_URL}")
        with request.urlopen(_ARCHIVE_URL) as response, archive_path.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_path)
        extracted_root = tmp_path / "cifar-10h-master"
        if not extracted_root.is_dir():
            msg = (
                f"Expected `{extracted_root}` after extraction; the upstream "
                f"archive layout may have changed. Inspect the contents of "
                f"`{tmp_path}` and update download_cifar10h.py accordingly."
            )
            raise RuntimeError(msg)
        target = dest / "cifar-10h-master"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted_root), str(target))

    if not counts_path.is_file():
        msg = (
            f"Extraction completed but `{counts_path}` is missing. "
            f"Inspect `{dest}` and update download_cifar10h.py."
        )
        raise RuntimeError(msg)
    return counts_path


def verify(counts_path: pathlib.Path) -> None:
    """Verify the SHA256 of the extracted counts file.

    Args:
        counts_path: Path to ``cifar10h-counts.npy``.

    Raises:
        RuntimeError: If the digest does not match
            ``EXPECTED_COUNTS_SHA256``.
    """
    digest = _sha256(counts_path)
    if EXPECTED_COUNTS_SHA256 is None:
        print(
            f"No expected sha256 for cifar10h-counts.npy yet -- got {digest}; "
            f"please paste this into download_cifar10h.py "
            f"(EXPECTED_COUNTS_SHA256)."
        )
        return
    if digest != EXPECTED_COUNTS_SHA256:
        msg = (
            f"SHA256 mismatch for `{counts_path}`: expected "
            f"{EXPECTED_COUNTS_SHA256}, got {digest}."
        )
        raise RuntimeError(msg)
    print(f"Verified sha256 of {counts_path}: {digest}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path("data/cifar10h"),
        help="Destination directory; the archive is extracted under <dest>/cifar-10h-master/.",
    )
    args = parser.parse_args(argv)
    dest = args.dest.expanduser().resolve()
    try:
        counts_path = download_and_extract(dest)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Failed to download CIFAR-10H archive: {exc}", file=sys.stderr)
        return 1
    verify(counts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
