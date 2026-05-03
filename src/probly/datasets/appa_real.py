"""First-order APPA-REAL dataset loader.

This module ships the ``AppaReal`` PyTorch dataset that yields
``(image_tensor, p_star_counts)`` per sample, where ``p_star_counts``
is a dense integer-vote vector indexed by ``self.age_support``. The
support is the union of integer ages observed in the chosen split (or
in the whole dataset, via :meth:`AppaReal.compute_global_support`).

See the project ``decisions.md`` -- "p*(y | x) construction" and
"APPA-REAL deployment losses" -- for the rationale behind the
sparse-over-integer-support representation and the squared/absolute
loss semantics applied downstream.
"""

from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING, Literal

import pandas as pd
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


_FILE_COLUMN_PATTERN = re.compile(r"file", re.IGNORECASE)
_AGE_COLUMN_PATTERN = re.compile(r"apparent.*age|age", re.IGNORECASE)


def _resolve_columns(df: pd.DataFrame, csv_path: pathlib.Path) -> tuple[str, str]:
    """Resolve the filename and apparent-age column names in ``df``.

    The CVPR 2017 APPA-REAL release uses column names that vary slightly
    between mirrors. We look for the first column whose name matches
    ``r"file"`` (case-insensitive) and the first column matching
    ``r"apparent.*age|age"``. If either cannot be found we raise with a
    clear listing of the columns present.

    Args:
        df: DataFrame parsed from one of the per-rater ``gt_<split>.csv``
            files.
        csv_path: Path to the CSV; used only to make the error message
            actionable.

    Returns:
        ``(file_column, age_column)`` -- the resolved column names.

    Raises:
        ValueError: If a filename or apparent-age column cannot be
            identified.
    """
    file_col: str | None = None
    age_col: str | None = None
    apparent_age_col: str | None = None
    for col in df.columns:
        if file_col is None and _FILE_COLUMN_PATTERN.search(col):
            file_col = col
        if "apparent" in col.lower() and "age" in col.lower():
            apparent_age_col = col
        elif age_col is None and _AGE_COLUMN_PATTERN.search(col):
            age_col = col
    # Prefer the apparent-age column when both an "age" and an
    # "apparent_age" column are present (some releases include both).
    if apparent_age_col is not None:
        age_col = apparent_age_col
    if file_col is None or age_col is None:
        msg = (
            f"Could not identify required columns in {csv_path}. "
            f"Looked for a filename column matching r'file' and an "
            f"apparent-age column matching r'apparent.*age|age'. "
            f"Columns present: {list(df.columns)}."
        )
        raise ValueError(msg)
    return file_col, age_col


def _validate_integer_ages(
    df: pd.DataFrame,
    file_col: str,
    age_col: str,
    csv_path: pathlib.Path,
) -> pd.Series:
    """Validate that ``df[age_col]`` holds non-negative integers.

    Args:
        df: DataFrame parsed from a per-rater CSV.
        file_col: Filename column name (used only for error messages).
        age_col: Apparent-age column name.
        csv_path: Path to the CSV (used only for error messages).

    Returns:
        A ``pd.Series`` of integer ages aligned with ``df``.

    Raises:
        ValueError: If any age is non-numeric, non-integer, or negative.
    """
    ages = df[age_col]
    ages_numeric = ages if pd.api.types.is_numeric_dtype(ages) else pd.to_numeric(ages, errors="coerce")
    if ages_numeric.isna().any():
        bad_rows = df.loc[ages_numeric.isna(), [file_col, age_col]].head(5)
        msg = (
            f"Non-numeric apparent-age values in {csv_path}. First offending rows: "
            f"{bad_rows.to_dict(orient='records')}."
        )
        raise ValueError(msg)
    ages_int = ages_numeric.astype(int)
    if not (ages_numeric == ages_int).all():
        bad = df.loc[ages_numeric != ages_int, [file_col, age_col]].head(5)
        msg = f"Non-integer apparent-age values in {csv_path}. First offending rows: {bad.to_dict(orient='records')}."
        raise ValueError(msg)
    if (ages_int < 0).any():
        bad = df.loc[ages_int < 0, [file_col, age_col]].head(5)
        msg = f"Negative apparent-age values in {csv_path}. First offending rows: {bad.to_dict(orient='records')}."
        raise ValueError(msg)
    return ages_int


def _resolve_age_support(
    observed_ages: list[int],
    age_support: Sequence[int] | None,
    csv_path: pathlib.Path,
) -> list[int]:
    """Return the integer age support for the chosen split.

    Args:
        observed_ages: Sorted ascending list of integer ages observed
            in the per-rater CSV for the loaded split.
        age_support: Optional user-supplied support; must be a
            superset of ``observed_ages``.
        csv_path: Path to the CSV (used only for error messages).

    Returns:
        Sorted ascending list of integer ages.

    Raises:
        ValueError: If ``age_support`` is supplied and omits any
            observed age.
    """
    if age_support is None:
        return list(observed_ages)
    support_list = sorted({int(a) for a in age_support})
    missing = [age for age in observed_ages if age not in support_list]
    if missing:
        msg = (
            f"age_support is missing observed ages from "
            f"{csv_path}: {missing[:10]}{'...' if len(missing) > 10 else ''}. "
            f"Pass a wider support or use AppaReal.compute_global_support(root)."
        )
        raise ValueError(msg)
    return support_list


class AppaReal(torch.utils.data.Dataset):
    """First-order APPA-REAL dataset (Agustsson et al. 2017).

    Yields ``(image_tensor, p_star_counts)`` per sample.
    ``p_star_counts`` is a dense tensor of shape
    ``(len(self.age_support),)`` containing raw integer vote counts; the
    caller normalises at use-time per ``decisions.md``'s "Normalise to a
    probability vector at use-time" spec. Indices align with
    ``self.age_support`` (sorted ascending).

    The dense-vector-over-integer-support representation was chosen
    over per-image dicts because (a) the global age support is only
    ~101 entries, (b) ``DataLoader`` collates dense tensors trivially,
    and (c) downstream oracle reductions (mean, variance, median, MAD)
    are vectorised one-liners. See ``decisions.md`` "p*(y | x)
    construction" for the spec rationale.

    The default transform resizes to 224x224 and applies ImageNet-style
    normalisation. Override ``transform`` at construction to match the
    deployed backbone's preprocessing exactly.

    Attributes:
        age_support: Sorted ascending list of integer ages covered by
            ``p_star_counts``.
        image_filenames: Per-image filenames in iteration order.
        num_raters_per_image: Mapping from filename to total vote count
            (handy for sanity-checking against the dataset card).
        vote_counts: Mapping from filename to a 1-D ``torch.Tensor`` of
            raw integer vote counts indexed by ``age_support``. Read-only
            outside of construction; ``__getitem__`` returns a clone.
    """

    age_support: list[int]
    image_filenames: list[str]
    num_raters_per_image: dict[str, int]
    vote_counts: dict[str, torch.Tensor]

    def __init__(
        self,
        root: str | pathlib.Path,
        split: Literal["train", "valid", "test"] = "test",
        transform: Callable[..., torch.Tensor] | None = None,
        age_support: Sequence[int] | None = None,
    ) -> None:
        """Initialise the dataset for the chosen split.

        Args:
            root: Root directory containing ``train/``, ``valid/``,
                ``test/`` and per-rater ``gt_<split>.csv`` files.
            split: Which split to load. Defaults to ``"test"``.
            transform: Optional transform applied to each image. If
                ``None`` we use a 224x224 resize + ImageNet
                normalisation. Override to match the deployed
                backbone's preprocessing.
            age_support: Optional integer age support. If ``None``, we
                use the union of integer ages observed in the chosen
                split. The paper's main results use
                :meth:`compute_global_support` so that ``A*`` / ``E*``
                oracle computations are comparable across splits.

        Raises:
            ValueError: If ``root`` does not exist, the per-rater CSV
                or split image folder is missing, the CSV contains
                non-integer or negative ages, or an observed age is
                outside the user-specified ``age_support``.
        """
        self._root = pathlib.Path(root).expanduser()
        self._split = split
        self.transform = transform if transform is not None else _DEFAULT_TRANSFORM

        csv_path, image_dir = self._validate_paths(self._root, split)
        self._image_dir = image_dir

        df = pd.read_csv(csv_path)
        file_col, age_col = _resolve_columns(df, csv_path)
        ages_int = _validate_integer_ages(df, file_col, age_col, csv_path)

        observed_ages = sorted(set(ages_int.tolist()))
        self.age_support = _resolve_age_support(observed_ages, age_support, csv_path)

        # Group per-rater rows by filename and accumulate the histogram.
        # We iterate row-wise instead of using DataFrame.groupby+pivot
        # because the per-row loop is negligible vs. image I/O for ~7k
        # images and keeps the construction code straightforward.
        age_to_idx = {age: idx for idx, age in enumerate(self.age_support)}
        per_image_counts: dict[str, torch.Tensor] = {}
        for filename, age in zip(df[file_col].tolist(), ages_int.tolist(), strict=True):
            counts = per_image_counts.get(filename)
            if counts is None:
                counts = torch.zeros(len(self.age_support), dtype=torch.float32)
                per_image_counts[filename] = counts
            counts[age_to_idx[age]] += 1.0

        self.image_filenames = list(per_image_counts.keys())
        self.vote_counts = per_image_counts
        self.num_raters_per_image = {
            filename: int(counts.sum().item()) for filename, counts in per_image_counts.items()
        }

    @staticmethod
    def _validate_paths(root: pathlib.Path, split: str) -> tuple[pathlib.Path, pathlib.Path]:
        """Validate filesystem layout and return ``(csv_path, image_dir)``."""
        if not root.is_dir():
            msg = f"Expected root directory at {root}, but it does not exist."
            raise ValueError(msg)
        csv_path = root / f"gt_{split}.csv"
        if not csv_path.is_file():
            msg = (
                f"Expected `{csv_path}` (per the CVPR 2017 release) "
                f"but it was not found. APPA-REAL must contain "
                f"per-rater files gt_train.csv / gt_valid.csv / gt_test.csv."
            )
            raise ValueError(msg)
        image_dir = root / split
        if not image_dir.is_dir():
            msg = f"Expected split image folder at {image_dir}, but it does not exist."
            raise ValueError(msg)
        return csv_path, image_dir

    def __len__(self) -> int:
        """Return the number of images in the loaded split."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image_tensor, p_star_counts)`` for ``index``.

        Args:
            index: Position into ``self.image_filenames``.

        Returns:
            Tuple of the transformed image and the dense vote-count
            tensor of shape ``(len(self.age_support),)``.
        """
        filename = self.image_filenames[index]
        image_path = self._image_dir / filename
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_tensor = self.transform(image)
        counts = self.vote_counts[filename].clone()
        return image_tensor, counts

    @classmethod
    def compute_global_support(cls, root: str | pathlib.Path) -> list[int]:
        """Return the union of integer ages observed across all splits.

        Use this for the paper's main results to keep ``A*`` / ``E*``
        oracle computations comparable across splits::

            support = AppaReal.compute_global_support(root)
            test_set = AppaReal(root, split="test", age_support=support)

        Args:
            root: Root directory containing the per-rater
                ``gt_<split>.csv`` files for every split.

        Returns:
            Sorted ascending list of integer ages observed in the
            union of train, valid, and test splits.

        Raises:
            ValueError: If a per-rater CSV cannot be located or any of
                the validations described in :meth:`__init__` fails.
        """
        root_path = pathlib.Path(root).expanduser()
        if not root_path.is_dir():
            msg = f"Expected root directory at {root_path}, but it does not exist."
            raise ValueError(msg)
        observed: set[int] = set()
        for split in ("train", "valid", "test"):
            csv_path = root_path / f"gt_{split}.csv"
            if not csv_path.is_file():
                msg = (
                    f"Expected `{csv_path}` (per the CVPR 2017 release) "
                    f"but it was not found. APPA-REAL must contain "
                    f"per-rater files gt_train.csv / gt_valid.csv / gt_test.csv."
                )
                raise ValueError(msg)
            df = pd.read_csv(csv_path)
            file_col, age_col = _resolve_columns(df, csv_path)
            ages_int = _validate_integer_ages(df, file_col, age_col, csv_path)
            observed.update(ages_int.tolist())
        return sorted(observed)
