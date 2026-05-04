"""Schema + sanity tests for the 9 DCIC dataset YAML configs.

Each of the 9 production DCIC datasets has a YAML config under
``experiments/epistemic_eval/configs/datasets/``. This test pins the
schema (so a future config edit can't silently regress the wiring
contract) and verifies cross-config uniformity of the locked recipe
declared in ``decisions.md`` -> "DCIC datasets".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.datasets.dcic import DCIC_LOADERS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CFG_DIR = _REPO_ROOT / "experiments" / "epistemic_eval" / "configs" / "datasets"

_DCIC_YAML_FILES: tuple[str, ...] = (
    "benthic.yaml",
    "mice_bone.yaml",
    "pig.yaml",
    "plankton.yaml",
    "quality_mri.yaml",
    "dcic_synthetic.yaml",
    "treeversity_1.yaml",
    "treeversity_6.yaml",
    "turkey.yaml",
)


@pytest.mark.parametrize("yaml_file", _DCIC_YAML_FILES)
def test_dcic_yaml_has_locked_schema(yaml_file: str) -> None:
    """Every DCIC config carries the fields the wiring branches read."""
    cfg_path = _CFG_DIR / yaml_file
    assert cfg_path.exists(), cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    # Family flag dispatches the DCIC branches in train_classifier,
    # fit_uncertainty, extract_uncertainties.
    assert cfg.get("family") == "dcic", yaml_file

    # Loader block must declare the canonical DCIC name so the
    # adapter can resolve it through DCIC_LOADERS.
    loader_block = cfg.get("loader") or {}
    dcic_name = loader_block.get("dcic_name")
    assert dcic_name in DCIC_LOADERS, (yaml_file, dcic_name, sorted(DCIC_LOADERS))

    # Classifier block must declare num_classes (used by the resnet18
    # head factory).
    classifier_cfg = cfg.get("classifier") or {}
    num_classes = classifier_cfg.get("num_classes")
    assert isinstance(num_classes, int) and num_classes >= 2, yaml_file

    # Training block must carry the locked AdamW recipe for basecls
    # plus method_overrides for the from-scratch UQ training path.
    training = cfg.get("training") or {}
    assert training.get("optimizer") == "adamw", yaml_file
    for key in ("lr", "weight_decay", "batch_size", "epochs", "patience", "val_fraction"):
        assert key in training, (yaml_file, key)
    overrides = training.get("method_overrides") or {}
    for key in ("epochs", "lr", "momentum", "nesterov", "weight_decay"):
        assert key in overrides, (yaml_file, key)


def test_all_dcic_yamls_share_the_same_recipe() -> None:
    """All 9 DCIC configs declare the same AdamW basecls recipe + overrides.

    Pin: cross-DCIC recipe uniformity. If a future PR tunes one
    dataset's lr or batch size, this test fails so the change is
    caught and the divergence either reverted or recorded in
    decisions.md.
    """
    canonical_basecls = {
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 32,
        "epochs": 20,
        "patience": 4,
        "val_fraction": 0.1,
    }
    canonical_overrides = {
        "epochs": 20,
        "lr": 1.0e-3,
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": 1.0e-4,
    }
    for yaml_file in _DCIC_YAML_FILES:
        cfg = yaml.safe_load((_CFG_DIR / yaml_file).read_text())
        training = cfg["training"]
        for key, expected in canonical_basecls.items():
            actual = training.get(key)
            assert actual == pytest.approx(expected) or actual == expected, (
                yaml_file,
                key,
                actual,
                expected,
            )
        overrides = training["method_overrides"]
        for key, expected in canonical_overrides.items():
            actual = overrides.get(key)
            assert actual == pytest.approx(expected) or actual == expected, (
                yaml_file,
                key,
                actual,
                expected,
            )


def test_all_nine_dcic_loaders_have_a_yaml() -> None:
    """The 9 wired DCIC loaders each have exactly one dataset config.

    Pin: dataset-set membership. Adding a new DCIC dataset must come
    with its own yaml, and removing a yaml must come with removing
    the loader from DCIC_LOADERS.
    """
    declared_dcic_names: set[str] = set()
    for yaml_file in _DCIC_YAML_FILES:
        cfg = yaml.safe_load((_CFG_DIR / yaml_file).read_text())
        declared_dcic_names.add(cfg["loader"]["dcic_name"])
    assert declared_dcic_names == set(DCIC_LOADERS), (
        sorted(declared_dcic_names),
        sorted(DCIC_LOADERS),
    )
