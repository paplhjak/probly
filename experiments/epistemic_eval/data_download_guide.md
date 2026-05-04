# Data download guide — epistemic-eval

This guide tells you how to fetch every dataset the
`experiments/epistemic_eval/` pipeline can run on, where to place
the files on disk, and how to verify the layout. The pipeline
expects `data/` at the repo root (the `loader_kwargs.root` field in
each dataset YAML); on the cluster the equivalent path is
`~/neurips2026/probly/data/`.

If a download fails, every dataset has a fallback verification
section so you can sanity-check what you got from any source against
what the loader expects.

## Storage budget at a glance

| Dataset family | Approx. disk |
|---|---|
| CIFAR-10H (CIFAR-10 + counts) | ~330 MB |
| ImageNet-ReaL labels (no images) | <10 MB |
| ImageNet-1k val images | ~6.3 GB |
| APPA-REAL | ~1 GB |
| DCIC (9 datasets) | ~5-7 GB total (varies; QualityMRI < 50 MB, Treeversity ~1 GB each) |

The pipeline never deletes anything from `data/`; once a dataset is
in place, all subsequent runs reuse it.

---

## CIFAR-10H

CIFAR-10H is the CIFAR-10 test set re-annotated with 50+ human
votes per image, by Peterson et al. 2019. Two artefacts are needed:
the canonical CIFAR-10 batches (for the train images the basecls
trains on) and the human-vote `npy` (for the soft labels p\*).

### Get it

Run the helper:

```bash
.venv/bin/python experiments/epistemic_eval/scripts/download_cifar10h.py \
    --dest data/cifar10h
```

If that fails (the canonical Toronto host has been flaky with 503s),
fall back to:

```bash
# Manually fetch the human votes from
# https://github.com/jcpeterson/cifar-10h
# Place cifar10h-counts.npy under:
#   data/cifar10h/cifar-10h-master/data/cifar10h-counts.npy
#
# Then materialise the canonical CIFAR-10 batches from the local
# cifar10-raw-images.zip via the project's helper:
.venv/bin/python experiments/epistemic_eval/scripts/build_canonical_cifar10_pickles.py
```

### Expected layout

```
data/cifar10h/
    cifar-10h-master/
        data/
            cifar10h-counts.npy            # (10000, 10) int64 vote counts
    cifar-10-batches-py/
        data_batch_1 ... data_batch_5      # CIFAR-10 train pickles
        test_batch                          # CIFAR-10 test pickles
        batches.meta
```

### Build the p\* sidecar

```bash
.venv/bin/python experiments/epistemic_eval/scripts/prepare_cifar10h_p_star.py
```

This writes `data/cifar10h/p_star.npz` (one-shot; loss-independent
because CIFAR-10H uses a fixed test set across seeds, no fold
rotation).

---

## ImageNet-ReaL

ImageNet val with re-labeled multi-label annotations from Beyer et al.
2020. We use ImageNet val + the ReaL JSON file.

### Get the ReaL labels

```bash
# https://github.com/google-research/reassessed-imagenet
# Place real.json under data/imagenet_real/real.json
```

The probly loader looks for `<root>/reassessed-imagenet-master/real.json`
(see `probly.datasets.torch.ImageNetReaL.__init__`); make sure the
file is under that path or set up a symlink.

### Get ImageNet val (cluster only)

ImageNet-1k requires registration at https://www.image-net.org. We
do not redistribute it. On the RCI cluster the val split is
typically pre-staged in a shared location; ask the maintainer for
the path and add a symlink under `data/imagenet_real/val/`.

### Expected layout

```
data/imagenet_real/
    real.json                                  # OR
    reassessed-imagenet-master/real.json
    val/                                       # ImageNet val images (1000-class subdirs)
        n01440764/...
        ...
```

ImageNet-ReaL is not yet wired through the basecls trainer (per
`decisions.md` -> "ImageNet-ReaL training data" we reuse
probly-distributed checkpoints rather than train from scratch).
Wiring is gated on the ensemble checkpoint set being staged.

---

## APPA-REAL

APPA-REAL (Agustsson et al. 2017) needs registration at
https://chalearnlap.cvc.uab.cat/dataset/26/. It cannot be fetched
anonymously.

### Get it

```bash
# Register at the URL above, download the train / val / test
# archives, and extract to:
#   data/appa_real/<images>
#   data/appa_real/gt_<train|valid|test>.csv
```

Then verify SHA256:

```bash
.venv/bin/python experiments/epistemic_eval/scripts/download_appa_real.py \
    --dest data/appa_real --skip-download
```

### Expected layout

```
data/appa_real/
    train/<image files>
    valid/<image files>
    test/<image files>
    gt_train.csv
    gt_valid.csv
    gt_test.csv
```

APPA-REAL is loaded via `probly.datasets.appa_real.APPAReal`; see
`decisions.md` -> "APPA-REAL backbone strategy" for the locked head
architecture and the linear-probe pipeline.

---

## DCIC datasets (9 total)

DCIC (Schmarje et al. 2022, "Is one annotation enough?") is a
collection of multi-rater image classification datasets. The
release is on Zenodo at https://zenodo.org/records/7180818.

### Datasets we wire

| Config file | DCIC name | Classes | Images |
|---|---|---|---|
| `benthic.yaml` | `Benthic` | 8 | 4 867 |
| `mice_bone.yaml` | `MiceBone` | 4 | 7 240 |
| `pig.yaml` | `Pig` | 4 | 10 237 |
| `plankton.yaml` | `Plankton` | 10 | 12 280 |
| `quality_mri.yaml` | `QualityMRI` | 2 | 310 |
| `dcic_synthetic.yaml` | `Synthetic` | 6 | 15 000 |
| `treeversity_1.yaml` | `Treeversity#1` | 6 | 9 489 |
| `treeversity_6.yaml` | `Treeversity#6` | 6 | 9 826 |
| `turkey.yaml` | `Turkey` | 3 | 8 040 |

`CIFAR10HDCIC` is excluded — the existing CIFAR-10H wiring already
covers the same images via a different on-disk layout.

### Get them all in one go

The probly tree ships an installer at
`experiments/first_order_data/install_dcic_datasets.py` which fetches
every DCIC archive from Zenodo and extracts to a single parent
directory. To put them where our pipeline expects (`data/`), point
its `--dest` at that path:

```bash
.venv/bin/python experiments/first_order_data/install_dcic_datasets.py \
    --dest data
```

The script is idempotent — it skips any dataset whose
`<DCICName>/annotations.json` already exists.

### Get one at a time (if you only need a subset)

The script downloads all 10 by default. To grab a single dataset,
edit the `ARCHIVES` dict in `install_dcic_datasets.py` to keep only
the entry you need, or fetch the archive directly:

```bash
# example for Plankton
mkdir -p data
cd data
curl -L -O 'https://zenodo.org/records/7180818/files/Plankton.zip?download=1'
unzip Plankton.zip          # extracts to ./Plankton/
rm Plankton.zip
```

### Expected layout (one dataset)

```
data/<DCICName>/
    annotations.json                # the master annotations file
    <fold>/<image files>            # 5 fold subdirs (typically part1..part5)
        ...
```

Two folder-name oddities to watch for:

- **Treeversity** uses a `#` in the folder name on disk
  (`Treeversity#1`, `Treeversity#6`); quote those if you script
  shell commands. Our YAMLs handle this transparently.
- **Fold prefix** is `part1`..`part5` per Oleg's confirmation. The
  adapter (`experiments/epistemic_eval/datasets/dcic.py`) discovers
  fold names from disk rather than hardcoding the prefix, so
  releases with `fold1`..`fold5` would also work.

### Build the per-seed p\* sidecars

DCIC test fold rotates per seed (seed `N` -> sorted-fold[`N % 5`];
see `decisions.md` -> "DCIC datasets"). The p\* sidecar is therefore
per-seed, not per-dataset. Build all 5 sidecars for one dataset:

```bash
for SEED in 0 1 2 3 4; do
  .venv/bin/python experiments/epistemic_eval/scripts/prepare_dcic_p_star.py \
      --dataset-config experiments/epistemic_eval/configs/datasets/plankton.yaml \
      --seed "$SEED"
done
```

Output goes to `data/<DCICName>/p_star_seed<N>.npz` (one file per
seed). The orchestrator should pass the matching path to
`compute_oracle.py --p-star-path` per run.

### Verify a DCIC install

```bash
.venv/bin/python -c '
from pathlib import Path
from experiments.epistemic_eval.datasets.dcic import _build_loader, fold_indices
loader = _build_loader("Plankton", Path("data"), transform=None)
print(f"{len(loader)} images, {loader.num_classes} classes")
print({k: len(v) for k, v in fold_indices(loader).items()})
'
```

This should report ~12 280 images and a fold-size dict whose values
sum to that total. If the fold dict is empty or has only one fold,
the on-disk layout is wrong (the second path component of each
image_path must encode the fold).

---

## Cluster-side workflow

On the RCI cluster (`~/neurips2026/probly`):

1. Pull the latest commit (`git pull --ff-only`).
2. Install dependencies if changed (`uv sync`).
3. Run the relevant download script(s) above; they all default to
   writing under `~/neurips2026/probly/data/`.
4. Build p\* sidecars (CIFAR-10H is one-shot; DCIC datasets need
   one per seed).
5. Launch the run grid via the existing `run_grid.sh`.

The local-development `cluster_probly/` SSHFS mount gives you
read-only access to the cluster's data and run dirs; downloads
themselves must run on the cluster.

---

## Troubleshooting

**A loader complains about missing `annotations.json`.** The DCIC
loader joins `<root>/<DCICName>/annotations.json`. Check the dataset
folder lives directly under `data/`, not nested deeper.

**`label_mappings` differs between train and extract.** The DCIC
adapter wraps every loader instance to enforce a deterministic
`str(label)`-sorted ordering, regardless of `PYTHONHASHSEED`. If
you're hitting this on a custom dataset that bypasses
`_DeterministicDCIC`, route through the adapter instead.

**Test fold has zero images.** The fold name is the second path
component of `image_path` in `annotations.json`. If your release
flattens that (e.g. `Plankton/img_001.png` instead of
`Plankton/part1/img_001.png`), the adapter buckets everything under
`unknown_fold` and rotation breaks. Fix the on-disk paths or write
a custom adapter.

**Treeversity#1 / #6 path errors.** Some shells choke on `#` in
quoted paths (it's interpreted as a comment marker outside quotes).
Always wrap the path in single quotes or escape the `#`.
