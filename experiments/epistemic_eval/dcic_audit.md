# DCIC dataset audit (epistemic-eval branch)

> **STATUS** (post-2026-05-04): the audit below is preserved for
> historical context. The wiring it scoped has since been
> implemented; see `decisions.md` -> "DCIC datasets" for the locked
> recipe and `data_download_guide.md` for the install procedure.
> Production code lives under
> `experiments/epistemic_eval/datasets/dcic.py` (adapter),
> `scripts/prepare_dcic_p_star.py` (per-seed sidecar prep), DCIC
> branches in `train_classifier.py` / `fit_uncertainty.py` /
> `extract_uncertainties.py`, and 9 dataset YAMLs under
> `configs/datasets/`. All datasets in this audit are now
> first-class.

Read-only audit of DCIC (Schmarje et al. 2022, "Is one annotation
enough?") infrastructure already present in this branch, and the
wiring work each candidate dataset would need to flow through the
existing pipeline (basecls training → fit methods → extract →
decompose → metrics).

References:
- DCIC paper: Schmarje et al. 2022, arxiv 2207.06214.
- Loaders cited: `src/probly/datasets/torch.py:98-428`.
- Reference download tool: `experiments/first_order_data/install_dcic_datasets.py:10-22` (Zenodo `records/7180818`).
- DCIC dataset stats table (size / classes / resolution): `experiments/first_order_data/README.md`.

---

## 1. Executive summary

- **Loaders for all 10 DCIC datasets are present in `probly.datasets.torch`** (`src/probly/datasets/torch.py:199-428`): `Benthic`, `CIFAR10HDCIC`, `MiceBone`, `Pig`, `Plankton`, `QualityMRI`, `Synthetic`, `Turkey`, `Treeversity1`, `Treeversity6`. They share a common base `DCICDataset` (`torch.py:98`) with one constructor signature and one soft-label format.
- **No DCIC dataset files are on disk** locally (`data/` has only `cifar10h/`, `cifar10-raw-images/`, `imagenet_real/`) or on the cluster mount (same three plus `tmp` stub). The download helper `experiments/first_order_data/install_dcic_datasets.py` pulls the Zenodo `records/7180818` archives but has not been run for this branch.
- **The epistemic-eval pipeline has 3 hardcoded `if name == "cifar10h"` dispatches** that any new DCIC dataset would need a sibling branch in: `scripts/train_classifier.py:479`, `scripts/fit_uncertainty.py:416`, `scripts/extract_uncertainties.py:160`. p* prep is per-dataset and lives outside that hardcode (one prep script per dataset).
- **CIFAR-10H is wired via the non-DCIC class** `probly.datasets.torch.CIFAR10H` (`torch.py:19`, derived from `torchvision.datasets.CIFAR10`), not via `CIFAR10HDCIC`. The two consume different on-disk layouts so the existing wiring does not transfer to other DCIC datasets — a new dataset needs its own train provider, test provider, and p* prep.
- **Train/test split is the load-bearing wiring difference.** DCIC ships 5 predefined folds embedded in image paths (`experiments/first_order_data/utils.py:310`'s `extract_fold_name`); `DCICDataset` itself returns all annotations and is fold-blind. Any wrapping must add fold filtering, since our pipeline expects a clean train-vs-test split.

The smallest non-trivial second dataset is **Plankton** (10 classes, 96×96, 12 280 images, train-from-scratch with a CIFAR-style ResNet-18 still works at 96² with a small stem change). **MiceBone** is a close second on different grounds (medical imagery, 4 classes, 224×224, needs torchvision ResNet-18/50). See §3.

---

## 2. Per-dataset sections

Conventions:
- "Loader location" cites `src/probly/datasets/torch.py`.
- "Disk layout expected" follows `DCICDataset.__init__` (`torch.py:125-170`): the loader reads `<root>/<DatasetName>/annotations.json`. Each entry has `annotations: [{image_path, class_label, ...}, ...]`. `targets[i]` is the per-image soft-label distribution (vote histogram normalised to a probability vector) — this is exactly what our `p_star` requires.
- "Effort": SMALL = yaml + p* prep + minor wiring branches; MEDIUM = SMALL + non-trivial backbone or fold-handling work; LARGE = significant pipeline change.
- All 10 datasets share one piece of work that is not per-dataset: a fold-filtering adapter (DCIC-wide). Estimates below assume that adapter exists; one of the SMALL datasets pays for it once.

### 2.1 Common to every DCIC dataset

- **Loader API**: `DCICDataset(root, transform=None, *, first_order=True)`. `first_order=True` (the default) populates `targets[i]` as a `(K,)` `float32` torch tensor summing to 1; `first_order=False` samples one hard label from that distribution. p* construction needs `first_order=True`.
- **Disk layout**: the dataset constructor receives `root=data/<DatasetName>` and reads `<root>/annotations.json`. The `image_path` field in each annotation is resolved relative to `root.parent` (`torch.py:143` sets `self.root = root.parent`, then `torch.py:190` joins `root / image_paths[i]`). So the full layout is:
  ```
  data/<DatasetName>/
      annotations.json
      <whatever image_path tree the release provides>
  ```
- **Soft-label format**: `targets[i]` is a row-stochastic `(K,)` tensor; equivalent to CIFAR-10H's row of `cifar10h-counts.npy` after row-normalisation. Building `p_star.npz` is just `np.stack([t.numpy() for t in dataset.targets])` plus `support = np.arange(num_classes)` plus `indices = np.arange(len(dataset))`. No DCIC-specific math.
- **Fold metadata**: each `image_path` encodes its fold ("part1" through "part5" by DCIC convention; see `experiments/first_order_data/utils.py:310-312`). The `DCICDataset` class **does not** filter by fold — train/test split must be applied by a wrapper at index construction time.
- **Number of classes**: `dataset.num_classes` (derived from the union of observed labels). Useful as the source of truth at config-write time.

### 2.2 Benthic

- **Loader location**: `torch.py:199-219`.
- **Disk layout expected**: `data/Benthic/annotations.json` + image tree.
- **Data on disk**: NO (`data/` lists `cifar10h cifar10-raw-images imagenet_real`).
- **Soft-label format**: 8-class softmax-shaped vote histogram per image (see DCIC README).
- **Image resolution / classes / size**: 112×112, 8 classes, 4 867 images.
- **Training recipe recommendation**: ResNet-18 at 112² is fine (single conv-stem rescale or a 3×3 stem like the CIFAR variant). Reuse the CIFAR-10H 200-epoch SGD-cosine recipe; consider a smaller batch (64) since the dataset is 5× smaller than CIFAR-10 train.
- **Wiring effort**: **SMALL**. (yaml + prep script + 3 wiring branches + fold filter.)

### 2.3 CIFAR10HDCIC

- **Loader location**: `torch.py:222-244`.
- **Disk layout expected**: `data/CIFAR10H/annotations.json` (DCIC-format, **distinct** from the existing `data/cifar10h/cifar-10h-master/data/cifar10h-counts.npy`).
- **Data on disk**: NO in the DCIC layout. The non-DCIC CIFAR-10H wiring already covers the same images via a different file path, so adding the DCIC variant is redundant for this paper.
- **Soft-label format**: 10-class vote histogram, identical to the existing CIFAR10H wiring.
- **Image resolution / classes / size**: 32×32, 10 classes, 10 000 images.
- **Training recipe recommendation**: same as CIFAR-10H — already locked in `cifar10h.yaml`.
- **Wiring effort**: **N/A — skip**. We already have CIFAR-10H wired via the torchvision-style class; adding `CIFAR10HDCIC` would duplicate it without a new image domain.

### 2.4 MiceBone

- **Loader location**: `torch.py:247-267`.
- **Disk layout expected**: `data/MiceBone/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 4-class vote histogram (collagen-fibre orientation classes per the DCIC README).
- **Image resolution / classes / size**: 224×224, 4 classes, 7 240 images.
- **Training recipe recommendation**: torchvision `resnet18` (or `resnet50`) at 224² with ImageNet pretraining; SGD-cosine, ~50-100 epochs (full 200 is overkill given dataset size). MUST use a 224²-compatible backbone — the `probly_benchmark.resnet.ResNet18` used by basecls is the CIFAR-10 variant (3×3 stem, no maxpool) and is wrong for 224² inputs.
- **Wiring effort**: **MEDIUM**. (yaml + prep + 3 wiring branches + fold filter + ImageNet-pretrained backbone choice + transforms for 224² imagery.)

### 2.5 Pig

- **Loader location**: `torch.py:270-290`.
- **Disk layout expected**: `data/Pig/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 4-class vote histogram (tail-injury severity per the DCIC README).
- **Image resolution / classes / size**: 96×96, 4 classes, 10 237 images.
- **Training recipe recommendation**: CIFAR-style `probly_benchmark.resnet.ResNet18` works at 96² with a stride adjustment (or use a 96²-friendly backbone). SGD-cosine, ~150 epochs.
- **Wiring effort**: **SMALL**.

### 2.6 Plankton

- **Loader location**: `torch.py:293-313`.
- **Disk layout expected**: `data/Plankton/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 10-class vote histogram (10 plankton species per the DCIC README).
- **Image resolution / classes / size**: 96×96, 10 classes, 12 280 images.
- **Training recipe recommendation**: same family as Pig — CIFAR-style ResNet at 96² is fine. SGD-cosine 200 epochs matches CIFAR-10H budget exactly. Class count (10) and size (~CIFAR-10's order of magnitude) make this the closest cousin to CIFAR-10H among the DCIC datasets.
- **Wiring effort**: **SMALL**.

### 2.7 QualityMRI

- **Loader location**: `torch.py:316-336`.
- **Disk layout expected**: `data/QualityMRI/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 2-class vote histogram (good / bad MRI quality).
- **Image resolution / classes / size**: 224×224, 2 classes, 310 images.
- **Training recipe recommendation**: 310 images is too small to train from scratch; needs ImageNet-pretrained backbone with a frozen-encoder + linear-probe approach (akin to the planned APPA-REAL flow), or strong cross-validation. Binary task limits how informative `(A_hat, E_hat)` decompositions can be — only one degree of freedom.
- **Wiring effort**: **MEDIUM-LARGE**. (Tiny dataset + binary task + 224² + needs pretrained backbone path. Architecturally reuses APPA-REAL's linear-probe flow but APPA-REAL itself is not yet wired through training.)

### 2.8 Synthetic

- **Loader location**: `torch.py:339-359`.
- **Disk layout expected**: `data/Synthetic/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 6-class vote histogram (single-coloured-circle classes).
- **Image resolution / classes / size**: 224×224, 6 classes, 15 000 images.
- **Training recipe recommendation**: synthetic, easy task → fast convergence. ResNet at 224² is fine but smaller backbones suffice. SGD-cosine ~50-100 epochs.
- **Wiring effort**: **SMALL-MEDIUM**. (224² inputs need a torchvision-style backbone path, but the dataset itself is forgiving.)

### 2.9 Turkey

- **Loader location**: `torch.py:362-382`.
- **Disk layout expected**: `data/Turkey/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 3-class vote histogram (turkey-injury severity).
- **Image resolution / classes / size**: 192×192, 3 classes, 8 040 images.
- **Training recipe recommendation**: 192² is awkward — neither CIFAR ResNet-18 (32-stem) nor torchvision ResNet-18 (224²) is a perfect fit. Easiest path: `Resize(224)` then torchvision ResNet-18 with ImageNet pretrain. SGD-cosine ~100 epochs.
- **Wiring effort**: **MEDIUM**.

### 2.10 Treeversity#1

- **Loader location**: `torch.py:385-405`. Note `super().__init__(Path(root) / "Treeversity#1", ...)` — the `#` in the path is not URL-escaped; depending on filesystem this may or may not need shell escaping in scripts.
- **Disk layout expected**: `data/Treeversity#1/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 6-class vote histogram, single-label-per-image variant.
- **Image resolution / classes / size**: 224×224, 6 classes, 9 489 images.
- **Training recipe recommendation**: torchvision ResNet-18/50 at 224² with ImageNet pretrain; SGD-cosine ~100 epochs.
- **Wiring effort**: **MEDIUM**. (224² + the `#` path quirk.)

### 2.11 Treeversity#6

- **Loader location**: `torch.py:408-428`. Same `#` path note as Treeversity1.
- **Disk layout expected**: `data/Treeversity#6/annotations.json` + image tree.
- **Data on disk**: NO.
- **Soft-label format**: 6-class vote histogram, **possibly multiple labels per image** (avg. non-zero target classes 3.11 vs. 1.68 for #1).
- **Image resolution / classes / size**: 224×224, 6 classes, 9 826 images.
- **Training recipe recommendation**: same as #1, but the multi-label nature means soft-label CE on a row-stochastic target is the natural formulation (which is exactly what the existing wrappers consume). Out of all DCIC datasets this one most naturally exercises the framework's claim that p* is a distribution and not a one-hot.
- **Wiring effort**: **MEDIUM**.

### 2.12 Datasets in `experiments/first_order_data` but NOT in `probly.datasets.torch`

None — every dataset in the DCIC release has a loader class in `src/probly/datasets/torch.py`. The 10 `DCIC_DATASET_LOADERS` keys in `experiments/first_order_data/dcic_ensemble_pipeline.py:55-64` map 1:1 onto the 10 classes audited above (modulo the `Treeversity1`/`Treeversity#1` name normalisation).

---

## 3. Recommended next step

**Wire Plankton first.**

Reasoning:
- (a) **Data availability**: not on disk anywhere, but the install helper is committed (`experiments/first_order_data/install_dcic_datasets.py`); the user can run it once on the cluster and the dataset is ~hundreds of MB. Same starting cost as any other DCIC dataset, so this isn't a discriminator.
- (b) **Wrapping effort**: SMALL. Same backbone family (CIFAR-style ResNet-18) and same class count (10) as CIFAR-10H, so the existing `probly_benchmark.resnet.ResNet18` factory works (with a minor stride/stem tweak for 96² input). The 3 dataset-specific dispatches in `train_classifier.py:479`, `fit_uncertainty.py:416`, `extract_uncertainties.py:160` need new branches but each branch can be a near-copy of the CIFAR-10H one. The fold-filter adapter is the one piece of new work, but it's DCIC-wide and pays for itself across any subsequent DCIC dataset.
- (c) **Paper-narrative value**: Plankton is **a different image domain** (microscope-imaged plankton, biological / scientific imagery) from CIFAR-10H (natural-image objects), with a comparable class count and similar dataset size. That tests whether the AuReC / Pareto-gap claims survive a domain shift while keeping the experimental complexity manageable. The MiceBone alternative (medical imagery) would be more diagnostic, but the 224² backbone work and ImageNet-pretraining decisions push it into MEDIUM and double the wiring surface.

Recommended ordering for any further DCIC datasets after Plankton, on a "useful diversity per unit work" basis:
1. **Treeversity#6** (multi-label exercise; biggest stress on the soft-label assumption — directly probes the framework's p* hypothesis).
2. **MiceBone** (medical, 4-class, 224² — first dataset that requires the torchvision-backbone path; pays the way for QualityMRI later).
3. **Synthetic** (easy task; useful upper-bound baseline).
4. **Turkey / Pig / Benthic** (livestock + benthic; lower marginal information).

Drop QualityMRI unless the paper specifically calls for a tiny-dataset stress test — 310 images + binary task is too small for the framework's `(A_hat, E_hat)` decomposition to be informative.

---

## 4. Open questions for the user

1. **Dataset scope for the paper.** Locked decisions in `decisions.md` mention only CIFAR-10H, ImageNet-ReaL, APPA-REAL. Adding any DCIC dataset is outside that scope — confirm before wiring. (Section 6 of `NeurIPS2026_paper_draft.pdf` doesn't list DCIC; revisit `decisions.md` "Datasets" before any code change.)
2. **Backbone budget.** Datasets at 224² need torchvision ResNet-18/50 with ImageNet pretraining, which the existing `cifar10h.yaml` recipe doesn't accommodate (`probly_benchmark.resnet.ResNet18` is the 3×3-stem CIFAR variant). Are we willing to introduce a second backbone family, or do we restrict the DCIC additions to ≤96² datasets (Pig, Plankton, CIFAR10HDCIC) where the existing backbone reuses cleanly?
3. **Fold convention.** DCIC ships 5 predefined folds; CIFAR-10H uses the canonical CIFAR-10 train/test split. For DCIC datasets, do we (a) pick one fold (e.g. `part1`) as test and the other four as train — most literature-aligned — or (b) do 5-fold cross-validation and report mean ± std? Option (a) is simpler and matches our existing 5-seeds-per-cell convention; option (b) doubles the run grid.
4. **Zenodo record.** `install_dcic_datasets.py` points at `records/7180818`; `experiments/first_order_data/README.md` says `records/8115942`. These are different DCIC releases; confirm which version the paper should cite and we should download.
5. **Disk-space budget.** All 10 DCIC datasets together are several GB. Do we need everything or only a subset? Plankton alone is ~hundreds of MB.
6. **Cluster data placement.** Per memory, `cluster_probly/` is read-only; the user runs downloads on the cluster directly. Confirm the target path on the cluster (presumably `~/neurips2026/probly/data/<DatasetName>/` to mirror the local layout the loader expects).
