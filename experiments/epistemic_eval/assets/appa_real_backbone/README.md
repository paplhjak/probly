# APPA-REAL backbone

The APPA-REAL backbone is **provided externally by the user** — a
model pretrained on a large external age-estimation corpus
(`decisions.md` → "APPA-REAL backbone strategy"). Weights are large
and live here without being committed to git; only metadata is
committed.

## Files expected in this directory

- **Weights** (gitignored): one of `weights.pt`, `weights.pth`,
  `weights.safetensors`, depending on the artifact format.
- **`backbone_meta.yaml`** (committed) describing the weights file:
  ```yaml
  # backbone_meta.yaml
  architecture: <torchvision/timm name, e.g. resnet50, vit_b_16>
  weights_file: weights.pt
  weights_sha256: <hex digest>
  pretraining_corpus: <name of the external corpus>
  pretraining_corpus_size: <integer or "unknown">
  feature_layer: <name of the layer used as the frozen feature
                  extractor, e.g. "avgpool" or "norm" — the head
                  attaches after this layer>
  feature_dim: <int>          # in_features for the head's first Linear
  expected_input:
    image_size: [H, W]        # e.g. [224, 224]
    channels: 3
    mean: [..., ..., ...]
    std:  [..., ..., ...]
    interpolation: <bicubic|bilinear|nearest>
  notes: |
    Free-form notes on caveats, license, provenance.
  ```

## Why metadata-but-not-weights

Reproducibility: the metadata is enough to identify the backbone, but
the weights file may be large or have license restrictions that
prevent committing. The `weights_sha256` field lets us verify a
local file matches the artifact assumed by the pipeline.

## .gitignore policy

`.gitignore` at the repo root carves out this directory:

```
experiments/epistemic_eval/assets/appa_real_backbone/*
!experiments/epistemic_eval/assets/appa_real_backbone/*.md
!experiments/epistemic_eval/assets/appa_real_backbone/backbone_meta.yaml
```

Do not commit weights even if the user asks — flag it and ask for
explicit confirmation first.

## Status

**Path: TBD** per `decisions.md` → "APPA-REAL backbone strategy".
The user has not yet shared the weights file or written
`backbone_meta.yaml`. Required before the APPA-REAL extraction run
actually executes; not required to write the code.
