# ImageNet-ReaL bucket fractions

**STATUS: PLACEHOLDER.** Run

    python experiments/epistemic_eval/scripts/check_imagenet_real_buckets.py \
        --real-json <path-to-reassessed-imagenet-master>/real.json \
        --output experiments/epistemic_eval/imagenet_real_buckets.md

to overwrite this file with the real numbers, then commit.

| Bucket           |  Count  | Fraction |
|------------------|--------:|---------:|
| empty            |       ? |        ? |
| single-label     |       ? |        ? |
| multi-label (2)  |       ? |        ? |
| multi-label (3)  |       ? |        ? |
| multi-label (4+) |       ? |        ? |
| **total**        |       ? |   100.00% |

## Notes

- Source: ``real.json`` from
  ``https://github.com/google-research/reassessed-imagenet``.
- "Empty" entries arise when the relabeling team marked an image as
  unidentifiable. Per ``decisions.md`` we drop these via
  ``ImageNetReaLDropEmpty``.
- "Multi-label" entries get uniform mass ``1/k`` over the ``k`` labels
  in the set, per ``decisions.md`` "p*(y | x) construction".
