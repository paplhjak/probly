# ImageNet-ReaL bucket fractions

| Bucket           |  Count  | Fraction |
|------------------|--------:|---------:|
| empty            |    3163 |    6.33% |
| single-label     |   39394 |   78.79% |
| multi-label (2)  |    5408 |   10.82% |
| multi-label (3)  |    1319 |    2.64% |
| multi-label (4+) |     716 |    1.43% |
| **total**        |   50000 | 100.00% |

## Notes

- Source: ``real.json`` from
  ``https://github.com/google-research/reassessed-imagenet``.
- "Empty" entries arise when the relabeling team marked an image as
  unidentifiable. Per ``decisions.md`` we drop these via
  ``ImageNetReaLDropEmpty``.
- "Multi-label" entries get uniform mass ``1/k`` over the ``k`` labels
  in the set, per ``decisions.md`` "p*(y | x) construction".
