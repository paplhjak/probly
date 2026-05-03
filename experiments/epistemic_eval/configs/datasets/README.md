# Dataset configs

One YAML per first-order dataset. Each config declares paths, splits,
and **which deployment losses are supported** (CE only for the
classification datasets; squared and absolute for APPA-REAL).

See `decisions.md` → "p*(y | x) construction" for per-dataset `p*`
construction rules and "Deployment loss" for the supported-losses
contract.

Files (Task 4):

- `cifar10h.yaml`
- `imagenet_real.yaml`
- `appa_real.yaml`
