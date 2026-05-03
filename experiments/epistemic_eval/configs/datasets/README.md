# Dataset configs

One YAML per first-order dataset. Each config declares paths, splits,
and **which deployment losses are supported** (CE only for the
classification datasets; squared and absolute for APPA-REAL).

See `decisions.md` -> "p*(y | x) construction" for per-dataset `p*`
construction rules and "Deployment loss" for the supported-losses
contract.

## Schema

```yaml
name: <short-name>             # canonical name; used in run_id
loader:
  module: <importable.module.path>
  class: <ClassName>
loader_kwargs:
  ...                          # passed verbatim to __init__
supported_losses: [...]        # subset of {cross_entropy, zero_one, squared, absolute}
metadata:
  ...                          # informational only; not consumed by code
```

`loader.module` + `loader.class` is the importable target instantiated
with `**loader_kwargs`. `supported_losses` is a guard rail: the
experiment driver checks that the experiment config's `loss` field is
in this list before running anything. `metadata` is for human readers
only and is not parsed.

Files (Task 4):

- `cifar10h.yaml` - probly's `CIFAR10H`, cross-entropy / zero-one.
- `imagenet_real.yaml` - drop-empty wrapper around probly's
  `ImageNetReaL`, cross-entropy / zero-one.
- `appa_real.yaml` - new `AppaReal` loader, squared / absolute.
