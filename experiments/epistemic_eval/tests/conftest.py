"""Test fixtures local to ``experiments/epistemic_eval/tests``.

Adds the experiment's ``scripts/`` directory to ``sys.path`` so that
test modules can import the script files as bare names (e.g.
``import imagenet_real_adapter``) without polluting probly's import
space.
"""

from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
