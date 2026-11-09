"""Pianist v2: deterministic ensemble + arrangement + performance.

All modules are dependency-light (stdlib + numpy) so the backend runs without
Kaggle. Learned models (STARS / Pop2Piano / VirtuosoNet) plug in on Kaggle
via kaggle/run_batch.py and export the same PianoEvent contracts.
"""
from .contracts import PianoEvent, validate_score
from .ensemble import merge_rosvot_crepe
from .arranger import arrange_for_piano
from .performance import apply_performance

__all__ = ["PianoEvent", "validate_score", "merge_rosvot_crepe", "arrange_for_piano", "apply_performance"]
