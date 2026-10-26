"""stainless: label-free in, stain out.

Per-marker virtual-stain models for imaging cytometers (BD FACSDiscover A8/S8 CellView,
Amnis ImageStream, Attune CytPix). See docs/plan_modular.md for the design.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository
SCRIPTS = os.path.join(ROOT, "scripts")                               # stdlib server tools, importable here too
__version__ = "0.1.0"
