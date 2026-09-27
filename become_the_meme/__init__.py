"""Become the Meme — match your webcam pose/expression to a folder of memes.

Fully local, zero-shot (no training data, no manual annotation).
"""

import os

# Let torch fall back to CPU for the few ops the Apple Silicon (MPS) backend
# doesn't implement, instead of crashing. Set before torch initializes MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
