"""Become the Meme — match your webcam pose/expression to a folder of memes.

Fully local, zero-shot (no training data, no manual annotation).
"""

import logging
import os
import warnings

# Let torch fall back to CPU for the few ops the Apple Silicon (MPS) backend
# doesn't implement, instead of crashing. Set before torch initializes MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# We only pull public model weights, so the Hugging Face Hub "no HF_TOKEN /
# unauthenticated requests" nag is noise. Silence both forms (a warnings.warn and
# a logger message). Set HF_HUB_OFFLINE=1 in your env to skip the hub check
# entirely once models are cached.
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

__version__ = "0.1.0"
