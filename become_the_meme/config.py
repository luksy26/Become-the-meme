"""Central configuration: filesystem paths and compute-device selection.

Kept dependency-light so importing paths never forces heavy imports (torch,
etc.). Anything that needs a third-party library imports it lazily inside the
function that uses it.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Repo root is the parent of this package directory.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

MEMES_DIR: Path = BASE_DIR / "memes"
CACHE_DIR: Path = BASE_DIR / "cache"

# Image file types we will treat as memes.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ensure_dirs() -> None:
    """Create the directories the app expects, if they don't exist yet."""
    MEMES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --- Compute device ----------------------------------------------------------
def get_device() -> str:
    """Pick the best available torch device: MPS (Apple Silicon) > CUDA > CPU.

    torch is imported lazily so this module stays cheap to import.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
