"""Package entry point.

    python -m become_the_meme          # launch the live meme matcher
    python -m become_the_meme --check  # environment smoke test
"""

from __future__ import annotations

import importlib
import platform
import sys

from . import __version__, config

# (import name, human label, attribute holding the version string)
_DEPENDENCIES = [
    ("numpy", "NumPy", "__version__"),
    ("PIL", "Pillow", "__version__"),
    ("cv2", "OpenCV", "__version__"),
    ("mediapipe", "MediaPipe", "__version__"),
    ("torch", "PyTorch", "__version__"),
    ("open_clip", "OpenCLIP", "__version__"),
]


def _check_dependencies() -> bool:
    all_ok = True
    for module_name, label, version_attr in _DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, version_attr, "?")
            print(f"  [ok]   {label:<10} {version}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            all_ok = False
            print(f"  [FAIL] {label:<10} {exc}")
    return all_ok


def check() -> int:
    """Environment smoke test: verify deps import and report the compute device."""
    print(f"Become the Meme v{__version__} — environment check")
    print(f"  Python     {platform.python_version()} ({sys.executable})")
    print(f"  Platform   {platform.platform()}")

    config.ensure_dirs()
    meme_count = sum(
        1
        for p in config.MEMES_DIR.iterdir()
        if p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    print(f"  Memes dir  {config.MEMES_DIR}  ({meme_count} image(s))")
    print(f"  Cache dir  {config.CACHE_DIR}")

    print("Dependencies:")
    deps_ok = _check_dependencies()
    if deps_ok:
        print(f"Compute device: {config.get_device()}")
        print("\nEnvironment looks good. ✅")
        return 0
    print("\nSome dependencies failed to import. Run: pip install -r requirements.txt")
    return 1


def main() -> int:
    if "--check" in sys.argv[1:]:
        return check()
    from .app import main as app_main

    return app_main([a for a in sys.argv[1:] if a != "--check"])


if __name__ == "__main__":
    raise SystemExit(main())
