"""Remove generated caches and, optionally, downloaded model weights.

    python -m become_the_meme.cleanup [--weights] [--all] [--dry-run]

Default removes the regenerable project caches under `cache/` (they rebuild on the
next run). `--weights` also prunes model weights no backend uses (the multi-GB
weights live in the Hugging Face / Torch caches outside the project; in-use ones
are kept). `--all` is a full reset: all project caches (including captured poses
and MediaPipe models) plus every downloaded weight (all re-download on next use).
`--dry-run` previews without deleting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config

# Derived caches that rebuild themselves — always safe to remove.
DERIVED_DIRS = ["meme_index", "testset_index", "concept_cache", "desc_index",
                "segmentation", "snapshots"]
# Project caches kept unless --all (re-capture / re-download cost).
HEAVY_DIRS = ["testset", "models"]

# Hugging Face repos any current code path can download — never pruned by --weights
# (only removed by --weights --all). Anything else in the HF cache is treated as an
# orphaned leftover and pruned.
USED_HF_REPOS = {
    "timm/ViT-B-16-SigLIP2",                       # default concept matcher
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",       # appearance backend + baseline
    "apple/DFN2B-CLIP-ViT-L-14",                   # comparison scripts
    "apple/DFN2B-CLIP-ViT-B-16",                   # registry extra
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",       # registry extra
    "Qwen/Qwen2-VL-2B-Instruct",                   # vlm backend
    "sentence-transformers/all-MiniLM-L6-v2",      # vlm text matching
}
TORCH_CKPT_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _clean_project(all_: bool, dry_run: bool) -> int:
    cache = config.CACHE_DIR
    targets: list[Path] = [cache / d for d in DERIVED_DIRS if (cache / d).exists()]
    targets += sorted(cache.glob("*.png"))          # loose debug / preview images
    if all_:
        targets += [cache / d for d in HEAVY_DIRS if (cache / d).exists()]
    if not targets:
        return 0
    import shutil

    print("Project cache:")
    freed = 0
    for t in targets:
        size = _size(t)
        freed += size
        print(f"  {'would remove' if dry_run else 'removed'}: cache/{t.relative_to(cache)}"
              f"  ({_human(size)})")
        if not dry_run:
            shutil.rmtree(t) if t.is_dir() else t.unlink()
    return freed


def _clean_weights(all_: bool, dry_run: bool) -> int:
    print("Model weights (Hugging Face / Torch caches):")
    freed = 0
    # --- Hugging Face cache ---
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        revisions: list[str] = []
        for repo in info.repos:
            orphaned = repo.repo_id not in USED_HF_REPOS
            if all_ or orphaned:
                tag = "" if orphaned else " (in use)"
                print(f"  {'would remove' if dry_run else 'removed'}: {repo.repo_id}"
                      f"  ({_human(repo.size_on_disk)}){tag}")
                freed += repo.size_on_disk
                revisions += [r.commit_hash for r in repo.revisions]
        if revisions and not dry_run:
            info.delete_revisions(*revisions).execute()
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  (skipped HF cache: {exc})")
    # --- Torch checkpoints (keypoint-RCNN from the abandoned pose spike; unused) ---
    if TORCH_CKPT_DIR.exists():
        for f in TORCH_CKPT_DIR.glob("*.pth"):
            size = f.stat().st_size
            freed += size
            print(f"  {'would remove' if dry_run else 'removed'}: torch/{f.name}"
                  f"  ({_human(size)})")
            if not dry_run:
                f.unlink()
    return freed


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove generated caches and, optionally, model weights.")
    parser.add_argument("--weights", action="store_true",
                        help="also prune downloaded model weights no backend uses (HF/Torch caches)")
    parser.add_argument("--all", action="store_true",
                        help="full reset: all project caches (incl. poses + MediaPipe models) AND all weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview what would be removed; delete nothing")
    args = parser.parse_args()

    freed = _clean_project(args.all, args.dry_run)
    if args.weights or args.all:
        freed += _clean_weights(all_=args.all, dry_run=args.dry_run)

    print(f"\n{'Would free' if args.dry_run else 'Freed'} {_human(freed)} total.")
    if not args.all:
        note = "Kept your captured poses and downloaded models"
        if args.weights:
            note += " (and model weights still in use)"
        print(note + " — use --all for a full reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
