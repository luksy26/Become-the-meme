"""Meme embedding index: precompute + cache CLIP vectors for the memes folder.

Scans ``memes/``, embeds each image with CLIP, and caches the result so startup
is instant on later runs. The cache is *incremental*: only new or changed files
are re-embedded, removed files are dropped, and if the embedding model changes
the whole index is rebuilt automatically (embeddings from different models are
not comparable).

Matching a query vector against the index is a single normalized matrix-vector
product (cosine similarity), so it stays fast even with many memes.

Demo:
    python -m become_the_meme.meme_index --build            # build / update
    python -m become_the_meme.meme_index --rebuild          # force full rebuild
    python -m become_the_meme.meme_index --stats            # show index info
    python -m become_the_meme.meme_index --match photo.jpg  # nearest memes
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .embedding import CLIPEmbedder

DEFAULT_INDEX_DIR: Path = config.CACHE_DIR / "meme_index"
_EMB_FILE = "embeddings.npy"
_MANIFEST_FILE = "manifest.json"


@dataclass
class BuildStats:
    total: int
    new: int
    reused: int
    removed: int
    failed: int

    def __str__(self) -> str:
        return (
            f"index: {self.total} memes "
            f"({self.new} embedded, {self.reused} reused, "
            f"{self.removed} removed, {self.failed} failed)"
        )


@dataclass
class Match:
    path: Path       # absolute path to the meme image
    score: float     # cosine similarity in [-1, 1] (typically 0..1)


class MemeIndex:
    """A cached, incrementally-updated CLIP index over a folder of memes."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        memes_dir: Path = config.MEMES_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> None:
        self.embedder = embedder
        self.memes_dir = Path(memes_dir)
        self.index_dir = Path(index_dir)
        self.paths: list[str] = []          # relative paths, aligned with rows
        self.embeddings: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)

    # --- persistence ---------------------------------------------------------
    @property
    def _emb_path(self) -> Path:
        return self.index_dir / _EMB_FILE

    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / _MANIFEST_FILE

    def _load_cached(self) -> dict[str, tuple[np.ndarray, int, float]]:
        """Load cached embeddings keyed by relative path.

        Returns an empty dict (forcing a full rebuild) if nothing is cached or
        the cache was produced by a different embedding model.
        """
        if not self._manifest_path.exists() or not self._emb_path.exists():
            return {}
        try:
            manifest = json.loads(self._manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        if manifest.get("model") != self.embedder.model_id:
            return {}  # different model -> old vectors are meaningless

        embeddings = np.load(self._emb_path)
        entries = manifest.get("entries", [])
        if len(entries) != embeddings.shape[0]:
            return {}  # corrupt/mismatched cache -> rebuild
        return {
            e["path"]: (embeddings[i], int(e["size"]), float(e["mtime"]))
            for i, e in enumerate(entries)
        }

    def _save(self, entries: list[tuple[str, int, float]]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path, self.embeddings)
        manifest = {
            "model": self.embedder.model_id,
            "dim": int(self.embedder.dim),
            "entries": [
                {"path": p, "size": s, "mtime": m} for (p, s, m) in entries
            ],
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    def load(self) -> bool:
        """Load an existing index into memory (no folder scan). Returns success."""
        cached = self._load_cached()
        if not cached:
            return False
        self.paths = list(cached.keys())
        self.embeddings = np.stack([cached[p][0] for p in self.paths]).astype(np.float32)
        return True

    # --- building ------------------------------------------------------------
    def _scan_files(self) -> list[Path]:
        if not self.memes_dir.exists():
            return []
        files = [
            p
            for p in self.memes_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        ]
        return sorted(files)

    def build(self, force: bool = False, chunk: int = 32) -> BuildStats:
        """Build or incrementally update the index. Returns build statistics."""
        cached = {} if force else self._load_cached()
        files = self._scan_files()

        kept: dict[str, np.ndarray] = {}
        to_embed: list[Path] = []
        rels: dict[Path, str] = {}
        stats_meta: dict[str, tuple[int, float]] = {}

        for f in files:
            rel = str(f.relative_to(self.memes_dir))
            rels[f] = rel
            st = f.stat()
            stats_meta[rel] = (st.st_size, st.st_mtime)
            prev = cached.get(rel)
            if prev is not None and prev[1] == st.st_size and prev[2] == st.st_mtime:
                kept[rel] = prev[0]  # unchanged -> reuse cached vector
            else:
                to_embed.append(f)

        # Embed the new/changed files in chunks (skip unreadable ones).
        computed: dict[str, np.ndarray] = {}
        failed = 0
        for i in range(0, len(to_embed), chunk):
            batch_files = to_embed[i : i + chunk]
            images: list[Image.Image] = []
            ok_rels: list[str] = []
            for f in batch_files:
                try:
                    images.append(Image.open(f).convert("RGB"))
                    ok_rels.append(rels[f])
                except (UnidentifiedImageError, OSError):
                    failed += 1
            if images:
                vectors = self.embedder.embed_images(images)
                for rel, vec in zip(ok_rels, vectors):
                    computed[rel] = vec

        # Assemble the final index in stable scan order.
        order = [rels[f] for f in files if rels[f] in kept or rels[f] in computed]
        if order:
            self.embeddings = np.stack(
                [kept.get(r, computed.get(r)) for r in order]
            ).astype(np.float32)
        else:
            self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.paths = order

        entries = [(r, stats_meta[r][0], stats_meta[r][1]) for r in order]
        self._save(entries)

        removed = len(cached) - len(kept)
        return BuildStats(
            total=len(order),
            new=len(computed),
            reused=len(kept),
            removed=max(0, removed),
            failed=failed,
        )

    # --- matching ------------------------------------------------------------
    def match(self, query_embedding: np.ndarray, top_k: int = 1) -> list[Match]:
        """Return the ``top_k`` closest memes to a (normalized) query vector."""
        if self.embeddings.shape[0] == 0:
            return []
        sims = self.embeddings @ query_embedding.astype(np.float32)
        k = min(top_k, sims.shape[0])
        # argpartition for top-k, then sort just those k by score desc.
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Match(path=self.memes_dir / self.paths[i], score=float(sims[i])) for i in idx]


# --- demo / entry point ------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Meme CLIP index.")
    parser.add_argument("--build", action="store_true", help="build/update the index")
    parser.add_argument("--rebuild", action="store_true", help="force a full rebuild")
    parser.add_argument("--stats", action="store_true", help="show current index info")
    parser.add_argument("--match", metavar="IMAGE", help="find nearest memes to an image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = CLIPEmbedder()
    index = MemeIndex(embedder)

    if args.rebuild or args.build:
        stats = index.build(force=args.rebuild)
        print(stats)

    if args.stats and not (args.build or args.rebuild):
        if index.load():
            print(f"index: {len(index.paths)} memes, model={embedder.model_id}, "
                  f"dim={embedder.dim}")
        else:
            print("no index cached yet — run with --build")

    if args.match:
        if not index.paths and not index.load():
            print("index is empty — add memes to memes/ and run --build")
            return 1
        query_img = Image.open(args.match).convert("RGB")
        q = embedder.embed_image(query_img)
        matches = index.match(q, top_k=args.top_k)
        print(f"top {len(matches)} matches for {args.match}:")
        for m in matches:
            print(f"  {m.score:.3f}  {m.path.name}")

    if not any((args.build, args.rebuild, args.stats, args.match)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
