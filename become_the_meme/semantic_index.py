"""Description-based meme index for the VLM matcher.

Each meme is described once (Qwen2-VL) and the description is embedded with a
small sentence model (all-MiniLM). Matching a live frame means: describe the
frame, embed that text, and find the nearest meme description by cosine
similarity. Descriptions abstract away identity, so this responds to actions and
expressions rather than appearance.

Like the CLIP index, the cache is incremental (only new/changed memes are
re-described) and self-invalidates if the VLM model, the prompt, or the text
model changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .meme_index import Match  # reuse (path, score) result type
from .vlm import DEFAULT_PROMPT, VLMDescriber

DEFAULT_INDEX_DIR: Path = config.CACHE_DIR / "desc_index"
DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"
_EMB_FILE = "embeddings.npy"
_MANIFEST_FILE = "manifest.json"


class TextEmbedder:
    """L2-normalized sentence embeddings via sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)


class SemanticIndex:
    """Cached, incrementally-updated description index over a folder of memes."""

    def __init__(
        self,
        describer: VLMDescriber | None = None,
        text_embedder: TextEmbedder | None = None,
        prompt: str = DEFAULT_PROMPT,
        memes_dir: Path = config.MEMES_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> None:
        self.describer = describer
        self.text = text_embedder or TextEmbedder()
        self.prompt = prompt
        self.memes_dir = Path(memes_dir)
        self.index_dir = Path(index_dir)
        self.paths: list[str] = []
        self.descriptions: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, self.text.dim), dtype=np.float32)

    # --- identity used to invalidate the cache -------------------------------
    @property
    def _vlm_model_id(self) -> str:
        return self.describer.model_id if self.describer else "?"

    @property
    def _emb_path(self) -> Path:
        return self.index_dir / _EMB_FILE

    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / _MANIFEST_FILE

    def _load_cached(self) -> dict[str, tuple[str, np.ndarray, int, float]]:
        """Return {relpath: (description, embedding, size, mtime)} or {} to rebuild."""
        if not self._manifest_path.exists() or not self._emb_path.exists():
            return {}
        try:
            manifest = json.loads(self._manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        # Any change to how descriptions or their embeddings are produced makes
        # the whole cache invalid.
        if (
            manifest.get("text_model") != self.text.model_name
            or manifest.get("prompt") != self.prompt
            or (self.describer is not None and manifest.get("vlm_model") != self._vlm_model_id)
        ):
            return {}
        embeddings = np.load(self._emb_path)
        entries = manifest.get("entries", [])
        if len(entries) != embeddings.shape[0]:
            return {}
        return {
            e["path"]: (e["desc"], embeddings[i], int(e["size"]), float(e["mtime"]))
            for i, e in enumerate(entries)
        }

    def _save(self, entries: list[tuple[str, str, int, float]]) -> None:
        """entries: list of (relpath, desc, size, mtime), aligned with embeddings."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path, self.embeddings)
        manifest = {
            "vlm_model": self._vlm_model_id,
            "text_model": self.text.model_name,
            "prompt": self.prompt,
            "dim": int(self.text.dim),
            "entries": [
                {"path": p, "desc": d, "size": s, "mtime": m}
                for (p, d, s, m) in entries
            ],
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    def load(self) -> bool:
        cached = self._load_cached()
        if not cached:
            return False
        self.paths = list(cached.keys())
        self.descriptions = [cached[p][0] for p in self.paths]
        self.embeddings = np.stack([cached[p][1] for p in self.paths]).astype(np.float32)
        return True

    def _scan_files(self) -> list[Path]:
        if not self.memes_dir.exists():
            return []
        return sorted(
            p
            for p in self.memes_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )

    def build(self, force: bool = False, progress: bool = True) -> int:
        """Describe new/changed memes and (re)embed. Returns count described."""
        cached = {} if force else self._load_cached()
        files = self._scan_files()

        keep_desc: dict[str, str] = {}
        keep_emb: dict[str, np.ndarray] = {}
        to_describe: list[Path] = []
        meta: dict[str, tuple[int, float]] = {}
        rels: dict[Path, str] = {}

        for f in files:
            rel = str(f.relative_to(self.memes_dir))
            rels[f] = rel
            st = f.stat()
            meta[rel] = (st.st_size, st.st_mtime)
            prev = cached.get(rel)
            if prev is not None and prev[2] == st.st_size and prev[3] == st.st_mtime:
                keep_desc[rel], keep_emb[rel] = prev[0], prev[1]
            else:
                to_describe.append(f)

        if to_describe and self.describer is None:
            raise RuntimeError(
                f"{len(to_describe)} meme(s) need describing but no VLM describer "
                "was provided. Construct SemanticIndex with a VLMDescriber."
            )

        # Describe new/changed images, then embed those descriptions in one batch.
        new_desc: dict[str, str] = {}
        for i, f in enumerate(to_describe, 1):
            if progress:
                print(f"  describing {i}/{len(to_describe)}: {f.name}")
            new_desc[rels[f]] = self.describer.describe(f)
        if new_desc:
            new_rels = list(new_desc)
            vecs = self.text.embed([new_desc[r] for r in new_rels])
            for r, v in zip(new_rels, vecs):
                keep_desc[r], keep_emb[r] = new_desc[r], v

        order = [rels[f] for f in files if rels[f] in keep_desc]
        self.paths = order
        self.descriptions = [keep_desc[r] for r in order]
        self.embeddings = (
            np.stack([keep_emb[r] for r in order]).astype(np.float32)
            if order else np.zeros((0, self.text.dim), dtype=np.float32)
        )
        self._save([(r, keep_desc[r], meta[r][0], meta[r][1]) for r in order])
        return len(new_desc)

    def match_text(self, query_desc: str, top_k: int = 1) -> list[Match]:
        """Nearest memes to a query description (by sentence-embedding cosine)."""
        if self.embeddings.shape[0] == 0:
            return []
        q = self.text.embed([query_desc])[0]
        sims = self.embeddings @ q
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Match(path=self.memes_dir / self.paths[i], score=float(sims[i])) for i in idx]

    def description_of(self, path: Path) -> str:
        """Return the cached description for a meme path (empty string if unknown)."""
        rel = str(Path(path).relative_to(self.memes_dir)) if Path(path).is_absolute() else str(path)
        try:
            return self.descriptions[self.paths.index(rel)]
        except ValueError:
            return ""
