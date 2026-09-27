"""Composable CLIP matching strategies: model x representation x method x normalization.

A `MatchStrategy` ranks memes for a query frame. Heavy work (loading a model,
embedding memes, building the concept matrix) is shared across strategies that use
the same model, via `ModelBackend`. See the plan for the algorithm details.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..embedding import CLIPEmbedder
from ..meme_index import Match, MemeIndex
from ..segmentation import PersonSegmenter
from ..webcam import Frame

# --- model registry (verified available in open_clip 3.3.0) ------------------
MODELS: dict[str, tuple[str, str]] = {
    "b32": ("ViT-B-32", "laion2b_s34b_b79k"),               # baseline
    "l14_dfn": ("ViT-L-14-quickgelu", "dfn2b"),             # strong (quickgelu arch matches dfn2b)
    "siglip2_b": ("ViT-B-16-SigLIP2", "webli"),             # different family
    # extras (available if we want to widen the sweep)
    "l14_laion": ("ViT-L-14", "laion2b_s32b_b82k"),
    "b16_dfn": ("ViT-B-16-quickgelu", "dfn2b"),
}

TEMPLATES = ["a photo of {c}", "an image of {c}", "{c}", "a meme of {c}"]
CONCEPTS_PATH = Path(__file__).parent / "concepts.txt"
TESTSET_INDEX_DIR = config.CACHE_DIR / "testset_index"
CONCEPT_CACHE_DIR = config.CACHE_DIR / "concept_cache"

REPRESENTATIONS = ("raw", "cutout", "bbox_crop", "upper_body_crop", "multi_crop")
METHODS = ("image", "concept")
# query_* norms calibrate each meme against a set of real poses (a query prior),
# which removes hub memes far better than meme-vs-meme calibration.
NORMALIZATIONS = ("none", "hubness", "meancenter", "zscore", "query_center", "query_zscore")


# --- small numeric helpers ---------------------------------------------------
def _l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        n = np.linalg.norm(a)
        return a / n if n > 0 else a
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def load_concepts(path: Path = CONCEPTS_PATH) -> list[str]:
    concepts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            concepts.append(line)
    return concepts


def fingerprint(X: np.ndarray, T: np.ndarray, mode: str = "raw",
                k: int = 8, tau: float = 0.01) -> np.ndarray:
    """Project image embeddings X (n,D) onto concepts T (C,D) -> (n,C)."""
    S = X @ T.T
    if mode == "raw":
        return S
    if mode == "softmax":
        Z = S / tau
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)
    if mode == "topk":
        out = np.zeros_like(S)
        idx = np.argsort(-S, axis=1)[:, :k]
        rows = np.arange(S.shape[0])[:, None]
        out[rows, idx] = S[rows, idx]
        return out
    raise ValueError(f"unknown fingerprint mode: {mode}")


# --- representations (query-side transforms) ---------------------------------
def _upper_body_crop(frame: Frame, mask: np.ndarray, threshold: float,
                     frac: float = 0.6) -> Frame:
    ys, xs = np.where(mask >= threshold)
    if xs.size == 0:
        return frame
    h, w = frame.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    y1 = y0 + max(1, int((y1 - y0) * frac))
    pad_x = int((x1 - x0) * 0.1)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x + 1)
    return frame[y0:y1, x0:x1]


def _center_square(frame: Frame) -> Frame:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def apply_representation(name: str, frame: Frame,
                         segmenter: PersonSegmenter | None) -> list[Frame]:
    """Return the list of crops to embed for a representation (mean-pooled later)."""
    if name == "raw":
        return [frame]
    if segmenter is None:
        raise ValueError(f"representation '{name}' needs a PersonSegmenter")
    mask = segmenter.segment(frame)
    if name == "cutout":
        return [segmenter.cutout(frame, mask)]
    if name == "bbox_crop":
        return [segmenter.bbox_crop(frame, mask)]
    if name == "upper_body_crop":
        return [_upper_body_crop(frame, mask, segmenter.threshold)]
    if name == "multi_crop":
        bbox = segmenter.bbox_crop(frame, mask)
        return [
            bbox,
            _upper_body_crop(frame, mask, segmenter.threshold),
            _center_square(frame),
            cv2.flip(bbox, 1),
        ]
    raise ValueError(f"unknown representation: {name}")


def embed_with_rep(embedder: CLIPEmbedder, frame: Frame, representation: str,
                   segmenter: PersonSegmenter | None) -> np.ndarray:
    """Embed a frame under a representation -> (D,) L2-normalized (mean over crops)."""
    crops = apply_representation(representation, frame, segmenter)
    vecs = embedder.embed_images(crops)          # (n_crops, D), each normalized
    return _l2(vecs.mean(axis=0))


# --- per-model backend (shared across its strategies) ------------------------
class ModelBackend:
    def __init__(self, model_key: str) -> None:
        name, pretrained = MODELS[model_key]
        self.model_key = model_key
        self.embedder = CLIPEmbedder(name, pretrained)
        self.model_id = self.embedder.model_id
        self._meme_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._concept_cache: dict[tuple, np.ndarray] = {}

    def meme_base_embeddings(
        self, meme_rep: str = "raw", segmenter: PersonSegmenter | None = None
    ) -> tuple[list[str], np.ndarray]:
        if meme_rep in self._meme_cache:
            return self._meme_cache[meme_rep]
        slug = _slug(self.model_id)
        if meme_rep == "raw":
            index = MemeIndex(self.embedder, index_dir=TESTSET_INDEX_DIR / f"{slug}__raw")
            index.build()
            result = (list(index.paths), _l2(index.embeddings.astype(np.float32)))
        else:
            paths, _ = self.meme_base_embeddings("raw")  # fix canonical ordering
            vecs = []
            for rel in paths:
                img = cv2.imread(str(config.MEMES_DIR / rel))
                vecs.append(embed_with_rep(self.embedder, img, meme_rep, segmenter))
            result = (paths, np.stack(vecs).astype(np.float32))
        self._meme_cache[meme_rep] = result
        return result

    def concept_matrix(self, concepts: list[str], templates: list[str]) -> np.ndarray:
        key = (tuple(concepts), tuple(templates))
        if key in self._concept_cache:
            return self._concept_cache[key]
        slug = _slug(self.model_id)
        digest = hashlib.sha1(
            ("\n".join(concepts) + "||" + "\n".join(templates)).encode()
        ).hexdigest()[:16]
        path = CONCEPT_CACHE_DIR / f"{slug}__{digest}.npy"
        if path.exists():
            T = np.load(path).astype(np.float32)
        else:
            texts = [t.format(c=c) for c in concepts for t in templates]
            E = self.embedder.embed_text(texts)              # (C*Tt, D)
            E = E.reshape(len(concepts), len(templates), -1).mean(axis=1)
            T = _l2(E)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, T)
        self._concept_cache[key] = T
        return T


# --- the strategy ------------------------------------------------------------
@dataclass
class MatchStrategy:
    name: str
    model_key: str
    representation: str
    method: str
    normalization: str
    meme_representation: str = "raw"
    fingerprint_mode: str = "raw"
    beta: float = 1.0

    # runtime state (set in prepare)
    _backend: ModelBackend | None = field(default=None, repr=False)
    _segmenter: PersonSegmenter | None = field(default=None, repr=False)
    _paths: list[str] = field(default_factory=list, repr=False)
    _V: np.ndarray | None = field(default=None, repr=False)
    _T: np.ndarray | None = field(default=None, repr=False)
    _cmean: np.ndarray | None = field(default=None, repr=False)
    _h: np.ndarray | None = field(default=None, repr=False)
    _mu_j: np.ndarray | None = field(default=None, repr=False)
    _sigma_j: np.ndarray | None = field(default=None, repr=False)
    _muv: np.ndarray | None = field(default=None, repr=False)
    _qmu: np.ndarray | None = field(default=None, repr=False)   # per-meme mean over calibration poses
    _qsigma: np.ndarray | None = field(default=None, repr=False)  # per-meme std over calibration poses

    def prepare(self, backend: ModelBackend, segmenter: PersonSegmenter | None,
                concepts: list[str], templates: list[str]) -> "MatchStrategy":
        self._backend = backend
        self._segmenter = segmenter
        self._paths, M = backend.meme_base_embeddings(self.meme_representation, segmenter)

        if self.method == "image":
            self._V = M
        elif self.method == "concept":
            self._T = backend.concept_matrix(concepts, templates)
            Fr = fingerprint(M, self._T, self.fingerprint_mode)
            self._cmean = Fr.mean(axis=0)
            self._V = _l2(Fr - self._cmean)
        else:
            raise ValueError(f"unknown method: {self.method}")

        # normalization stats derived from the meme-meme similarity in this space
        if self.normalization in ("hubness", "zscore"):
            H = self._V @ self._V.T
            off = H.copy()
            np.fill_diagonal(off, np.nan)
            self._h = np.nanmean(off, axis=1).astype(np.float32)
            if self.normalization == "zscore":
                self._mu_j = np.nanmean(off, axis=1).astype(np.float32)
                self._sigma_j = np.nanstd(off, axis=1).astype(np.float32)
        elif self.normalization == "meancenter":
            self._muv = self._V.mean(axis=0)
            self._V = _l2(self._V - self._muv)
        return self

    def _query_vector_from_emb(self, q_img: np.ndarray) -> np.ndarray:
        v = q_img
        if self.method == "concept":
            v = _l2(fingerprint(q_img[None], self._T, self.fingerprint_mode)[0] - self._cmean)
        if self.normalization == "meancenter":
            v = _l2(v - self._muv)
        return v

    def raw_meme_scores(self, q_img: np.ndarray) -> np.ndarray:
        """Pre-normalization similarity of a query embedding to every meme (n_memes,)."""
        return self._V @ self._query_vector_from_emb(q_img)

    def calibrate(self, query_img_embeddings: np.ndarray) -> "MatchStrategy":
        """Set the query prior for query_* norms from calibration pose embeddings (M, D)."""
        P = np.stack([self.raw_meme_scores(e) for e in query_img_embeddings])  # (M, n_memes)
        self._qmu = P.mean(axis=0).astype(np.float32)
        self._qsigma = P.std(axis=0).astype(np.float32)
        return self

    def rank_from_emb(self, q_img: np.ndarray, top_k: int | None = None) -> list[Match]:
        """Rank memes from an already-computed query image embedding (D,)."""
        s = self._V @ self._query_vector_from_emb(q_img)
        if self.normalization == "hubness":
            s = s - self.beta * self._h
        elif self.normalization == "zscore":
            s = (s - self._mu_j) / (self._sigma_j + 1e-6)
        elif self.normalization == "query_center" and self._qmu is not None:
            s = s - self._qmu
        elif self.normalization == "query_zscore" and self._qmu is not None:
            s = (s - self._qmu) / (self._qsigma + 1e-6)
        order = np.argsort(-s)
        if top_k is not None:
            order = order[:top_k]
        return [Match(path=config.MEMES_DIR / self._paths[i], score=float(s[i])) for i in order]

    def rank(self, frame: Frame, top_k: int | None = None) -> list[Match]:
        q_img = embed_with_rep(self._backend.embedder, frame, self.representation, self._segmenter)
        return self.rank_from_emb(q_img, top_k)


# --- presets -----------------------------------------------------------------
def default_strategies() -> list[MatchStrategy]:
    """The curated 11-strategy comparison set (see plan)."""
    S = MatchStrategy
    return [
        S("b32_raw_image", "b32", "raw", "image", "none"),
        S("b32_bbox_image", "b32", "bbox_crop", "image", "none"),
        S("b32_bbox_image_hub", "b32", "bbox_crop", "image", "hubness"),
        S("b32_bbox_concept", "b32", "bbox_crop", "concept", "none"),
        S("b32_bbox_concept_z", "b32", "bbox_crop", "concept", "zscore"),
        S("b32_upper_concept_hub", "b32", "upper_body_crop", "concept", "hubness"),
        S("b32_multi_concept_mc", "b32", "multi_crop", "concept", "meancenter"),
        S("l14_bbox_image_hub", "l14_dfn", "bbox_crop", "image", "hubness"),
        S("l14_upper_concept_z", "l14_dfn", "upper_body_crop", "concept", "zscore"),
        S("siglip2_bbox_concept_hub", "siglip2_b", "bbox_crop", "concept", "hubness"),
        S("l14_bbox_concept_hub", "l14_dfn", "bbox_crop", "concept", "hubness"),
        S("siglip2_upper_concept_z", "siglip2_b", "upper_body_crop", "concept", "zscore"),
        # current best (query-calibrated): the one to tune against. Needs a
        # calibration pose set (evaluate does leave-one-out; the app uses cache/testset).
        S("siglip2_bbox_concept_qz", "siglip2_b", "bbox_crop", "concept", "query_zscore"),
    ]


def strategies_by_name(names: list[str]) -> list[MatchStrategy]:
    lookup = {s.name: s for s in default_strategies()}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"unknown strategies {missing}; choices: {list(lookup)}")
    return [lookup[n] for n in names]


def prepare_strategies(strategies: list[MatchStrategy], segmenter: PersonSegmenter | None,
                       concepts: list[str] | None = None,
                       templates: list[str] | None = None) -> list[MatchStrategy]:
    """Load each model once (grouped) and prepare every strategy."""
    concepts = concepts or load_concepts()
    templates = templates or TEMPLATES
    backends: dict[str, ModelBackend] = {}
    for st in strategies:
        if st.model_key not in backends:
            name, pretrained = MODELS[st.model_key]
            print(f"loading model '{st.model_key}' ({name}/{pretrained})...")
            backends[st.model_key] = ModelBackend(st.model_key)
        st.prepare(backends[st.model_key], segmenter, concepts, templates)
    return strategies
