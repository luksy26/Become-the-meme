"""Frame -> best-matching meme.

Ties the pieces together: take a webcam frame, optionally isolate the person
(segmentation), embed it with CLIP, and look up the nearest meme in the index.

Matching is appearance/expression based (CLIP). It captures "which meme do you
look/vibe like", not precise body actions — see the project notes for why exact
pose matching was deferred.

The query *representation* is switchable so it's easy to compare live:
* ``bbox_crop`` — crop to the person (removes room clutter, keeps natural look).
* ``cutout``    — hard background removal onto neutral gray.
* ``raw``       — the whole frame, unmodified.
"""

from __future__ import annotations

from . import config
from .embedding import CLIPEmbedder
from .meme_index import Match, MemeIndex
from .segmentation import PersonSegmenter
from .webcam import Frame

REPRESENTATIONS = ("bbox_crop", "cutout", "raw")
DEFAULT_REPRESENTATION = "bbox_crop"


class MemeMatcher:
    """Matches webcam frames to the meme index using CLIP embeddings."""

    def __init__(
        self,
        representation: str = DEFAULT_REPRESENTATION,
        segmenter_model: str = "multiclass",
        embedder: CLIPEmbedder | None = None,
        index: MemeIndex | None = None,
        auto_build: bool = True,
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")

        self.embedder = embedder or CLIPEmbedder()
        self.index = index or MemeIndex(self.embedder)
        # Incrementally build/update so newly added memes are picked up on start;
        # this is a no-op cost when nothing changed.
        if auto_build:
            self.index.build()
        else:
            self.index.load()

        self.representation = representation
        self._segmenter_model = segmenter_model
        self._segmenter: PersonSegmenter | None = None
        if representation in ("bbox_crop", "cutout"):
            self._segmenter = PersonSegmenter(model=segmenter_model)

    @property
    def num_memes(self) -> int:
        return len(self.index.paths)

    def set_representation(self, representation: str) -> None:
        """Switch query representation, lazily creating the segmenter if needed."""
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        if representation in ("bbox_crop", "cutout") and self._segmenter is None:
            self._segmenter = PersonSegmenter(model=self._segmenter_model)
        self.representation = representation

    def preprocess(self, frame: Frame) -> Frame:
        """Turn a raw frame into the image we actually embed."""
        if self.representation == "raw" or self._segmenter is None:
            return frame
        mask = self._segmenter.segment(frame)
        if self.representation == "cutout":
            return self._segmenter.cutout(frame, mask)
        return self._segmenter.bbox_crop(frame, mask)

    def match_frame(self, frame: Frame, top_k: int = 1) -> tuple[list[Match], Frame]:
        """Return (matches, preprocessed_query_image) for a frame."""
        processed = self.preprocess(frame)
        query = self.embedder.embed_image(processed)
        return self.index.match(query, top_k=top_k), processed


class VLMMatcher:
    """Matches frames by describing the action/expression (VLM) and comparing text.

    Slower than :class:`MemeMatcher` (~1-4s/frame) but responds to what you're
    *doing* rather than who you look like. Run it off the UI thread.
    """

    def __init__(self, describer=None, index=None, auto_build: bool = True) -> None:
        from .semantic_index import SemanticIndex
        from .vlm import VLMDescriber

        self.describer = describer or VLMDescriber()
        self.index = index or SemanticIndex(describer=self.describer)
        if auto_build:
            self.index.build()
        else:
            self.index.load()

    @property
    def num_memes(self) -> int:
        return len(self.index.paths)

    def match_frame(self, frame: Frame, top_k: int = 1) -> tuple[list[Match], str]:
        """Return (matches, query_description) for a frame."""
        description = self.describer.describe(frame)
        return self.index.match_text(description, top_k=top_k), description

    def description_of(self, path) -> str:
        return self.index.description_of(path)


class ConceptMatcher:
    """Fast CLIP-concept matcher — the winning strategy from the harness.

    SigLIP2 + bbox crop + concept projection + query-calibrated z-score. Matches
    on *what you're doing* (via a generic action/expression concept vocabulary)
    rather than who you look like, at ~130ms/frame. The z-score is calibrated
    against your captured poses (cache/testset), which removes "hub" memes that
    otherwise win everything. The on-screen captions show the top concepts fired
    by you and by the matched meme.
    """

    def __init__(self, strategy_name: str = "siglip2_bbox_concept_qz",
                 segmenter: PersonSegmenter | None = None) -> None:
        import numpy as np

        from .testing.capture import TESTSET_DIR
        from .testing.strategies import (
            TEMPLATES,
            embed_with_rep,
            load_concepts,
            prepare_strategies,
            strategies_by_name,
        )

        self._np = np
        self._embed_with_rep = embed_with_rep
        self.segmenter = segmenter or PersonSegmenter()
        self.strategy = strategies_by_name([strategy_name])[0]
        prepare_strategies([self.strategy], self.segmenter)
        self.backend = self.strategy._backend
        self.concepts = load_concepts()
        self._T = self.backend.concept_matrix(self.concepts, TEMPLATES)

        # Calibrate the query-prior norm from captured poses (removes hub memes).
        if self.strategy.normalization in ("query_center", "query_zscore"):
            import cv2

            pose_files = sorted(TESTSET_DIR.glob("img_*.png"))
            if pose_files:
                cal = np.stack([
                    embed_with_rep(self.backend.embedder, cv2.imread(str(p)),
                                   self.strategy.representation, self.segmenter)
                    for p in pose_files
                ])
                self.strategy.calibrate(cal)
                print(f"calibrated matching from {len(pose_files)} captured poses.")
            else:
                # No calibration poses yet -> query norm degrades to raw; fall back
                # to a self-contained z-score strategy so matching still de-hubs.
                # Reuse the already-loaded backend (don't reload the model).
                print("no calibration poses (cache/testset) — using meme z-score fallback.")
                self.strategy = strategies_by_name(["siglip2_upper_concept_z"])[0]
                self.strategy.prepare(self.backend, self.segmenter, self.concepts, TEMPLATES)

        # Precompute each meme's top concepts for the on-screen caption.
        paths, meme_embs = self.backend.meme_base_embeddings("raw")
        self._meme_concepts = {
            str(config.MEMES_DIR / rel): self._top_concepts(meme_embs[i])
            for i, rel in enumerate(paths)
        }

    @property
    def num_memes(self) -> int:
        return len(self.strategy._paths)

    def _top_concepts(self, vec, k: int = 3) -> str:
        scores = vec @ self._T.T
        idx = self._np.argsort(-scores)[:k]
        return ", ".join(self.concepts[i] for i in idx)

    def match_frame(self, frame: Frame, top_k: int = 1) -> tuple[list[Match], str]:
        """Return (matches, top-concepts-you-fired) for a frame."""
        q_img = self._embed_with_rep(
            self.backend.embedder, frame, self.strategy.representation, self.segmenter
        )
        matches = self.strategy.rank_from_emb(q_img, top_k=top_k)
        return matches, self._top_concepts(q_img)

    def description_of(self, path) -> str:
        return self._meme_concepts.get(str(path), "")
