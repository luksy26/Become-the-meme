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
