"""CLIP image embeddings (local, via open_clip).

Turns an image into a unit-length vector so images can be compared by cosine
similarity — no labels, no training. The *same* embedder is used for both the
meme index (step 4) and the live webcam frame (step 5), so the two live in the
same vector space and are directly comparable.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import config

# Default model: strong, small, fast on MPS (512-dim, ~4 ms/image).
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

# Accepts a BGR OpenCV frame (ndarray) or a PIL image.
ImageLike = "np.ndarray | Image.Image"


class CLIPEmbedder:
    """Encodes images into L2-normalized CLIP vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or config.get_device()
        self._torch = torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        # Embedding dimensionality. `visual.output_dim` is absent on some
        # backbones (e.g. timm-based SigLIP2), so fall back to a dummy forward.
        dim = getattr(getattr(self.model, "visual", None), "output_dim", None)
        if dim is None:
            with torch.no_grad():
                dummy = self.preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(self.device)
                dim = int(self.model.encode_image(dummy).shape[-1])
        self.dim = int(dim)

    @property
    def model_id(self) -> str:
        """Stable identifier stored in the index cache to detect model changes."""
        return f"{self.model_name}/{self.pretrained}"

    @staticmethod
    def _to_pil(image: ImageLike) -> Image.Image:
        """Normalize input to a PIL RGB image (converting BGR frames as needed)."""
        if isinstance(image, np.ndarray):
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return image.convert("RGB")

    def embed_images(self, images: list[ImageLike], batch_size: int = 32) -> np.ndarray:
        """Embed a list of images -> (N, dim) float32, each row L2-normalized."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        torch = self._torch
        tensors = [self.preprocess(self._to_pil(im)) for im in images]
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i : i + batch_size]).to(self.device)
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def embed_image(self, image: ImageLike) -> np.ndarray:
        """Embed a single image -> (dim,) float32, L2-normalized."""
        return self.embed_images([image])[0]

    def embed_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed text -> (N, dim) float32, each row L2-normalized.

        Same vector space as :meth:`embed_images`, so image and text embeddings
        are directly comparable (used for concept projection). The tokenizer is
        created lazily and matches the model family (e.g. SigLIP2 has its own).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        import open_clip

        torch = self._torch
        if getattr(self, "_tokenizer", None) is None:
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self._tokenizer(texts[i : i + batch_size]).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)
