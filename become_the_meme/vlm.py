"""Local vision-language description of what a person is doing (Qwen2-VL).

Given an image, returns a short phrase describing the person's pose, gesture and
facial expression — deliberately ignoring identity/appearance so that matching
responds to *actions and expressions* rather than "who you look like".

This is the engine behind the VLM matcher backend. It's heavier and slower than
CLIP (~1-4s/image on Apple Silicon), so callers run it off the UI thread.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .webcam import Frame

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# Focus the model on action/expression and away from appearance, so two very
# different-looking people doing the same thing get similar descriptions.
DEFAULT_PROMPT = (
    "Describe ONLY the person's body pose, hand or arm gesture, and facial "
    "expression, in under 12 words. Do NOT mention clothing, background, age, "
    "gender, or looks. Name the action or gesture if there is one."
)


class VLMDescriber:
    """Wraps a local Qwen2-VL model to caption pose/gesture/expression."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._torch = torch
        self.model_id = model_id
        self.prompt = prompt
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    @staticmethod
    def _to_pil(image: Frame | Image.Image | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image.convert("RGB")

    def describe(self, image: Frame | Image.Image | str | Path, max_new_tokens: int = 40) -> str:
        """Return a short action/expression phrase for the image."""
        torch = self._torch
        img = self._to_pil(image)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
