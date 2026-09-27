"""Person segmentation / background removal (MediaPipe Tasks ImageSegmenter).

Isolates the person in a frame so downstream matching focuses on *you*, not
your room. Two outputs are offered so we can A/B test which matches memes best
in a later step:

* ``cutout``    — hard background removal; person composited on a neutral color.
* ``bbox_crop`` — crop to the person's bounding box, keeping natural pixels
                  inside the box (CLIP tends to like natural images).

Model choice matters a lot. The default ``multiclass`` model segments a whole
person (hair + skin + clothes), which is what we need for full-body *actions*.
The general ``selfie`` model is faster but tuned for close-up head-and-shoulders
selfies and drops the torso/arms. Models are auto-downloaded into
``cache/models/`` on first use.

Demo:
    python -m become_the_meme.segmentation --snapshot --show      # from webcam
    python -m become_the_meme.segmentation --image photo.jpg
    python -m become_the_meme.segmentation --model selfie --threshold 0.3
    python -m become_the_meme.segmentation --selftest             # headless
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from . import config
from .webcam import Frame

MODEL_DIR: Path = config.CACHE_DIR / "models"
DEFAULT_BACKGROUND: tuple[int, int, int] = (128, 128, 128)  # neutral gray (BGR)
_BASE_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter"


@dataclass(frozen=True)
class SegModel:
    """Describes a segmentation model and how to turn its output into a mask."""

    key: str
    url: str
    filename: str
    output: str  # "confidence" or "category"
    channel: int = 0  # which confidence mask to read (confidence output)
    complement: bool = False  # confidence output: person = 1 - channel
    person_classes: tuple[int, ...] = field(default_factory=tuple)  # category output

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename


# Registry of supported models. `multiclass` is the default: it segments the
# whole person via 1 - P(background), which captures hair/torso/arms far better
# than the general selfie model.
MODELS: dict[str, SegModel] = {
    "multiclass": SegModel(
        key="multiclass",
        url=f"{_BASE_URL}/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        filename="selfie_multiclass_256x256.tflite",
        output="confidence",
        channel=0,          # class 0 is background
        complement=True,    # person = 1 - P(background)
    ),
    "selfie": SegModel(
        key="selfie",
        url=f"{_BASE_URL}/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        filename="selfie_segmenter.tflite",
        output="confidence",
        channel=0,
        complement=False,   # single foreground-probability mask
    ),
    "deeplab": SegModel(
        key="deeplab",
        url=f"{_BASE_URL}/deeplab_v3/float32/latest/deeplab_v3.tflite",
        filename="deeplab_v3.tflite",
        output="category",
        person_classes=(15,),  # Pascal VOC "person"
    ),
}
DEFAULT_MODEL = "multiclass"


def ensure_model(model: SegModel) -> Path:
    """Download ``model`` if it isn't cached yet. Returns the local path."""
    path = model.path
    if path.exists() and path.stat().st_size > 0:
        return path

    # HTTPS-only: never fetch a model over an unencrypted/untrusted transport.
    if not model.url.lower().startswith("https://"):
        raise ValueError(f"Refusing to download model over non-HTTPS URL: {model.url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{model.key}' segmentation model -> {path} ...")
    # Download to a temp file first so an interrupted download can't leave a
    # truncated model in place.
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(model.url, tmp)  # noqa: S310 - scheme checked above
    tmp.replace(path)
    print(f"Model ready ({path.stat().st_size} bytes).")
    return path


def _squeeze(mask: np.ndarray) -> np.ndarray:
    """MediaPipe returns HxWx1; collapse to a clean 2D HxW array."""
    return mask[:, :, 0] if mask.ndim == 3 else mask


class PersonSegmenter:
    """Wraps MediaPipe's ImageSegmenter to produce a person mask.

    >>> with PersonSegmenter() as seg:          # default: multiclass model
    ...     mask = seg.segment(frame)           # float32 HxW in [0, 1]
    ...     person = seg.cutout(frame)          # background removed
    ...     crop = seg.bbox_crop(frame)         # cropped to the person
    """

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = 0.5) -> None:
        # Imported here so importing this module stays cheap and a model file is
        # only required when a segmenter is actually constructed.
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choices: {list(MODELS)}")
        self.model = MODELS[model]
        self.threshold = threshold
        path = ensure_model(self.model)

        base_options = python.BaseOptions(model_asset_path=str(path))
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=(self.model.output == "confidence"),
            output_category_mask=(self.model.output == "category"),
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    # --- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None

    def __enter__(self) -> "PersonSegmenter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- core ----------------------------------------------------------------
    def segment(self, frame_bgr: Frame) -> np.ndarray:
        """Return a person-probability mask (float32 HxW, values in [0, 1])."""
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # MediaPipe needs a contiguous array it owns a clean view of.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment(mp_image)

        if self.model.output == "confidence":
            # numpy_view() aliases an internal buffer reused on the next call,
            # so copy it out before returning.
            raw = np.array(result.confidence_masks[self.model.channel].numpy_view(),
                           dtype=np.float32)
            mask = _squeeze(raw)
            return (1.0 - mask) if self.model.complement else mask

        # category output: person is the union of the model's person classes.
        cats = _squeeze(np.array(result.category_mask.numpy_view()))
        return np.isin(cats, self.model.person_classes).astype(np.float32)

    def _alpha(self, frame_bgr: Frame, mask: np.ndarray | None, feather: int) -> np.ndarray:
        """Build a soft alpha matte (float32 HxW in [0,1]) from a mask.

        We threshold first (crisp person/background decision) then feather just
        the boundary, rather than using the raw probabilities as alpha — the
        raw mask is diffuse over the background and would leak it in.
        """
        if mask is None:
            mask = self.segment(frame_bgr)
        alpha = (mask >= self.threshold).astype(np.float32)
        if feather > 0:
            k = feather * 2 + 1  # odd kernel size
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        return np.clip(alpha, 0.0, 1.0)

    def cutout(
        self,
        frame_bgr: Frame,
        mask: np.ndarray | None = None,
        background: tuple[int, int, int] = DEFAULT_BACKGROUND,
        feather: int = 3,
    ) -> Frame:
        """Composite the person over a solid ``background`` (BGR), removing the rest."""
        alpha = self._alpha(frame_bgr, mask, feather)[:, :, None]
        bg = np.full_like(frame_bgr, background, dtype=np.uint8)
        blended = frame_bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
        return blended.astype(np.uint8)

    def cutout_rgba(
        self,
        frame_bgr: Frame,
        mask: np.ndarray | None = None,
        feather: int = 3,
    ) -> Frame:
        """Return a BGRA image with a transparent background (handy for saving PNGs)."""
        alpha = self._alpha(frame_bgr, mask, feather)
        alpha_u8 = (alpha * 255).astype(np.uint8)
        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_u8
        return bgra

    def bbox_crop(
        self,
        frame_bgr: Frame,
        mask: np.ndarray | None = None,
        padding: float = 0.1,
    ) -> Frame:
        """Crop the frame to the person's bounding box (natural pixels kept).

        ``padding`` adds a margin as a fraction of the box size. Falls back to
        the full frame if no foreground is detected.
        """
        if mask is None:
            mask = self.segment(frame_bgr)
        ys, xs = np.where(mask >= self.threshold)
        if xs.size == 0 or ys.size == 0:
            return frame_bgr  # no person found; return as-is

        h, w = frame_bgr.shape[:2]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad_x = int((x1 - x0) * padding)
        pad_y = int((y1 - y0) * padding)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w, x1 + pad_x + 1)
        y1 = min(h, y1 + pad_y + 1)
        return frame_bgr[y0:y1, x0:x1]


# --- demo / entry point ------------------------------------------------------
def _load_input(args: argparse.Namespace) -> Frame:
    """Get a frame from --image, or grab one from the webcam."""
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        return frame
    from .webcam import capture_frame

    return capture_frame(camera_index=args.camera)


def run_selftest(model: str = DEFAULT_MODEL) -> int:
    """Headless: run the segmenter on a synthetic image to verify the pipeline."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    with PersonSegmenter(model=model) as seg:
        mask = seg.segment(frame)
        cut = seg.cutout(frame)
        crop = seg.bbox_crop(frame)
    print(f"model={model}  mask: shape={mask.shape} dtype={mask.dtype} "
          f"min={mask.min():.3f} max={mask.max():.3f}")
    print(f"cutout: shape={cut.shape}   bbox_crop: shape={crop.shape}")
    ok = mask.shape == frame.shape[:2] and cut.shape == frame.shape
    print("Segmentation self-test passed. ✅" if ok else "Self-test FAILED.")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Person segmentation demo.")
    parser.add_argument("--image", help="segment this image file")
    parser.add_argument("--snapshot", action="store_true",
                        help="grab a frame from the webcam and segment it")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                        help=f"segmentation model (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="foreground cutoff, 0-1 (lower = more inclusive)")
    parser.add_argument("--feather", type=int, default=3,
                        help="edge softening radius in pixels")
    parser.add_argument("--selftest", action="store_true",
                        help="run a headless pipeline check on a synthetic image")
    parser.add_argument("--show", action="store_true", help="display results in a window")
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(model=args.model)

    frame = _load_input(args)
    with PersonSegmenter(model=args.model, threshold=args.threshold) as seg:
        mask = seg.segment(frame)
        cutout = seg.cutout(frame, mask, feather=args.feather)
        crop = seg.bbox_crop(frame, mask)
        rgba = seg.cutout_rgba(frame, mask, feather=args.feather)

    out_dir = config.CACHE_DIR / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "input.png"), frame)
    cv2.imwrite(str(out_dir / "mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "cutout.png"), cutout)
    cv2.imwrite(str(out_dir / "cutout_rgba.png"), rgba)
    cv2.imwrite(str(out_dir / "bbox_crop.png"), crop)
    coverage = float((mask >= seg.threshold).mean())
    print(f"model={args.model} threshold={args.threshold} -> foreground coverage: "
          f"{coverage:.1%}. Wrote results to {out_dir}/")

    if args.show:
        cv2.imshow("input", frame)
        cv2.imshow("cutout", cutout)
        cv2.imshow("bbox_crop", crop)
        print("Press any key in a window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
