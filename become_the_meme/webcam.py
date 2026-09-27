"""Platform-agnostic webcam capture.

The :class:`Webcam` class is the single interface the rest of the app uses to
get "the current frame". It wraps OpenCV's ``VideoCapture`` with sensible
defaults, a warm-up, transient-read retries, and optional mirroring so the
preview feels like looking in a mirror.

Run the interactive demo:

    python -m become_the_meme.webcam            # live preview window
    python -m become_the_meme.webcam --selftest # headless: grab a few frames

Controls in the preview window:
    space / c   save a snapshot to cache/snapshots/
    q / Esc     quit
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from . import config

# A frame is an OpenCV BGR image: HxWx3 uint8 ndarray.
Frame = np.ndarray

SNAPSHOTS_DIR: Path = config.CACHE_DIR / "snapshots"


class CameraError(RuntimeError):
    """Raised when the camera can't be opened or a frame can't be read."""


class Webcam:
    """A webcam as a context manager yielding BGR frames.

    Example
    -------
    >>> with Webcam() as cam:
    ...     frame = cam.read()   # HxWx3 BGR uint8 ndarray
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int | None = 1280,
        height: int | None = 720,
        mirror: bool = True,
        backend: int = cv2.CAP_ANY,
        warmup_frames: int = 5,
        read_retries: int = 3,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.backend = backend
        self.warmup_frames = warmup_frames
        self.read_retries = read_retries
        self._cap: cv2.VideoCapture | None = None

    # --- lifecycle -----------------------------------------------------------
    def open(self) -> "Webcam":
        """Open the camera and warm it up. Returns self for chaining."""
        # Letting OpenCV pick the backend (CAP_ANY) keeps this portable across
        # macOS (AVFoundation), Windows (MSMF/DSHOW) and Linux (V4L2).
        cap = cv2.VideoCapture(self.camera_index, self.backend)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.camera_index}. "
                "Is another app using it, or is camera permission denied?"
            )

        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._cap = cap

        # Some cameras (and the macOS permission handshake) return empty frames
        # for the first fraction of a second; discard a few to warm up.
        for _ in range(self.warmup_frames):
            cap.read()

        return self

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Webcam":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # --- capture -------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Frame:
        """Grab the current frame as a BGR ndarray.

        Retries a few times to ride out transient grab failures, then raises
        :class:`CameraError` if it still can't get a frame.
        """
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() or use 'with Webcam()'.")

        for _ in range(self.read_retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                return frame

        raise CameraError("Failed to read a frame from the camera.")

    @property
    def actual_resolution(self) -> tuple[int, int]:
        """(width, height) the driver actually gave us (may differ from request)."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)


def capture_frame(camera_index: int = 0, mirror: bool = True) -> Frame:
    """Convenience: open the camera, grab one frame, release. Returns BGR ndarray."""
    with Webcam(camera_index=camera_index, mirror=mirror) as cam:
        return cam.read()


def save_snapshot(frame: Frame, directory: Path = SNAPSHOTS_DIR) -> Path:
    """Save a frame to ``directory`` with a timestamped filename. Returns the path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"snapshot_{stamp}.png"
    cv2.imwrite(str(path), frame)
    return path


# --- demos / entry point -----------------------------------------------------
def _draw_overlay(frame: Frame, fps: float) -> None:
    """Draw FPS and control hints onto the frame in place."""
    lines = [f"{fps:4.1f} FPS", "space/c: snapshot   q/Esc: quit"]
    # Dark backing strip for readability (a thick text outline renders with
    # artifacts on this OpenCV build, so we use a backing rectangle instead).
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 12 + 30 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    y = 28
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 30


def run_preview(camera_index: int = 0, mirror: bool = True,
                width: int | None = 1280, height: int | None = 720) -> int:
    """Open a live preview window until the user quits. Returns an exit code."""
    window = "Become the Meme — webcam"
    last = time.time()
    fps = 0.0

    try:
        with Webcam(camera_index, width, height, mirror) as cam:
            print(f"Camera opened at {cam.actual_resolution[0]}x{cam.actual_resolution[1]}. "
                  "space/c = snapshot, q/Esc = quit.")
            while True:
                frame = cam.read()

                # Smooth FPS with an exponential moving average.
                now = time.time()
                dt = now - last
                last = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                display = frame.copy()
                _draw_overlay(display, fps)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if key in (ord(" "), ord("c")):
                    path = save_snapshot(frame)  # save the clean frame
                    print(f"Saved snapshot -> {path}")
    except CameraError as exc:
        print(f"[camera error] {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


def run_selftest(camera_index: int = 0, num_frames: int = 10) -> int:
    """Headless check: grab a few frames and report, no GUI window."""
    try:
        with Webcam(camera_index=camera_index) as cam:
            w, h = cam.actual_resolution
            print(f"Camera {camera_index} opened at {w}x{h}.")
            shapes = set()
            for i in range(num_frames):
                frame = cam.read()
                shapes.add(frame.shape)
            print(f"Grabbed {num_frames} frames. dtype={frame.dtype}, shapes={shapes}")
            print("Webcam self-test passed. ✅")
    except CameraError as exc:
        print(f"[camera error] {exc}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam capture demo.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true", help="disable mirror flip")
    parser.add_argument("--selftest", action="store_true",
                        help="grab a few frames headless instead of showing a window")
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(camera_index=args.camera)
    return run_preview(camera_index=args.camera, mirror=not args.no_mirror,
                       width=args.width, height=args.height)


if __name__ == "__main__":
    raise SystemExit(main())
