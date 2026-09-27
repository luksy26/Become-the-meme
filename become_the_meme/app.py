"""Live 'Become the Meme' app: webcam on the left, best-matching meme on the right.

    python -m become_the_meme          # run it
    python -m become_the_meme --check  # environment smoke test instead

Controls:
    q / Esc   quit
    s         save the current side-by-side view to cache/snapshots/
    r         cycle query representation (bbox_crop -> cutout -> raw)
    p         toggle showing the processed query image on the left
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from . import config
from .matcher import MemeMatcher
from .webcam import CameraError, Frame, Webcam, save_snapshot

PANEL_HEIGHT = 540           # height of each side panel
HEADER_HEIGHT = 40           # info bar height
MATCH_INTERVAL = 0.12        # seconds between matches (display stays smooth)
SMOOTHING_WINDOW = 6         # frames of history for flicker suppression


class MemeImageCache:
    """Lazily loads and caches meme images (BGR) by path."""

    def __init__(self) -> None:
        self._cache: dict[str, Frame | None] = {}

    def get(self, path: Path) -> Frame | None:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = cv2.imread(key)  # None if unreadable
        return self._cache[key]


def _fit_to_height(img: Frame, height: int) -> Frame:
    """Resize keeping aspect ratio so the result is exactly `height` tall."""
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _text(img: Frame, s: str, org: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _placeholder(width: int, height: int, message: str) -> Frame:
    panel = np.full((height, width, 3), 40, dtype=np.uint8)
    _text(panel, message, (20, height // 2), scale=0.7)
    return panel


def _compose(
    left: Frame,
    meme: Frame | None,
    meme_name: str,
    score: float,
    info: str,
) -> Frame:
    """Build the side-by-side canvas with a header bar."""
    left_panel = _fit_to_height(left, PANEL_HEIGHT)
    if meme is not None:
        right_panel = _fit_to_height(meme, PANEL_HEIGHT)
    else:
        right_panel = _placeholder(left_panel.shape[1], PANEL_HEIGHT, "no match")

    canvas = np.hstack([left_panel, right_panel])
    header = np.full((HEADER_HEIGHT, canvas.shape[1], 3), 25, dtype=np.uint8)
    _text(header, f"YOU  |  {info}", (12, 27))
    label = f"{meme_name}  ({score:.2f})" if meme_name else "..."
    _text(header, label, (left_panel.shape[1] + 12, 27))
    return np.vstack([header, canvas])


def run(
    camera_index: int = 0,
    representation: str = "bbox_crop",
    top_k: int = 3,
) -> int:
    print("Loading models and meme index (first run downloads weights)...")
    matcher = MemeMatcher(representation=representation)
    if matcher.num_memes == 0:
        print("No memes found. Add images to memes/ and run again "
              "(or: python -m become_the_meme.meme_index --build).")
        return 1
    print(f"Ready — {matcher.num_memes} memes indexed. Controls: q quit, "
          f"s snapshot, r representation, p processed-view.")

    meme_cache = MemeImageCache()
    recent = deque(maxlen=SMOOTHING_WINDOW)  # recent top-1 paths for smoothing
    scores: dict[str, float] = {}            # latest score per meme path
    last_match = 0.0
    show_processed = False
    fps = 0.0
    prev = time.time()
    window = "Become the Meme"

    try:
        with Webcam(camera_index) as cam:
            while True:
                frame = cam.read()

                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                processed = frame
                if now - last_match >= MATCH_INTERVAL:
                    last_match = now
                    matches, processed = matcher.match_frame(frame, top_k=top_k)
                    if matches:
                        recent.append(str(matches[0].path))
                        for m in matches:
                            scores[str(m.path)] = m.score

                # Smoothed choice: most common top-1 over the recent window.
                best_path, best_score, best_name = None, 0.0, ""
                if recent:
                    best_path = Counter(recent).most_common(1)[0][0]
                    best_score = scores.get(best_path, 0.0)
                    best_name = Path(best_path).name

                left = processed if show_processed else frame
                meme_img = meme_cache.get(Path(best_path)) if best_path else None
                info = f"{fps:4.1f}fps  rep={matcher.representation}"
                canvas = _compose(left, meme_img, best_name, best_score, info)
                cv2.imshow(window, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = save_snapshot(canvas)
                    print(f"Saved -> {path}")
                if key == ord("r"):
                    order = ["bbox_crop", "cutout", "raw"]
                    nxt = order[(order.index(matcher.representation) + 1) % len(order)]
                    matcher.set_representation(nxt)
                    recent.clear()
                    print(f"representation -> {nxt}")
                if key == ord("p"):
                    show_processed = not show_processed
    except CameraError as exc:
        print(f"[camera error] {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Become the Meme — live matcher.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--representation", choices=["bbox_crop", "cutout", "raw"],
                        default="bbox_crop")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    return run(camera_index=args.camera, representation=args.representation,
               top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
