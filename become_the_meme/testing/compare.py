"""Eyeball several matching strategies on one query, side by side.

    python -m become_the_meme.testing.compare                 # webcam snapshot
    python -m become_the_meme.testing.compare --image cache/testset/img_003.png
    python -m become_the_meme.testing.compare --strategies b32_raw_image,l14_upper_concept_z --save

Composes a canvas (query on top, one row per strategy with its top-3 memes) and
also prints the rankings to the terminal (robust fallback if the GUI misbehaves).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..app import MemeImageCache, _fit_to_height, _text, _wrap
from ..segmentation import PersonSegmenter
from ..webcam import CameraError, Webcam
from . import capture
from .strategies import default_strategies, prepare_strategies, strategies_by_name

THUMB_H = 150
CAPTION_H = 24
LABEL_W = 220
WINDOW = "strategy compare"

# A small, legible default subset (one per interesting idea).
DEFAULT_SUBSET = [
    "b32_raw_image",
    "b32_bbox_image_hub",
    "b32_bbox_concept_z",
    "l14_upper_concept_z",
    "siglip2_bbox_concept_hub",
]


def _caption_strip(width: int, text: str, scale: float = 0.5) -> np.ndarray:
    strip = np.full((CAPTION_H, width, 3), 25, dtype=np.uint8)
    _text(strip, text, (6, 17), scale)
    return strip


def _labeled_thumb(img: np.ndarray, caption: str, height: int = THUMB_H) -> np.ndarray:
    thumb = _fit_to_height(img, height)
    return np.vstack([thumb, _caption_strip(thumb.shape[1], caption)])


def _label_cell(text: str, height: int) -> np.ndarray:
    cell = np.full((height, LABEL_W, 3), 45, dtype=np.uint8)
    y = 24
    for line in _wrap(text, 0.5, LABEL_W - 12):
        _text(cell, line, (8, y), 0.5)
        y += 22
    return cell


def _pad_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] >= width:
        return img
    pad = np.full((img.shape[0], width - img.shape[1], 3), 15, dtype=np.uint8)
    return np.hstack([img, pad])


def _compose(query: np.ndarray, rows: list[tuple[str, list]], cache: MemeImageCache):
    panels = [_labeled_thumb(query, "QUERY", height=220)]
    for name, matches in rows:
        thumbs = []
        for m in matches:
            img = cache.get(str(m.path))
            if img is None:
                img = np.full((THUMB_H, THUMB_H, 3), 40, dtype=np.uint8)
            thumbs.append(_labeled_thumb(img, f"{m.path.name}  {m.score:.2f}"))
        row_h = THUMB_H + CAPTION_H
        row = np.hstack([_label_cell(name, row_h), *thumbs])
        panels.append(row)
    width = max(p.shape[1] for p in panels)
    return np.vstack([_pad_width(p, width) for p in panels])


def _get_query(args) -> np.ndarray | None:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}")
        return frame
    cv2.namedWindow(capture.WINDOW, cv2.WINDOW_NORMAL)
    try:
        with Webcam(args.camera) as cam:
            cv2.setWindowProperty(capture.WINDOW, cv2.WND_PROP_TOPMOST, 1)
            input("Strike a pose, then press Enter to capture (Ctrl-C to cancel)... ")
            frame = capture._countdown_and_grab(cam)
    except (KeyboardInterrupt, CameraError) as exc:
        print(f"\ncancelled ({exc})" if isinstance(exc, CameraError) else "\ncancelled")
        return None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return frame


def _show(canvas: np.ndarray) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        cv2.imshow(WINDOW, canvas)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        while True:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare matching strategies on one query.")
    parser.add_argument("--image", help="use this image instead of a webcam snapshot")
    parser.add_argument("--strategies", help="comma-separated strategy names")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="save the canvas to cache/snapshots/")
    args = parser.parse_args()

    names = args.strategies.split(",") if args.strategies else DEFAULT_SUBSET
    strategies = strategies_by_name(names)
    need_seg = any(s.representation != "raw" or s.meme_representation != "raw"
                   for s in strategies)
    segmenter = PersonSegmenter() if need_seg else None
    prepare_strategies(strategies, segmenter)

    query = _get_query(args)
    if query is None:
        return 1

    rows = []
    print("\nRankings:")
    for strat in strategies:
        matches = strat.rank(query, top_k=3)
        rows.append((strat.name, matches))
        pretty = ", ".join(f"{m.path.name} ({m.score:.2f})" for m in matches)
        print(f"  {strat.name:26}: {pretty}")

    canvas = _compose(query, rows, MemeImageCache())
    if args.save:
        from ..webcam import save_snapshot
        print(f"saved -> {save_snapshot(canvas)}")
    _show(canvas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
