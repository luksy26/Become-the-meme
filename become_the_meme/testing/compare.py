"""Eyeball several matching strategies on one query, side by side.

    python -m become_the_meme.testing.compare                 # in-window webcam snapshot
    python -m become_the_meme.testing.compare --image cache/testset/img_003.png
    python -m become_the_meme.testing.compare --strategies b32_raw_image,l14_upper_concept_z --save

Capture is in-window (Space / click to snap with a corner countdown, Q to cancel).
Composes a canvas (query on top, one row per strategy with its top-3 memes) and
also prints the rankings to the terminal (robust fallback if the GUI misbehaves).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..app import FONT, MemeImageCache, _fit_to_height, _text, _wrap
from ..segmentation import PersonSegmenter
from ..webcam import CameraError, Webcam
from . import capture
from .strategies import (
    default_strategies,
    embed_with_rep,
    prepare_strategies,
    strategies_by_name,
)

THUMB_H = 150
CAPTION_H = 24
LABEL_W = 220
WINDOW = "strategy compare"

CAPTION_SCALE = 0.5
MIN_CELL_W = 120      # narrow/portrait thumbs still get room for their caption
MAX_CELL_W = 460      # cap runaway widening for very long names (name gets ellipsized)

# A small, legible default subset showing the progression to the winner.
DEFAULT_SUBSET = [
    "b32_raw_image",             # naive CLIP baseline (appearance / look-alike)
    "b32_bbox_concept_z",        # concept projection on the small model
    "siglip2_bbox_concept_hub",  # SigLIP2 concept, self-contained normalization
    "siglip2_bbox_concept_qz",   # the winner (query-calibrated)
]


def _text_w(s: str, scale: float = CAPTION_SCALE) -> int:
    return cv2.getTextSize(s, FONT, scale, 1)[0][0]


def _ellipsize(text: str, max_w: int, scale: float = CAPTION_SCALE) -> str:
    """Trim text from the right (adding …) until it fits within max_w pixels."""
    if _text_w(text, scale) <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_w(text[:mid] + "…", scale) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…" if lo else "…"


def _caption_strip(width: int, text: str, scale: float = CAPTION_SCALE) -> np.ndarray:
    strip = np.full((CAPTION_H, width, 3), 25, dtype=np.uint8)
    _text(strip, text, (6, 17), scale)
    return strip


def _pad_center(img: np.ndarray, width: int) -> np.ndarray:
    """Center img on a dark strip of the given width (no-op if already wide enough)."""
    if img.shape[1] >= width:
        return img
    left = (width - img.shape[1]) // 2
    right = width - img.shape[1] - left
    return np.hstack([
        np.full((img.shape[0], left, 3), 20, dtype=np.uint8),
        img,
        np.full((img.shape[0], right, 3), 20, dtype=np.uint8),
    ])


def _labeled_thumb(img: np.ndarray, name: str, score: float | None = None,
                   height: int = THUMB_H) -> np.ndarray:
    """A thumbnail with its caption below. The cell is widened so the caption fits
    (narrow/portrait thumbs no longer crop the filename or score); for very long
    names the name is ellipsized but the score is always kept."""
    thumb = _fit_to_height(img, height)
    score_txt = "" if score is None else f"  {score:.2f}"
    cell_w = int(min(max(thumb.shape[1], _text_w(f"{name}{score_txt}") + 12, MIN_CELL_W),
                     MAX_CELL_W))
    name_room = cell_w - 12 - _text_w(score_txt)  # keep the score, trim the name
    caption = f"{_ellipsize(name, name_room)}{score_txt}"
    thumb = _pad_center(thumb, cell_w)
    return np.vstack([thumb, _caption_strip(cell_w, caption)])


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
            thumbs.append(_labeled_thumb(img, m.path.name, m.score))
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
    try:
        with Webcam(args.camera) as cam:
            frame = capture.snap_in_window(cam)   # in-window: Space/click to capture, Q to cancel
    except (KeyboardInterrupt, CameraError) as exc:
        print(f"[camera error] {exc}" if isinstance(exc, CameraError) else "\ncancelled")
        return None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if frame is None:
        print("cancelled.")
    return frame


def _calibrate_query_strategies(strategies, segmenter) -> None:
    """Calibrate query_* setups from captured poses (cache/testset), like the live app.

    Without this, query-calibrated setups fall back to raw scores that don't reflect
    real performance.
    """
    query_strats = [s for s in strategies
                    if s.normalization in ("query_center", "query_zscore")]
    if not query_strats:
        return
    pose_files = sorted(capture.TESTSET_DIR.glob("img_*.png"))
    if not pose_files:
        print("  (no cache/testset poses — query-calibrated setups show raw scores)")
        return
    frames = [cv2.imread(str(p)) for p in pose_files]
    for st in query_strats:
        cal = np.stack([embed_with_rep(st._backend.embedder, f, st.representation, segmenter)
                        for f in frames])
        st.calibrate(cal)
    print(f"  calibrated {len(query_strats)} setup(s) from {len(pose_files)} poses.")


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
    _calibrate_query_strategies(strategies, segmenter)

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
