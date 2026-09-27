"""Live 'Become the Meme' app: webcam on the left, best-matching meme on the right.

    python -m become_the_meme                    # concept backend (fast, matches actions)
    python -m become_the_meme --backend vlm      # slow VLM captions
    python -m become_the_meme --backend clip     # fast CLIP (appearance only)
    python -m become_the_meme --check            # environment smoke test

Matching runs on a background thread so the webcam stays smooth. The default
'concept' backend (SigLIP2 + concept projection) refreshes a few times a second;
the 'vlm' backend is slower (~1-4s/frame).

Controls:
    q / Esc   quit           s   save the side-by-side view
    r         (CLIP) cycle representation      p   (CLIP) toggle processed view
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .webcam import CameraError, Frame, Webcam, save_snapshot

PANEL_HEIGHT = 540
HEADER_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX


class MemeImageCache:
    """Lazily loads and caches meme images (BGR) by path."""

    def __init__(self) -> None:
        self._cache: dict[str, Frame | None] = {}

    def get(self, path: str) -> Frame | None:
        if path not in self._cache:
            self._cache[path] = cv2.imread(path)
        return self._cache[path]


def _fit_to_height(img: Frame, height: int) -> Frame:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _text(img: Frame, s: str, org: tuple[int, int], scale: float = 0.6) -> None:
    # White text only. This OpenCV build renders a thick outline pass (thickness>1)
    # with artifacts, so readability comes from the dark header/caption backing
    # instead of a stroke outline.
    cv2.putText(img, s, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _wrap(text: str, scale: float, max_width: int) -> list[str]:
    """Greedy word-wrap so lines fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(trial, FONT, scale, 1)
        if tw <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_caption(panel: Frame, text: str) -> None:
    """Draw wrapped caption text over a translucent strip at the bottom of a panel."""
    if not text:
        return
    scale = 0.55
    lines = _wrap(text, scale, panel.shape[1] - 20)
    line_h = 24
    strip_h = line_h * len(lines) + 12
    y0 = panel.shape[0] - strip_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, panel, 0.35, 0, panel)
    y = y0 + 24
    for ln in lines:
        _text(panel, ln, (10, y), scale)
        y += line_h


def _compose(
    left: Frame,
    meme: Frame | None,
    meme_name: str,
    score: float,
    info: str,
    query_desc: str = "",
    meme_desc: str = "",
) -> Frame:
    left_panel = _fit_to_height(left, PANEL_HEIGHT)
    if meme is not None:
        right_panel = _fit_to_height(meme, PANEL_HEIGHT)
    else:
        right_panel = np.full((PANEL_HEIGHT, left_panel.shape[1], 3), 40, dtype=np.uint8)
        _text(right_panel, "describing...", (20, PANEL_HEIGHT // 2), 0.7)

    _draw_caption(left_panel, query_desc)
    _draw_caption(right_panel, meme_desc)

    canvas = np.hstack([left_panel, right_panel])
    header = np.full((HEADER_HEIGHT, canvas.shape[1], 3), 25, dtype=np.uint8)
    _text(header, f"YOU  |  {info}", (12, 27))
    label = f"{meme_name}  ({score:.2f})" if meme_name else "..."
    _text(header, label, (left_panel.shape[1] + 12, 27))
    return np.vstack([header, canvas])


class _MatchWorker(threading.Thread):
    """Runs matching off the UI thread; always works on the most recent frame."""

    def __init__(self, matcher, backend: str, top_k: int) -> None:
        super().__init__(daemon=True)
        self.matcher = matcher
        self.backend = backend
        self.top_k = top_k
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.result: dict | None = None

    def submit(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame

    def _take(self) -> Frame | None:
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def run(self) -> None:
        while not self._stop.is_set():
            frame = self._take()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                matches, extra = self.matcher.match_frame(frame, top_k=self.top_k)
            except Exception as exc:  # noqa: BLE001 - keep the app alive on a bad frame
                print(f"[match error] {exc}")
                continue
            if matches:
                best = matches[0]
                # `extra` is a text caption for vlm/concept backends, a processed
                # frame for clip; only strings are shown as the query caption.
                query_desc = extra if isinstance(extra, str) else ""
                describe = getattr(self.matcher, "description_of", None)
                self.result = {
                    "path": str(best.path),
                    "score": best.score,
                    "query_desc": query_desc,
                    "meme_desc": describe(best.path) if describe else "",
                }

    def stop(self) -> None:
        self._stop.set()


def _build_matcher(backend: str, representation: str):
    if backend == "clip":
        from .matcher import MemeMatcher

        return MemeMatcher(representation=representation)
    if backend == "vlm":
        from .matcher import VLMMatcher

        return VLMMatcher()
    from .matcher import ConceptMatcher

    return ConceptMatcher()


def run(
    camera_index: int = 0,
    backend: str = "concept",
    representation: str = "bbox_crop",
    top_k: int = 3,
) -> int:
    print(f"Loading {backend.upper()} backend and meme index "
          "(first run downloads weights / describes memes)...")
    matcher = _build_matcher(backend, representation)
    if matcher.num_memes == 0:
        print("No memes found. Add images to memes/ and run again.")
        return 1
    print(f"Ready — {matcher.num_memes} memes. Click the window for focus. "
          "Quit: q, Esc, Ctrl-C, or the close button.")

    worker = _MatchWorker(matcher, backend, top_k)
    worker.start()

    meme_cache = MemeImageCache()
    fps = 0.0
    prev = time.time()
    show_processed = False
    window = "Become the Meme"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    raised = False

    try:
        with Webcam(camera_index) as cam:
            while True:
                frame = cam.read()
                worker.submit(frame)

                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                res = worker.result
                meme_img = meme_cache.get(res["path"]) if res else None
                name = Path(res["path"]).name if res else ""
                score = res["score"] if res else 0.0
                q_desc = res["query_desc"] if res else ""
                m_desc = res["meme_desc"] if res else ""

                info = f"{fps:4.1f}fps  {backend}"
                if backend == "clip":
                    info += f"  rep={matcher.representation}"
                left = matcher.preprocess(frame) if (show_processed and backend == "clip") else frame

                canvas = _compose(left, meme_img, name, score, info, q_desc, m_desc)
                cv2.imshow(window, canvas)
                if not raised:
                    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
                    raised = True
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    print(f"Saved -> {save_snapshot(canvas)}")
                if key == ord("r") and backend == "clip":
                    order = ["bbox_crop", "cutout", "raw"]
                    nxt = order[(order.index(matcher.representation) + 1) % len(order)]
                    matcher.set_representation(nxt)
                    print(f"representation -> {nxt}")
                if key == ord("p") and backend == "clip":
                    show_processed = not show_processed
    except KeyboardInterrupt:
        print("\nInterrupted — quitting.")
    except CameraError as exc:
        print(f"[camera error] {exc}")
        return 1
    finally:
        worker.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Become the Meme — live matcher.")
    parser.add_argument("--backend", choices=["concept", "vlm", "clip"], default="concept",
                        help="concept = fast action/expression match (default); "
                             "vlm = slow captions; clip = fast appearance-only")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--representation", choices=["bbox_crop", "cutout", "raw"],
                        default="bbox_crop", help="(CLIP backend only)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    return run(camera_index=args.camera, backend=args.backend,
               representation=args.representation, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
