"""Live 'Become the Meme' app: webcam on the left, best-matching meme on the right.

    python -m become_the_meme                    # concept backend (fast, matches actions)
    python -m become_the_meme --backend vlm      # slow VLM captions
    python -m become_the_meme --backend clip     # fast CLIP (appearance only)
    python -m become_the_meme --check            # environment smoke test

Matching runs on a background thread so the webcam stays smooth. The default
'concept' backend (SigLIP2 + concept projection) refreshes a few times a second;
the 'vlm' backend is slower (~1-4s/frame).

Controls:
    q / Esc   quit                      s   save the side-by-side view
    f         toggle fullscreen         d   toggle text captions (off by default)
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

PANEL_HEIGHT = 720
PANEL_WIDTH = 1280          # fixed panel size so the window never resizes
HEADER_HEIGHT = 44
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Display stability so the matched meme doesn't strobe (but stays responsive).
SUBMIT_INTERVAL = 0.25      # seconds between frames handed to the matcher
SWITCH_STREAK = 2           # consecutive agreeing matches needed to switch memes
MIN_HOLD = 0.5             # min seconds a meme stays on screen before it can change


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


def _screen_size() -> tuple[int, int]:
    """Best-effort screen resolution (Tkinter), with a safe fallback."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w >= 640 and h >= 480:
            return int(w), int(h)
    except Exception:  # noqa: BLE001 - Tk may be unavailable; fall back
        pass
    return 1728, 1117


def _fit_into_box(img: Frame, box_w: int, box_h: int) -> Frame:
    """Resize preserving aspect and letterbox into a fixed box (constant output size)."""
    box_w, box_h = max(1, box_w), max(1, box_h)
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    box = np.full((box_h, box_w, 3), 20, dtype=np.uint8)
    y0, x0 = (box_h - nh) // 2, (box_w - nw) // 2
    box[y0:y0 + nh, x0:x0 + nw] = resized
    return box


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
    panel_w: int = PANEL_WIDTH,
    panel_h: int = PANEL_HEIGHT,
    header_h: int = HEADER_HEIGHT,
    header_scale: float = 0.6,
) -> Frame:
    left_panel = _fit_into_box(left, panel_w, panel_h)
    if meme is not None:
        right_panel = _fit_into_box(meme, panel_w, panel_h)
    else:
        right_panel = np.full((panel_h, panel_w, 3), 40, dtype=np.uint8)
        _text(right_panel, "matching...", (20, panel_h // 2), 0.7)

    _draw_caption(left_panel, query_desc)
    _draw_caption(right_panel, meme_desc)

    canvas = np.hstack([left_panel, right_panel])
    header = np.full((header_h, canvas.shape[1], 3), 25, dtype=np.uint8)
    hy = int(header_h * 0.62)
    _text(header, f"YOU  |  {info}", (12, hy), header_scale)
    label = f"{meme_name}  ({score:.2f})" if meme_name else "..."
    _text(header, label, (panel_w + 12, hy), header_scale)
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
        self.version = 0  # bumped on each new result so the UI can detect changes

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
                self.version += 1

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
    show_captions: bool = False,
    fullscreen: bool = True,
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
    last_submit = 0.0
    last_version = -1
    shown: dict | None = None     # the match currently displayed (stabilized)
    shown_since = 0.0
    cand: str | None = None       # challenger meme + how many results in a row
    streak = 0
    show_processed = False
    window = "Become the Meme"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_size = _screen_size() if fullscreen else None
    raised = False

    def panel_dims() -> tuple[int, int, int, float]:
        """(panel_w, panel_h, header_h, header_scale) for the current mode."""
        if fullscreen and screen_size:
            sw, sh = screen_size
            hh = max(HEADER_HEIGHT, min(sh // 16, sh - 120))
            ph, pw = max(1, sh - hh), max(1, sw // 2)
            return pw, ph, hh, min(1.3, max(0.6, hh / 60))
        return PANEL_WIDTH, PANEL_HEIGHT, HEADER_HEIGHT, 0.6

    try:
        with Webcam(camera_index) as cam:
            while True:
                frame = cam.read()
                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                # Throttle how often the (heavier) matcher runs.
                if now - last_submit >= SUBMIT_INTERVAL:
                    worker.submit(frame)
                    last_submit = now

                # Fold in a new result with hysteresis so the shown meme doesn't
                # strobe: a challenger must win SWITCH_STREAK times in a row and the
                # current pick must have been up at least MIN_HOLD seconds.
                res = worker.result
                if res is not None and worker.version != last_version:
                    last_version = worker.version
                    top = res["path"]
                    if shown is None:
                        shown, shown_since = res, now
                    elif top == shown["path"]:
                        shown, cand, streak = res, None, 0  # refresh score/desc
                    else:
                        cand, streak = (top, streak + 1) if top == cand else (top, 1)
                        if streak >= SWITCH_STREAK and (now - shown_since) >= MIN_HOLD:
                            shown, shown_since, cand, streak = res, now, None, 0

                meme_img = meme_cache.get(shown["path"]) if shown else None
                name = Path(shown["path"]).name if shown else ""
                score = shown["score"] if shown else 0.0
                q_desc = shown["query_desc"] if (shown and show_captions) else ""
                m_desc = shown["meme_desc"] if (shown and show_captions) else ""

                info = f"{fps:4.1f}fps  {backend}"
                if backend == "clip":
                    info += f"  rep={matcher.representation}"
                left = matcher.preprocess(frame) if (show_processed and backend == "clip") else frame

                pw, ph, hh, hs = panel_dims()
                canvas = _compose(left, meme_img, name, score, info, q_desc, m_desc,
                                  pw, ph, hh, hs)
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
                if key == ord("d"):
                    show_captions = not show_captions
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        window, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    screen_size = _screen_size() if fullscreen else None
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
    parser.add_argument("--captions", action="store_true",
                        help="show the text concept captions (off by default; toggle with 'd')")
    parser.add_argument("--windowed", action="store_true",
                        help="start windowed instead of fullscreen (toggle with 'f')")
    args = parser.parse_args(argv)
    return run(camera_index=args.camera, backend=args.backend,
               representation=args.representation, top_k=args.top_k,
               show_captions=args.captions, fullscreen=not args.windowed)


if __name__ == "__main__":
    raise SystemExit(main())
