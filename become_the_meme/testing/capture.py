"""Capture and label webcam poses — entirely in the window (no terminal).

Interleaved flow: watch your webcam, press Space (or click) to capture with a
small corner countdown, then click the meme thumbnail(s) that match and Save.
Repeat, quit with Q. Labeling is by clicking thumbnails (mouse clicks are reliable
on this macOS OpenCV build).

    python -m become_the_meme.testing.capture           # capture + label
    python -m become_the_meme.testing.capture --relabel # re-tag / delete existing

Controls (in-window):
    Space / click feed   capture a pose
    click a thumbnail    toggle it as a label (multiple allowed)
    Enter                save & next        R   retake
    Q / Esc / close      finish
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..app import MemeImageCache, _fit_into_box, _screen_size, _text, _wrap
from ..webcam import CameraError, Frame, Webcam

TESTSET_DIR = config.CACHE_DIR / "testset"
LABELS_PATH = TESTSET_DIR / "labels.json"
WINDOW = "Become the Meme — capture"

BAR_H = 52          # bottom action/instruction bar
CAP_H = 22          # thumbnail caption strip
GRID_COLS = 4
COUNTDOWN_SECONDS = 3


@dataclass
class TestItem:
    image: str            # filename relative to TESTSET_DIR
    labels: list[str] = field(default_factory=list)  # meme relpaths (acceptable matches)
    note: str = ""


@dataclass
class TestSet:
    created: str = ""
    memes_dir: str = "memes"
    entries: list[TestItem] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = LABELS_PATH) -> "TestSet":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            created=data.get("created", ""),
            memes_dir=data.get("memes_dir", "memes"),
            entries=[TestItem(**e) for e in data.get("entries", [])],
        )

    def save(self, path: Path = LABELS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "memes_dir": self.memes_dir,
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))

    def image_path(self, item: TestItem) -> Path:
        return TESTSET_DIR / item.image


def scan_memes() -> list[str]:
    """Sorted meme relpaths (same ordering the index uses)."""
    if not config.MEMES_DIR.exists():
        return []
    return sorted(
        str(p.relative_to(config.MEMES_DIR))
        for p in config.MEMES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def _blit(canvas: Frame, img: Frame, x: int, y: int) -> None:
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


class CaptureUI:
    """In-window capture + labeling state machine."""

    def __init__(self, camera_index: int = 0, num: int = 0) -> None:
        self.camera_index = camera_index
        self.target = num  # stop after this many new captures (0 = until quit)

        self.memes = scan_memes()
        self.testset = TestSet.load()
        self._by_image = {e.image: e for e in self.testset.entries}
        self._cache = MemeImageCache()

        # Fixed canvas sized to the screen so WINDOW_AUTOSIZE maps clicks 1:1.
        sw, sh = _screen_size()
        self.W = min(1600, int(sw * 0.9))
        self.H = min(900, int(sh * 0.85))
        self.content_h = self.H - BAR_H

        # Precompute meme thumbnails + their grid rectangles (right pane).
        self._left_w = int(self.W * 0.45)
        self._right_x = self._left_w
        right_w = self.W - self._left_w
        rows = max(1, (len(self.memes) + GRID_COLS - 1) // GRID_COLS)
        self._cell_w = right_w // GRID_COLS
        self._cell_h = self.content_h // rows
        self._meme_rects: list[tuple[int, int, int, int, str]] = []
        self._thumbs: dict[str, Frame] = {}
        tw, th = self._cell_w - 10, self._cell_h - 10 - CAP_H
        for i, rel in enumerate(self.memes):
            c, r = i % GRID_COLS, i // GRID_COLS
            x0 = self._right_x + c * self._cell_w
            y0 = r * self._cell_h
            self._meme_rects.append((x0, y0, x0 + self._cell_w, y0 + self._cell_h, rel))
            img = self._cache.get(str(config.MEMES_DIR / rel))
            self._thumbs[rel] = (_fit_into_box(img, max(1, tw), max(1, th))
                                 if img is not None
                                 else np.full((max(1, th), max(1, tw), 3), 40, np.uint8))

        self._button_rects: list[tuple[int, int, int, int, str]] = []
        self._click: tuple[int, int] | None = None
        self._hover: tuple[int, int] = (-1, -1)
        self._saved = 0

    # --- mouse -----------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._hover = (x, y)

    @staticmethod
    def _in(rect, pt) -> bool:
        x0, y0, x1, y1 = rect[:4]
        return x0 <= pt[0] < x1 and y0 <= pt[1] < y1

    # --- rendering -------------------------------------------------------------
    def _bar(self, canvas: Frame) -> None:
        cv2.rectangle(canvas, (0, self.content_h), (self.W, self.H), (25, 25, 25), -1)

    def _draw_buttons(self, canvas: Frame, buttons: list[tuple[str, str]]) -> None:
        """buttons: list of (action, label), drawn right-to-left in the bar."""
        self._button_rects = []
        bw, gap = 190, 12
        x1 = self.W - gap
        for action, label in buttons:
            x0 = x1 - bw
            y0, y1 = self.content_h + 8, self.H - 8
            hovered = self._in((x0, y0, x1, y1), self._hover)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (95, 95, 95) if hovered else (60, 60, 60), -1)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            _text(canvas, label, (x0 + (bw - tw) // 2, y1 - 14), 0.6)
            self._button_rects.append((x0, y0, x1, y1, action))
            x1 = x0 - gap

    def _render_live(self, frame: Frame, remaining: float | None) -> Frame:
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        _blit(canvas, _fit_into_box(frame, self.W, self.content_h), 0, 0)
        self._bar(canvas)
        hint = f"Space / click = capture     Q = done     saved: {self._saved}"
        _text(canvas, hint, (14, self.H - 18), 0.6)
        if remaining is not None:  # small countdown chip, top-right (off the face)
            n = int(remaining) + 1
            cx, cy = self.W - 60, 60
            cv2.circle(canvas, (cx, cy), 40, (0, 0, 0), -1)
            (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            cv2.putText(canvas, str(n), (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        return canvas

    def _render_label(self, image: Frame, selected: set[str],
                      buttons: list[tuple[str, str]], hint: str) -> Frame:
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        _blit(canvas, _fit_into_box(image, self._left_w, self.content_h), 0, 0)
        for x0, y0, x1, y1, rel in self._meme_rects:
            thumb = self._thumbs[rel]
            _blit(canvas, thumb, x0 + 5, y0 + 5)
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2 - CAP_H),
                          (0, 200, 0) if rel in selected
                          else ((150, 150, 150) if self._in((x0, y0, x1, y1), self._hover)
                                else (70, 70, 70)),
                          3 if rel in selected else 1)
            name = Path(rel).stem[:18]
            _text(canvas, name, (x0 + 6, y1 - 6), 0.45)
        self._bar(canvas)
        _text(canvas, hint, (14, self.H - 18), 0.6)
        self._draw_buttons(canvas, buttons)
        return canvas

    # --- flows -----------------------------------------------------------------
    def _open_window(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _closed(self) -> bool:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1

    def _next_name(self) -> str:
        existing = sorted(TESTSET_DIR.glob("img_*.png"))
        return f"img_{len(existing):03d}.png"

    def _commit(self, name: str, labels: set[str]) -> None:
        item = self._by_image.get(name) or TestItem(image=name)
        item.labels = sorted(labels)
        self._by_image[name] = item
        self.testset.entries = [self._by_image[k] for k in sorted(self._by_image)]
        self.testset.save()

    def run_capture(self) -> None:
        if not self.memes:
            print("No memes found in memes/. Add some first.")
            return
        TESTSET_DIR.mkdir(parents=True, exist_ok=True)
        self._open_window()
        try:
            with Webcam(self.camera_index) as cam:
                mode = "live"
                counting_since: float | None = None
                captured: Frame | None = None
                selected: set[str] = set()
                while True:
                    if mode == "live":
                        frame = cam.read()
                        remaining = None
                        if counting_since is not None:
                            remaining = COUNTDOWN_SECONDS - (time.time() - counting_since)
                            if remaining <= 0:
                                captured, selected, mode = frame.copy(), set(), "label"
                                counting_since = None
                        canvas = self._render_live(frame, remaining)
                    else:
                        canvas = self._render_label(
                            captured, selected,
                            [("save", "SAVE & NEXT"), ("retake", "RETAKE")],
                            "Click the meme(s) that match, then SAVE.")
                    cv2.imshow(WINDOW, canvas)
                    if self._closed():
                        break
                    key = cv2.waitKey(1) & 0xFF
                    click = self._click
                    self._click = None

                    if key in (ord("q"), 27):
                        break
                    if mode == "live":
                        if counting_since is None and (key == ord(" ") or click is not None):
                            counting_since = time.time()
                    else:  # label
                        act = self._label_action(key, click, selected)
                        if act == "save":
                            name = self._next_name()
                            cv2.imwrite(str(TESTSET_DIR / name), captured)
                            self._commit(name, selected)
                            self._saved += 1
                            mode = "live"
                            if self.target and self._saved >= self.target:
                                break
                        elif act == "retake":
                            mode = "live"
        except CameraError as exc:
            print(f"[camera error] {exc}")
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print(f"Saved {self._saved} pose(s) -> {LABELS_PATH}")

    def run_relabel(self) -> None:
        images = [p.name for p in sorted(TESTSET_DIR.glob("img_*.png"))]
        if not images:
            print("No captured images to relabel.")
            return
        if not self.memes:
            print("No memes found in memes/. Add some first.")
            return
        self._open_window()
        i = 0
        try:
            while i < len(images):
                name = images[i]
                image = cv2.imread(str(TESTSET_DIR / name))
                if image is None:
                    i += 1
                    continue
                selected = set(self._by_image.get(name, TestItem(image=name)).labels)
                while True:
                    canvas = self._render_label(
                        image, selected,
                        [("save", "SAVE & NEXT"), ("delete", "DELETE")],
                        f"{name}  ({i + 1}/{len(images)})  — click memes, then SAVE.")
                    cv2.imshow(WINDOW, canvas)
                    if self._closed():
                        return
                    key = cv2.waitKey(1) & 0xFF
                    click = self._click
                    self._click = None
                    if key in (ord("q"), 27):
                        return
                    act = self._label_action(key, click, selected)
                    if act == "save":
                        self._commit(name, selected)
                        i += 1
                        break
                    if act == "delete":
                        (TESTSET_DIR / name).unlink(missing_ok=True)
                        self._by_image.pop(name, None)
                        self.testset.entries = [self._by_image[k] for k in sorted(self._by_image)]
                        self.testset.save()
                        i += 1
                        break
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print(f"Relabeling done -> {LABELS_PATH}")

    def _label_action(self, key: int, click, selected: set[str]) -> str | None:
        """Handle a key/click in the LABEL view; returns 'save'/'retake'/'delete'/None."""
        if key in (13, 10):  # Enter
            return "save"
        if key == ord("r"):
            return "retake"
        if click is None:
            return None
        for rect in self._button_rects:
            if self._in(rect, click):
                return rect[4]
        for rect in self._meme_rects:
            if self._in(rect, click):
                rel = rect[4]
                selected.discard(rel) if rel in selected else selected.add(rel)
                return None
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture + label a pose test set (in-window).")
    parser.add_argument("--num", type=int, default=0,
                        help="stop after N new captures (default: until you quit)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--relabel", action="store_true",
                        help="skip capture; re-tag / delete existing images")
    args = parser.parse_args()

    ui = CaptureUI(camera_index=args.camera, num=args.num)
    if args.relabel:
        ui.run_relabel()
    else:
        ui.run_capture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
