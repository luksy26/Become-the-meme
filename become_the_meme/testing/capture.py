"""Capture and label a small test set of webcam poses.

Terminal-driven so it never depends on OpenCV keyboard handling (which is flaky
on this macOS build): you press Enter in the terminal, a countdown renders in the
window, and the frame is grabbed automatically. Labeling is a numbered menu in
the terminal.

    python -m become_the_meme.testing.capture --num 12
    python -m become_the_meme.testing.capture --relabel   # re-label existing images
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from .. import config
from ..webcam import CameraError, Frame, Webcam

TESTSET_DIR = config.CACHE_DIR / "testset"
LABELS_PATH = TESTSET_DIR / "labels.json"
WINDOW = "capture test set"


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


# --- capture -----------------------------------------------------------------
def _draw_countdown(frame: Frame, n: int) -> Frame:
    disp = frame.copy()
    h, w = disp.shape[:2]
    text = str(n)
    scale, thick = 5.0, 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx, cy = w // 2, h // 2
    cv2.circle(disp, (cx, cy), int(th * 0.9), (0, 0, 0), -1)
    cv2.putText(disp, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return disp


def _countdown_and_grab(cam: Webcam, seconds: int = 3) -> Frame:
    start = time.time()
    while True:
        frame = cam.read()
        remaining = seconds - (time.time() - start)
        if remaining > 0:
            disp = _draw_countdown(frame, int(remaining) + 1)
        else:
            disp = frame
        cv2.imshow(WINDOW, disp)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt
        cv2.waitKey(1)
        if remaining <= 0:
            return frame


def _show_captured(frame: Frame) -> None:
    """Freeze-display the just-captured frame so the user can see what was taken."""
    disp = frame.copy()
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (disp.shape[1], 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)
    cv2.putText(disp, "CAPTURED  (keep it? answer in the terminal)", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # Paint a few times so macOS actually renders the static frame.
    for _ in range(5):
        cv2.imshow(WINDOW, disp)
        cv2.waitKey(30)


def capture(num: int, camera_index: int) -> list[str]:
    """Capture `num` snapshots; returns the list of saved filenames."""
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(TESTSET_DIR.glob("img_*.png"))
    start_idx = len(existing)
    saved: list[str] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        with Webcam(camera_index) as cam:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            for i in range(num):
                idx = start_idx + i
                while True:
                    input(f"\nPose {i + 1}/{num} — get ready, then press Enter "
                          "(Ctrl-C to stop)... ")
                    frame = _countdown_and_grab(cam)
                    _show_captured(frame)
                    if input("  Keep this shot? [Enter=keep / r=retake]: "
                             ).strip().lower() == "r":
                        print("  retaking...")
                        continue
                    break
                name = f"img_{idx:03d}.png"
                cv2.imwrite(str(TESTSET_DIR / name), frame)
                saved.append(name)
                print(f"  saved {name}")
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    except CameraError as exc:
        print(f"[camera error] {exc}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return saved


# --- labeling ----------------------------------------------------------------
def _show(image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for _ in range(3):
        cv2.imshow(WINDOW, img)
        cv2.waitKey(30)


def _prompt_labels(memes: list[str]) -> tuple[list[str], str]:
    while True:
        raw = input("  meme number(s), comma-separated (blank=skip): ").strip()
        if not raw:
            return [], ""
        try:
            nums = [int(x) for x in raw.replace(" ", "").split(",") if x]
            picks = [memes[n] for n in nums]
        except (ValueError, IndexError):
            print("  invalid — enter numbers from the menu.")
            continue
        note = input("  optional note (blank=none): ").strip()
        return picks, note


def label(images: list[str]) -> None:
    memes = scan_memes()
    if not memes:
        print("No memes found in memes/. Add some first.")
        return
    print("\nMemes:")
    for i, m in enumerate(memes):
        print(f"  [{i}] {m}")

    testset = TestSet.load()
    by_image = {e.image: e for e in testset.entries}
    for name in images:
        print(f"\n--- {name} ---")
        _show(TESTSET_DIR / name)
        picks, note = _prompt_labels(memes)
        item = by_image.get(name) or TestItem(image=name)
        item.labels, item.note = picks, note
        by_image[name] = item
        print(f"  labels: {picks or '(skipped)'}")

    testset.entries = [by_image[k] for k in sorted(by_image)]
    testset.save()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    labeled = sum(1 for e in testset.entries if e.labels)
    print(f"\nSaved {LABELS_PATH} — {labeled}/{len(testset.entries)} images labeled.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture + label a pose test set.")
    parser.add_argument("--num", type=int, default=12, help="number of poses to capture")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--relabel", action="store_true",
                        help="skip capture; re-label existing images")
    args = parser.parse_args()

    if args.relabel:
        images = [p.name for p in sorted(TESTSET_DIR.glob("img_*.png"))]
        if not images:
            print("No captured images to relabel.")
            return 1
        label(images)
        return 0

    saved = capture(args.num, args.camera)
    if saved:
        print("\nNow label the captures you just took:")
        label(saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
