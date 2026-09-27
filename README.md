# Become the Meme

Point your webcam at yourself, and the tool finds the meme in your `memes/`
folder that you most look/vibe like — in real time, fully local, no cloud, no
training data, no manual annotation.

## How it works (the short version)

1. **Capture** a frame from the webcam (OpenCV, cross-platform).
2. **Isolate the person** by removing the background (MediaPipe segmentation).
3. **Embed** the processed frame into a vector using a pretrained CLIP model.
4. **Match** that vector against precomputed embeddings of every image in
   `memes/` (cosine similarity, nearest neighbor).
5. **Show** you next to the winning meme.

No labels or training required — matching is zero-shot via CLIP embeddings.

**A note on what it matches:** CLIP keys on overall appearance/expression/vibe,
not precise body actions. It's great at "which meme do you look like"; matching a
specific pose (e.g. hands raised) to a specific action meme is not reliable and
was deliberately deferred (off-the-shelf pose estimation proved too flaky on a
mixed, low-res meme corpus).

## Project layout

```
become_the_meme/     # the Python package (all app code lives here)
  __init__.py
  __main__.py        # `python -m become_the_meme` — env smoke test for now
  config.py          # paths + device (MPS/CUDA/CPU) selection
memes/               # drop your meme images here (jpg/png/…)
cache/               # generated embedding index (git-ignored)
requirements.txt
```

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
source .venv/bin/activate
python -m become_the_meme            # live: you on the left, best meme on the right
```

Controls: **q** quit · **s** save the side-by-side view · **r** cycle query
representation (`bbox_crop` → `cutout` → `raw`) · **p** toggle the processed view.

The meme index builds/updates automatically on start, so just drop new images
into `memes/` and relaunch. To (re)build it manually:

```bash
python -m become_the_meme.meme_index --build      # or --rebuild to force
```

## Verify the environment

```bash
python -m become_the_meme --check
```

This prints the detected compute device and confirms every dependency imports
correctly.
