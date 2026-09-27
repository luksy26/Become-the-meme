# Become the Meme

Point your webcam at yourself and Become the Meme shows the meme from your
`memes/` folder that best matches **what you're doing** — your pose, gesture, and
expression, not just who you look like. It runs fully locally: no cloud, no
training, no manual labels.

Drop some images in `memes/`, run it, and strike a pose — you appear on the left,
your best-match meme on the right, updating live.

## Setup

Requires Python 3.11+ (developed on Apple Silicon / macOS; the webcam layer is
cross-platform).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add your meme images (`.jpg` / `.png` / …) to the `memes/` folder.

## Run

```bash
source .venv/bin/activate
python -m become_the_meme
```

It opens fullscreen with your webcam on the left and the matched meme on the
right. The first run downloads the vision model (~1.5 GB, one-time) and caches
meme embeddings; later starts are fast. Drop new memes into `memes/` anytime and
relaunch — they're picked up automatically.

**Controls:** `q` / `Esc` quit · `f` toggle fullscreen · `d` toggle text
captions · `s` save the current view. Start windowed with `--windowed`.

---

## How it works (short version)

Each meme and each webcam frame is turned into an embedding with a local
vision-language model (SigLIP2), then projected onto a generic vocabulary of
~100 action / expression / pose concepts (`concepts.txt`). Matching compares
those concept "fingerprints", so it responds to *what you're doing* rather than
your identity. Scores are calibrated against the images you capture (below), which
stops any single "hub" meme from winning everything.

## Getting the best results: capture some of your own poses

The matcher calibrates its scores against a set of your own pose **images** in
`cache/testset/` — this is what stops one "hub" meme from always winning and makes
matching feel accurate. Capture a dozen once, entirely in the window:

```bash
python -m become_the_meme.testing.capture
```

- **Live view:** press **Space** (or click) to capture — a small 3-2-1 countdown
  runs in the corner so you can strike the pose.
- **Label view:** your shot appears next to a grid of meme thumbnails; click the
  meme(s) it matches, then **SAVE & NEXT** (or **RETAKE**). **Q** to finish.
- Aim for ~a dozen varied poses. Relaunch the app and it uses them automatically.

Re-tag or delete existing shots with `--relabel`.

**Images vs. labels:** the live app only uses the pose *images* (to calibrate and
de-hub scores) — it ignores which meme you tagged. The labels are used only by the
**testing harness** (below) to measure accuracy and tune the vocabulary. So for
the app alone you can even save poses without labeling them; add labels when you
want to measure or tune.

## Tweaking the concept vocabulary

The vocabulary in [`become_the_meme/testing/concepts.txt`](become_the_meme/testing/concepts.txt)
drives matching. To tune it:

```bash
# See which concepts each pose and its target meme fire (find mismatches):
python -m become_the_meme.testing.probe --testset

# edit concepts.txt (add / rephrase / remove lines), then measure:
python -m become_the_meme.testing.evaluate --strategies siglip2_bbox_concept_qz
```

Edits take effect automatically (the concept cache is keyed by the file's
contents). A concept only helps a match when it appears in *both* the pose's and
the meme's top concepts.

## Testing & comparing approaches

The `testing/` harness lets you measure and compare matching strategies on your
labeled poses:

```bash
python -m become_the_meme.testing.evaluate        # rank all strategies by top-1 / top-3 / MRR
python -m become_the_meme.testing.compare         # snapshot -> several strategies side by side
```

Running the full `evaluate` downloads two extra models the first time
(ViT-L-14 and the baseline CLIP) to compare against; the default backend only
needs SigLIP2.

## Other backends

```bash
python -m become_the_meme                 # concept  (default; matches actions/expressions)
python -m become_the_meme --backend clip  # CLIP     (fast, appearance/look-alike only)
python -m become_the_meme --backend vlm   # VLM      (describes you in words; slower, ~1-4s)
```

## Environment check

```bash
python -m become_the_meme --check
```

Confirms every dependency imports and prints the detected compute device
(MPS / CUDA / CPU).
