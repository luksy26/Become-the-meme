# Become the Meme

Point your webcam at yourself and Become the Meme shows the meme from your
`memes/` folder that best matches **what you're doing** — your pose, gesture, and
expression, not just who you look like. It runs fully locally: no cloud, no
training, no manual labels.

Drop some images in `memes/`, run it, and strike a pose — you appear on the left,
your best-match meme on the right, updating live.

## Setup

- Python 3.11+ (download for your platform at [python.org](https://www.python.org/)).
  Developed on Apple Silicon / macOS; the webcam layer is cross-platform.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add your meme images (`.jpg` / `.png` / …) to the `memes/` folder, then verify the
install (confirms dependencies import and prints the compute device — MPS / CUDA / CPU):

```bash
python -m become_the_meme --check
```

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
**evaluation scripts** (below) to measure accuracy and tune the vocabulary. So for
the app alone you can even save poses without labeling them; add labels when you
want to measure or tune.

## Cleanup

```bash
python -m become_the_meme.cleanup [--weights] [--all] [--dry-run]
```

Removes generated caches (which rebuild automatically). By default it clears only
the regenerable project caches under `cache/`, keeping your captured poses and
downloaded models. Flags:

- **`--weights`** — also prune downloaded model weights no backend uses (the
  multi-GB weights live in the Hugging Face / Torch caches outside the project;
  in-use ones are kept).
- **`--all`** — full reset: all project caches (including captured poses and
  MediaPipe models) **and** every downloaded weight (all re-download on next use).
- **`--dry-run`** — preview what would be removed without deleting anything.

---

> **Everything below is optional.** The sections above are all you need to use the
> app. What follows is for tinkering with match quality, benchmarking setups, trying
> alternate backends, and understanding the internals — none of it is required for
> normal use.

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

## Comparing matching setups

The scripts in `testing/` let you measure and compare different matching setups
(model + settings) on your labeled poses — this is how the default was chosen:

```bash
python -m become_the_meme.testing.evaluate        # score each setup by top-1 / top-3 / MRR
python -m become_the_meme.testing.compare         # snapshot -> several setups side by side
```

`evaluate` scores each setup on your labeled poses (0→1, higher is better):
- **top-1** — how often the single best match is correct.
- **top-3** — how often a correct match lands in the top 3.
- **MRR** (mean reciprocal rank) — `1 ÷ rank of the first correct match`, averaged
  (1.0 = always #1, 0.50 ≈ usually #2); rewards ranking the right meme higher even
  when it isn't #1.

Running the full `evaluate` downloads two extra models the first time
(ViT-L-14 and ViT-B-32) to compare against; the app's default only needs SigLIP2.

### Choosing which setups to compare

Both tools take `--strategies` to pick specific setups by name:

```bash
python -m become_the_meme.testing.compare  --strategies b32_raw_image,siglip2_bbox_concept_qz
python -m become_the_meme.testing.evaluate --strategies siglip2_bbox_concept_qz
```

`evaluate` with no `--strategies` lists every available setup. Each name reads as
`model_representation_method_norm`:

- **model** — which vision model:
  - `b32` — CLIP ViT-B-32
  - `l14` — CLIP ViT-L-14 (DFN)
  - `siglip2` — SigLIP2
- **representation** — which part of you gets embedded:
  - `raw` — the whole frame
  - `bbox` — cropped to you (background trimmed)
  - `upper` — head + torso
  - `cutout` — background removed onto grey
  - `multi` — average of several crops
- **method** — how images are compared:
  - `image` — raw image-to-image similarity → matches *appearance*
  - `concept` — projected onto action/expression concepts → matches *what you're doing*
- **norm** — score normalization to stop one "hub" meme winning everything:
  - `hub` — hubness
  - `z` — z-score
  - `mc` — mean-center
  - `qz` — query-calibrated z-score (calibrated against your poses — **the winner**)
  - *(absent)* — raw scores

So `siglip2_bbox_concept_qz` = SigLIP2 · cropped-to-you · concept-projection ·
query-calibrated — the setup the app uses by default.

## Other backends

```bash
python -m become_the_meme                       # concept     (default; matches actions/expressions)
python -m become_the_meme --backend appearance  # appearance  (fast look-alike; raw image similarity)
python -m become_the_meme --backend vlm         # vlm         (describes you in words; slower, ~1-4s)
```

## Technologies

Everything runs locally in [Python](https://www.python.org/) 3.11+ on **PyTorch**
(Apple Silicon MPS / CUDA / CPU).

**Models**
- **SigLIP2** (`ViT-B-16-SigLIP2`, via open_clip) — the default concept-projection matcher.
- **MediaPipe ImageSegmenter** (selfie multiclass) — isolates the person from the background.
- **CLIP ViT-B-32** (via open_clip) — the appearance matcher (`--backend appearance`), and the reference the comparison scripts measure against.
- **CLIP ViT-L-14** — a larger CLIP tried when comparing setups; it didn't beat SigLIP2 on this task, so the app never uses it.
- **Qwen2-VL** (via Transformers) — the optional caption (`vlm`) backend.

**Libraries**
- **open_clip** / **Transformers** / **sentence-transformers** — image & text embeddings.
- **OpenCV** — webcam capture and the fullscreen / capture windows.
- **MediaPipe** — segmentation.
- **NumPy** & **Pillow** — array and image handling.
