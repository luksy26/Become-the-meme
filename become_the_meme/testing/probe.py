"""Inspect which concepts each meme / pose activates — the tuning aid for concepts.txt.

    python -m become_the_meme.testing.probe                     # top concepts per meme
    python -m become_the_meme.testing.probe --testset           # each pose vs its target meme
    python -m become_the_meme.testing.probe --image path.png    # one image

Use it to spot vocabulary problems:
- a meme's top concepts don't describe it   -> add/rephrase a concept that does
- two memes you want to tell apart share the same top concepts -> add a separating one
- a pose and its target meme activate different concepts -> add a bridging concept

After editing concepts.txt, just re-run evaluate — the concept cache auto-invalidates
(it is keyed by a hash of the file), so changes take effect immediately.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .. import config
from ..segmentation import PersonSegmenter
from .capture import TESTSET_DIR, TestSet
from .strategies import (
    TEMPLATES,
    ModelBackend,
    embed_with_rep,
    load_concepts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe concept activations.")
    parser.add_argument("--model", default="siglip2_b", help="model key (default: winner)")
    parser.add_argument("--rep", default="upper_body_crop", help="query representation")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--testset", action="store_true",
                        help="show each labeled pose vs its target meme")
    parser.add_argument("--image", help="probe a single image file")
    args = parser.parse_args()

    concepts = load_concepts()
    backend = ModelBackend(args.model)
    T = backend.concept_matrix(concepts, TEMPLATES)      # (C, D)
    seg = PersonSegmenter()

    def top(vec: np.ndarray) -> str:
        s = vec @ T.T
        idx = np.argsort(-s)[: args.topk]
        return ", ".join(f"{concepts[i]} ({s[i]:.2f})" for i in idx)

    if args.image:
        q = embed_with_rep(backend.embedder, cv2.imread(args.image), args.rep, seg)
        print(f"{args.image}:\n  {top(q)}")
        return 0

    paths, M = backend.meme_base_embeddings("raw")        # memes embedded raw (as in matching)
    meme_top = {rel: top(M[i]) for i, rel in enumerate(paths)}

    if args.testset:
        ts = TestSet.load()
        for e in ts.entries:
            if not e.labels:
                continue
            frame = cv2.imread(str(TESTSET_DIR / e.image))
            q = embed_with_rep(backend.embedder, frame, args.rep, seg)
            print(f"\n{e.image}  (pose):\n  {top(q)}")
            for lab in e.labels:
                print(f"  target {lab}:\n    {meme_top.get(lab, '(unknown meme)')}")
        return 0

    print(f"Top {args.topk} concepts per meme ({args.model}, memes embedded raw):")
    for rel in paths:
        print(f"\n{rel}:\n  {meme_top[rel]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
