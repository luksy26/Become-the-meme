"""Score matching strategies on the labeled test set.

    python -m become_the_meme.testing.evaluate
    python -m become_the_meme.testing.evaluate --strategies b32_raw_image,l14_upper_concept_z
    python -m become_the_meme.testing.evaluate --per-item

Reports top-1 accuracy, top-3 accuracy, and MRR per strategy, sorted best first.
"""

from __future__ import annotations

import argparse
import time

import cv2

from .. import config
from ..segmentation import PersonSegmenter
from .capture import TESTSET_DIR, TestSet
from .strategies import (
    MatchStrategy,
    default_strategies,
    prepare_strategies,
    strategies_by_name,
)


def _rel(match_path) -> str:
    return str(match_path.relative_to(config.MEMES_DIR))


def evaluate_strategy(strat: MatchStrategy, items, per_item: bool = False):
    top1 = top3 = rr = 0.0
    total_ms = 0.0
    n = len(items)
    for it in items:
        frame = cv2.imread(str(TESTSET_DIR / it.image))
        t0 = time.time()
        ranking = strat.rank(frame)
        total_ms += (time.time() - t0) * 1000
        rels = [_rel(m.path) for m in ranking]
        labels = set(it.labels)
        hit1 = rels[0] in labels
        hit3 = bool(labels & set(rels[:3]))
        rank_rr = 0.0
        for i, r in enumerate(rels):
            if r in labels:
                rank_rr = 1.0 / (i + 1)
                break
        top1 += hit1
        top3 += hit3
        rr += rank_rr
        if per_item:
            mark = "OK " if hit1 else ("~  " if hit3 else "X  ")
            print(f"    {mark}{it.image}: got {rels[0]}  (want {it.labels})")
    return {
        "top1": top1 / n,
        "top3": top3 / n,
        "mrr": rr / n,
        "ms": total_ms / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CLIP matching strategies.")
    parser.add_argument("--strategies", help="comma-separated strategy names (default: all)")
    parser.add_argument("--per-item", action="store_true", help="print each image's result")
    args = parser.parse_args()

    testset = TestSet.load()
    items = [e for e in testset.entries
             if e.labels and (TESTSET_DIR / e.image).exists()]
    if not items:
        print("No labeled test items. Run: python -m become_the_meme.testing.capture")
        return 1
    print(f"Evaluating on {len(items)} labeled images.\n")

    strategies = (strategies_by_name(args.strategies.split(","))
                  if args.strategies else default_strategies())
    need_seg = any(s.representation != "raw" or s.meme_representation != "raw"
                   for s in strategies)
    segmenter = PersonSegmenter() if need_seg else None
    prepare_strategies(strategies, segmenter)

    results = []
    for strat in strategies:
        if args.per_item:
            print(f"[{strat.name}]")
        results.append((strat, evaluate_strategy(strat, items, args.per_item)))

    results.sort(key=lambda r: (r[1]["top1"], r[1]["top3"], r[1]["mrr"]), reverse=True)

    print(f"\n{'strategy':26} {'model':10} {'rep':16} {'method':8} {'norm':10} "
          f"{'top1':>5} {'top3':>5} {'MRR':>5} {'ms':>6}")
    print("-" * 108)
    for strat, m in results:
        print(f"{strat.name:26} {strat.model_key:10} {strat.representation:16} "
              f"{strat.method:8} {strat.normalization:10} "
              f"{m['top1']:5.2f} {m['top3']:5.2f} {m['mrr']:5.2f} {m['ms']:6.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
