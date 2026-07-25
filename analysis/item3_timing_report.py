"""Item 3 - Direct extrapolation-vs-DUAL time comparison.

Rebuttal target: reviewer bWqr W3/Q2 ("4.9x measured vs the most expensive
baseline; no direct extrapolation-vs-DUAL time comparison"). This assembles a
single wall-clock comparison table from measured component timings (App
``sec:temporal``): the FULL-pipeline cost of computing a dynamic score (e.g.
DUAL) on the whole set, versus the extrapolation pipeline cost = subset scoring
+ embedding extraction + KNN/GNN extrapolation.

It does not change the training pipeline; it consumes a small timing JSON of the
form:
    {
      "full": {"dual_full_scoring_s": 51230.0},
      "extrapolation": {
          "subset_scoring_s": 5120.0,
          "embedding_extraction_s": 640.0,
          "knn_extrapolation_s": 95.0
      }
    }
and reports total, breakdown, and speed-up. Multiple named methods can be given.

Self-test:
    python analysis/item3_timing_report.py --smoke
"""

# ---------------------------------------------------------------------------
# DATA TO LOAD (real run). No embeddings/scores needed -- this assembles a timing
# table from measured stage costs. Provide a JSON with, per (prune-rate, method),
# the three stage costs (i) score computation, (ii) extrapolation, (iii) eval-train.
# The real numbers are already reported in the ICML rebuttal (DUAL/GNN-DUAL,
# old_Review_icml.txt lines ~453-468) and are ALREADY RENDERED in rebuttal.md W3/Q2:
#
#   Rate  Method     (i)Score  (ii)Extrap  (iii)Train  Total
#   0.9   DU            910        0           456      1366
#         GNN-DU        182      192           456       830
#         DUAL          787        0           456      1243
#         GNN-DUAL      157      182           456       795
#   0.5   DU            910        0           182      1092
#         GNN-DU        182      192           182       556
#         DUAL          787        0           182       969
#         GNN-DUAL      157      182           182       521
#   0.1   DU            910        0            46       956
#         GNN-DU        182      192            46       420
#         DUAL          787        0            46       833
#         GNN-DUAL      157      182            46       385
# Extrapolation (GNN) replaces most of the score-computation cost: GNN-DUAL scoring
# 787->157+182 (~2.32x), GNN-DU 910->182+192 (~2.43x); totals up to ~2.16x / 2.28x.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from typing import Dict

import rebuttal_common as rc


def build_report(timings: Dict) -> Dict:
    full = timings["full"]
    full_total = float(sum(full.values()))
    report = {"full_total_s": full_total, "full_breakdown": full, "methods": {}}
    for name, comps in timings.get("extrapolation_methods", {}).items():
        total = float(sum(comps.values()))
        report["methods"][name] = {
            "breakdown_s": comps,
            "total_s": total,
            "speedup_vs_full": (full_total / total) if total > 0 else float("inf"),
            "fraction_of_full": total / full_total if full_total > 0 else float("nan"),
        }
    return report


def _print(rep: Dict) -> None:
    print(f"  FULL pipeline total: {rep['full_total_s']:.1f} s")
    for comp, v in rep["full_breakdown"].items():
        print(f"      {comp:>28}: {v:>12.1f} s")
    print(f"  {'method':>16} {'total_s':>12} {'speedup':>9} {'frac_full':>10}")
    for name, m in rep["methods"].items():
        print(f"  {name:>16} {m['total_s']:>12.1f} {m['speedup_vs_full']:>9.2f}x "
              f"{m['fraction_of_full']:>9.2%}")


def run_smoke() -> Dict:
    print("[item3] SMOKE: extrapolation-vs-DUAL timing report")
    timings = {
        "full": {"dual_full_scoring_s": 51230.0},
        "extrapolation_methods": {
            "knn": {"subset_scoring_s": 5120.0, "embedding_extraction_s": 640.0,
                    "knn_extrapolation_s": 95.0},
            "gnn": {"subset_scoring_s": 5120.0, "embedding_extraction_s": 640.0,
                    "gnn_training_s": 820.0, "gnn_inference_s": 60.0},
        },
    }
    rep = build_report(timings)
    _print(rep)
    assert rep["methods"]["knn"]["speedup_vs_full"] > 1.0
    print("[item3] SMOKE PASSED")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 3: extrapolation-vs-DUAL timing")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timings", help="JSON with measured component timings")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if not args.timings:
        ap.error("need --timings JSON (or --smoke)")

    import json
    with open(args.timings) as f:
        timings = json.load(f)
    rep = build_report(timings)
    _print(rep)
    if args.out:
        rc.save_json(rep, args.out)
        print(f"[item3] wrote {args.out}")


if __name__ == "__main__":
    main()
