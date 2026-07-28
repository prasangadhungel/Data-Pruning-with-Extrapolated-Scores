"""Run every rebuttal analysis script's --smoke self-test (no data, no torch).

    python analysis/run_all_smoke.py
"""

from __future__ import annotations

import importlib
import sys

MODULES = [
    ("item1_knn_k_selection", "Item 1  non-oracle KNN k-selection"),
    ("item4_regression_baselines", "Item 4  regression/interpolation baselines"),
    ("item8_composition", "Item 8  pruning-composition analysis"),
    ("item3_timing_report", "Item 3  extrapolation-vs-DUAL timing"),
    ("item2_behavior_preservation", "Item 2  error-set / subgroup preservation"),
    ("q2_pretrained_backbone_knn", "Dcja Q2  pretrained-backbone KNN extrapolation"),
    ("b1_calibrate_scores", "B1      ranking-preserving calibration"),
    ("b3p1_uncertainty_error", "B3.1    uncertainty as error predictor"),
    ("longtail_fidelity", "LongTail score-fidelity (replaces dist-shift)"),
    ("perturbations", "OOD      ImageNet-C-style on-the-fly test corruptions"),
]


def main() -> int:
    failures = []
    for mod_name, desc in MODULES:
        print(f"\n===== {desc} ({mod_name}) =====")
        try:
            mod = importlib.import_module(mod_name)
            mod.run_smoke()
        except Exception as exc:  # noqa: BLE001
            failures.append((mod_name, repr(exc)))
            print(f"  !! FAILED: {exc!r}")
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} / {len(MODULES)} smoke tests FAILED:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 1
    print(f"ALL {len(MODULES)} smoke tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
