# Rebuttal — *Effective Data Pruning through Score Extrapolation* (NeurIPS 2026, #27899)

> **How to use this document.**
> **Part A** is an internal summary analysis (strengths, weaknesses, priorities) — *not* for posting.
> **Part B** contains one ready-to-post response per reviewer, plus a short reply to the AC meta-review.
> Each reviewer response is written to stay **under the 10,000-character portal limit** (current length noted at the end of each section).
> `[PLACEHOLDER: …]` markers indicate where you must insert **new results/numbers** before posting. Do not post placeholders.

---

# Part A — Summary Analysis (internal)

## Scoreboard (current NeurIPS)
| Reviewer | Rating | Confidence | Stance |
|---|---|---|---|
| bWqr | **4** Borderline accept | 4 | Positive, deeply engaged (W1–W6, Q1–Q5) |
| Dcja | **4** Borderline accept | 3 | Positive, wants clarity + tempered claims |
| myyu | **2** Reject | 4 | Robustness/behavior-preservation concerns |
| RpJS | **2** Reject | 3 | Novelty + weak baselines |
| AC K4gS | meta | — | Accuracy modest, novelty limited, claims too strong, variance/baselines |

**Split decision (4/4/2/2).** myyu and RpJS are the swing votes.

## Strengths (consensus across reviewers + prior venues)
- **Real, well-framed problem** — the "scoring costs as much as training" time paradox (bWqr S1, Dcja, RpJS, ICML Dbqa).
- **Clean conceptual shift** — importance as a *learnable function over the embedding manifold* rather than a per-sample measured property (bWqr S1, ICML Dbqa).
- **Orthogonal / drop-in acceleration layer**, composes with existing metrics; honestly scoped as bounded by the base metric (bWqr S2).
- **Broad validation** — 2 metrics (DU, TDDS), 4 datasets to million scale, 3 paradigms; adversarial 2M regime enables pruning otherwise infeasible (bWqr S3, Dcja, RpJS).
- **KNN Pareto result** — training-free O(nd) extrapolator on the time–accuracy frontier, beating geometric "training-free" baselines (bWqr S4).
- **Good-faith failure analysis** — oversmoothing / OOD error concentration (bWqr S5, ICLR mc1X).

## Weaknesses / issues, grouped by theme (with reviewer + recurrence tags)
| # | Theme | Raised by (NeurIPS) | Recurs at ICML/ICLR? |
|---|---|---|---|
| T1 | **Modest score correlation** vs "accurate" claim; Spearman < Pearson | bWqr W1/Q1, myyu, Dcja, RpJS, AC | ICML decision, ICLR |
| T2 | **Behavior preservation beyond avg accuracy** (error-set overlap, subgroup, distribution shift) | myyu, AC | ICLR (oversmoothing) |
| T3 | **Uncertainty/variance** — single-seed downstream, no error bars | bWqr W6/Q5, Dcja, AC | ICML M3bU |
| T4 | **Overstrong claims** ("extrapolation", "resolves paradox", Pareto, billion-scale, "highly accurate") | Dcja, AC | — |
| T5 | **Baselines too weak / too few** (stronger regression/interpolation, AL, EL2N, coreset, Top-K-of-subset) | RpJS, AC | ICML Dbqa/o91M, ICLR |
| T6 | **Limited novelty** (KNN/GNN in embedding space) | RpJS, myyu, AC | ICML o91M |
| T7 | **Speedup measured vs most expensive baseline**; no extrapolation-vs-DUAL time comparison | bWqr W3/Q2, AC | ICML o91M |
| T8 | **KNN complexity O(nd)** vs exact NN O(nmd) | Dcja | ICML Dbqa (graph build) |
| T9 | **Theory loosely coupled** — motivation, not guarantee | bWqr W5, Dcja, RpJS | ICML o91M |
| T10 | **GNN vs KNN cannibalize** — no regime where GNN is the right call | bWqr W4/Q3 | — |
| T11 | **DU below random past 50%** — narrow useful band | bWqr W2/Q4 | — |
| T12 | **Setup clarity** — hyperparameter selection, proxy vs downstream arch, TDDS-on-ImageNet gap, oracle-k selection | Dcja (Qs), | ICML x1Pq |
| T13 | **Proxy-model sensitivity / foundation embeddings** | Dcja (Q), myyu | ICML o91M/x1Pq |
| T14 | **Typos / formatting / figure issues** | Dcja | ICML Dbqa (Fig 6) |

## Rebuttal priorities
1. **Reframe correlation → pruning utility** (T1) — the single most-repeated concern; separate *score fidelity* from *selection quality* and back it with the rank/neighborhood evidence already in App. C.6 (Tab. 16) + Fig. 4.
2. **Add variance / error bars** (T3) and **behavior-preservation evidence** (T2) — concrete new results that flip myyu and reassure Dcja/bWqr.
3. **Temper claims + fix oracle-k + clarify setup** (T4, T12) — cheap, high-credibility wins for Dcja.
4. **Non-oracle DUAL time comparison + stronger baselines** (T5, T7) — address AC + RpJS.
5. **Novelty framing** (T6) — position as a *new paradigm/first framework*, not a new estimator.

---

# Part B — Per-Reviewer Responses (ready to post)

---

## Response to Reviewer bWqr (Rating 4)

We thank you for the exceptionally careful and constructive review, and for clearly articulating the contribution: the "scoring costs as much as training" reframing (**S1**), the honest scoping as a metric-bounded acceleration layer (**S2**), the breadth of validation including the otherwise-infeasible adversarial 2M-sample regime (**S3**), the training-free **KNN Pareto** result (**S4**), and our good-faith failure analysis (**S5**). Your S1–S5 summary captures our intent precisely, and W1–W6 are all actionable. Point-by-point below.

**W1 / Q1 — Low correlation; Spearman < Pearson exactly where the rank argument leans.**
You are right that this needs a real explanation rather than a hand-wave. Our claim is *not* that global rank is well recovered; it is that **pruning only needs the *tails* of the ranking to be approximately right** — which samples are safely removable (low-importance, dense class-homogeneous regions) vs. which must be kept (high-importance, near decision boundaries). Two pieces of existing evidence support this: (i) App. C.6 / Tab. 16 shows ground-truth DU importance is strongly tied to *local neighborhood composition* (same-label neighbor count ρ=−0.30, different-label distance ρ=+0.29), i.e., the signal our estimators exploit is concentrated exactly in the tails; (ii) Fig. 4 shows downstream accuracy rises monotonically with correlation, so even moderate correlation lands on the useful part of that curve. Spearman<Pearson arises from **oversmoothing collapsing the mid-range ranks** (App. C.3), which is precisely where mis-ranking is *harmless* for top-k selection.
`[PLACEHOLDER: add a "tail-rank" metric — Spearman restricted to the top-x% and bottom-x% of scores, and/or recall of the retained set vs. ground-truth retained set — to quantify that the *decision-relevant* rank is far higher than the global 0.35/0.22.]`

**W2 / Q4 — DU drops below random past 50%; headline speedups come from TDDS@95%.**
Agreed, and we will state the operating range explicitly. The intended use of extrapolation is *within the band where the base metric beats random* — for DU that is ≲50%, for TDDS up to 95% (Tab. 1). Extrapolation never claims to rescue a metric outside its own effective range; it accelerates it inside that range. We will add a sentence to §4 and the abstract making this scope explicit rather than implying uniformly high rates.

**W3 / Q2 — 4.9× is vs the most expensive baseline; no extrapolation-vs-DUAL time comparison.**
Fair. The 4.9× is against full multi-epoch scoring, and we already show extrapolation *composes* with DUAL (App. C.6, Fig. 10) rather than competing with it. We will add the direct time-axis comparison you ask for.
`[PLACEHOLDER: table/figure — wall-clock of (a) full DU, (b) DUAL (reduced-epoch), (c) extrapolation-of-DU, (d) extrapolation-of-DUAL, on ≥1 large dataset, so the residual speedup against a cheap scorer is explicit.]`

**W4 / Q3 — GNN and KNN cannibalize; no "use GNN when X".**
GNN gives higher correlation (Tab. 2) but KNN is Pareto-optimal on time–accuracy (Fig. 3). The honest current answer: **use KNN under a time budget; use GNN when score *fidelity itself* is the product** (e.g., extrapolating scores for data attribution / re-use across many downstream selections, where the extra one-off cost amortizes). We will state this decision rule explicitly.
`[PLACEHOLDER: if available, one downstream setting (e.g., a higher/edge pruning rate or a fidelity-sensitive task) where GNN's downstream accuracy exceeds KNN beyond noise — otherwise we will explicitly scope GNN as the fidelity-oriented variant and KNN as the deployment default.]`

**W5 — Theory is loosely coupled.**
We agree and will reframe. The influence-function argument (Eqs. 1–5) establishes *local smoothness of importance in embedding space*; it deliberately does **not** claim importance is radial in Euclidean distance (we say this in §3, "Practical Considerations"). We will relabel it a **motivating prior**, not a justification of the estimator, and lean on the empirical validation in App. C.6 (Tab. 16) as the actual support.

**W6 / Q5 — Single-seed downstream training; ± only captures scoring noise.**
Correct — scores use 3 seeds but each pruned model is trained once, so sub-1pp gaps are not statistically resolved.
`[PLACEHOLDER: re-run the tightest extrapolated-vs-ground-truth downstream comparisons (Tab. 1 / Figs. 2, 5) over ≥3 training seeds and report mean±std, so the reader can see which gaps are significant.]`

We believe W1, W5, W2 are addressable by reframing + existing evidence, and W3/W6 by the targeted additions above. Thank you again — we would gladly raise scope of the operating-range and DUAL-time additions if that would move your assessment.

<!-- length check: keep < 10000 chars -->

---

## Response to Reviewer myyu (Rating 2)

Thank you for the review and for recognizing that improving training efficiency via data pruning is a well-motivated, practically relevant goal, and for the constructive pointers toward robustness and subgroup behavior. Your central point — that aggregate accuracy may hide *where* the pruned model fails — is exactly the kind of scrutiny that strengthens the paper, and we address it directly below.

**On low score correlation.**
We agree correlation is modest (e.g., ImageNet DU 20%: Spearman 0.35) and we will stop describing scores as "highly accurate." Our claim is narrower: pruning needs only the **decision-relevant tails** of the ranking, not exact scores. App. C.6 (Tab. 16) shows ground-truth importance is governed by local neighborhood class-composition, the exact signal our estimators recover, and Fig. 4 shows accuracy tracks correlation. So moderate correlation is sufficient *for selection*, which is what the downstream results in Tab. 1 / Fig. 2 confirm.

**On "does the pruned model fail on the same examples?" (error-set overlap, subgroup / long-tail, distribution shift).**
This is the most valuable suggestion in the review and we will add these diagnostics. Our own failure analysis (App. C.3) already shows extrapolation error concentrates on atypical/OOD samples, which is exactly why per-group evaluation matters. We will quantify it rather than only note it.
`[PLACEHOLDER: (1) error-set overlap / agreement between models pruned with ground-truth vs. extrapolated scores (e.g., Jaccard of misclassified sets, prediction-agreement rate); (2) subgroup / long-tail (per-class or rare-class) accuracy for both; (3) a natural distribution-shift eval, e.g., ImageNet→ImageNet-v2/-C or CIFAR→CIFAR-C, comparing ground-truth-pruned vs. extrapolation-pruned vs. random.]`

We note the adversarial results (Tab. 4–5) already provide *some* robustness evidence — extrapolation retains clean *and* robust accuracy close to ground truth (ℓ∞: 63.56% vs. 63.13% random) — but we agree **natural** distribution shift is the more relevant stressor and will add it as above.

**On pruning side-effects for large models (your cited arXiv:2605.19407).**
We appreciate the pointer and will add a discussion of pruning's downside for large-scale training. 
We would to empaize that this work was **posted to arXiv on 19 May 2026, i.e., after the NeurIPS submission deadline of 15 May 2026**, so it was not availible for the submitted version. 
We will incorporate it in the revision and discuss how score extrapolation interacts with those findings (our method *inherits* the base metric's behavior, so it neither adds nor removes such side-effects beyond what the underlying metric already induces — which our new error-set/subgroup analysis will make measurable).

**On novelty / significance.**
We would gently reframe the contribution: it is not a new per-sample estimator but the **first framework to treat importance as an extrapolatable quantity**, turning any expensive full-dataset scorer into a subset-scored one. The KNN/GNN instantiations are deliberately simple to show the paradigm is feasible and general (App. C.7 shows they beat naive interpolation, score-sampling, and an MLP baseline).

We hope the concrete behavior-preservation experiments above address the core of your concern; we are committed to adding all three (error-set overlap, subgroup, distribution shift) and would welcome guidance on which shift benchmark you would find most convincing.

<!-- length check: keep < 10000 chars -->

---

## Response to Reviewer Dcja (Rating 4)

Thank you for the unusually thorough and constructive review, and for recognizing that the paper **addresses an important practical bottleneck** in data pruning, with a **well-motivated empirical setup covering a broad set of datasets**, and a method that is **simple to implement, practically appealing, and yields promising speedups with only mild accuracy drops**. Most of your points are directly actionable and improve the paper; we group our responses by your headings.

**W (uncertainty) — no std/error bars on downstream accuracy (Figs. 2, 5, 9).**
Agreed; this is our most important omission. Scores are averaged over 3 seeds but downstream models were trained once.
`[PLACEHOLDER: add error bars (state whether std or SE, and over how many seeds) to Figs. 2/5/9 and Tab. 1; prioritize low-pruning-rate regimes and add ≥1 high-rate point to show variance growth as you predict.]`

**W (setup clarity) + Questions on hyperparameters / architectures / TDDS-on-ImageNet.**
- **Hyperparameter selection:** proxy/scoring hyperparameters follow the original DU/TDDS papers' defaults (App. B). For the extrapolators, KNN's *k* and the GNN config are set by validation (see oracle-k caveat below). We will state each selection criterion explicitly.
- **Proxy vs downstream architecture (Q):** the proxy F_s uses the **same architecture and setup** the base metric would use on the full data (§3, "Proxy Model"), and the downstream model matches it. We will state this explicitly and flag any exception.
- **TDDS on ImageNet:** you and the minor-issues note (App. B.2 K=90 vs. missing Table 7 column) correctly caught an inconsistency — TDDS-ImageNet was set up but not completed in time.
`[PLACEHOLDER: either add the TDDS-ImageNet results, or remove the B.2 K=90 line and state plainly that TDDS-ImageNet is out of scope, so the claim "2 metrics × 4 datasets" is tightened to what is actually run.]`

**W (overstrong claims) — several, all fair.** We will revise wording throughout:
- **"Extrapolation":** we will clarify terminology up front — KNN is closer to **interpolation/smoothing** and GNN to **graph-based score regression**; we keep "extrapolation" only as the umbrella name for predicting scores on unscored samples and say so.
- **"Highly accurate / highly correlated":** removed; replaced with the honest "moderate correlation, sufficient for selection" framing, citing the modest Spearman values (0.35 ImageNet DU 20%, 0.26 Places TDDS 20%, 0.22 Places DU 10%).
- **"Resolves the time paradox":** softened to **"substantially mitigates"** the scoring-cost bottleneck for static score-based pruning.
- **Pareto / billion-scale:** we will state Pareto-optimality is **relative to the plotted methods/datasets/rates**, and mark billion-scale as motivation extrapolated from million-scale evidence.

**W (KNN complexity O(nd)).**
You are right that exact NN from n targets to m scored points is O(nmd). Our reported runtimes use **approximate nearest-neighbor search** (batched, index-based), for which the per-query cost is sub-linear in m; the O(nd) statement refers to that regime.
`[PLACEHOLDER: state the exact ANN method/index used and correct the complexity to reflect the approximate search (or report exact-search O(nmd) with the constant), so the stated complexity matches the measured runtimes.]`

**W (theory as motivation).** Agreed — we will present the influence-function/local-linearity argument as **motivation**, with App. C.6 (Tab. 16) as the empirical support, not as a guarantee that DU/TDDS are recoverable by KNN/GNN.

**W (separate score quality from pruning utility).** This is exactly the distinction we will foreground: the method need not recover the full ranking, only enough of the high/low tails to choose a good retained set. We will add this framing where correlation is first reported (§4) and in the abstract.

**Q (proxy embedding sensitivity / foundation embeddings).** We did not stress-test proxy over/under-fitting in the submission.
`[PLACEHOLDER: report proxy train/val accuracy or an embedding-quality proxy vs. extrapolation quality; and, since the unsupervised setting already uses DINOv2 (App. B.3, Tab. 9), add a supervised run using pretrained/foundation embeddings for F_s to show sensitivity.]`

**Q (qualitative effect on *which* data is pruned).** Good suggestion; Fig. 7 already points this way.
`[PLACEHOLDER: compare the composition of the retained/pruned sets (easy-redundant vs. outlier vs. borderline proportions) under ground-truth vs. extrapolated scores, to show the speedup preserves the metric's qualitative behavior.]`

**Q + minor issue (KNN k possibly oracle-selected, App. B.4).**
You correctly identified this: App. B.4 currently selects *k* by the highest Pearson correlation **against the full scores S on D_r**, which are unavailable in the intended setting — an oracle. This must be fixed. GNN, in contrast, already selects its checkpoint on a validation split of the *scored* subset S_s (available), which is the correct protocol.
`[PLACEHOLDER: re-select KNN k using only the scored subset S_s (e.g., held-out split of S_s), report non-oracle results, and show the gap to the oracle-k numbers in Tab. 2 is small.]`

**Minor issues / typos.** Thank you for the detailed list — we will fix all of them (Fig. 3 legend/Pareto entry + pruning-rate caption + reference lines; Fig. 9 redundancy note; the self-referential T_extra symbol in App. A.6; and the full typo/capitalization list on lines 154/226, abstract, Fig. 1 "comprising", §4 "CIFAR-10. On synthetic", Limitations "datasets", App. A.1 "denoising", C.2 "Initial", C.3 "Analysis"/"misprediction", B.3 "original", "arXiv", the stray bracket in 2302.12366, Fig. 10 placement).

We are grateful for the depth here — every one of these is being incorporated. Given the fixes are largely wording, clarity, and clearly-scoped additions, we hope you would consider these sufficient to raise your assessment.

<!-- length check: ~5.7k chars, within the 10000 limit. If added placeholder text pushes it over, split into (Post 1) uncertainty + setup + claims and (Post 2) complexity + theory + score-vs-utility + proxy/qualitative + oracle-k + typos. -->

---

## Response to Reviewer RpJS (Rating 2)

Thank you for the review and for recognizing the core value of the approach: that it **reduces computational cost by scoring only a subset** rather than the full dataset, **improves efficiency through a smaller-scale training-and-extrapolation setup**, and — by **building on existing importance-scoring methods** — stays **practically applicable and is empirically validated across multiple datasets and training settings**. We address the four weaknesses directly below.

**W1 — Limited novelty; "closer to a practical technique than a theoretically grounded contribution."**
We would reframe the contribution's locus. The novelty is **not** the KNN/GNN estimators (deliberately simple) but the **paradigm**: to our knowledge this is the first work to treat importance as an **extrapolatable quantity** and to define a framework that converts *any* expensive full-dataset scorer (DU, TDDS, and, as we show, DUAL) into a subset-scored one — directly targeting the scoring/training time paradox. The estimators are minimal on purpose, to show the paradigm is general rather than tied to one clever model; App. C.7 shows even these simple choices beat naive interpolation, uniform score-sampling, and an MLP. We will make this framing explicit so the contribution is not read as "just KNN/GNN in embedding space."

**W2 — Are the experiments limited to settings where scores are *easy* to predict (classes group naturally)?**
A fair concern. We note the paradigm is *not* restricted to classification: the **unsupervised** setting (Turtle/DINOv2, Tab. 3) has no class grouping yet extrapolation still correlates strongly (KNN Spearman 0.66 at 20%), and the **adversarial** 2M-sample regime (Tab. 5) extrapolates from a 5% subset where ground-truth scoring is infeasible. The class-composition structure we exploit (App. C.6, Tab. 16) is a *reason it works*, not a hidden requirement.
`[PLACEHOLDER: if feasible, one non-class-structured or regression-style scoring task (or an imbalanced/long-tailed split) to show extrapolation holds when samples are not neatly class-grouped.]`

**W3 — Oversmoothing / struggles on atypical or ambiguous samples (your own quote of our failure analysis).**
We agree and report this honestly (App. C.3, Limitations): extrapolation narrows the bimodal ground-truth distribution and errs most on OOD/atypical samples. Crucially, these are a *minority*, and — because pruning keeps high-importance samples — the practical effect on the retained set is limited, which is why downstream accuracy stays close to ground truth (Tab. 1). We already point to concrete mitigations (residual GNN connections, larger subset m). To make the impact measurable rather than asserted:
`[PLACEHOLDER: quantify error on the atypical/OOD subset vs. the bulk (e.g., error stratified by sample typicality), and show how much the retained-set quality actually degrades due to oversmoothing.]`

**W4 — Baselines are relatively weak; stronger regression/interpolation baselines needed.**
We do compare against Score Sampling, Naive Theoretical Interpolation, and an MLP regressor (App. C.7), plus training-free geometric methods (ZCoreSet, SSP) in the main results — but we agree a stronger regression suite would strengthen the claim that structure-aware extrapolation is doing real work.
`[PLACEHOLDER: add stronger predictors — e.g., kernel/Nadaraya–Watson regression, random-forest/gradient-boosted regression on embeddings, and a label-propagation baseline — reporting both score correlation and downstream accuracy against KNN/GNN.]`

We believe W1 (framing) and W2/W3 (existing unsupervised/adversarial + failure evidence) are largely addressable now, and W4 with the added regression baselines above. We would be glad to prioritize whichever of these additions would most affect your assessment.

<!-- length check: keep < 10000 chars -->

---

## Response to the Area Chair (Meta-Review K4gS)

Thank you for the balanced summary. We briefly note how the shared concerns are addressed across the individual responses:

- **Modest extrapolated-score accuracy →** we separate *score fidelity* from *selection quality*: pruning needs only the decision-relevant tails of the ranking, supported by App. C.6 (Tab. 16) and Fig. 4, and we temper all "highly accurate" wording (Dcja, myyu, bWqr-W1).
- **Limited methodological novelty →** we reframe the contribution as the **first extrapolation *framework*** for importance scores, not a new estimator; the simple KNN/GNN choices still beat naive baselines in App. C.7 (RpJS-W1, myyu).
- **Behavior preservation beyond average accuracy →** we are adding **error-set overlap, subgroup/long-tail accuracy, and natural distribution-shift** evaluations (myyu).
- **Claims stronger than warranted →** "resolves the time paradox" → "substantially mitigates"; Pareto/billion-scale explicitly scoped to plotted evidence; "extrapolation" terminology clarified (Dcja).
- **Insufficient variance quantification →** adding downstream error bars over multiple training seeds to Figs. 2/5/9 and Tab. 1 (bWqr-W6, Dcja).
- **Stronger baselines →** adding kernel/tree regression and label-propagation predictors, plus a non-oracle *k*-selection protocol and a direct extrapolation-vs-DUAL time comparison (RpJS-W4, bWqr-W3, Dcja).

We also flag, for the record, that Reviewer myyu's cited work (arXiv:2605.19407) appeared **after** the 15 May 2026 submission deadline; we will nonetheless discuss it in the revision.

<!-- length check: keep < 10000 chars -->

---

## Global to-do before posting (checklist for the authors)
- [ ] Fill every `[PLACEHOLDER]` with real numbers/figures; delete any that cannot be completed and soften the corresponding text.
- [ ] Verify each posted reviewer block is **< 10,000 characters** (all currently pass: bWqr ~4.4k, myyu ~3.3k, Dcja ~5.7k, RpJS ~3.5k, AC ~2.1k).
- [ ] Fix the full typo/consistency list (Dcja minor issues) and the oracle-*k* protocol (App. B.4).
