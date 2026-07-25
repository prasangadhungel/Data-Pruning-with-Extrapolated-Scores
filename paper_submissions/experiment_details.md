# Experiment Protocols — Rebuttal for #27899

Detailed, runnable specifications for the experiments in `experiments_tbd.md`. Notation follows the paper: full set $\mathcal{D}$ ($n$ samples), scored subset $\mathcal{D}_s$ ($m$ samples, e.g. 10–20%), residual set $\mathcal{D}_r=\mathcal{D}\setminus\mathcal{D}_s$; ground-truth scores $S\in\mathbb{R}^n$ (from full training) and $S_s$ (subset); embeddings $z=\phi_s(x)\in\mathbb{R}^d$ from proxy $\mathcal{F}_s$; extrapolated scores $S_r$ (KNN or GNN). Pruning keeps the top-$k$ highest-importance samples.

**Standard setup (reuse across items):** metrics DU, TDDS; datasets + backbones ImageNet/ResNet-18, Places365/ResNet-50, Synthetic CIFAR-100-1M/ResNet-50, CIFAR-10/ResNet-18, adversarial CIFAR-10/WRN-28-10; pruning rates {10,20,50,80,90,95}%; subset sizes {10,20}%; 3 score seeds. Hardware: A100. Report mean ± std.

**Artifacts you should already have** (needed to avoid recomputation): per-seed ground-truth scores $S$, subset scores $S_s$, proxy embeddings $z$ for all of $\mathcal{D}$, the extrapolated scores $S_r$ (KNN/GNN), and the trained downstream checkpoints (or at least their test-set predictions). Items 1–4 mostly need scores + embeddings + predictions, not retraining.

---

## Item 1 — Non-oracle *k*-selection for KNN  🟢 (mandatory)

**Concern:** Dcja-W / Dcja-Q6. App. B.4 (`\label{Sec:Extrapolation}`) selects KNN's $k$ by the highest Pearson correlation **against full scores $S$ on $\mathcal{D}_r$** — those scores don't exist at deployment (oracle). Must be replaced with selection using only $S_s$.

**Objective:** Show a *deployable* $k$-selection rule attains correlation close to the oracle-$k$ currently reported in Tab. 2 (`tab:correlation-results`).

**Procedure (no proxy retraining; operates on saved $S_s$ and embeddings):**
1. For each (dataset, metric, subset size, seed): split $\mathcal{D}_s$ into $\mathcal{D}_s^{\text{fit}}$ (80%) and $\mathcal{D}_s^{\text{val}}$ (20%), stratified by class.
2. For each candidate $k\in\{10,20,50,100\}$: extrapolate scores for $\mathcal{D}_s^{\text{val}}$ using only $\mathcal{D}_s^{\text{fit}}$ neighbors (Eq. 6, distance-weighted average), and compute Pearson$(\hat S,\;S_s)$ on $\mathcal{D}_s^{\text{val}}$.
3. Pick $k^\star=\arg\max$ that validation Pearson. **This uses only subset scores $S_s$ — fully deployable.**
4. Re-extrapolate all of $\mathcal{D}_r$ with $k^\star$ (fit on full $\mathcal{D}_s$) and recompute Pearson/Spearman vs $S$ on $\mathcal{D}_r$ (the real evaluation).
5. Also report a fixed-$k$ baseline (e.g. $k{=}20$ everywhere) as an even simpler rule.

**Output table** (replaces/augments KNN columns of Tab. 2):
| Dataset | Metric | Subset | Oracle-$k$ ρ (current) | Non-oracle $k^\star$ ρ | Fixed $k{=}20$ ρ | Δ(oracle−nonoracle) | Selected $k^\star$ |

**Success criterion:** |Δ| small (target ≲0.01–0.02 Pearson). Then state in App. B.4 that $k$ is chosen on an $S_s$ held-out split and that results are insensitive to $k$ (Tab. `tab:all_correlation_knn` already shows the full sweep is flat).

**Also fix:** the label-ablation text (`tab:class_ablation`) says "oracle scores on $\mathcal{D}_r$" — reword to reflect the deployable protocol.

**Effort:** minutes–hours of CPU (KNN over saved embeddings). No GPU/retraining.

**Rebuttal snippet (Dcja):** "We agree the reported $k$ used an oracle. Selecting $k$ on a held-out split of the *scored* subset $S_s$ yields ρ=… vs the oracle ρ=… (Δ=…), and a fixed $k{=}20$ gives ρ=…; the choice of $k$ is not doing hidden work (full sweep in Tab. `tab:all_correlation_knn`)."

---

## Item 2 — Behavior preservation: error-set overlap + subgroup accuracy  🟢 (flips myyu)

**Concern:** myyu-W2, AC-M3 — does the extrapolation-pruned model fail on the *same* examples, and does it hurt subgroups/long-tail? Currently only qualitative (`ap:Visual`).

**Objective:** Quantify that pruning with extrapolated scores preserves not just average accuracy but *where* the model succeeds/fails, relative to ground-truth-score pruning.

**Models compared** (all already trained; a representative rate, e.g. DU@50% on ImageNet/Places, TDDS@95% on synthetic): (a) GT-score pruned, (b) KNN-extrapolation pruned, (c) GNN-extrapolation pruned, (d) Random pruned, (e) Unpruned reference.

**Metrics (post-hoc, need only test-set predictions):**
- **Error-set overlap.** Let $E_A,E_B$ be misclassified test-sample sets. Report Jaccard $J=\frac{|E_A\cap E_B|}{|E_A\cup E_B|}$ and **prediction-agreement** $=\frac1{|T|}\sum_x \mathbb{1}[\hat y_A(x)=\hat y_B(x)]$ over test set $T$; optionally Cohen's κ. Key comparison: GT-vs-KNN and GT-vs-GNN should have **higher** overlap/agreement than GT-vs-Random (the null).
- **Subgroup / long-tail accuracy.** Per-class accuracy; then bucket classes by frequency in the *retained* set (or by ground-truth long-tail rank) into head/mid/tail thirds. Report mean accuracy per bucket and **worst-group** accuracy. Show GT-vs-extrapolation gap per bucket is small and comparable to GT-vs-GT-across-seeds noise.
- **Optional: error concentration.** Fraction of extra errors (vs GT-pruned) that fall on atypical/OOD samples flagged in `ap:Visual` — ties the quantitative result back to the existing qualitative story.

**Output tables:**
1. Overlap: rows = {GT–KNN, GT–GNN, GT–Random}; cols = Jaccard, Agreement, κ.
2. Subgroup: rows = {GT, KNN, GNN, Random}; cols = head / mid / tail / worst-group accuracy.

**Success criterion:** GT–KNN/GNN agreement ≫ GT–Random; tail/worst-group gap (GT vs extrapolation) ≲ the training-seed noise from Item 5.

**Effort:** post-hoc on saved predictions; if predictions weren't stored, one forward pass per model over the test set (minutes on A100). No retraining.

**Rebuttal snippet (myyu):** "Extrapolation-pruned and GT-pruned models agree on X% of predictions (Jaccard Y on the error set) vs only Z% for random, and the worst-group accuracy gap is ≤…pp — so the speedup preserves *where* the model fails, not just average accuracy."

---

## Item 3 — Direct extrapolation-vs-DUAL time comparison  🟢

**Concern:** bWqr-W3/Q2 — the 4.9× is vs full-training scoring; how much remains against a *cheap* scorer (DUAL, reduced epochs)? Currently DUAL is only shown as *composable* (`sec:DynamicPruning`, `Fig:DUAL`).

**Objective:** Isolate the residual speedup of extrapolation on top of a cheap scorer.

**Configurations (same hardware, same eval-training budget):**
| Config | Scoring cost (i) | Extrapolation (ii) | Eval-train (iii) | Total |
|---|---|---|---|---|
| Full DU | full-set, all epochs | 0 | fixed | … |
| DUAL | reduced-epoch full-set | 0 | fixed | … |
| Extrap-DU | subset-only | KNN/GNN | fixed | … |
| Extrap-DUAL | subset-only (reduced-epoch) | KNN/GNN | fixed | … |

Reuse the per-phase decomposition already defined in App. "Temporal Analysis" (`sec:temporal`); only the DUAL scoring-time rows may need one clean run. Pick ≥1 large dataset (Places365 or ImageNet).

**Output:** the table above + a bar/point plot on the time axis; report speedup(Extrap-DUAL vs DUAL) and speedup(Extrap-DUAL vs full DU).

**Success criterion:** Extrap-DUAL < DUAL in wall-clock while matching downstream accuracy — demonstrating extrapolation still helps against a cheap scorer, not only the expensive one.

**Effort:** bookkeeping + possibly one DUAL timing run.

**Rebuttal snippet (bWqr):** "Against full DU we save 4.9×; against reduced-epoch DUAL, extrapolation still yields a further …× (Extrap-DUAL … min vs DUAL … min) at equal accuracy, and the two compose."

---

## Item 4 — Stronger regression / interpolation baselines  🟡 (flips RpJS)

**Concern:** RpJS-W4 — existing baselines (score-sampling `tab:sampling`, naive interpolation `tab:naiveInter`, MLP `tab:MLP`, plus Top-K-of-subset from the ICML rebuttal) are called weak. Add stronger *regressors*.

**Objective:** Show structure/geometry-aware extrapolation (KNN/GNN) beats strong generic regressors trained on the same embeddings, i.e., the neighborhood/graph structure adds value.

**Baselines to add (all fit on $(\phi_s(x), S_s)$ over $\mathcal{D}_s$, predict on $\mathcal{D}_r$; no proxy retraining):**
1. **Kernel / Nadaraya–Watson regression** (RBF kernel; bandwidth via $S_s$-val) — a principled generalization of KNN's distance weighting.
2. **Gradient-boosted trees** (XGBoost/LightGBM) or **random forest** on the $d$-dim embeddings.
3. **Label propagation** on the $k$-NN graph (propagate $S_s$ over the same graph the GNN uses) — isolates "GNN learning" vs "pure diffusion".
4. (Optional) **Ridge / kernel-ridge** as a linear reference.

**Evaluation:** (a) Pearson & Spearman vs $S$ on $\mathcal{D}_r$ for **all** (cheap — no downstream training); (b) downstream accuracy for the **best 1–2** baselines only, on **small datasets** (CIFAR-10, synthetic CIFAR-100) at 2–3 rates, to bound cost. Selection rule for each baseline's hyperparameters must use $S_s$-val only (consistency with Item 1).

**Output:** extend the `sec:Interpolation` tables — rows = {Score-sampling, Naive-interp, MLP, Kernel/NW, GBT/RF, Label-prop, **KNN (ours)**, **GNN (ours)**}; cols = Pearson, Spearman, (Acc on CIFAR-10 / synthetic).

**Success criterion:** KNN/GNN ≥ every generic regressor on correlation *and* downstream accuracy — especially label-propagation, which shares the graph but lacks learning; if a regressor ties KNN on correlation, emphasize KNN's O(nd) cost and Pareto position (Fig. 3).

**Effort:** correlation rows are cheap (CPU on saved embeddings); downstream rows are the moderate part (small datasets only).

**Rebuttal snippet (RpJS):** "Against kernel regression, gradient-boosted trees, and label propagation on the *same* embeddings, KNN/GNN retain the best correlation (…) and downstream accuracy (…), confirming the neighborhood/graph structure — not just regression on embeddings — drives the result."

---

## Item 5 — Downstream seed variance / error bars  🟡

**Concern:** bWqr-W6/Q5, Dcja-W1, AC-M5. App. `ap:statistical_significance`: 3 score-seeds but each pruned model **trained once** → figures report **mean, no error bars**, and *training-seed* variance is absent for the sub-1pp GT-vs-extrapolation gaps.

**Objective:** Make the gaps interpretable — show which are significant vs within-noise.

**Two parts:**
- **(A) Free error bars.** You already have 3 accuracy values per rate (one per score-seed, trained once). Add std/SE error bars to Figs. 2/5/9 (`fig:secondExperiment`, `fig:appendix_pruning`, `fig:ParetoPlots`) and to Tab. 1. State explicitly these capture **scoring-seed** variance and are std (or SE, with $n{=}3$).
- **(B) Training-seed variance on tight comparisons.** For the Tab. 1 rows where |GT − extrapolation| < 1pp, fix one score-seed and retrain the pruned model with ≥3 **training** seeds for: GT, KNN-20%, GNN-20%, Random. Report mean±std and a paired test (e.g. paired $t$ / Wilcoxon) for GT-vs-extrapolation. Cheap on CIFAR-10/synthetic; restrict ImageNet/Places to the 1–2 headline points (DU@50%).

**Output:** updated Figs 2/5/9 with error bars; a focused table: rows = {GT, KNN, GNN, Random} × {ImageNet DU@50%, Places DU@50%, synthetic TDDS@95%}; cols = mean±std (training seeds), p-value vs GT.

**Success criterion:** either gaps are within overlapping error bars (supporting "extrapolation ≈ GT") or, where significant, honestly reported. Both outcomes strengthen credibility.

**Effort:** (A) trivial (re-plot). (B) 🟡 — several small-dataset trainings + ≤2 large-scale points.

**Rebuttal snippet (bWqr/Dcja):** "We add error bars to Figs. 2/5/9 (scoring-seed std, $n{=}3$) and, for the tight <1pp comparisons, report training-seed variance over 3 seeds: e.g. ImageNet DU@50% GT …±… vs KNN …±… ($p{=}$…)."

---

## Item 6 — Natural distribution-shift evaluation  🟡 (Tier 3)

**Concern:** myyu-W2 (natural shift > adversarial here). No natural-shift eval exists.

**Objective:** Show robustness to natural shift is preserved under extrapolation pruning, comparably to GT pruning.

**Procedure (evaluation-only if checkpoints exist — no retraining):** take the already-trained {GT, KNN, GNN, Random, Unpruned} models and evaluate on shifted test sets:
- ImageNet-trained → **ImageNet-v2** and **ImageNet-C** (report mean corruption accuracy / mCE).
- CIFAR-10-trained → **CIFAR-10-C** (and optionally CIFAR-10.1).

**Output:** rows = {GT, KNN, GNN, Random, Unpruned}; cols = clean acc, shifted acc, clean−shifted gap.

**Success criterion:** extrapolation's shifted-accuracy gap ≈ GT's, and both > Random — pruning quality is inherited from the base metric under shift too.

**Effort:** 🟡 eval-only if checkpoints saved; 🔴 if retraining needed.

---

## Item 7 — Proxy-embedding sensitivity + foundation embeddings  🟡 (Tier 3)

**Concern:** Dcja-Q4. Partially covered (DINOv2 in unsupervised `ap:Models`; ICLR preliminary supervised SSCD/DINOv2 tests). Not tabulated.

**Objective:** Quantify how extrapolation quality depends on proxy embedding quality, and whether foundation embeddings help supervised pruning.

**Procedure:**
1. **Proxy-quality curve:** for subset sizes {5,10,20,40}%, record proxy $\mathcal{F}_s$ train/val accuracy (a proxy for embedding quality) and the resulting extrapolation Pearson/Spearman. Plot correlation vs proxy-val-accuracy.
2. **Foundation embeddings:** repeat supervised extrapolation on ≥1 dataset using a frozen foundation encoder (DINOv2 ViT; already available from the unsupervised pipeline) as $\phi_s$ instead of the in-domain proxy; report correlation + downstream accuracy vs the in-domain proxy. (ICLR preliminary result: pretrained embeddings slightly *reduced* performance and add a dependency — formalize this.)

**Output:** (1) a small correlation-vs-proxy-accuracy table/plot; (2) in-domain vs foundation-embedding comparison table.

**Success criterion:** monotone-ish correlation↑ with proxy quality (bounded, robust for ≥10% subsets); foundation embeddings competitive but not clearly better, justifying the in-domain default.

**Effort:** 🟡 (mostly reuse; one foundation-embedding extrapolation run).

---

## Item 8 — Qualitative pruning-composition analysis  🟢/🟡 (Tier 3)

**Concern:** Dcja-Q5 — does approximating scores change *which* data is pruned? Fig. `fig:VisualAndDistribution` hints; no direct composition comparison.

**Objective:** Show the retained/pruned set composition under extrapolated scores matches GT scores.

**Procedure:** Categorize samples into {easy-redundant, borderline, outlier/noisy} using ground-truth signals already available — e.g. terciles of the GT score, and/or the neighborhood statistics of `tab:correlation_analysis` (same-label neighbor count, distance to different-label). For a fixed pruning rate, compute the fraction of each category in the **retained** set under GT vs KNN vs GNN vs Random. Optionally report retained-set overlap (Jaccard of kept indices GT-vs-extrapolation).

**Output:** stacked-bar or table: category proportions in retained set × {GT, KNN, GNN, Random}; plus retained-set Jaccard(GT, ·).

**Success criterion:** extrapolation's category mix and retained-set overlap track GT far more closely than Random — the speedup preserves the metric's qualitative selection behavior.

**Effort:** 🟢 if using existing GT-score buckets; 🟡 if new categorization needed.

---

## Cross-cutting practical notes
- **Consistency rule:** every hyperparameter (KNN $k$, kernel bandwidth, tree depth, GNN checkpoint) must be selected on an $S_s$ **validation split**, never on $S$ over $\mathcal{D}_r$ — this is exactly the oracle mistake Item 1 fixes; apply it everywhere so reviewers can't extend the critique.
- **Report $n$ and dispersion** (std vs SE) in every new table caption — Dcja-W1/AC-M5 will re-check this.
- **Reuse artifacts:** Items 1, 2, 3, 4-correlation, 6, 8 need **no retraining** — only saved scores, embeddings, and test predictions/checkpoints. Only Items 4-downstream, 5B, and 7 require (small) training.
- **Minimal high-leverage set if time is tight:** Items **1 + 2** (mandatory + flips myyu) then **4-correlation + 3** — all essentially post-hoc, and together they answer both rejects (myyu, RpJS) and the AC.

---

# Method-Modification Protocols (B1–B3)

These are *extensions* that turn recurring caveats (oversmoothing, OOD failure, limited novelty) into results. **Rebuttal-politics rule for all three:** present them as analyses/extensions that address the concern, validate on the datasets you can, and defer the full study to camera-ready — do **not** rewrite headline claims. Feasible under a "post-hoc + small-dataset retraining" budget: the GNN is ~1.4M params on **saved embeddings**, so all B-item training is graph-regression, *not* task-model retraining. Selection of every new hyperparameter follows the **$S_s$-validation consistency rule** (see above).

Shared distributional-fidelity metrics (used by B1 and B2): KS statistic $D_{KS}$ and Wasserstein-1 distance between the extrapolated and GT score distributions; a **bimodality** measure (bimodality coefficient, or #modes from a KDE) to quantify recovery of the GT bimodal shape (`fig:VisualAndDistribution`a). **Tail correlation:** Spearman restricted to the top-$x$% and bottom-$x$% of GT scores (the decision-relevant ranks).

---

## B1 — Ranking-preserving score calibration  🟢 *(safest win; deployable, no retraining)*

**Concern:** RpJS-W3, ICLR-mc1X (oversmoothing not fixed); simultaneously clinches bWqr-W1 / myyu-W1 / AC-M1 (ranking, not magnitude, is what pruning uses).
**Coverage: ❌ not addressed** — oversmoothing is only *described* (`ap:Visual`, `fig:VisualAndDistribution`), never mitigated.

**Objective:** Restore the GT score *distribution shape* (bimodality) from a deployable signal, while proving the correction changes **no** pruning decision — thereby demonstrating that oversmoothing is a magnitude artifact, not a ranking failure.

**Procedure (uses only $S_s$ — deployable):**
1. Build the empirical target CDF $\hat F$ from the **subset** ground-truth scores $S_s$ (the only GT available at deployment; valid because $\mathcal{D}_s$ is i.i.d.).
2. Let $G$ be the empirical CDF of the extrapolated scores $S_r$. Define the strictly-monotone map $T=\hat F^{-1}\circ G$ and set $S_r^{\text{cal}}=T(S_r)$. (Implementation: `sklearn.preprocessing.QuantileTransformer` fit to $S_s$, or rank-based histogram matching.)
3. **Sanity check the invariance:** because $T$ is strictly monotone, $\operatorname{argsort}(S_r^{\text{cal}})=\operatorname{argsort}(S_r)$ ⇒ identical top-$k$ ⇒ **retained-set Jaccard $=1.0$ and downstream $\Delta\text{acc}=0$**. Verify empirically and report as the punchline.
4. Measure fidelity **before vs after** calibration against the full GT distribution $S$: $D_{KS}$, Wasserstein-1, bimodality. Overlay histograms (mirror `fig:VisualAndDistribution`a).

**Output:** table (dataset, metric, subset) × {$D_{KS}$ before/after, W1 before/after, bimodality before/after, Jaccard of retained set (=1.0), $\Delta$acc (=0)} + a before/after histogram figure.

**Success criterion:** large drop in $D_{KS}$/W1 and restored bimodality, with selection and accuracy provably unchanged.

**Effort:** 🟢 CPU, hours. Zero risk (cannot degrade results by construction).

**Rebuttal snippet (RpJS/bWqr):** "Oversmoothing affects score *magnitudes*, not the *ranking*: a ranking-preserving recalibration to the $S_s$-estimated distribution reduces KS from … to … and restores bimodality **without changing a single pruning decision** (retained-set Jaccard = 1.0, $\Delta$acc = 0) — direct evidence that pruning depends on rank, not exact scores."

---

## B2 — Residual / initial-residual GNN  🟡 *(concrete methodological novelty)*

**Concern:** RpJS-W1 (limited novelty), RpJS-W3 / ICLR-mc1X (oversmoothing), AC-M2.
**Coverage: ❌ not addressed** — currently only named as future work in the Outlook (Sec. 5). Running it converts a promise into a result.

**Objective:** Reduce GNN oversmoothing so tail correlation and distributional fidelity improve, and enable deeper GNNs without correlation collapse.

**Procedure (graph regression on saved embeddings — no task-model retraining):**
- **Baseline:** current 3-layer GCN (hidden 512, 256), dropout 0.5, neighbor sampling, 25 epochs, checkpoint by $S_s$-val Pearson (per App. B.4).
- **Variants (everything else fixed):**
  - **(a) Residual GCN:** $H^{l+1}=\sigma(\hat A H^l W^l)+H^l$ (linear projection if dims differ / pre-activation residual).
  - **(b) GCNII** [Chen et al. 2020]: initial-residual + identity mapping $H^{l+1}=\sigma\big(((1-\alpha)\hat A H^l+\alpha H^0)((1-\beta_l)I+\beta_l W^l)\big)$, with $\alpha\in\{0.1,0.2\}$, $\beta_l=\log(\lambda/l+1)$; select on $S_s$-val.
  - **(c) Depth ablation:** 2/3/4/8 layers for vanilla vs residual, to show residuals prevent the collapse.
- **Metrics on $\mathcal{D}_r$ vs $S$:** global Pearson/Spearman, **tail-Spearman**, $D_{KS}$/bimodality (link to B1), and downstream accuracy on **small datasets** (CIFAR-10, synthetic) at 2–3 rates.

**Output:** rows = {GCN (vanilla), +Residual, GCNII, GCNII-deep}; cols = Pearson, Spearman, tail-Spearman, $D_{KS}$-to-GT, Acc(CIFAR/synthetic). Plus a "correlation vs #layers" plot (vanilla collapses, residual doesn't).

**Success criterion:** residual variants raise **tail** correlation and lower $D_{KS}$ vs vanilla GNN, at ≥ parity downstream, and sustain depth without collapse.

**Effort:** 🟡 a handful of GNN runs on saved embeddings (minutes–hours each) + small-dataset downstream eval. **Headline only if it clearly helps**; otherwise report as a positive ablation.

**Rebuttal snippet (RpJS):** "Addressing your novelty/oversmoothing point, an initial-residual GNN (GCNII) raises tail-Spearman from … to … and cuts KS-to-GT by …%, while supporting up to 8 layers without the correlation collapse the vanilla GCN suffers — a concrete architectural remedy rather than only a caveat."

---

## B3 — Uncertainty-aware extrapolation (error prediction + safeguard/hybrid)  🟡 *(most compelling; part 1 is free)*

**Concern:** bWqr-S5 (why it degrades), myyu-W2 (OOD/behavior preservation), AC-M3; also novelty (AC-M2).
**Coverage: ❌ not addressed** — error concentration on atypical/OOD samples is described qualitatively (`ap:Visual`) but uncertainty is never *quantified* or *used*.

**Objective:** (1) Show predictive uncertainty predicts *where* extrapolation errs; (2) exploit it to protect those samples, improving tail/worst-group behavior for little or no extra budget.

### Part 1 — Uncertainty as an error predictor (post-hoc, cheap) 🟢
- **GNN uncertainty (free):** MC-dropout — $T{=}20\text{–}50$ stochastic forward passes with the existing dropout $0.5$; per-sample $u_i=\operatorname{Var}_t[\hat S_i^{(t)}]$.
- **KNN uncertainty:** distance-weighted variance of the $k$ neighbor scores (natural companion to the Eq. 6 mean).
- **Validate:** with offline GT $S$ on $\mathcal{D}_r$, compute per-sample error $e_i=|\,\hat S_i-S_i\,|$ and report (a) Spearman$(u_i,e_i)$ and (b) **AUROC** of $u_i$ detecting top-decile errors. Cross-tab $u_i$ against the atypical/OOD flags from `ap:Visual`.

### Part 2 — Uncertainty-aware selection (small-dataset retrain) 🟡
- **Policy A — safeguard (no new scores):** always **retain** samples with $u_i>\tau$ (protect uncertain samples from pruning); fill the remaining budget by extrapolated score. Threshold $\tau$ set on $S_s$-val. Evaluate downstream + **tail/worst-group** accuracy vs vanilla extrapolation.
- **Policy B — active hybrid (optional, more expensive):** move the top-$\varepsilon$% highest-$u_i$ samples into $\mathcal{D}_s$ and compute their *true* scores. **Honesty caveat:** DU/TDDS scores require a sample to be *in training*, so obtaining true scores means (re)training a scored model that includes them — an active-learning cost, not post-hoc; keep $\varepsilon$ small (1–5%).
- **Report a budget–quality curve:** accuracy and tail accuracy vs extra scoring budget $\varepsilon$ (Policy A at $\varepsilon{=}0$, Policy B at $\varepsilon{=}1,5\%$).

**Output:** (1) uncertainty→error table {Spearman, AUROC} for KNN & GNN; (2) selection table rows = {Extrapolation, +Safeguard($\tau$), +Hybrid($\varepsilon{=}1\%,5\%$)}, cols = {acc, tail/worst-group acc, error-set overlap with GT (ties to Item 2), extra budget}; (3) budget–quality curve.

**Success criterion:** AUROC$(u\!\to\!$error$)\gg0.5$; safeguard improves tail/worst-group at ~no cost; hybrid closes most of the GT gap at small $\varepsilon$ — a clean "little extra budget on the *right* samples" story.

**Effort:** Part 1 🟢 (free via existing dropout); Part 2 Policy A 🟡 (small-dataset retrains); Policy B 🟡/🔴.

**Rebuttal snippet (myyu/bWqr):** "Predictive uncertainty (MC-dropout) predicts extrapolation error with AUROC …, concentrating exactly on the atypical/OOD samples of App. `ap:Visual`. Simply retaining high-uncertainty samples raises worst-group accuracy by …pp at no extra scoring cost; scoring the top-1% most-uncertain samples closes …% of the remaining gap to ground truth."

---

## B-item quick reference
| ID | What | Coverage | Effort | Primary concerns | Rebuttal risk |
|---|---|---|---|---|---|
| B1 | Ranking-preserving calibration | ❌ | 🟢 | RpJS-W3, bWqr-W1, myyu-W1, AC-M1 | none (selection unchanged) |
| B2 | Residual / GCNII GNN | ❌ | 🟡 | RpJS-W1/W3, AC-M2 | low (report as ablation) |
| B3 | Uncertainty-aware (error pred + safeguard/hybrid) | ❌ | 🟢 (Pt1) / 🟡 (Pt2) | bWqr-S5, myyu-W2, AC-M2/M3 | low–moderate (frame as extension) |

**Recommended order:** B1 (free, two-reject payoff) → B3 Part 1 (free via existing dropout) → B3 Part 2 Policy A → B2 (stretch, headline only if it clearly helps).
