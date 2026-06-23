# Peer Review & Revision Prompt for the AEGIS Paper

You are acting as a senior peer reviewer for a top-tier venue (IEEE S&P / USENIX Security / NDSS) reviewing a paper on Byzantine-resilient Federated Learning. The paper introduces "AEGIS," a novel robust aggregation protocol. Below is a detailed, section-by-section review identifying factual errors, missing content, overclaimed results, structural weaknesses, and specific revision instructions. Your task is to revise the paper to address every point below while maintaining academic rigor and intellectual honesty.

---

## CRITICAL ISSUE #1: The Abstract and Introduction Overclaim ALIE and IPM Resilience

The abstract states: "Aegis effectively neutralizes both naive (Sign-Flipping) and optimization-based (ALIE, IPM) attacks."

This is factually false based on the actual experimental results. The empirical data shows:
- ALIE (z=1.0, omniscient): AEGIS collapses to ~10% accuracy (random guessing). Detection rate = 0.0% across all rounds. Every ALIE attacker passes both the cosine filter and the Euclidean filter in every single round. The model never learns.
- IPM (ε=1.0, omniscient): AEGIS collapses to ~10% accuracy within 100 rounds. The two-pass median decontamination helps at large ε (where cosine ≈ -1 catches attackers), but the original single-pass version failed completely.
- Sybil (with 2 clones per attacker): AEGIS degrades to ~23-28% accuracy. The model survives but is severely impaired.

The paper MUST NOT claim resilience against attacks that demonstrably defeat AEGIS. This is the single most damaging issue — a reviewer who runs the code or checks the results will immediately reject the paper for dishonesty.

**Revision instruction:** Rewrite the abstract and introduction to accurately characterize AEGIS's resilience hierarchy:
- Strong resilience: Sign-flip, label-flip, additive noise, orthogonal, volume spam (~70-75% accuracy, within 2-5% of no-attack baseline)
- Partial resilience: Sign-flip with median contamination (improved by two-pass decontamination)
- Known limitations: ALIE and IPM defeat AEGIS's per-round statistical filtering, consistent with Baruch et al.'s theoretical impossibility result for coordinate-wise defenses

Frame ALIE/IPM honestly as: "We characterize the fundamental boundary of per-round statistical filtering and show that AEGIS's dual-metric approach, while effective against conventional attacks, cannot detect variance-envelope attacks (ALIE) that are theoretically proven to evade all coordinate-wise filters."

---

## CRITICAL ISSUE #2: The Results Section Shows Only 2 Figures and Omits Most Experimental Data

Section VI presents only two figures:
- Fig 1: AEGIS across 4 attack types (sign-flip, label-flip, additive noise, orthogonal)
- Fig 2: AEGIS vs FoolsGold under sign-flip

This is severely insufficient for a research paper. The following are MISSING and MUST be added:

### Missing Comparison Table (most important)
A table comparing AEGIS against ALL baselines (FedAvg, Multi-Krum, CWMed, Bulyan, FoolsGold, FLTrust) across ALL attack types, showing final accuracy. This is the standard format in every Byzantine FL paper (see FLTrust Table 1, FLAME Table 2, Shejwalkar & Houmansadr Table 1). Without this table, the paper has no comparative evaluation.

### Missing Results That Must Be Added
1. FedAvg under attack (the "no defense" baseline) — reviewers need to see that FedAvg collapses to 10% under sign-flip to appreciate that AEGIS's 63-75% is meaningful
2. AEGIS vs Multi-Krum, CWMed, Bulyan under the same attacks
3. ALIE results (honestly showing failure at 10%)
4. IPM results across ε ∈ {0.01, 0.1, 0.5, 1.0}
5. Sybil results showing partial survival at 23-28%
6. Volume spam results showing successful defense at 72-73%
7. Ablation study: Full AEGIS vs each component removed (no Euclidean filter, no directional filter, no cosine penalty, no volume clipping, no adaptive threshold)
8. Byzantine fraction sweep: 10%, 20%, 30%, 40% showing degradation curve

### Missing Statistical Rigor
The paper reports zero confidence intervals and appears to show single-run results. Standard practice requires 3-5 independent runs with mean ± standard deviation reported. At minimum, acknowledge single-run results as a limitation.

---

## CRITICAL ISSUE #3: The Paper Does Not Mention the Two-Pass Median Decontamination

The actual AEGIS implementation uses a two-pass robust center computation:
- Pass 1: Compute preliminary median over all clients, remove clients with cos_sim < 0
- Pass 2: Recompute median from cleaned pool, then run full dual-metric filtering against the debiased median

This is a significant algorithmic contribution (inspired by Bulyan's two-phase architecture and FLAME's cosine-based pre-screening, but maintaining O(kd) complexity). It solves the median contamination problem under sign-flip and large-ε IPM attacks. The paper describes only the single-pass version in Section IV, which does not match the actual implementation.

**Revision instruction:** Update Section IV-B (Step 2) and IV-D (Step 4) to describe the two-pass process. Add a paragraph explaining the median contamination problem (Byzantine entries bias the coordinate-wise median, causing false positives among honest non-IID clients) and how the two-pass approach resolves it. Cite Bulyan [9] and FLAME [8] as architectural precedents while noting AEGIS's O(kd) advantage.

---

## CRITICAL ISSUE #4: The Paper Does Not Mention the Reputation System

The actual implementation includes a cross-round reputation tracking mechanism:
- Per-client EMA of cosine penalty across rounds
- Integrated into the credit score denominator as λ × R_k
- Designed to suppress ALIE-like clients with consistently elevated cosine penalties

Even though this mechanism did not successfully defeat omniscient ALIE (because the model dies before reputation can accumulate signal), it IS part of the implemented protocol and represents an algorithmic contribution worth discussing. It also provides partial mitigation under colluding ALIE (weaker attacker model).

**Revision instruction:** Add a subsection (e.g., IV-F "Cross-Round Reputation Tracking") describing the mechanism, its mathematical formulation (EMA update rule, steady-state analysis), its integration into the credit score, and honestly state that it provides marginal improvement under colluding ALIE but cannot overcome omniscient ALIE due to the dead-model feedback loop.

---

## ISSUE #5: Section IV-E Volume Clipping Formula Discrepancy

The paper states (Eq. 12): v_k = min(n_k, 2.0 × n_avg)

The actual implementation uses the MEDIAN of approved clients' data sizes, not the average:
robust_median_size = torch.median(approved_data_sizes)
approved_clipped_sizes = torch.clamp(approved_data_sizes, max=2.0 * robust_median_size)

Using the median is a strictly better design (the mean can be inflated by a single volume spammer who passes the filter, the median cannot). Update Eq. 12 to reflect the actual implementation and explain why the median is more robust than the mean for this purpose.

---

## ISSUE #6: Hyperparameter Table Does Not Match Current Implementation

Section V-C states K_safe_floor = 4.5. The actual current implementation uses K_safe_floor = 3.0. The paper must report the actual values used in the experiments whose results are shown.

Additionally, the paper does not mention:
- REPUTATION_ENABLED = True, REPUTATION_DECAY = 0.95, REPUTATION_LAMBDA = 20.0
- The two-pass median decontamination
- Batch size = 32 (critical for reproducibility)
- GroupNorm (not BatchNorm) in the CNN architecture
- RANDOM_SEED = 42 for reproducibility

**Revision instruction:** Add a complete hyperparameter table listing every parameter value, or provide these in an appendix. Reproducibility is a minimum requirement for any venue.

---

## ISSUE #7: The Model Architecture Section Is Vague

Section V-A says "The local models utilize a Convolutional Neural Network (CNN) architecture" without specifying:
- Number of layers, channels, kernel sizes
- GroupNorm configuration (groups=2)
- Dropout rate (0.25)
- Parameter count (~290K)
- That GroupNorm was chosen specifically because BatchNorm's running statistics are incompatible with non-IID FL (cite McMahan et al. and the GroupNorm FL literature)

**Revision instruction:** Add a model architecture table or description with full specification. This is standard in every FL paper.

---

## ISSUE #8: The Threat Model Section Needs Correction

Section III-A says "We consider... a swarm of k = 30 clients" and "we utilize a Dirichlet distribution (α = 0.5) to allocate data, restricting each client to a maximum of 4 data shards (classes)."

This conflates two different partitioning methods. The actual implementation uses SHARD-BASED partitioning (McMahan et al. 2017), not Dirichlet. Each client receives exactly 4 randomly assigned shards from a label-sorted dataset. Dirichlet partitioning with α = 0.5 is a different method that produces a continuous distribution of class proportions. The paper should accurately describe which method is used.

Section III-B says "up to 33% of the clients are entirely compromised." The current config uses FRACTION_BYZANTINE = 0.30 (30%, which is 9 out of 30 clients). Verify which fraction was used for the results shown and report it accurately.

---

## ISSUE #9: The Complexity Analysis Claim Needs Nuance

The paper repeatedly claims O(kd) complexity. With the two-pass median decontamination, the actual complexity is O(2kd) = O(kd) asymptotically, but the constant factor doubles. This should be acknowledged: "The two-pass approach doubles the constant factor but preserves the O(kd) asymptotic complexity."

Additionally, the paper should include an empirical timing comparison table showing wall-clock aggregation time per round for AEGIS vs each baseline, not just the total training time comparison with FoolsGold.

---

## ISSUE #10: Missing "Limitations" Section

Every rigorous paper needs an explicit limitations section. Based on the actual experimental results, the following limitations MUST be acknowledged:

1. ALIE with omniscient knowledge completely defeats AEGIS (cite Baruch et al. impossibility result)
2. IPM with small ε produces ALIE-like stealth that evades detection
3. Sybil attacks that inflate the Byzantine fraction beyond 50% overwhelm the median-based defense
4. The reputation system cannot bootstrap in environments where the model dies before reputation signal accumulates
5. Under extreme non-IID (2 shards per client), false positive rates increase significantly
6. Results are from single runs without confidence intervals

---

## ISSUE #11: Missing "Future Work" Content

The current future work section mentions only SMPC integration, which is tangential. The actual future work should discuss:

1. Cross-round temporal memory (reputation system) as a path toward ALIE resilience
2. Integration with spectral methods (DnC) for detecting correlated perturbation patterns
3. Server-side validation (FLTrust/Zeno hybrid) as a complement to statistical filtering
4. Sybil identity verification mechanisms beyond gradient-level detection
5. Testing on larger models (ResNet-18), more datasets (CIFAR-100, FEMNIST), and Dirichlet partitioning
6. Multiple-run statistical evaluation with confidence intervals

---

## ISSUE #12: The Related Work Section Is Missing Key Recent References

Add the following references that are directly relevant:
- FLAME (Nguyen et al., USENIX Security 2022) — already cited as [8] but not discussed in related work, despite being the closest precedent for AEGIS's cosine-based pre-screening
- DnC / Divide and Conquer (Shejwalkar & Houmansadr, NDSS 2021) — spectral defense that specifically addresses ALIE
- "Do We Really Need to Design New Byzantine-robust Aggregation Rules?" (NDSS 2025) — recent meta-analysis of defense effectiveness
- Hsu et al. "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" (NeurIPS Workshop 2019) — the FedAvgM server momentum paper that AEGIS's Step 6 is based on

---

## STRUCTURAL REVISION: Suggested Paper Outline

The revised paper should follow this structure:

I. Introduction (with honest characterization of capabilities and limitations)
II. Related Work (expanded with FLAME discussion, DnC, FedAvgM)
III. Threat Model (corrected data partitioning description, accurate Byzantine fraction)
IV. The AEGIS Protocol
   A. Distribution & Local Training
   B. Two-Pass Robust Center (NEW: describe both passes)
   C. Dual Anomaly Scoring
   D. Adaptive Thresholding & Gatekeeper Logic
   E. Volume Bounding & Credit Scoring (corrected formula)
   F. Cross-Round Reputation Tracking (NEW)
   G. Aggregation & Server Momentum
V. Experimental Setup (with complete hyperparameter table, model architecture)
VI. Results and Discussion
   A. AEGIS Resilience Across Attack Types (expanded with all attacks including ALIE, IPM, Sybil, volume spam)
   B. Comparative Evaluation vs Baselines (table comparing against FedAvg, Krum, CWMed, Bulyan, FoolsGold)
   C. Ablation Study (each component removed individually)
   D. Byzantine Fraction Sensitivity (10%-40% sweep)
   E. Computational Efficiency (per-round timing comparison)
VII. Limitations (NEW: honest discussion of ALIE/IPM/Sybil failures)
VIII. Conclusion and Future Work (expanded)

---

## WHAT THE PAPER DOES WELL (preserve these strengths)

1. The algorithm description (Section IV) is mathematically rigorous and clearly written
2. The O(kd) complexity argument is well-motivated and important
3. The five pillars of novelty in the introduction are clearly articulated
4. The fixed-adversary with random-participation model (Section V-B) with Hypergeometric distribution analysis is a thoughtful experimental design choice
5. The reference list covers the key papers in the field
6. The dual-metric insight (combining Euclidean distance and cosine similarity) is genuinely novel

---

## REVISION PRIORITY ORDER

1. Remove false claims about ALIE/IPM resilience (CRITICAL — paper will be rejected otherwise)
2. Add comprehensive comparison table against all baselines
3. Add two-pass median description and reputation system
4. Add ablation study results
5. Add limitations section
6. Fix technical discrepancies (volume clipping formula, K_safe_floor value, data partitioning method)
7. Add model architecture details and complete hyperparameter table
8. Expand future work
9. Add ALIE/IPM/Sybil results with honest discussion
10. Add statistical rigor (multiple runs or acknowledge as limitation)
