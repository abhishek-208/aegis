# Aegis FL Project — Research Paper References
*Organized by concept area. Each entry maps directly to a feature or technique used in the project.*

---

## 1. Federated Learning — Foundation

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 1 | **Communication-Efficient Learning of Deep Networks from Decentralized Data** | H.B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas | AISTATS 2017 | Introduces **FedAvg** — the baseline aggregation algorithm our project compares against. |
| 2 | **Federated Optimization in Heterogeneous Networks (FedProx)** | T. Li, A.K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith | MLSys 2020 | Addresses convergence under heterogeneity with a proximal term; relevant to our LR decay and momentum stabilization. |

---

## 2. Byzantine-Resilient Aggregation (Baselines we compare against)

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 3 | **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum)** | P. Blanchard, E.M. El Mhamdi, R. Guerraoui, J. Stainer | NeurIPS 2017 | Introduces **Krum / Multi-Krum** — selects updates closest to neighbors. Directly implemented in `aggregator.py`. |
| 4 | **Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates** | D. Yin, Y. Chen, R. Kannan, P. Bartlett | ICML 2018 | Introduces **Coordinate-wise Median** and **Trimmed Mean** — both provably robust aggregators. CW-Median is implemented in `aggregator.py`. |

---

## 3. Server-Side Momentum (FedAvgM)

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 5 | **Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification** | T.-M.H. Hsu, H. Qi, M. Brown | arXiv 2019 | Introduces **FedAvgM** — server momentum to stabilize training under non-IID data. Directly implemented in `server.py` as `SERVER_MOMENTUM`. |

---

## 4. Non-IID & Heterogeneous Data

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 6 | **Federated Learning with Non-IID Data** | Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, V. Chandra | arXiv 2018 | Foundational study showing FedAvg accuracy drops up to 55% on non-IID data. Motivates our Non-IID and Dirichlet partitioning. |
| 7 | **Federated Learning on Non-IID Data Silos: An Experimental Study** | Q. Li, Y. Diao, Q. Chen, B. He | IEEE ICDE 2022 | Studies Dirichlet-based label-skew partitioning (α parameter). Directly used in our `data_utils.py` for `DIRICHLET_ALPHA`. |

---

## 5. Cosine Similarity–Based Defenses

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 8 | **The Limitations of Deep Learning in Adversarial Settings (FoolsGold)** | C. Fung, C.J.M. Yoon, I. Beschastnikh | arXiv 2018 / IEEE S&P 2020 | Uses cosine similarity of gradient histories to reduce Sybil attacker learning rates. Inspires Aegis's cosine penalty. |
| 9 | **CONTRA: Defending Against Poisoning Attacks in Federated Learning** | A. Awan, B. Li, et al. | ESORICS 2021 | Cosine-similarity credibility scoring + dynamic reputation. Closely related to our credit score + cosine penalty formulation. |
| 10 | **CosDefense: Cosine-Similarity Based Attacker Detection** | — | arXiv 2023 | Detects malicious clients via last-layer cosine similarity with the global model; complementary to our whole-model approach. |

---

## 6. Byzantine Attack Types

### 6a. Sign Flip Attack

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 11 | **Detecting Poisoning Nodes in Federated Learning by Ranking Gradients** | — | arXiv / IEEE 2023 | Formally defines the sign-flip attack and proposes gradient-ranking detection. Directly tests our `sign_flip` implementation. |
| 12 | **A Little Is Enough: Circumventing Defenses For Distributed Learning** | M. Baruch, G. Baruch, Y. Goldberg | NeurIPS 2019 | Shows that even small-magnitude attacks (like scaled sign flip) can circumvent Krum and Trimmed Mean. Motivates Aegis's dual filtering. |

### 6b. Label Flip Attack

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 13 | **LFighter: Defending Against the Label-Flipping Attack in Federated Learning** | N.T. Jebreel, J. Domingo-Ferrer, D. Sánchez, A. Blanco-Justicia | Neural Networks 2024 | Deep analysis of label-flip behavior + gradient-cluster defense. Validates our `label_flip` threat model. |
| 14 | **Analyzing Federated Learning through an Adversarial Lens** | A.N. Bhagoji, S. Chakraborty, P. Mittal, S. Calo | ICML 2019 | Studies model poisoning via label manipulation in FL; shows existing aggregators are more vulnerable than expected. |

### 6c. Additive Noise / Model Poisoning

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 15 | **Local Model Poisoning Attacks to Byzantine-Robust Federated Learning** | M. Fang, X. Cao, J. Jia, N.Z. Gong | USENIX Security 2020 | Systematic study of model poisoning optimized to defeat Krum, Trimmed Mean, etc. Validates our `additive_noise` delta-scaled attack. |

### 6d. Sybil / Volume Spam Attack

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 16 | **Sybil-based Data Poisoning Attacks in Federated Learning** | — | arXiv 2023 | Studies Sybil nodes amplifying attacker influence via fake identities and inflated sizes. Motivates our `volume_spam` + Volume Clipping defense. |
| 17 | **Free-rider Attacks on Model Aggregation in Federated Learning** | J. Lin, M. Du, J. Liu | AISTATS 2021 | Studies clients faking meaningful contributions. Related to our volume clipping as a general defense against inflated participation. |

---

## 7. Differential Privacy in Deep Learning

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 18 | **Deep Learning with Differential Privacy (DP-SGD)** | M. Abadi, A. Chu, I. Goodfellow, H.B. McMahan, I. Mironov, K. Talwar, L. Zhang | CCS 2016 | Introduces DP-SGD with gradient clipping + Gaussian noise. Our Local/Central DP modes in `client.py` / `server.py` follow this framework. |

---

## 8. Adaptive Thresholding & Outlier Detection

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 19 | **Robust Statistics for Signal Processing** (MAD) | P.J. Huber, E.M. Ronchetti | Wiley 2009 (Book) | Classical reference for **Median Absolute Deviation** as a robust spread estimator. Foundation of our MAD-based rejection threshold. |
| 20 | **Identifying Outliers Using the Hampel Identifier** | — | Multiple classical stats references | The Hampel X84 method flags outliers at `median ± k × MAD`. Directly used in our adaptive threshold: `T = E_median + (κ × D)`. |

---

## 9. Convergence & Learning Rate Decay

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 21 | **Adaptive Federated Optimization (FedAdam / FedAdaGrad / FedYogi)** | S.J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konečný, S. Kumar, H.B. McMahan | ICLR 2021 | Studies server-side adaptive optimizers and decay in FL. Our server LR + decay design draws from these principles. |

---

## 10. Surveys & Overviews (Recommended for Background)

| # | Paper | Authors | Venue / Year | Relevance |
|---|-------|---------|-------------|-----------|
| 22 | **Advances and Open Problems in Federated Learning** | P. Kairouz, H.B. McMahan et al. (58 authors) | Foundations & Trends in ML 2021 | Comprehensive 200+ page survey on FL. Covers nearly every concept in our project. Essential background reading. |
| 23 | **A Survey on Security and Privacy of Federated Learning** | P. Lyu, X. Han, et al. | Future Generation Computer Systems 2022 | Surveys Byzantine attacks, data/model poisoning, and defense mechanisms in FL. |
| 24 | **Byzantine-Robust Aggregation in Federated Learning Empowered Industrial IoT** | Z. Ma, J. Ma et al. | IEEE TII 2022 | Compares Krum, Trimmed Mean, CW-Median, and credit-scoring aggregators in industrial settings. |
