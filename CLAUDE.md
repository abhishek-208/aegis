# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch **Federated Learning simulation** built to develop and benchmark **Aegis**, a Byzantine-resilient aggregation rule, against an attack suite and five baseline defenses. The whole system is a single-process simulator (no real networking): clients, server, and attacks are all Python objects driven by one experiment runner. The research target is **CIFAR-10 under 4-shard Non-IID data with ~30% Byzantine clients**.

There is no separate paper in this repo; the design rationale and results history live in Claude's persistent memory (`aegis-protocol-evolution`, `presentation-build-setup`) and in `Aegis_Reference_Papers.md`.

## Running

There is no build step, test suite, or linter. Everything runs through one entry point:

```powershell
python main.py        # runs whatever EXPERIMENT_CONFIGS / toggles config.py selects
```

- **All behavior is controlled by editing [config.py](config.py)**, not CLI flags. There are no command-line arguments.
- `python setup_data.py` — one-time dataset download/partition prep (CIFAR-10).
- `python profile_complexity.py` — standalone O(kd) complexity verification plots.
- `run_modal.py` — runs the simulation on Modal cloud GPUs (`modal run --detach run_modal.py::execute_aegis`); results land in the `aegis-results` Modal volume.
- Output (logs, plots, saved models) goes to `RESULTS_DIR`, which auto-selects by environment in [config.py](config.py): `D:\IITD\MTP 2\Results` on Windows, `/kaggle/working/...` on Kaggle, `./saved_models` on Modal, `./results` on Linux lab server.

To run a single scenario, set the `'run'` flag to `1`/`0` on entries in `EXPERIMENT_CONFIGS` in [main.py](main.py), and set `ATTACK_TYPE` / `FRACTION_BYZANTINE` / `DATA_SPLIT_TYPE` in [config.py](config.py).

## Experiment-mode toggles (config.py §2)

Exactly one of these "macro" modes should be enabled at a time; if all are `False`, the manual `EXPERIMENT_CONFIGS` list in [main.py](main.py) runs instead:

- `COMPARE_AEGIS_SCENARIOS` — Aegis vs. baselines across data splits/attacks.
- `RUN_ABLATION_STUDY` — toggles individual Aegis components (the `ablate_*` args of `aegis()`).
- `ABLATION_NON_IID_SWEEP` / `ABLATION_BYZANTINE_SWEEP` — sweep shards-per-client or attacker fraction.
- `MULTI_SEED_EVAL` — 3-seed reproducibility (`EVAL_SEEDS`).
- `ABLATION_PARAM_SWEEP` — sweep one Aegis hyperparameter (`SWEEP_PARAM` ∈ `variance_sensitivity`, `pass1_cos_threshold`, `cosine_penalty_weight`).

## Architecture / data flow

One round flows: **`main.run_simulation` → `Server.run_round` → `Client.train` (×N) → attack injection → `aggregator_func` → global model update → `Server.evaluate`.**

- **[main.py](main.py)** — experiment runner. Defines `EXPERIMENT_CONFIGS` (a list of dicts: `aggregator`, `attack_type`, `fraction_byzantine`, `data_split`, …), expands the macro sweep modes into config lists, calls `run_simulation()` per experiment, collects per-round metrics, and hands everything to the plotter. Baselines that need extra args (Krum, Bulyan) are bound with `functools.partial`. Also contains the `Tee` stdout-to-logfile helper.
- **[server.py](server.py)** — `Server` orchestrates a round: client selection (`MIN/MAX_CLIENTS_PER_ROUND`), decides per-round attackers (a **fixed** Byzantine set chosen once in `__init__`, gated by `ATTACK_PROBABILITY`), runs local training (serial GPU or `ThreadPoolExecutor` when `MAX_PARALLEL_CLIENTS > 1`), **injects coordinated attacks** (ALIE, IPM, Sybil cloning — see below), calls the aggregator, optionally applies FedAvgM server momentum (Aegis-only), and computes filter diagnostics (TP/FP/FN/TN, detection rate, precision). Also runs PCA on the update matrix for scatter visualization.
- **[client.py](client.py)** — `Client.train()` does one round of local SGD, then `apply_attack()` corrupts the resulting weights for Byzantine clients. Returns `(client_id, weights_dict, n_samples)`.
- **[aggregator.py](aggregator.py)** — all six aggregation rules: `fed_avg`, **`aegis`**, `cw_med`, `multi_krum`, `bulyan`, `fools_gold`. Each returns `(new_global_state_dict, stats_or_None)`. Holds two module-level state dicts reset between experiments: `AEGIS_REPUTATION` (cross-round EMA) and `FOOLSGOLD_HISTORY` (gradient history).
- **[model.py](model.py)** — `get_model()` dispatches on `MODEL_TYPE`: `MLP` (MNIST), `CNN` (LeNet-5), `ImprovedCNN` (CIFAR-10, ~290K params, BatchNorm + Dropout — the default).
- **[data_utils.py](data_utils.py)** / **[setup_data.py](setup_data.py)** — dataset loading and partitioning (`BALANCED_IID`, `UNBALANCED_IID`, `NON_IID` shard split via `SHARDS_PER_CLIENT`).
- **[plotter.py](plotter.py)** / **[replot_from_log.py](replot_from_log.py)** — accuracy curves (EMA-smoothed), filter-diagnostic bars, PCA scatters; IEEE publication formatting toggles. `replot_from_log.py` regenerates plots from a saved log without rerunning.

### Important cross-cutting conventions

- **Updates are `(client_id, weights_dict, n_samples)` tuples** everywhere. Weights are always moved to **CPU** before leaving `Client.train` (avoids CUDA pickling issues under multiprocessing).
- **Aegis scores on deltas, aggregates on weights.** Anomaly scores (Euclidean, cosine) are computed on `delta_i = w_i − w_global`, but the final weighted average is over raw `w_i` to reconstruct the model. The `global_model` is passed into the aggregator as a kwarg; `server.run_round` calls the aggregator with `current_round=` and `global_model=` and falls back via `try/except TypeError` for baselines that don't accept them.
- **BatchNorm stats are special-cased.** `num_batches_tracked` must be cast back to `Long` after float averaging; server momentum **skips** `running_mean`/`running_var`/`num_batches_tracked` to avoid driving `running_var` negative → `NaN`.
- **Attacks are split between client and server.** Per-client corruptions (`sign_flip`, `*_noise`, `orthogonal`, and label-flip during training for `label_flip`/`sybil`/`volume_spam`) happen in [client.py](client.py). **Coordinated, cross-client attacks (`alie`, `ipm`) and Sybil cloning are orchestrated centrally in [server.py](server.py)** because they need statistics over all/other clients' deltas. `volume_spam` is honest weights with a falsified `n_samples` (1e9).

## The Aegis algorithm (aggregator.py `aegis()`)

Two-pass robust aggregation, O(kd) in clients×dim. Steps:

1. **Compute deltas** against the passed-in global model.
2. **Pass 1 — directional decontamination:** preliminary coordinate-wise median → cosine similarity → drop clients with `cos < PASS1_COS_THRESHOLD` **from the median computation only** (not a rejection gate). Removes sign-flip/IPM bias from the reference.
3. **Pass 2 — debiased median** from the cleaned pool; everyone is re-scored against it (Euclidean distance + cosine penalty `P_k = 1 − cos`).
4. **Adaptive threshold** `T = median_dist + k·MAD`, where `k` combines round-based decay (`K_MAX→K_MIN` over `WARMUP_ROUNDS`) and variance-normalization (`VARIANCE_SENSITIVITY`), floored at `K_SAFE_FLOOR`. **The Euclidean filter is the only hard rejection gate.**
5. **Credit score:** `score_k = clipped_volume_k / (E_k + α·P_k + λ·R_k + ε)`, normalized to sum to 1. `α = COSINE_PENALTY_WEIGHT`, `λ = REPUTATION_WEIGHT`. `R_k` is a cross-round EMA of cosine penalty (`R ← γR + (1−γ)P`, `γ = REPUTATION_DECAY`) — the **ALIE defense**. A `CREDIT_WARMUP_ROUNDS` ramp softens credit scoring in early rounds.
6. **Aggregate** approved weights by `score_k`.

Key design decisions (the "why"), captured in memory `[[aegis-protocol-evolution]]`: the hard cosine gate was replaced by the soft penalty (it amputated honest Non-IID specialists); EMA reputation was added to catch stealth attackers; variance must be computed on the *unfiltered* set to avoid a "variance-collapse death spiral." Aegis is dominant on label-flip, competitive on sign-flip/IPM/Sybil, but **fails on ALIE** and collapses past `f≈0.40` (median breakdown point).

The `ablate_*` parameters of `aegis()` disable individual components for the ablation study (volume clipping, directional Pass-1, cosine penalty, adaptive threshold, Euclidean filter).

## Attack suite (config `ATTACK_TYPE`)

`none`, `sign_flip`, `additive_noise` / `pure_additive_noise` / `catastrophic_noise`, `label_flip`, `orthogonal`, `volume_spam`, `sybil`, `alie` ("A Little Is Enough", Baruch 2019 — coordinated `μ − zσ`), `ipm` ("Fall of Empires", Xie 2020 — `−ε·μ`). Per-attack parameters (`ALIE_Z`, `ALIE_USE_OMNISCIENT`, `IPM_EPSILON`, `IPM_USE_OMNISCIENT`, `NUM_SYBILS_PER_ATTACKER`, `ATTACK_NOISE_STD`) are in config §8.

## Gotchas

- `DATALOADER_WORKERS` **must be 0** if `MAX_PARALLEL_CLIENTS > 1` (thread oversubscription).
- Bulyan requires `n ≥ 4f + 3`; the experiment entry overrides `fraction_byzantine=0.2` so the constraint holds for ~15 clients/round — it will assert-fail at higher fractions.
- Always reset state between experiments via `reset_aegis_reputation()` / `reset_foolsgold_history()` (done in `main.run_simulation`).
- `RESULTS_DIR` is environment-detected; on a new machine confirm the branch in config §11 resolves to a writable path.
