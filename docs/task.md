# Task Plan (task.md)

A sequenced, bite‑size plan. Each task is **small**, **testable**, has a clear **start** and **end**, and focuses on **one** problem.

> Conventions: run from repo root. Python >=3.10. Use `pytest -q`. All scripts use the `src/mmrl` layout from `architecture.md`.

---

## Phase 0 — Repository Bootstrap

### 0.1 Create repo skeleton
**Start:** empty folder.  
**Do:** create directories and stub files per `architecture.md` minimal set.  
**End:** tree matches minimal skeleton; `src/mmrl/__init__.py` exists.

**Test:**
- `python -c "import mmrl; print('ok')"` prints `ok`.

---

### 0.2 Packaging & dependencies
**Start:** skeleton present.  
**Do:** add `pyproject.toml` (set module name `mmrl`), `requirements.txt` (gymnasium, numpy, scipy, torch, tensorboard, pyyaml, pandas, pytest).  
**End:** `pip install -e .` works.

**Test:**
- `pip install -e .` succeeds.
- `python -c "import mmrl; print(mmrl.__name__)"` prints `mmrl`.

---

### 0.3 CI smoke (local only)
**Start:** installed package.  
**Do:** add `tests/test_import.py` with one import test.  
**End:** pytest passes.

**Test:**
- `pytest -q` shows 1 passed.

---

## Phase 1 — Core Environment Primitives

### 1.1 Card ranks & utilities
**Start:** empty `cards.py`.  
**Do:** implement rank constants, to/from value maps, sampling without replacement, sum helpers.  
**End:** deterministic sampler with seed.

**Test:**
- Unit: `tests/test_cards.py` verifies 3 draws are unique and in 2..14; sum in [6,42].

---

### 1.2 Event definitions (fixed list) & toggles
**Start:** empty `events.py`.  
**Do:** implement events: `none`, `remap_value(x,y)`, `ge10_only`, `le7_only`, `even_only`, `odd_only`; config `flags.enable_events: bool`; optional `event_persist` with frequencies.  
**End:** function `apply_event(deck, event)` and `sample_event(rng, cfg, last_event)`.

**Test:**
- Unit: `tests/test_events.py` checks each filter; `remap_value` replaces one rank’s value.
- When `enable_events=false`, always returns `none`.

---

### 1.3 Posterior moments for hidden sum
**Start:** empty `quotes.py`.  
**Do:** implement `posterior_mean_var(hints, event_filtered_deck)` using without‑replacement logic; fall back to i.i.d. approx if needed; expose `mu, sigma`.  
**End:** returns correct mean/variance for edge cases (0..3 hints).

**Test:**
- Unit: `tests/test_posterior.py` compares against brute‑force enumeration for small cases.

---

### 1.4 Quote generation (mid, spread, bid/ask)
**Start:** `quotes.py`.  
**Do:** implement `make_quote(mu, sigma, cfg, rng)` with `mid = mu + N(0, sigma_q)`, `spread = clip(s0 + beta*sigma)`, `X=mid-spread/2`, `Y=mid+spread/2`.  
**End:** numerically stable, clipped.

**Test:**
- Unit: `tests/test_quotes.py` asserts spread bounds; mean of many mids approx `mu`.

---

### 1.5 Tier‑2 Liquidity draw & display
**Start:** empty `liquidity.py`.  
**Do:** implement `draw_true_depth(sigma, spread, k, tau, caps)`, `displayed_depth(L_true, L_cap)`.  
**End:** lognormal draw, clipped, display cap applied.

**Test:**
- Unit: `tests/test_liquidity_draw.py` checks monotonicity: higher sigma → lower median depth; display ≤ true.

---

### 1.6 Execution price with impact (no pro‑rata)
**Start:** `liquidity.py`.  
**Do:** implement `exec_price_buy(Y, q_total, L_true, alpha, enable_impact)` and symmetric sell.  
**End:** overflow maps to linear impact; `enable_impact=false` returns quote.

**Test:**
- Unit: `tests/test_impact.py` verifies prices with/without impact and overflow edge cases.

---

### 1.7 Validity & budget constraints
**Start:** empty `execution.py`.  
**Do:** implement `feasible_sizes(side, W, X, Y, S_max)` for long affordability and short worst‑case; action mask builder for 21‑action space.  
**End:** returns boolean mask and `i_max`.

**Test:**
- Unit: `tests/test_validity.py` covers boundary conditions and event‑adjusted `S_max`.

---

## Phase 2 — Single‑Player Environment

### 2.1 Observation & action spaces
**Start:** empty `spaces.py`.  
**Do:** define shapes and encoders: quote scalars, displayed depths, event one‑hot, 13‑dim hint counts, bankroll fraction, time fraction, flag bits.  
**End:** `build_obs(...) -> np.ndarray` and `ACTION_SPACE=21`.

**Test:**
- Unit: `tests/test_spaces.py` checks dimensions and dtype.

---

### 2.2 SingleCardEnv reset()
**Start:** empty `single_env.py`.  
**Do:** implement `reset(seed)` producing first observation with budgets initialized and event sampling obeying `enable_events`.  
**End:** returns obs, info with `mu, sigma, event` for logging.

**Test:**
- Unit: `tests/test_single_reset.py` asserts fields present and valid.

---

### 2.3 SingleCardEnv step(): cards → quote → liquidity → mask → exec → reward
**Start:** `single_env.py`.  
**Do:** implement full step pipeline using modules above; support `flags.enable_impact`.  
**End:** deterministic under fixed seed; info includes `exec_price, S, slippage, masks`.

**Test:**
- Unit: `tests/test_single_step.py` constructs a known scenario and checks payoff math.

---

### 2.4 Logging hooks
**Start:** empty `logging_hooks.py`.  
**Do:** implement minimal hook to produce a dict with observation snapshot, action, mask validity, P&L, `mu, sigma, S`, depths, flags.  
**End:** env returns `info['log']` per step.

**Test:**
- Unit: `tests/test_logging.py` ensures required keys present.

---

## Phase 3 — Baselines

### 3.1 Random‑Valid policy
**Start:** empty `baselines/random_valid.py`.  
**Do:** sample uniformly from valid action mask.  
**End:** exposes `act(obs, mask, rng)`.

**Test:**
- Unit: `tests/test_baseline_random.py` verifies always valid.

---

### 3.2 Level‑0 EV oracle
**Start:** empty `baselines/ev_oracle.py`.  
**Do:** compute direction via `mu` vs `X,Y`, size via `feasible_sizes` then simple rule (e.g., max).  
**End:** returns action id.

**Test:**
- Unit: `tests/test_baseline_ev.py` checks no‑edge → pass; positive edge → correct side.

---

### 3.3 Level‑1 crowding‑aware
**Start:** empty `baselines/level1_crowding.py`.  
**Do:** maintain moving estimate of opponent hit probability per side; if expected overflow vs displayed depth is high, downsize/skip.  
**End:** policy deterministic given history and obs.

**Test:**
- Unit: `tests/test_baseline_level1.py` crafts overflow case → smaller size than Level‑0.

---

## Phase 4 — Two‑Player Environment

### 4.1 TwoPlayerCardEnv reset()
**Start:** empty `two_player_env.py`.  
**Do:** extend Single env to produce two observations (same quote & depths), separate RNG streams for hints if desired; include opponent last action placeholders.  
**End:** returns `(obs_a, obs_b)`.

**Test:**
- Unit: `tests/test_two_reset.py` checks symmetry and independence of private hints if configured.

---

### 4.2 TwoPlayerCardEnv step() with shared liquidity
**Start:** `two_player_env.py`.  
**Do:** aggregate actions per side to `q_buy,q_sell`; compute executed prices with impact toggle; compute each agent P&L; update opponent features.  
**End:** returns next `(obs_a, obs_b)`, `(r_a, r_b)`, and info dicts.

**Test:**
- Unit: `tests/test_two_step.py` verifies crowding causes higher buy exec price and lower sell exec price when impact enabled.

---

## Phase 5 — Agents

### 5.1 DQN agent (single‑player)
**Start:** empty `agents/dqn/`.  
**Do:** implement MLP policy head for 21 actions, masked epsilon‑greedy, replay buffer, target net, Huber loss.  
**End:** `train_dqn.py` runs and logs episode returns.

**Test:**
- Smoke: `pytest -q tests/test_agents_smoke.py::test_dqn_smoke` ensures > Random‑Valid after short training.

---

### 5.2 IPPO (two‑player, parameter sharing)
**Start:** empty `agents/ippo/`.  
**Do:** actor‑critic with shared policy across agents; GAE, PPO clip; entropy bonus; action masking.  
**End:** `train_ippo.py` trains in two‑player env.

**Test:**
- Smoke: `pytest -q tests/test_agents_smoke.py::test_ippo_smoke` beats Random‑Valid.

---

### 5.3 MAPPO (centralized critic)
**Start:** empty `agents/mappo/`.  
**Do:** centralized critic gets concatenated obs (or obs + last actions), decentralized actors identical to IPPO; training loop.  
**End:** `train_mappo.py` runs and improves over IPPO in crowding scenarios.

**Test:**
- Smoke: `pytest -q tests/test_agents_smoke.py::test_mappo_smoke` ≥ IPPO on configured seed.

---

## Phase 6 — Evaluation & Datasets

### 6.1 Evaluation harness
**Start:** empty `eval/evaluate.py`.  
**Do:** roll fixed seeds and episodes; compute metrics: mean/var P&L, Sharpe, MDD, valid‑order rate, fill ratio, slippage, ruin prob.  
**End:** CSV results into `data/results/`.

**Test:**
- Unit: `tests/test_eval_metrics.py` verifies metric formulas on toy sequences.

---

### 6.2 Diagnostics plots
**Start:** empty `eval/diagnostics.py`.  
**Do:** action histograms vs edge, size vs displayed depth, slippage vs overflow; save PNGs.  
**End:** figures saved to `data/results/plots/`.

**Test:**
- Unit: `tests/test_diagnostics.py` generates files and checks presence.

---

### 6.3 OOD tests
**Start:** empty `eval/ood_tests.py`.  
**Do:** evaluate saved policies under shifted spreads (`beta`), impact (`alpha`), and liquidity variability (`tau`), and event frequencies.  
**End:** report generalization deltas.

**Test:**
- Unit: `tests/test_ood.py` runs tiny OOD eval and produces CSV.

---

### 6.4 Rollout dataset script
**Start:** empty `scripts/make_dataset.py`.  
**Do:** generate offline dataset (obs, action, reward, info) to parquet for N steps using any policy.  
**End:** file saved under `data/rollouts/`.

**Test:**
- Manual: run script and confirm parquet shape and required columns.

---

## Phase 7 — Human Play

### 7.1 CLI human vs random agent (single and two)
**Start:** empty `human/cli_play.py`.  
**Do:** curses/simple prompt UI: display quote, displayed depth, event (if enabled), hints, bankroll; parse action; step env; show exec price, S, and P&L. Support flags `--events on|off`, `--impact on|off`.  
**End:** playable loop for 10 rounds.

**Test:**
- Manual: play a full episode; invalid actions are never offered.

---

## Phase 8 — Configuration & Defaults

### 8.1 YAML config files
**Start:** empty `config/*.yaml`.  
**Do:** `defaults.yaml`, `env.yaml` (with `flags.enable_events`, `flags.enable_impact`, event frequencies, `event_persist`), `dqn.yaml`, `ippo.yaml`, `mappo.yaml`, `eval.yaml`.  
**End:** loaders read and validate configs.

**Test:**
- Unit: `tests/test_config.py` loads all YAMLs and asserts required keys.

---

## Phase 9 — Documentation & Readme

### 9.1 README and quickstarts
**Start:** empty README.  
**Do:** installation, quickstarts for baselines and agents, config knobs, references to `docs/architecture.md`.  
**End:** copy‑pasteable commands run.

**Test:**
- Manual: follow quickstarts on clean venv.

---

## Phase 10 — Polishing & CI (optional)

### 10.1 Lint & format
**Start:** none.  
**Do:** add `ruff` and `black` configs; format code; add `pre-commit`.  
**End:** `ruff` and `black` pass.

**Test:**
- `ruff check src tests` and `black --check src tests` succeed.

---

### 10.2 Minimal GitHub Actions (optional)
**Start:** none.  
**Do:** add a workflow to set up Python, install, run `pytest -q` on push.  
**End:** CI green.

**Test:**
- Observe green check on PR.

---

# Checklist Summary (for the executor LLM)
- [ ] 0.1 Repo skeleton
- [ ] 0.2 Packaging
- [ ] 0.3 CI smoke
- [ ] 1.1 Cards
- [ ] 1.2 Events
- [ ] 1.3 Posterior
- [ ] 1.4 Quotes
- [ ] 1.5 Liquidity draw
- [ ] 1.6 Impact exec
- [ ] 1.7 Validity/mask
- [ ] 2.1 Spaces
- [ ] 2.2 Single reset
- [ ] 2.3 Single step
- [ ] 2.4 Logging hooks
- [ ] 3.1 Random‑Valid
- [ ] 3.2 EV oracle
- [ ] 3.3 Level‑1 crowding
- [ ] 4.1 Two reset
- [ ] 4.2 Two step
- [ ] 5.1 DQN
- [ ] 5.2 IPPO
- [ ] 5.3 MAPPO
- [ ] 6.1 Evaluate
- [ ] 6.2 Diagnostics
- [ ] 6.3 OOD tests
- [ ] 6.4 Dataset script
- [ ] 7.1 CLI human play
- [ ] 8.1 YAML configs
- [ ] 9.1 README
- [ ] 10.1 Lint/format
- [ ] 10.2 CI workflow

