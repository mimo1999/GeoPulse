# GeoPulse — Review Findings TODO

Source: senior-dev peer review, 2026-09-10. Grouped by priority — work top to bottom;
items in 1–3 decide whether this is a credible system, the rest is polish.

Check items off as we close them out.

---

## P0 — Modelling credibility

- [ ] **Retarget to real ground truth.** Stop training/evaluating primarily against
      GDELT-derived proxy labels (circular: labels come from the same event ratios as
      the features). Pick a real target — UCDP/ACLED onset or fatality threshold over
      1–3 months — and make GDELT features-only.
- [ ] **Benchmark against ViEWS / ACLED CAST** on the new target so the accuracy claims
      mean something outside this repo.
- [ ] **Pull GNN network-adjusted risk from the UI** until it beats the no-GNN baseline.
      `evaluation/results/gnn_eval.json` currently shows it *hurts* skill (−6.5% to −7.2%)
      and it's still surfaced as a headline number with its own tab + global graph.
- [ ] Decide on and document who the actual user/decision is (analyst? trader? NGO?) —
      "risk 0.72" needs a stated consumer and action to mean anything.

## P1 — API correctness / idempotency

- [ ] **Make GETs idempotent.** `/riskscore`, `/country/{c}/forecast`, `/gnn_influence`
      all write to the DB or recompute synchronously on a plain GET. Move persistence
      into scheduled jobs / explicit POST triggers only.
- [ ] Add a **connection pool** (psycopg2 pool, or move to psycopg3/asyncpg) — currently
      a new connection per call, 14+ sites in `backend/main.py` alone.
- [ ] Replace the **N+1 feature extraction** (one query per country per day) with a
      single `GROUP BY` aggregation. The 162-day backfill made ~35k round trips.
- [ ] Cut **MC-Dropout to a sane pass count or drop it** — 50 passes/request on CPU for
      a variance estimate the README itself says collapses and gets overwritten by
      conformal bands.
- [ ] Parallelize `score_all_countries` (currently sequential, one country at a time).

## P2 — Error handling & security

- [ ] Replace blanket `except Exception: return {}` with explicit handling + logging.
      This pattern is what let the FIPS→ISO-3 map bug (zero countries rendered) and the
      null-field spillover crash ship silently.
- [ ] Show "insufficient data" instead of a fabricated `0.000 / 0% / LOW` when a country
      has no real signal (reproduced live: United States scored this way).
- [ ] Move `gldt_secret` and all hardcoded credentials to env vars (8 hardcoded sites).
- [ ] Restrict CORS from `*` to actual allowed origins.
- [ ] Add auth on admin endpoints (`/ingest/trigger`, `/analyze/*`) — `backfill_days=365`
      unauthenticated is a one-request DoS.

## P3 — Codebase cleanup

- [ ] Single source of truth for the composite risk formula (0.4/0.3/0.2/0.1) — currently
      duplicated ~30 places.
- [ ] Delete or wire up `backend/routers/risk.py` (currently dead code, unmounted).
- [ ] Remove unused infra: pgvector (`event_embeddings`, 0 references), PostGIS
      (0 spatial calls), SQLAlchemy (never instantiated) — or actually use them.
- [ ] Add `pyproject.toml` + ruff + mypy + pre-commit; add CI running lint + `pytest`.
- [ ] Fix `pytest` collection: rename `scripts/test_pg_conn.py` so it isn't picked up
      as a test module (it connects to Postgres at import time and breaks a bare
      `pytest` run).
- [ ] Bump/pin Streamlit properly (README says 1.40, installed is 1.57) and fix the
      `use_container_width` deprecation warnings flooding the logs.

## P4 — UI / UX

- [ ] Surface `data_confidence` / `coverage_tier` in the country header — it's returned
      by the API already and is the best trust signal in the product, currently hidden.
- [ ] Add provenance: link scores/events back to `source_url`.
- [ ] Add a freshness indicator ("data as of 2026-09-01") to the header.
- [ ] Remove developer-facing strings from user-visible UI (e.g. "Run POST /analyze/gnn").
- [ ] Fix the "317 countries tracked" stat — currently includes territories/FIPS noise,
      inflating the count.
- [ ] Add deep links (`?country=SD`) so a specific view is shareable.
- [ ] Add a "what changed since last week" view — biggest movers, new alerts.
- [ ] Add a watchlist with threshold-based alerts.
- [ ] Add side-by-side comparison for 2–3 countries.
- [ ] Add an as-of date picker for historical lookback.
- [ ] Add export (PDF/CSV) of a one-page country briefing.
- [ ] Map: add country search, region zoom presets, and a legend in risk *levels*
      instead of raw 0–1.
- [ ] Lazy-load tab content (or `st.fragment`) so switching country doesn't trigger
      ~10 backend calls across every tab on every rerun.
- [ ] Accessibility pass: red/green palette is not colorblind-safe, `#555` on `#0d0d0d`
      fails contrast, body text uses Courier, no mobile layout.

---

## Notes while working through this

**2026-09-11 — FX gate: FAILED (this is a good outcome).** Tested whether GDELT signal
predicts next-month FX moves, to decide whether to pivot the project to economic
targets. Pre-registered rule (written before the analysis code, see
`scripts/fx_gate_test.py` docstring): dir_acc > 0.53 OR OOS R² > 0, AND ≥1 feature
surviving BH-FDR q=0.05. Result on 420 country-months / 14 floating-FX countries /
238 OOS predictions:
- directional accuracy **0.445** (worse than chance)
- OOS R² vs random walk **−0.289** (model is 29% worse than predicting zero)
- **0** features survive FDR; 17 hit p<0.05 uncorrected vs **~24 expected by chance**

Econ-target pivot **closed on evidence**. Consistent with Meese-Rogoff (1983). Full
result in `evaluation/results/fx_gate.json`. Economic data still enters as *features*
for the conflict targets (plan A5) — coverage is good exactly where market data isn't.

**2026-09-11 — Step 0 honesty hygiene done.**
- Fixed the f5/f6 transposition in `scripts/seed_db_from_cache.py` (confirmed against
  three independent sources: the writer's `FEATURE_NAMES`, `forecaster_dataset.py:9-11`,
  and the parquet itself). Re-seed still required before any backtest number is valid.
- Deprecation notices on `preprocessing/label_generator.py` and
  `train_real_data.py::compute_proxy_labels` — the latter computes labels as a *linear
  function of the input features*, which is the real source of the AUC 0.98.
- README performance numbers marked withdrawn pending re-run; the UCDP section flagged
  separately as **not reproducible** (no producing script exists in the repo at all).

**2026-09-11 — POLECAT deprecated as a data source.** Confirmed: POLECAT
stopped publishing after the Cline Center's funding ended; the raw file
in `data/POLECAT/` has no rows past June 2024. Not viable as a live
feature source going forward. Code is **not removed**, just marked:
- `ingestion/polecat_parser.py`, `ingestion/polecat_pipeline.py` — deprecated,
  docstring notes explain why and what would need to change if POLECAT (or
  a successor) resumes publishing.
- `scripts/eval_polecat.py` — **not** deprecated, unaffected — it's a
  fixed-dataset historical benchmark, not live ingestion, so it keeps
  working regardless.

**2026-09-11 — Step 2 (PIT label pipeline) done.**
`scripts/init_ucdp.sql` (5 tables in a dedicated `ucdp` schema — PG15+ revokes
CREATE on `public`), `ingestion/ucdp_loader.py` (417,968 GED rows, 1,928
actors), `data/gw_country_map.py` (all 126 GED countries resolve to FIPS-2,
zero collisions, verified by `tests/test_gw_country_map.py`), and
`preprocessing/pit_labels.py` (16,506 country-months / 53,637 dyad-months,
2015-01→2025-11 monthly, both tables populated). Regression numbers against
the plan's pre-measured baselines (2023-2025 window, clarity=1 only): 1,485
violence country-months (target 1,547), 833 at ≥25 deaths (target 894), 476
at ≥100 (target 518), 392 new-dyad cases (target 412), dyad candidate-set
**median exactly 3** (target 3) — all within ~5-8%, consistent with the
target numbers having used a `clarity IN (1,2)` sensitivity filter rather
than `clarity=1`. One country-month (Ukraine, 2024-06) hand-verified exactly
against a direct query of the raw table: 1,403 events, 2,978 deaths_best.
`tests/test_pit_labels.py::test_pit_boundary_country_labels` recomputes a
random sample of stored aggregates directly from `ucdp.ged_events` with a
strict `date_start > run_date` bound and asserts equality — this is the
plan's "single most important test," proving the PIT boundary actually holds
rather than being merely asserted. 13/13 new tests pass; full suite 133/133.

**P0 label source found and fixed.** The UCDP file originally in
`data/UCDP/` (`GEDEvent_v26_1_4_1101_2512.csv`) turned out to be the
*Violent Political Protest* variant, not the full GED (1,429 rows / 31
countries / 2011–2025). Replaced with the actual **UCDP GED Global v26.1**
(`ged261-csv.zip`): 417,968 events, 126 countries, 1989–2025. Also pulled
the UCDP Actor dataset (`ucdp-actor-261-csv.zip`) — has canonical numeric
actor IDs + rename/split tracking, useful as the entity-resolution target
for the actor-graph work below rather than building alias-matching from
scratch.

**2026-09-11 — Country-code key-space bug found and fixed (blocked the UCDP
join).** `ingestion/event_cleaner.py`'s `FIPS_TO_ISO` map silently rewrote 8 of
GDELT's native FIPS 10-4 codes to ISO 3166-1 alpha-2 on ingest (RS→RU, CH→CN,
GM→DE, JA→JP, SP→ES, UK→GB, UP→UA, PO→PL) while leaving the other ~250
countries in FIPS. Every other consumer (`data/iso3_to_fips.py`, the parquet
feature cache, `streamlit_app/ui.py`'s `fips_to_iso3`) assumes FIPS-only.
Found because the UCDP GW→ISO3→FIPS2 join returned **zero gdelt_events rows**
for Russia, Ukraine, China, Germany and Spain — their 939,592 events were
split silently across two codes each. Fixed: `FIPS_TO_ISO` disabled (empty
dict, future ingestion is a no-op passthrough); `scripts/fix_country_code_split.py`
UPDATEd the 939,592 historical rows back to FIPS (safe — `gdelt_events`' PK is
`(global_event_id, event_date)`, doesn't include country, no collision risk).
`country_daily_features` was **not** repaired — it has pre-existing FIPS rows
from the parquet-seeded path that would collide with the corrected rows on
`(country, feature_date)`, and that table is superseded by the PIT panel
(P0 above), not worth reconciling in place.

**Also found while building `data/gw_country_map.py`: four more FIPS bugs in
`data/iso3_to_fips.py`, all now fixed and verified against GDELT event
geography (bounding boxes) in this DB:**
- `COD` (DR Congo) was `None` — should be `CG`.
- `COG` (Congo-Brazzaville) was `CG` — should be `CF`. (COD/COG were
  transposed relative to the ISO3 intuition.)
- `CAF` (Central African Republic) was `CF` — should be `CT`.
- `SSD` (South Sudan) was `None` — should be `OD` (FIPS did assign one after
  2011 independence; GDELT uses it, 5,438 events in this DB).
- `SCG` (Serbia and Montenegro, dissolved 2006) was `RI`, colliding with
  `SRB`'s `RI` — set to `None`, removing the ambiguity that previously forced
  a `DISTINCT ON ... ORDER BY iso3 != 'SRB'` workaround downstream.

**2026-09-12 — Econ-target search closed for good.** User proposed two more
financial-target framings after the FX gate failed (bucketed GDP growth;
national equity indices as a proxy). Both rejected without running new gates:
- GDP growth buckets: monthly GDP is essentially a UK-only phenomenon: IMF's
  own capacity assessment states CPI/national-accounts/GDP data are the
  *weakest* categories specifically in Fragile and Conflict-Affected States
  (70%+ of countries with serious statistical shortcomings are FCS) — same
  coverage failure as FX, relocated from "no liquid market" to "no
  statistics."
- National equity indices (NIFTY50/Sensex, S&P 500/Nasdaq): efficient-markets
  objection is *stronger* here than for FX (more liquid, more analyst
  coverage); the country overlap with viable indices is nearly the same
  ~18-country set FX already failed on; and index returns are dominated by
  global market beta, not domestic conflict — a real risk of a *spurious*
  positive (global bad-news volume moving both GDELT counts and markets
  together) rather than a clean null. The plan's own Step-1 text pre-committed
  to this: "If FX shows nothing, equities/credit/property will not" — written
  before the FX gate ran, specifically to prevent testing asset classes one
  at a time until something clears significance by chance. Not reopened.
  Economic indicators remain in scope only as PIT *features* (plan A5).

**2026-09-12 — Step 3 (evaluation harness + baselines) done.**
`evaluation/pit_backtest.py`: purged expanding-window walk-forward (3-month
purge), monthly grid, evaluated over the plan's locked 2023-2025 window (35
folds) using the full 2015-2025 history as training data. Reports pooled,
active-only (77 countries — matches the plan's measured figure exactly), and
macro-by-country. Four baselines implemented:
- `global_base_rate` and `persistence_flat` (always "flat") are numerically
  identical — confirms "flat" is genuinely the training-window mode, exactly
  what the plan's chosen persistence baseline assumes.
- `country_base_rate`: lower accuracy (64.2%) but much higher macro-F1 (0.378)
  than the global baselines — trades majority-class accuracy for minority
  (up/down) recall on chronically-escalating/de-escalating countries.
- `lagged_ucdp_only` (HistGradientBoostingClassifier — no `lightgbm` package
  installed, sklearn's native equivalent used instead; A3 features computed
  purely from `ucdp.country_pit_labels`' own stored history, zero GDELT):
  **macro-F1 0.592 vs 0.264-0.378 for the trivial baselines.** This is the
  number any GDELT-based Phase A/B model has to beat — "the baseline that
  matters," per the plan.
Active-country persistence accuracy (65.7%) is higher than the plan's
quoted 54% — expected: my population is every month of every 2023-2025-active
country (including their quiet months), the plan's n=974 figure was a
further-restricted subset. Not chased to an exact match; documented instead.
6/6 new tests pass (fold-purge boundary, PIT-lag correctness of the A3
feature builder, baseline sanity checks); full suite 139/139.

**Knowledge-graph prototype (separate from P0, see earlier discussion):**
built a relational-as-graph schema (`graph` Postgres schema, Apache AGE
confirmed not viable — not installable on this Postgres 18/Windows setup)
over real June 2026 GDELT data. Validated: actor entity fragmentation is
real and bad (one actor code, 275 name variants, within GDELT alone),
actor-country identity needed a fix (bare role codes like GOV carry no
country field at all — inferred from event location per CAMEO convention,
tested against GOV/USA vs GOV/CAN per-country, confirmed correctly
separated), and actor-cooccurrence edges are far richer than the existing
Pearson-correlation spillover table (8,244 vs 93 edges/month). POLECAT and
UCDP are staged (`graph.polecat_stage`, `graph.ucdp_stage`) but not yet
wired into the unified graph tables — UCDP needs rebuilding against the
correct GED file above before that happens.
