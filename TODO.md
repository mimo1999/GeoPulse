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
macro-by-country.

| baseline | active-only accuracy | active-only macro-F1 | pooled accuracy |
|---|---|---|---|
| `global_base_rate` | 65.7% | 0.264 | 79.0% |
| `persistence_flat` | 65.7% | 0.264 | 79.0% |
| `country_base_rate` | 64.2% | 0.378 | 78.1% |
| **`lagged_ucdp_only`** | **71.7%** | **0.592** | 82.7% |

Four baselines implemented:
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

**2026-09-12 — Step 4 (Phase A model): GDELT does not beat the UCDP-lag
baseline. Reporting as found, not tuned.** `preprocessing/pit_features.py`
builds A1 temporal derivatives from `data/real_cache/*.parquet` (56 columns:
current/Δ1/Δ2/rolling-mean-3-6-13/rolling-std-13/z-score-13 per raw feature)
and joins them onto the label panel via an as-of merge respecting the same
7-day PIT buffer used in `pit_labels.py`. Country-code normalization
(join hazard #2 from the plan): CAMEO-3 codes are GDELT's own alphabet for
`actor1_country_code`, same as ISO 3166-1 alpha-3 — resolved via
`data/iso3_to_fips.py` (fixed earlier this session), 68/71 distinct 3-char
codes resolve; the 3 that don't (AFR, EUR, SEA) are regional/supranational
CAMEO actors, correctly dropped, not countries. 97% A1 coverage within the
2023-2025 eval window (parquet starts exactly at 2023-01-01, so the first
few 2023 folds have thin rolling history — expected, not a bug).

Result, same harness, same 35 folds, A3 (UCDP lag) + A1 (GDELT) combined vs.
A3 alone:

| model | active-only accuracy | active-only macro-F1 |
|---|---|---|
| `lagged_ucdp_only` (A3 only, no GDELT) | 71.7% | **0.592** |
| `phase_a_gdelt` (A3 + A1 GDELT) | 70.5% | **0.547** |

**GDELT made it worse, not better.** Permutation importance on the model
(last fold) shows GDELT features are not dead weight — 9 of the top 15
features by importance are A1 columns with real positive contribution — but
the net effect across all 35 folds is still negative. Most likely
explanation: feature-volume dilution (65 total features, 56 of them A1, vs
9 for the UCDP-only baseline) on a modest per-fold training size, compounded
by the A1 feature quality issues the plan already flagged before this test
ran (`f2`/`f6` collinear with `avg_goldstein`, `f6` percentile-inverted
despite its name, within-day cross-sectional ranking meaning a country's
values move when *other* countries move, not just itself).

**Did not chase a better number.** No hyperparameter tuning, no feature
pruning, no reruns with different configs to try to flip this result —
that would be exactly the kind of post-hoc rationalization the FX gate's
pre-registration existed to prevent, just applied to model configs instead
of asset classes. `lagged_ucdp_only` stays the best model on record until
something beats it under the same harness.

**What this does and doesn't mean.** It does not mean "GDELT can't help" —
it means *this specific, already-flagged-as-lossy feature source* (7 raw
parquet columns, collapsed by percentile-ranking, missing 35 of GDELT's 58
raw columns per the plan's Phase B section) doesn't clear the bar. Phase B
(fine-grained CAMEO codes, actor-structural features, media/geographic
dispersion — the plan's B1-B7) is the next real test, per the plan's own
sequencing ("Phase B features → re-run the same harness, report the A→B
delta"). 7/7 new tests pass (no-future-leakage on the A1 builder, PIT-buffer
correctness, CAMEO-3 resolution); full suite 146/146.

**2026-09-12 — Step 5 (Phase B ingest) in progress.**
- `ingestion/gdelt_parser.py` + `ingestion/db_writer.py`: 8 previously-dropped
  raw GDELT v1 columns now parsed and stored — `action_geo_adm1_code`,
  `actor1/2_known_group_code`, `actor1/2_type2/3_code`, `is_root_event`
  (plan's B2/B5 feature families). Purely additive; 9/9 parser tests pass.
- `scripts/add_phase_b_columns.sql`: 8 new nullable columns on `gdelt_events`
  + an `(action_geo_country, action_geo_adm1, event_date)` index. Blocked
  initially on ownership: `gdelt_events` and its 84 monthly partitions were
  owned by `postgres`, not `gldt`; `ALTER TABLE ... OWNER TO` doesn't cascade
  to partitions, so `ADD COLUMN` on the partitioned parent failed atomically
  the moment it reached the first partition `gldt` didn't own. Resolved via
  `scripts/admin_grant_gldt_ownership.sql` (user ran as `postgres`).
- `preprocessing/feature_extractor.py`: **fixed the N+1** (TODO.md P1) —
  one SELECT + one UPSERT per country per day (~200 active countries/day,
  ~35k round trips over a 162-day backfill) replaced with one SELECT for
  the whole date grouped in Python + one bulk UPSERT. Verified against live
  data: `total_events` matches a direct `COUNT(*)` exactly. 2/2 new
  integration tests pass.
- **Backfill running**: `python scripts/backfill.py --start 2023-01-01 --end
  2025-12-31` (1,096 days), started 2026-09-12 15:25, log at
  `logs/phase_b_backfill.log`. First attempt was accidentally killed at
  7/1096 by a shell-backgrounding mistake (`&` combined with the harness's
  own background-task tracking — the wrapper shell exited immediately and
  orphaned the real process); restarted cleanly, already-ingested dates skip
  fast so the redo cost was negligible.

**2026-09-12 — Backfill complete: 1,084/1,096 dates, 47.8M raw events,
2023-01-01→2025-12-31.** 12 dates (2025-06-14 through 2025-07-01) are a
**genuine gap on GDELT's own server** — confirmed via direct HTTP check:
these return an empty placeholder response (MD5-of-empty-string ETag,
`Last-Modified: 2014`) where adjacent dates return a normal 9MB+ zip. Not a
bug in this codebase's URL construction or a transient network issue; not
retried further.

**Found and fixed a real bug while chasing the gap**: the backfill's own
summary line ("1096/1096 dates OK") was wrong — `stream_csv_rows()` returned
an *empty generator* on a failed download instead of raising, so
`ingest_date` saw zero chunks, ran to completion normally, and logged
`status="success"` with 0 events. Indistinguishable from a genuinely quiet
day; the gap was only caught by a direct `COUNT(DISTINCT event_date)`
against the requested range, not by anything the pipeline itself reported.
Fixed: raises `GDELTDownloadError` now, which `ingest_date`'s existing
(and already-correct) exception handling picks up like any other failure.
2 new tests; full suite 152/152.

Step 5 (Phase B ingest) is now done: parser extended, schema migrated, N+1
fixed, 2023-2025 raw events backfilled (98.9% date coverage, gap documented
and understood).

**2026-09-12 — Step 6 (Phase B features): closer, still short of the
UCDP-only bar.** `preprocessing/pit_features_b.py` built B1 (22 named
fine-grained CAMEO codes — riot/arrest/repression/assault/fight/mass-violence
subtypes — as mention-weighted share, log-count, and log-count delta each),
B2 (actor-sector shares for GOV/MIL/REB/INS/OPP/COP/CVL, REB↔GOV dyad share,
cross-border share), B4 (media amplification), B5 partial (adm1 count +
Shannon entropy — capital-distance and border-proximity skipped, no
reference data for either), and B7 (tone distribution) — 93 feature columns,
built directly from the Step 5 raw-event backfill rather than the parquet
cache Phase A used. B3 (graph scalars) not attempted — the existing `graph`
Postgres schema covers a single month (June 2026) as a separate KG
prototype, not this backfill's 2023-2025 window.

| model | active-only accuracy | active-only macro-F1 |
|---|---|---|
| `lagged_ucdp_only` (A3 only, no GDELT) | 71.7% | **0.592** |
| `phase_a_gdelt` (A3 + A1 parquet) | 70.5% | 0.547 |
| `phase_b_gdelt` (A3 + B1/B2/B4/B5/B7) | 70.3% | **0.551** |

Phase B modestly beats Phase A (0.551 vs 0.547) but still does not clear the
UCDP-lag-only bar. Permutation importance on the last fold shows a
qualitatively cleaner signal than Phase A had: **11 of the top 15 features
are B-family** (vs 9/15 for Phase A), with several fine-grained CAMEO codes
showing real, distinct contribution — `b1_175_share` (repression),
`b1_190_log_count` (fight), `b1_201_share`, `b2_actor1_cop_share`. The B
family is clearly picking up *something* GDELT-specific that the compressed
parquet couldn't, which is progress — it just isn't enough to overcome
feature-volume dilution against A3's 9 clean columns (93 B features is
close to Phase A's 56, same class of problem). No hyperparameter tuning or
feature pruning attempted to reverse this, same discipline as Step 4.

**Where this leaves the project honestly**: `lagged_ucdp_only` remains the
best model on record after two independent, differently-sourced GDELT
feature attempts. This is itself a real, reportable finding — not a
failure to find the right features, necessarily, but consistent with what
the literature already expects (conflict is dominated by its own recent
history; news-event aggregates add only a small, hard-to-extract margin on
top). Two ways to still test this fairly, neither attempted yet: (a) a
pre-registered feature-selection pass — pick the ~15 highest-permutation-
importance B/A1 features *before* looking at the aggregate score, refit,
and see if a leaner GDELT feature set clears the bar without the dilution
cost (same falsification discipline as the FX gate: decide the rule before
running it); (b) B3 graph scalars, rebuilt over 2023-2025 instead of the
June-2026 prototype window — the plan's own evidence (8,244 vs 93
edges/month over the existing spillover table) suggests this family has
real, unexploited structure the tabular B-features above don't capture.

4/4 new tests pass (PIT boundary matches a direct raw-event count exactly,
log-count matches by hand); full suite 152/152.

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

---

## 2026-09-22 — Pivot: back to the core CI use case

The actual brief (see [usecase.md](usecase.md), transcribed from the source slide)
is a Competitive Intelligence tool: transform GDELT into an **actor-interaction
graph over time**, map CAMEO codes to **interpretable interaction types**
(cooperation/consultation/conflict), and **analyze how the network evolves**.
This is materially different from the UCDP/PIT conflict-*prediction* pipeline
(`evaluation/pit_backtest.py`, `preprocessing/pit_labels.py` etc., Steps 0-6
above) — no forecast target, no UCDP ground truth, no walk-forward backtest.

**Decision**: pivot to the graph use case as the primary deliverable. The
PIT/UCDP pipeline is **paused, not deleted** — it stays in the repo as
working, tested code (all 152+ tests still pass) in case the project returns
to conflict prediction later, but active work now targets `usecase.md`.

**Found**: the existing `graph.*` Postgres schema (see the prototype note
above) had **no producing script anywhere in the repo** — it was built by ad
hoc commands from an earlier session and only existed as live DB state.
Given the project's standing discipline about reproducibility, this was
formalized first (`scripts/init_graph_schema.sql`, matches the live schema
exactly) before adding anything new to it.

**Built**:
- `data/cameo_codes.py` — the CAMEO root-code → `{cooperation, consultation,
  conflict}` mapping, the single source of truth for usecase.md's
  interaction-type requirement. `consultation` (root 04) isolated
  specifically because the brief names it as its own bucket; CAMEO's
  `quad_class` alone can't isolate it, only the root code can.
- `data/cameo_actor_types.py` — the ~30 well-known CAMEO institutional role
  codes (GOV, MIL, REB, ...), used to detect a "bare role" actor (e.g. GOV,
  GOVMIL) whose country can only be inferred from event context.
- `ingestion/graph_builder.py` — populates `graph.event/event_actor/actor/
  location` from `gdelt_events`. Entity resolution: GDELT's own
  `actor{1,2}_country` field first; falls back to inferring country from
  the event's own location **only** when the actor code is a recognized
  bare role (CAMEO's own convention, not a guess); otherwise left NULL —
  usecase.md's own named limitation ("generic references... too general"),
  surfaced as queryable NULLs rather than papered over. Actor identity is
  `(source, raw_code, country_iso3)`, so GOV/USA and GOV/CAN are always
  distinct — the exact bug class the June-2026 prototype had partially
  fixed already, now backed by tested, committed code instead of an ad hoc
  one-off. Additive and idempotent throughout (`ON CONFLICT`), so it runs
  safely alongside the existing June-2026 prototype rows, which — per
  instruction — were kept rather than wiped.
- 19 new tests across the three modules (all pass), including an
  actor-country-collapse check and a rerun-produces-no-duplicates check.

**In progress**: full graph build over 2023-2025 (`logs/graph_builder.log`),
same window as the PIT pipeline's raw backfill — no new data collection
needed, purely processing the 47.8M events already in `gdelt_events`.
~70 minutes estimated from the observed rate (90/1096 days in ~6 minutes).

**Not yet done**: the network-evolution analysis itself (WHAT's third bullet
— "analyze the resulting interaction network to identify patterns... and how
they evolve over time"). That's the next real deliverable once the build
finishes: temporal graph snapshots, interaction-type mix over time,
actor-centrality / community-detection changes, and a way to surface this
(dashboard or notebook) for the CI/Corporate-Security/Governmental-Affairs
audience the brief names.
