# GeoPulse: Global Risk Intelligence Platform

> **Geopolitical escalation monitoring and forecasting powered by GDELT**

Full-stack ML platform that ingests daily GDELT event exports, extracts per-country risk features, and runs a three-tier inference pipeline, all surfaced through a 7-page dark-themed intelligence dashboard.

---

## Architecture

```
GDELT (daily ZIP)
      |
      v
ingestion/
gdelt_downloader -> gdelt_parser -> event_cleaner -> db_writer
      |
      v  PostgreSQL / TimescaleDB
country_daily_features -+-> risk_scorer        -> country_risk_predictions
                        +-> label_generator    -> country_multitask_labels
                        +-> event_clusterer    -> event_clusters
                        +-> spillover          -> country_spillover
                        +-> escalation_forecaster -> country_escalation_forecasts
                        +-> gnn_spillover      -> gnn_node_embeddings
                        +-> rag_engine         -> advisory_corpus
      |
      v
FastAPI backend (30+ endpoints, MCP-compatible /riskscore)
      |
      v
Streamlit dashboard (7 pages)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| API | FastAPI 0.115 + Uvicorn |
| Database | PostgreSQL + TimescaleDB |
| ML | PyTorch 2.4: Transformer encoder, seq2seq LSTM, 2-layer GAT |
| Dashboard | Streamlit 1.40 + Plotly 5.24 |
| Explainability | Integrated Gradients (custom) |
| RAG | TF-IDF cosine retrieval |
| Deploy | Docker Compose |

---

## ML Models

**HybridRiskTransformer:** 3-layer Transformer encoder over a 90-day feature window with 5 parallel risk heads (risk_score, instability, war, terrorism, financial). Checkpoint: `models/real_data_model.pt`.

**EscalationForecaster:** seq2seq LSTM with 4 autoregressive bi-weekly steps. Uses MC-Dropout for variance estimation. Checkpoint: `models/checkpoints/forecaster_v1_best.pt`.

**RiskGNN:** 2-layer GAT over a hybrid adjacency matrix combining Pearson-correlation spillover edges with geographic contiguity priors (`data/structural_edges.csv`).

**Integrated Gradients:** Signed feature attribution over the 14 input features for each prediction.

**RAG Advisory Engine:** TF-IDF retrieval over seed situation templates and event-cluster entries.

---

## API

FastAPI on `http://localhost:8000` with 30+ endpoints. Swagger UI at `/docs`.

Key endpoints:
- `GET /global/heatmap` — all countries' latest risk scores
- `GET /country/{code}/timeline` — historical risk timeline
- `POST /riskscore` — MCP-compatible risk score
- `GET /country/{code}/forecast` — 4-step escalation forecast
- `GET /country/{code}/gnn_influence` — GNN contagion score
- `GET /country/{code}/rag_advisory` — RAG advisory
- `GET /country/{code}/attributions` — Integrated Gradients

---

## Dashboard

| Page | What it shows |
|---|---|
| Global Risk Map | Plotly choropleth world map + top-20 risk table + 90-day timeline |
| Country Drilldown | Risk timeline, event clusters, IG attribution, spillover neighbours |
| Event Explorer | GDELT cluster browser with category/time filters |
| Spillover Network | Global risk bar chart or ego-network view |
| Escalation Forecast | 4-step forecast ribbon + per-task breakdown + global alerts |
| GNN Network | GAT contagion graph with country inspector panel |
| RAG Advisory | RAG vs rule-based advisory side-by-side |

---

## Quick Start

**Prerequisites:** Python 3.10+, PostgreSQL with `timescaledb`.

```bash
git clone <repo-url> && cd geopulse
pip install -r requirements.txt
cp .env.example .env
```

```bash
python scripts/seed_db_from_cache.py
python -m uvicorn backend.main:app --port 8000 --reload
streamlit run streamlit_app/app.py --server.port 8502
```

**Docker:**
```bash
cd docker && docker compose up -d
```

---

## Training & Evaluation

```bash
python scripts/train_real_data.py
python scripts/train_forecaster.py
python scripts/run_backtest.py
python scripts/eval_risk_transformer.py
python scripts/eval_gnn_spillover.py
```

Results are written to `evaluation/results/`.

---

> **Disclaimer:** For geopolitical analytics and risk monitoring only, not for operational targeting or military planning.
