# Smart Search API

A search API for research papers. It uses two ways to search:
normal keyword search, and "smart" search that understands meaning,
not just exact words.

You can search two ways:
1. **Normal search**    find papers that have your exact words
2. **Smart search**    find papers that mean the same thing, even if
   the words are different

Example: if you search "how do machines learn", smart search can
find a paper that talks about "deep learning models" , even though
the words don't match exactly.

## What it can do

- Search papers by keyword
- Filter papers by category
- See how many papers exist in each category
- Smart search by meaning (using AI embeddings)

## How it works (simple version)

Step 1: A script downloads papers and saves them into Elasticsearch
Step 2: The API reads from Elasticsearch when someone searches
Step 3: Both parts run inside Docker containers

The download script and the API are two separate things. The script
only runs when I want new data. The API just answers search
requests fast.

## Tools used

- **FastAPI**  to build the API
- **Elasticsearch** to store and search the data
- **sentence-transformers** to turn text into numbers (for smart
  search)
- **Docker** to run everything the same way, anywhere
- **arXiv** free public source for the papers

## Observability

The API is instrumented with **OpenTelemetry** and ships with a small
monitoring stack, so you can see what it does in production:

- **Traces**: every request is traced from FastAPI, through the query
  embedding (`embed_query` span), to the Elasticsearch call. Open
  Jaeger at http://localhost:16686
- **Metrics**: request rate, p95 latency and error rate per endpoint,
  plus product KPIs: searches by type and category, zero-result rate,
  embedding time. Open Grafana at http://localhost:3000 (dashboard
  "Smart Search API")
- **Logs**: every log line carries `trace_id` and `span_id`, so a log
  can be matched to its trace

How the data flows:

API (OTLP) → OpenTelemetry Collector → Jaeger (traces) and
Prometheus (metrics) → Grafana

Config lives in `observability/`. Set `OTEL_ENABLED=false` to turn it off.

## API endpoints

GET /health check if everything is working
GET /search normal keyword search
GET /search/semantic smart search (by meaning)
GET /facets see paper counts per category

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for the ingestion script)

### 1. Start the infrastructure

\```bash
docker compose up -d --build
\```

### 2. Ingest data (fetch + index arXiv papers)

\```bash
cd scripts
python -m venv venv
source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python ingest.py
\```

### 3. Explore the API

Open http://localhost:8000/docs

## Example queries

Keyword search:

\```bash
curl "http://localhost:8000/search?q=neural+network&category=cs.AI"
\```

Semantic search (finds conceptually related results, even with no shared keywords):

\```bash
curl "http://localhost:8000/search/semantic?q=how+do+machines+learn+representations"
\```

Facets:

\```bash
curl "http://localhost:8000/facets"
\```