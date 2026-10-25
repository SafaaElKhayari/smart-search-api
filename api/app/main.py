import time

from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import ConnectionTimeout, NotFoundError
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from app.es_client import es
from app.schemas import PaperResult
from app.telemetry import embedding_duration, record_search, setup_telemetry, tracer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


app = FastAPI(title="Smart Search API")
setup_telemetry(app)

INDEX_NAME = "papers"
MAX_SIZE = 50
NUM_CANDIDATES = 100


@app.exception_handler(NotFoundError)
def index_not_found(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Index '{INDEX_NAME}' not found. Run the ingestion script first."},
    )


@app.exception_handler(ESConnectionError)
@app.exception_handler(ConnectionTimeout)
def es_unavailable(request: Request, exc: Exception):
    return JSONResponse(status_code=503, content={"detail": "Elasticsearch is unavailable."})


def to_results(response) -> list[PaperResult]:
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append(
            PaperResult(
                id=source["id"],
                title=source["title"],
                abstract=source["abstract"],
                authors=source["authors"],
                category=source["category"],
                published_date=source["published_date"],
                score=hit["_score"],
            )
        )
    return results


@app.get("/health")
def health():
    if es.ping():
        return {"status": "ok", "elasticsearch": True}
    return JSONResponse(status_code=503, content={"status": "degraded", "elasticsearch": False})


@app.get("/search", response_model=list[PaperResult])
def search(
    q: str = Query(..., description="Search query"),
    category: str | None = None,
    size: int = Query(10, ge=1, le=MAX_SIZE),
):
    must_clauses = [{"multi_match": {"query": q, "fields": ["title^2", "abstract"]}}]
    filter_clauses = []
    if category:
        filter_clauses.append({"term": {"category": category}})

    response = es.search(
        index=INDEX_NAME,
        query={"bool": {"must": must_clauses, "filter": filter_clauses}},
        size=size,
        source_excludes=["embedding"],
    )
    results = to_results(response)
    record_search("keyword", category, len(results))
    return results

@app.get("/facets")
def facets():
    response = es.search(
        index=INDEX_NAME,
        size=0,
        aggs={"categories": {"terms": {"field": "category", "size": 20}}},
    )
    buckets = response["aggregations"]["categories"]["buckets"]
    return [{"category": b["key"], "count": b["doc_count"]} for b in buckets]


@app.get("/search/semantic", response_model=list[PaperResult])
def search_semantic(q: str = Query(...), size: int = Query(10, ge=1, le=MAX_SIZE)):
    with tracer.start_as_current_span("embed_query"):
        start = time.perf_counter()
        query_vector = model.encode(q).tolist()
        embedding_duration.record((time.perf_counter() - start) * 1000)

    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": NUM_CANDIDATES,
        },
        size=size,
        source_excludes=["embedding"],
    )
    results = to_results(response)
    record_search("semantic", None, len(results))
    return results
