from fastapi import FastAPI, Query
from app.es_client import es
from app.schemas import PaperResult
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


app = FastAPI(title="Smart Search API")

INDEX_NAME = "papers"


@app.get("/health")
def health():
    return {"status": "ok", "elasticsearch": es.ping()}


@app.get("/search", response_model=list[PaperResult])
def search(
    q: str = Query(..., description="Search query"),
    category: str | None = None,
    size: int = 10,
):
    must_clauses = [{"multi_match": {"query": q, "fields": ["title^2", "abstract"]}}]
    filter_clauses = []
    if category:
        filter_clauses.append({"term": {"category": category}})

    body = {
        "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
        "size": size,
    }

    response = es.search(index=INDEX_NAME, body=body)

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

@app.get("/facets")
def facets():
    body = {
        "size": 0,
        "aggs": {
            "categories": {
                "terms": {"field": "category", "size": 20}
            }
        },
    }
    response = es.search(index=INDEX_NAME, body=body)
    buckets = response["aggregations"]["categories"]["buckets"]
    return [{"category": b["key"], "count": b["doc_count"]} for b in buckets]


@app.get("/search/semantic", response_model=list[PaperResult])
def search_semantic(q: str = Query(...), size: int = 10):
    query_vector = model.encode(q).tolist()

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": 50,
        }
    }

    response = es.search(index=INDEX_NAME, body=body)

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