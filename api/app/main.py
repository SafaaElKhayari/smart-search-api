from fastapi import FastAPI, Query
from app.es_client import es
from app.schemas import PaperResult

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