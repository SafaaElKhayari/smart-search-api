import requests
import feedparser  # easier than manual XML parsing
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

es = Elasticsearch("http://localhost:9200") 
INDEX_NAME = "papers"

MAPPING = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "authors": {"type": "keyword"},
            "category": {"type": "keyword"},
            "published_date": {"type": "date"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}

ARXIV_API = "http://export.arxiv.org/api/query"


def fetch_papers(category: str = "cs.AI", max_results: int = 200):
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API, params=params)
    response.raise_for_status()
    return response.text


def parse_papers(xml_data: str) -> list[dict]:
    feed = feedparser.parse(xml_data)
    papers = []
    for entry in feed.entries:
        papers.append(
            {
                "id": entry.id.split("/abs/")[-1],
                "title": entry.title.replace("\n", " ").strip(),
                "abstract": entry.summary.replace("\n", " ").strip(),
                "authors": [a.name for a in entry.authors],
                "category": entry.arxiv_primary_category["term"],
                "published_date": entry.published,
            }
        )
    return papers

def create_index():
    try:
        exists = es.indices.exists(index=INDEX_NAME)
    except Exception as e:
        print("ERROR CHECKING INDEX EXISTENCE:", repr(e))
        raise

    if exists:
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(
        index=INDEX_NAME,
        mappings=MAPPING["mappings"],
    )
    print(f"Index '{INDEX_NAME}' created.")

def index_papers(papers: list[dict]):
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": p["id"],
            "_source": p,
        }
        for p in papers
    ]
    success, errors = bulk(es, actions)
    print(f"Indexed {success} papers, {len(errors)} errors.")

def add_embeddings(papers: list[dict]) -> list[dict]:
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = model.encode(texts, show_progress_bar=True)
    for paper, embedding in zip(papers, embeddings):
        paper["embedding"] = embedding.tolist()
    return papers


if __name__ == "__main__":
    create_index()
    xml_data = fetch_papers(max_results=200)
    papers = parse_papers(xml_data)
    papers = add_embeddings(papers)
    index_papers(papers)
    
