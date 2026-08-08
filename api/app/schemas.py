from pydantic import BaseModel


class PaperResult(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str]
    category: str
    published_date: str
    score: float