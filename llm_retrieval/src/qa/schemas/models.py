from pydantic import BaseModel
from typing import List, Optional


class Document(BaseModel):
    id: str
    length: int
    title: str
    text: str


class Chunk(BaseModel):
    id: str
    length: int
    order_in_doc: int = 0
    num_chunks_in_doc: int = 1
    doc_title: str
    text: str
    embedding: Optional[List[float]] = None


class ChunkSearchResult(Chunk):
    score: float = 0.0


class SearchCondition(BaseModel):
    id: str
