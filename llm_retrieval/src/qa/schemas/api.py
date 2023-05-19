from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
