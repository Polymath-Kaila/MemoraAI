
from pydantic import BaseModel, Field
from typing import List, Optional

class IngestRequest(BaseModel):
    project_id: str = Field(..., description="Logical project namespace")
    text: str = Field(..., description="Raw text to store")
    tags: Optional[List[str]] = None

class AskRequest(BaseModel):
    project_id: str
    query: str
    k: int = 8

class AskResponse(BaseModel):
    response: str
    used_snippets: int
    tokens_estimate: int
