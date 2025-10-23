
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = Field(default="MemoraAI", env="APP_NAME")

    # Elastic
    elastic_url: str = Field(..., env="ELASTIC_URL")
    elastic_api_key: str = Field(..., env="ELASTIC_API_KEY")
    elastic_index: str = Field(default="memora_memory", env="ELASTIC_INDEX")

    # Vertex / GCP
    gcp_project_id: str = Field(..., env="GCP_PROJECT_ID")
    gcp_location: str = Field(default="us-central1", env="GCP_LOCATION")

    # Retrieval / Prompt
    max_context_snippets: int = Field(default=8, env="MAX_CONTEXT_SNIPPETS")
    embed_dims: int = Field(default=768, env="EMBED_DIMS")
    token_budget: int = Field(default=6000, env="TOKEN_BUDGET")
    system_preamble: str = Field(default="You are MemoraAI, built by Polymath Kaila, a project memory engine.", env="SYSTEM_PREAMBLE")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()
