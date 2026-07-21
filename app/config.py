from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Reads config from .env (or environment variables) into typed fields.
    pydantic-settings validates these automatically at startup, so a missing
    required value fails fast instead of surfacing as a confusing error
    later."""

    database_url: str
    groq_api_key:str
    groq_model:str


    embedding_model: str
    embedding_dim : int =384
    

    chunk_size: int = 800
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env",extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()