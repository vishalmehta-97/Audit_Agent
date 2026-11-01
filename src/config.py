"""Shared settings. No network clients are created here; no API key is required to import."""
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    # Groq's model catalog changes often: list your account's models (see README) and override in .env if needed.
    GROQ_MODEL_NAME: str = "openai/gpt-oss-120b"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RETRIEVAL_TOP_K: int = Field(default=4, ge=1, le=20)
    RRF_K: int = Field(default=60, ge=1)
    CHUNK_SIZE: int = Field(default=800, ge=1)  # characters
    CHUNK_OVERLAP: int = Field(default=150, ge=0)  # characters
    MAX_RETRY_COUNT: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()