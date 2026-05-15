"""
config.py - Centralised configuration via environment variables.

Uses pydantic-settings so every value is validated at startup. 
If WEBHOOK_API_KEY is still "change-me-to-a-real-secret", the app will
start but log a loud warning.
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    webhook_api_key: str = "change-me-to-a-real-secret"
    redis_url: str = "redis://localhost:6379/0"
    max_retries: int = 3
    worker_concurrency: int = 2

    # PostgreSQL - where completed pipeline results are stored
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/news_pipeline"

    # Ollama - for Agent 1 (classifier) running on your local GPUs
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"

    # Claude API - for Agents 2, 3, 4 (reasoning + writing)
    anthropic_api_key: str = "sk-ant-..."
    claude_model: str = "claude-sonnet-4-6"
    claude_haiku: str = "claude-haiku-4-5"

    # Qdrant - vector store for product catalog and EU directive
    qdrant_url: str = "http://192.168.2.185:6333"
    qdrant_products_collection: str = "products"
    qdrant_directives_collection: str = "directives"

    # Embedding model - runs locally on Ollama
    embedding_model: str = "nomic-embed-text"
    embedding_dimenstions: int = 768

    # Redis key names - no need to change these unlesss you want namespacing
    stream_name: str = "news:incoming"
    consumer_group: str = "pipeline-wokers"
    dead_letter_stream: str = "news:dead_letter"
    dedup_set: str = "news:seen_urls"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()