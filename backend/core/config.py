from functools import lru_cache
from pydantic_settings import BaseSettings

from core.env import ENV_FILE



class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.minimax.chat/v1"
    llm_model: str = "MiniMax-M1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    session_memory_max_tokens: int = 3500
    rag_augmentation_enabled: bool = True
    rag_augmentation_top_k: int = 4
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    admin_username: str = ""
    admin_password: str = ""
    max_upload_mb: int = 50
    max_image_upload_mb: int = 10
    seed_demo_data: bool = False

    data_generation_model: str = ""
    data_generation_product_count: int = 30
    data_generation_user_count: int = 8
    data_generation_behaviors_per_user: int = 8
    data_generation_categories: str = "耳机,手机,配件,户外电源,户外,个护,电脑,平板"
    data_generation_max_tokens: int = 8192
    data_generation_batch_size: int = 10
    data_generation_reset_tables: bool = True
    data_generation_reset_vectors: bool = True

    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_seconds: int = 86400

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "product_embeddings"
    milvus_product_collection: str = "product_embeddings"
    milvus_user_collection: str = "user_embeddings"

    # Database
    database_url: str = "sqlite:///./ecommerce.db"
    product_database_url: str = ""
    user_database_url: str = ""

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds)
    agent_timeout_product_rec: float = 8.0

    model_config = {
        "env_file": ENV_FILE,
        "env_prefix": "ECOM_",
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
