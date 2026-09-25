from functools import lru_cache

from core.env import ENV_FILE
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # 四类模型分别配置，不共享连接信息。
    text_api_key: str = Field(
        default="",
        validation_alias="TEXT_API_KEY",
    )
    text_base_url: str = Field(
        default="",
        validation_alias="TEXT_BASE_URL",
    )
    text_llm: str = Field(
        default="",
        validation_alias="TEXT_LLM",
    )
    vision_api_key: str = Field(
        default="",
        validation_alias="VISION_API_KEY",
    )
    vision_base_url: str = Field(
        default="",
        validation_alias="VISION_BASE_URL",
    )
    vision_llm: str = Field(
        default="",
        validation_alias="VISION_LLM",
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    session_memory_max_tokens: int = 3500
    rag_augmentation_enabled: bool = True
    rag_augmentation_top_k: int = 4
    agent_config_file: str = "config/agents.yaml"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    admin_username: str = ""
    admin_password: str = ""
    max_upload_mb: int = 50
    max_image_upload_mb: int = 10
    seed_demo_data: bool = False

    data_generation_product_count: int = 30
    data_generation_user_count: int = 8
    data_generation_behaviors_per_user: int = 8
    data_generation_categories: str = "耳机,手机,配件,户外电源,户外,个护,电脑,平板"
    data_generation_max_tokens: int = 8192
    data_generation_batch_size: int = 10
    data_generation_reset_tables: bool = True
    data_generation_reset_vectors: bool = True

    # 向量模型
    embedding_api_key: str = Field(default="", validation_alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", validation_alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(
        default=1536, gt=0, validation_alias="EMBEDDING_DIMENSION"
    )
    embedding_batch_size: int = Field(
        default=10, gt=0, validation_alias="EMBEDDING_BATCH_SIZE"
    )

    rerank_model: str = Field(
        default="", validation_alias="RERANK_MODEL"
    )
    rerank_api_key: str = Field(
        default="",
        validation_alias="RERANK_API_KEY",
    )
    rerank_base_url: str = Field(
        default="",
        validation_alias="RERANK_BASE_URL",
    )
    rerank_model_path: str = Field(
        default="", validation_alias="RERANK_MODEL_PATH"
    )
    rerank_device: str = Field(default="cpu", validation_alias="RERANK_DEVICE")
    rerank_batch_size: int = Field(
        default=8, gt=0, validation_alias="RERANK_BATCH_SIZE"
    )
    rerank_max_length: int = Field(
        default=512, gt=0, validation_alias="RERANK_MAX_LENGTH"
    )

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
    database_url: str = "sqlite:///data/ecommerce.db"
    product_database_url: str = ""
    user_database_url: str = ""

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    model_config = {
        "env_file": ENV_FILE,
        "env_prefix": "ECOM_",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
