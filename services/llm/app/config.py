from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    llm_model_name: str = "google/flan-t5-small"
    llm_max_new_tokens: int = 256

    # Бэкенд инференса: "transformers" (CPU-friendly, по умолчанию) или "vllm"
    # (быстрый, требует GPU — включается через docker-compose.vllm.yml).
    llm_backend: str = "transformers"

    # Накопление батча: запросы копятся в очередь и обрабатываются пачкой.
    # Батч уходит в модель, когда набралось max_batch_size ЛИБО прошло
    # batch_timeout_ms с момента первого запроса в пачке — что раньше.
    llm_max_batch_size: int = 8
    llm_batch_timeout_ms: int = 50

    log_level: str = "INFO"


settings = Settings()
