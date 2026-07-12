from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    asr_model_name: str = "openai/whisper-tiny"
    log_level: str = "INFO"


settings = Settings()
