from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DataAnalystAgent"
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    artifacts_dir: str = "./artifacts"

    model_config = {"env_prefix": "DAA_", "extra": "ignore"}


settings = Settings()
