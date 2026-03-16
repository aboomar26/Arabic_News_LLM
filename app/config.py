# config.py

from pydantic_settings  import BaseSettings


class Settings(BaseSettings):


    #vllm server
    VLLM_BASE_URL : str =  "http://localhost:8000"
    VLLM_MODEL_ID : str =  "news-lora"

    #base model
    BASE_MODEL_ID : str = "Qwen/Qwen2.5-1.5B-Instruct"

    #generation defaults 
    REQUEST_TIMEOUT: float = 60.0
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2

    #language
    TARGET_LANG: str = "English"


    class Config:

        env_file = ".env"
        env_file_encoding = "utf-8"



settings = Settings()


