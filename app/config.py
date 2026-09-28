"""
app/config.py
Central configuration – reads from .env and environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application-wide settings."""

    # VLM model name (Ollama tag)
    VLM_MODEL: str = os.getenv("VLM_MODEL", "qwen2.5vl")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
