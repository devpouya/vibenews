import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = ""
    
    # Rate limiting for Gemini
    gemini_rpm_limit: int = 5  # requests per minute
    gemini_daily_limit: int = 25  # requests per day
    
    # Database
    database_url: str = "sqlite:///./vibenews.db"
    
    # Scraping
    scrape_interval_hours: int = 24
    max_articles_per_source: int = 100
    
    # Models
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    bias_model_path: str = "./models/bias_classifier.pt"
    
    # Data paths
    data_dir: Path = Path("backend/data")
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    annotations_dir: Path = data_dir / "annotations"
    
    # Swiss news sources
    news_sources: List[str] = [
        "nzz.ch",
        "srf.ch", 
        "20min.ch",
        "watson.ch"
    ]
    
    class Config:
        env_file = ".env"


settings = Settings()

# Ensure data directories exist
for directory in [settings.raw_data_dir, settings.processed_data_dir, settings.annotations_dir]:
    directory.mkdir(parents=True, exist_ok=True)