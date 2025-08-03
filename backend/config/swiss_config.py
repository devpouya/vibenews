"""
Swiss News System Configuration
Settings and configuration for Swiss news scraping
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Optional
import os

class SwissNewsConfig(BaseSettings):
    """Swiss news scraping configuration"""
    
    # Scraping settings
    SCRAPING_INTERVAL_HOURS: int = Field(4, description="Hours between scraping cycles")
    SCRAPING_TIMEOUT_SECONDS: int = Field(10, description="HTTP request timeout")
    ARTICLES_PER_SOURCE_LIMIT: int = Field(50, description="Max articles per source per cycle")
    
    # Rate limiting
    REQUEST_DELAY_SECONDS: float = Field(1.0, description="Delay between requests to same source")
    SOURCE_DELAY_SECONDS: float = Field(2.0, description="Delay between different sources")
    
    # Paywall detection
    PAYWALL_DETECTION_ENABLED: bool = Field(True, description="Enable paywall detection")
    SKIP_PAYWALL_ARTICLES: bool = Field(True, description="Skip articles behind paywalls")
    
    # Database settings
    SWISS_ARTICLES_RETENTION_DAYS: int = Field(180, description="Days to keep articles")
    SESSION_RETENTION_COUNT: int = Field(100, description="Number of scraping sessions to keep")
    
    # BiasScanner integration
    BIAS_ANALYSIS_ENABLED: bool = Field(True, description="Enable bias analysis for Swiss articles")
    BIAS_ANALYSIS_BATCH_SIZE: int = Field(5, description="Articles per bias analysis batch")
    BIAS_ANALYSIS_INTERVAL_HOURS: int = Field(1, description="Hours between bias analysis runs")
    
    # Swiss sources configuration
    SWISS_SOURCES: Dict[str, Dict] = Field(default={
        "srf": {
            "name": "SRF",
            "base_url": "https://www.srf.ch",
            "language": "de",
            "enabled": True,
            "sections": ["/news", "/news/schweiz", "/news/international", "/news/wirtschaft"],
            "canton": "national"
        },
        "watson": {
            "name": "Watson",
            "base_url": "https://www.watson.ch",
            "language": "de", 
            "enabled": True,
            "sections": ["/schweiz", "/international", "/wirtschaft"],
            "canton": "national"
        },
        "blick": {
            "name": "Blick",
            "base_url": "https://www.blick.ch",
            "language": "de",
            "enabled": True,
            "sections": ["/schweiz", "/ausland", "/wirtschaft"],
            "canton": "national"
        },
        "rts": {
            "name": "RTS",
            "base_url": "https://www.rts.ch",
            "language": "fr",
            "enabled": True,
            "sections": ["/info", "/info/suisse", "/info/monde", "/info/economie"],
            "canton": "national"
        },
        "rsi": {
            "name": "RSI", 
            "base_url": "https://www.rsi.ch",
            "language": "it",
            "enabled": True,
            "sections": ["/news", "/news/svizzera", "/news/mondo", "/news/economia"],
            "canton": "ticino"
        }
    })
    
    # Regional sources (can be expanded)
    REGIONAL_SOURCES: Dict[str, Dict] = Field(default={
        "aargauer_zeitung": {
            "name": "Aargauer Zeitung",
            "base_url": "https://www.aargauerzeitung.ch",
            "language": "de",
            "enabled": False,  # Disabled until paywall status confirmed
            "canton": "aargau"
        },
        "luzerner_zeitung": {
            "name": "Luzerner Zeitung", 
            "base_url": "https://www.luzernerzeitung.ch",
            "language": "de",
            "enabled": False,  # Disabled until paywall status confirmed
            "canton": "luzern"
        }
    })
    
    # Topic keywords for content classification
    TOPIC_KEYWORDS: Dict[str, List[str]] = Field(default={
        "politics": [
            # German
            "politik", "regierung", "bundesrat", "parlament", "wahlen", "abstimmung",
            "initiative", "referendum", "partei", "politiker",
            # French  
            "politique", "gouvernement", "conseil fédéral", "parlement", "élections",
            "votation", "initiative", "référendum", "parti", "politicien",
            # Italian
            "politica", "governo", "consiglio federale", "parlamento", "elezioni",
            "votazione", "iniziativa", "referendum", "partito", "politico"
        ],
        "economy": [
            # German
            "wirtschaft", "inflation", "arbeitslosigkeit", "konjunktur", "börse", 
            "unternehmen", "handel", "export", "import", "steuern",
            # French
            "économie", "inflation", "chômage", "conjoncture", "bourse",
            "entreprise", "commerce", "export", "import", "impôts", 
            # Italian
            "economia", "inflazione", "disoccupazione", "congiuntura", "borsa",
            "impresa", "commercio", "esportazione", "importazione", "tasse"
        ],
        "international": [
            # German
            "international", "aussenpolitik", "eu", "europa", "nato", "uno",
            "diplomatie", "ausland", "bilateral", "multilateral",
            # French
            "international", "politique extérieure", "ue", "europe", "otan", "onu",
            "diplomatie", "étranger", "bilatéral", "multilatéral",
            # Italian  
            "internazionale", "politica estera", "ue", "europa", "nato", "onu",
            "diplomazia", "estero", "bilaterale", "multilaterale"
        ],
        "environment": [
            # German
            "umwelt", "klima", "energie", "nachhaltigkeit", "co2", "klimawandel",
            "erneuerbar", "solar", "wind", "atom", "kernenergie",
            # French
            "environnement", "climat", "énergie", "durabilité", "co2", "changement climatique", 
            "renouvelable", "solaire", "éolien", "nucléaire", "énergie nucléaire",
            # Italian
            "ambiente", "clima", "energia", "sostenibilità", "co2", "cambiamento climatico",
            "rinnovabile", "solare", "eolico", "nucleare", "energia nucleare"
        ],
        "society": [
            # German
            "gesellschaft", "bildung", "gesundheit", "sozial", "kultur", "religion",
            "migration", "integration", "gleichberechtigung", "diskriminierung",
            # French
            "société", "éducation", "santé", "social", "culture", "religion",
            "migration", "intégration", "égalité", "discrimination",
            # Italian
            "società", "educazione", "salute", "sociale", "cultura", "religione", 
            "migrazione", "integrazione", "uguaglianza", "discriminazione"
        ]
    })
    
    # Monitoring and alerting
    MONITORING_ENABLED: bool = Field(True, description="Enable monitoring and logging")
    ALERT_ON_SCRAPING_FAILURES: bool = Field(True, description="Alert on scraping failures")
    MAX_CONSECUTIVE_FAILURES: int = Field(3, description="Max failures before alerting")
    
    # Performance settings  
    MAX_CONCURRENT_REQUESTS: int = Field(5, description="Max concurrent HTTP requests")
    MEMORY_LIMIT_MB: int = Field(512, description="Memory limit for scraping process")
    
    class Config:
        env_prefix = "SWISS_NEWS_"
        case_sensitive = False

# Global configuration instance
swiss_config = SwissNewsConfig()

# Helper functions
def get_enabled_sources() -> Dict[str, Dict]:
    """Get only enabled Swiss news sources"""
    enabled = {}
    
    for source_id, config in swiss_config.SWISS_SOURCES.items():
        if config.get("enabled", True):
            enabled[source_id] = config
            
    for source_id, config in swiss_config.REGIONAL_SOURCES.items():
        if config.get("enabled", False):
            enabled[source_id] = config
            
    return enabled

def get_sources_by_language(language: str) -> Dict[str, Dict]:
    """Get sources filtered by language"""
    enabled_sources = get_enabled_sources()
    return {
        source_id: config 
        for source_id, config in enabled_sources.items()
        if config.get("language") == language
    }

def get_topic_keywords(topic: str, language: Optional[str] = None) -> List[str]:
    """Get keywords for a specific topic, optionally filtered by language"""
    keywords = swiss_config.TOPIC_KEYWORDS.get(topic.lower(), [])
    
    if language and language in ["de", "fr", "it"]:
        # Filter keywords by language (basic heuristic)
        if language == "de":
            # German keywords typically don't have accents
            filtered = [k for k in keywords if not any(c in k for c in "àáâãäåæçèéêëìíîïñòóôõöøùúûüý")]
        elif language == "fr": 
            # French keywords may have accents
            filtered = [k for k in keywords if any(c in k for c in "àáâãäåæçèéêëìíîïñòóôõöøùúûüý") or k in ["politique", "gouvernement", "économie", "environnement", "société"]]
        elif language == "it":
            # Italian keywords may have specific endings
            filtered = [k for k in keywords if k.endswith(("a", "e", "i", "o")) and "ä" not in k]
        else:
            filtered = keywords
        
        return filtered if filtered else keywords
    
    return keywords

def validate_config() -> bool:
    """Validate Swiss news configuration"""
    errors = []
    
    # Check required settings
    if swiss_config.SCRAPING_INTERVAL_HOURS < 1:
        errors.append("SCRAPING_INTERVAL_HOURS must be at least 1")
        
    if swiss_config.SCRAPING_TIMEOUT_SECONDS < 5:
        errors.append("SCRAPING_TIMEOUT_SECONDS must be at least 5")
        
    # Check that at least one source is enabled
    enabled_sources = get_enabled_sources()
    if not enabled_sources:
        errors.append("At least one Swiss news source must be enabled")
        
    # Check bias analysis settings
    if swiss_config.BIAS_ANALYSIS_ENABLED and swiss_config.BIAS_ANALYSIS_BATCH_SIZE < 1:
        errors.append("BIAS_ANALYSIS_BATCH_SIZE must be at least 1")
        
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        return False
        
    return True