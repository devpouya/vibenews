import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

import google.generativeai as genai
from backend.config import settings
from backend.models.article import Article, BiasLabel

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for Gemini API calls"""
    
    def __init__(self, requests_per_minute: int = 5, requests_per_day: int = 25):
        self.rpm_limit = requests_per_minute
        self.daily_limit = requests_per_day
        self.requests_this_minute = []
        self.requests_today = 0
        self.last_reset_date = datetime.now().date()
    
    async def wait_if_needed(self):
        """Wait if rate limits would be exceeded"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.last_reset_date:
            self.requests_today = 0
            self.last_reset_date = now.date()
        
        # Check daily limit
        if self.requests_today >= self.daily_limit:
            logger.warning(f"Daily limit of {self.daily_limit} requests exceeded")
            raise Exception("Daily rate limit exceeded")
        
        # Clean old requests (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        self.requests_this_minute = [
            req_time for req_time in self.requests_this_minute 
            if req_time > minute_ago
        ]
        
        # Wait if per-minute limit would be exceeded
        if len(self.requests_this_minute) >= self.rpm_limit:
            wait_time = 60 - (now - self.requests_this_minute[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests_this_minute.append(now)
        self.requests_today += 1


class GeminiAnnotator:
    """Gemini-based bias annotation for articles"""
    
    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured")
        
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.rate_limiter = RateLimiter(
            settings.gemini_rpm_limit, 
            settings.gemini_daily_limit
        )
    
    def _create_bias_prompt(self, article: Article, topic: str) -> str:
        """Create prompt for bias annotation"""
        return f"""
Analyze the political bias of this Swiss news article regarding the topic: "{topic}"

Article Title: {article.title}
Article Content: {article.content[:2000]}...
Source: {article.source}

Please analyze the bias on a scale from -1.0 to +1.0 where:
- -1.0 = Strongly against/negative stance on the topic
- -0.5 = Moderately against/negative stance  
- 0.0 = Neutral/balanced coverage
- +0.5 = Moderately for/positive stance
- +1.0 = Strongly for/positive stance on the topic

Respond with a JSON object containing:
{{
    "bias_score": <float between -1.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the bias assessment>"
}}

Focus specifically on how the article presents the topic "{topic}" and consider:
- Language choice and framing
- Selection of facts and sources
- Overall tone and sentiment
- What perspectives are included or excluded
"""
    
    async def annotate_article(self, article: Article, topic: str) -> Optional[Dict[str, Any]]:
        """Annotate article bias for a specific topic"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            prompt = self._create_bias_prompt(article, topic)
            response = await self.model.generate_content_async(prompt)
            
            # Parse JSON response
            import json
            result = json.loads(response.text)
            
            # Validate response
            if not all(key in result for key in ['bias_score', 'confidence', 'reasoning']):
                logger.error(f"Invalid response format: {result}")
                return None
            
            # Clamp values to valid ranges
            result['bias_score'] = max(-1.0, min(1.0, float(result['bias_score'])))
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
            
            logger.info(f"Annotated article {article.id[:8]}... for topic '{topic}': {result['bias_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error annotating article {article.id}: {e}")
            return None