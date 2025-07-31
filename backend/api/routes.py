from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import os

from backend.vector_store.chroma_store import ChromaVectorStore
from backend.bias_detection.biasscanner_pipeline import BiasDetectionPipeline

router = APIRouter()

# Initialize vector store
vector_store = ChromaVectorStore()

# Initialize BiasScanner pipeline (if API key available)
bias_pipeline = None
if os.getenv('GEMINI_API_KEY'):
    bias_pipeline = BiasDetectionPipeline(os.getenv('GEMINI_API_KEY'))


class ArticleResponse(BaseModel):
    id: str
    title: str
    content: str
    url: str
    source: str
    author: Optional[str]
    published_at: str
    bias_score: float
    topic: str
    distance: Optional[float] = None
    bias_analysis: Optional[Dict[str, Any]] = None


class BiasSpectrumResponse(BaseModel):
    topic: str
    total_articles: int
    articles: List[ArticleResponse]


@router.get("/topics", response_model=List[str])
async def get_topics():
    """Get all available topics"""
    try:
        topics = vector_store.get_topics()
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching topics: {str(e)}")


@router.get("/search", response_model=List[ArticleResponse])
async def search_articles(
    query: str = Query(..., description="Search query"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    min_bias: float = Query(-1.0, ge=-1.0, le=1.0, description="Minimum bias score"),
    max_bias: float = Query(1.0, ge=-1.0, le=1.0, description="Maximum bias score"),
    limit: int = Query(20, ge=1, le=100, description="Number of results")
):
    """Search articles with optional topic and bias filtering"""
    try:
        bias_range = (min_bias, max_bias) if min_bias != -1.0 or max_bias != 1.0 else None
        
        results = vector_store.search_by_topic(
            query=query,
            topic=topic,
            bias_range=bias_range,
            limit=limit
        )
        
        articles = []
        for result in results:
            metadata = result['metadata']
            article = ArticleResponse(
                id=result['id'],
                title=metadata['title'],
                content=result['content'][:500] + "..." if len(result['content']) > 500 else result['content'],
                url=metadata['url'],
                source=metadata['source'],
                author=metadata.get('author'),
                published_at=metadata['published_at'],
                bias_score=metadata['bias_score'],
                topic=metadata['topic'],
                distance=result.get('distance')
            )
            articles.append(article)
        
        return articles
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching articles: {str(e)}")


@router.get("/bias-spectrum/{topic}", response_model=BiasSpectrumResponse)
async def get_bias_spectrum(
    topic: str,
    min_bias: float = Query(-1.0, ge=-1.0, le=1.0, description="Minimum bias score"),
    max_bias: float = Query(1.0, ge=-1.0, le=1.0, description="Maximum bias score"),
    limit: int = Query(50, ge=1, le=100, description="Number of articles")
):
    """Get articles for a topic sorted by bias score (bias spectrum view)"""
    try:
        results = vector_store.get_articles_by_bias_range(
            topic=topic,
            min_bias=min_bias,
            max_bias=max_bias,
            limit=limit
        )
        
        articles = []
        for result in results:
            metadata = result['metadata']
            article = ArticleResponse(
                id=result['id'],
                title=metadata['title'],
                content=result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
                url=metadata['url'],
                source=metadata['source'],
                author=metadata.get('author'),
                published_at=metadata['published_at'],
                bias_score=metadata['bias_score'],
                topic=metadata['topic']
            )
            articles.append(article)
        
        return BiasSpectrumResponse(
            topic=topic,
            total_articles=len(articles),
            articles=articles
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching bias spectrum: {str(e)}")


@router.get("/article/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: str):
    """Get full article details by ID"""
    try:
        # ChromaDB get by ID
        results = vector_store.collection.get(
            ids=[article_id],
            include=["documents", "metadatas"]
        )
        
        if not results['documents']:
            raise HTTPException(status_code=404, detail="Article not found")
        
        metadata = results['metadatas'][0]
        article = ArticleResponse(
            id=article_id,
            title=metadata['title'],
            content=results['documents'][0],
            url=metadata['url'],
            source=metadata['source'],
            author=metadata.get('author'),
            published_at=metadata['published_at'],
            bias_score=metadata['bias_score'],
            topic=metadata['topic']
        )
        
        return article
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching article: {str(e)}")


@router.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        total_articles = vector_store.collection.count()
        topics = vector_store.get_topics()
        
        return {
            "total_articles": total_articles,
            "total_topics": len(topics),
            "topics": topics,
            "biasscanner_enabled": bias_pipeline is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@router.post("/analyze-bias")
async def analyze_article_bias(
    article_text: str,
    title: Optional[str] = None,
    url: Optional[str] = None
):
    """Analyze bias in article text using BiasScanner"""
    if not bias_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="BiasScanner not available - missing GEMINI_API_KEY"
        )
    
    try:
        # Create temporary article data
        article_data = {
            "title": title or "Untitled",
            "content": article_text,
            "url": url or "",
            "source": "manual_input"
        }
        
        # Process through BiasScanner pipeline
        result = bias_pipeline.process_article(article_data)
        
        return {
            "success": True,
            "bias_analysis": result.get("bias_analysis", {}),
            "summary": {
                "bias_level": result.get("bias_analysis", {}).get("summary", {}).get("bias_level", "Unknown"),
                "political_leaning": result.get("bias_analysis", {}).get("swiss_bias_spectrum", {}).get("political_leaning", "center"),
                "spectrum_score": result.get("bias_analysis", {}).get("swiss_bias_spectrum", {}).get("spectrum_score", 0.0),
                "confidence": result.get("bias_analysis", {}).get("bias_score", {}).get("confidence", 0.0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing bias: {str(e)}")


@router.get("/bias-spectrum-report/{topic}")
async def get_bias_spectrum_report(topic: str):
    """Get comprehensive bias spectrum report for a topic"""
    try:
        # Get articles for topic from vector store
        results = vector_store.get_articles_by_bias_range(
            topic=topic,
            min_bias=-1.0,
            max_bias=1.0,
            limit=100
        )
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No articles found for topic: {topic}")
        
        # Convert to article format
        articles = []
        for result in results:
            metadata = result['metadata']
            article = {
                "title": metadata['title'],
                "content": result['content'],
                "url": metadata['url'],
                "source": metadata['source'],
                "published_at": metadata['published_at'],
                "bias_score": metadata['bias_score'],
                "topic": metadata['topic']
            }
            articles.append(article)
        
        # Generate spectrum report if BiasScanner available
        if bias_pipeline:
            # Process articles through BiasScanner
            processed_articles = bias_pipeline.process_article_batch(articles)
            report = bias_pipeline.create_bias_spectrum_report(processed_articles)
        else:
            # Fallback report using existing bias scores
            report = {
                "timestamp": "BiasScanner not available",
                "total_articles": len(articles),
                "spectrum_distribution": {"left": 0, "center": len(articles), "right": 0},
                "articles_by_spectrum": [
                    {
                        "title": a["title"],
                        "url": a["url"],
                        "spectrum_score": a["bias_score"],
                        "political_leaning": "center",
                        "overall_bias": abs(a["bias_score"])
                    }
                    for a in sorted(articles, key=lambda x: x["bias_score"])
                ],
                "algorithm": "Legacy bias scores"
            }
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating bias spectrum report: {str(e)}")


@router.get("/bias-types")
async def get_bias_types():
    """Get information about the 27 bias types detected by BiasScanner"""
    if not bias_pipeline:
        raise HTTPException(
            status_code=503,
            detail="BiasScanner not available - missing GEMINI_API_KEY"
        )
    
    from backend.bias_detection.bias_types import BIAS_DEFINITIONS
    
    bias_types_info = {}
    for bias_type, definition in BIAS_DEFINITIONS.items():
        bias_types_info[bias_type.value] = {
            "name": definition.name,
            "description": definition.description,
            "examples": definition.examples,
            "severity_weight": definition.severity_weight
        }
    
    return {
        "total_bias_types": len(bias_types_info),
        "bias_types": bias_types_info,
        "algorithm": "BiasScanner v1.0.0"
    }