from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime

# Simple vector store imports
from backend.vector_store.article_vector_store import ArticleVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VibeNews API",
    description="Simplified news article serving from vector database",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
vector_store = None

def get_vector_store():
    """Get or initialize vector store"""
    global vector_store
    if vector_store is None:
        try:
            vector_store = ArticleVectorStore()
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            vector_store = None
    return vector_store

@app.get("/")
async def root():
    return {
        "message": "VibeNews API v2.0 - Simplified",
        "version": "2.0.0",
        "description": "Streamlined article serving from vector database",
        "endpoints": {
            "articles": "/articles",
            "search": "/search",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    store = get_vector_store()
    return {
        "status": "healthy" if store else "degraded",
        "version": "2.0.0",
        "vector_store": "available" if store else "unavailable",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/articles")
async def get_articles(
    limit: int = Query(50, ge=1, le=200, description="Number of articles to return"),
    skip: int = Query(0, ge=0, description="Number of articles to skip")
):
    """Get articles from vector database"""
    try:
        store = get_vector_store()
        if not store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Get articles from vector store (use get_all_articles for now to avoid date filtering issues)
        articles = store.get_all_articles(limit=limit + skip)
        
        # Apply pagination
        paginated_articles = articles[skip:skip + limit] if articles else []
        
        return {
            "articles": paginated_articles,
            "count": len(paginated_articles),
            "total": len(articles) if articles else 0,
            "limit": limit,
            "skip": skip
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_articles(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return")
):
    """Search articles using semantic search"""
    try:
        store = get_vector_store()
        if not store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Perform search
        articles = store.search_articles(query=q, limit=limit)
        
        return {
            "articles": articles,
            "count": len(articles),
            "query": q
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/{article_id}")
async def get_article(article_id: str):
    """Get a specific article by ID"""
    try:
        store = get_vector_store()
        if not store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Get the specific article
        result = store.articles_collection.get(
            ids=[f"article_{article_id}"],
            include=["documents", "metadatas"]
        )
        
        if not result['documents']:
            raise HTTPException(status_code=404, detail="Article not found")
        
        # Format the article
        article = {
            "id": article_id,
            "title": result['metadatas'][0].get('title', 'Untitled'),
            "content": result['documents'][0],
            "url": result['metadatas'][0].get('url', ''),
            "source": result['metadatas'][0].get('source', 'Unknown'),
            "language": result['metadatas'][0].get('language', 'en'),
            "published_date": result['metadatas'][0].get('published_date', ''),
            "bias_score": result['metadatas'][0].get('bias_score', 0.0),
            "word_count": result['metadatas'][0].get('word_count', 0),
            "metadata": result['metadatas'][0]
        }
        
        return {"article": article}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/{article_id}/similar")
async def get_similar_articles(
    article_id: str,
    limit: int = Query(5, ge=1, le=20, description="Number of similar articles to return")
):
    """Get articles similar to the specified article"""
    try:
        store = get_vector_store()
        if not store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Get similar articles using simple similarity (for testing with zero embeddings)
        similar_articles = store.get_simple_similar_articles(int(article_id), limit)
        
        return {
            "article_id": article_id,
            "similar_articles": similar_articles,
            "count": len(similar_articles)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar articles for {article_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get vector database statistics"""
    try:
        store = get_vector_store()
        if not store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        stats = store.get_statistics()
        
        return {
            "vector_store": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")