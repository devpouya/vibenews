"""
Test the dual storage system with scraped articles
"""
import asyncio
import logging
from backend.scraper.twentymin_scraper import TwentyMinScraper
from backend.storage.dual_storage import DualStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_storage_system():
    """Test scraping and storing articles"""
    
    # Step 1: Scrape articles
    logger.info("=== STEP 1: Scraping Articles ===")
    scraper = TwentyMinScraper()
    articles = scraper.scrape_politics_articles(limit=20)
    logger.info(f"Scraped {len(articles)} articles")
    
    if not articles:
        logger.error("No articles scraped, aborting test")
        return
    
    # Step 2: Initialize dual storage
    logger.info("=== STEP 2: Initializing Storage ===")
    storage = DualStorage(load_embedding_model=False)  # Skip embeddings for now
    
    # Step 3: Store articles in JSON Lines format
    logger.info("=== STEP 3: Storing Articles ===")
    results = storage.store_articles(
        articles=articles,
        jsonl_filename=None,  # Auto-generate filename
        store_in_vector_db=False  # Skip vector DB for now
    )
    
    logger.info(f"Storage results: {results}")
    
    # Step 4: Get storage statistics
    logger.info("=== STEP 4: Storage Statistics ===")
    stats = storage.get_storage_stats()
    
    print("\n" + "="*50)
    print("STORAGE STATISTICS")
    print("="*50)
    
    print(f"\nJSON Lines Files:")
    for file_type, files in stats['jsonl_files'].items():
        print(f"  {file_type}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"    - {file}")
    
    print(f"\nVector Store:")
    print(f"  {stats['vector_store']}")
    
    # Step 5: Test loading data
    logger.info("=== STEP 5: Testing Data Loading ===")
    if stats['jsonl_files']['raw']:
        latest_file = stats['jsonl_files']['raw'][-1]  # Get latest file
        logger.info(f"Loading articles from: {latest_file}")
        
        loaded_articles = storage.json_storage.load_articles(latest_file)
        logger.info(f"Loaded {len(loaded_articles)} articles")
        
        # Show sample article
        if loaded_articles:
            sample = loaded_articles[0]
            print(f"\nSample Article:")
            print(f"  ID: {sample['id']}")
            print(f"  Title: {sample['title']}")
            print(f"  Source: {sample['source']}")
            print(f"  Content length: {len(sample['content'])} chars")
            print(f"  Published: {sample['published_at']}")
    
    print("\n" + "="*50)
    print("✅ STORAGE TEST COMPLETED SUCCESSFULLY")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(test_storage_system())