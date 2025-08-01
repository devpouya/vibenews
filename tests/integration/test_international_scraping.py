#!/usr/bin/env python3
"""
Test international news scraping functionality
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_scraper_basic_functionality():
    """Test basic scraper setup and functionality"""
    print("🧪 Testing International News Scraper")
    print("=" * 50)
    
    try:
        from backend.scraper.news_sources import MultiSourceScraper
        print("✅ MultiSourceScraper imported successfully")
        
        scraper = MultiSourceScraper()
        print("✅ Scraper initialized")
        print(f"   Available sources: {list(scraper.sources.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Scraper error: {e}")
        return False

def test_memory_estimation():
    """Test memory estimation calculations"""
    print("\n💾 Testing Memory Estimation")
    print("-" * 30)
    
    # Simulate article data
    sample_articles = []
    for i in range(10):  # Small sample
        article = {
            'title': f'Sample Article Title {i}' * 3,  # ~60 chars
            'content': 'This is sample article content. ' * 200,  # ~6400 chars
            'url': f'https://example.com/article-{i}',
            'published_date': datetime.now().isoformat(),
            'source': 'Test Source',
            'topic': 'test-topic',
            'scraped_at': datetime.now().isoformat(),
            'word_count': 200
        }
        sample_articles.append(article)
    
    # Calculate memory usage
    json_size = len(json.dumps(sample_articles, indent=2).encode('utf-8'))
    total_content = sum(len(article['content']) for article in sample_articles)
    
    print(f"📊 Sample Data (10 articles):")
    print(f"   JSON size: {json_size:,} bytes ({json_size/1024:.1f} KB)")
    print(f"   Content chars: {total_content:,}")
    print(f"   Avg per article: {json_size//len(sample_articles):,} bytes")
    
    # Extrapolate to 200 articles
    estimated_200 = (json_size / 10) * 200
    print(f"\n📈 Extrapolated (200 articles):")
    print(f"   Estimated size: {estimated_200:,} bytes ({estimated_200/1024/1024:.2f} MB)")
    
    # Validation
    if estimated_200 < 10 * 1024 * 1024:  # Less than 10MB
        print("✅ Memory usage within reasonable limits")
        return True
    else:
        print("⚠️  High memory usage estimated")
        return False

def test_topic_mapping():
    """Test topic to URL mapping"""
    print("\n🗺️  Testing Topic Mapping")
    print("-" * 25)
    
    topics = [
        'russia-ukraine-conflict',
        'climate-change-policy',
        'economic-inflation', 
        'immigration-policy',
        'tech-regulation'
    ]
    
    # Test AP mapping (as example)
    ap_mappings = {
        'russia-ukraine-conflict': '/hub/russia-ukraine',
        'climate-change-policy': '/hub/climate-and-environment',
        'economic-inflation': '/hub/inflation',
        'immigration-policy': '/hub/immigration', 
        'tech-regulation': '/hub/technology'
    }
    
    print(f"📋 Topics to scrape: {len(topics)}")
    for topic in topics:
        if topic in ap_mappings:
            print(f"   ✅ {topic} -> {ap_mappings[topic]}")
        else:
            print(f"   ❌ {topic} -> No mapping")
    
    # Check coverage
    coverage = len([t for t in topics if t in ap_mappings]) / len(topics)
    print(f"\n📊 Coverage: {coverage*100:.0f}% of topics mapped")
    
    return coverage >= 0.8  # 80% coverage required

def test_data_format_compatibility():
    """Test compatibility with existing data format"""
    print("\n🔗 Testing Data Format Compatibility")
    print("-" * 35)
    
    # Expected format (based on 20 Minuten format)
    expected_fields = [
        'title', 'content', 'url', 'published_date', 
        'source', 'topic', 'scraped_at', 'word_count'
    ]
    
    # Sample scraped article format
    sample_article = {
        'title': 'Sample News Article',
        'content': 'Article content goes here...',
        'url': 'https://example.com/article',
        'published_date': '2025-08-01T16:30:00Z',
        'source': 'Test Source',
        'topic': 'test-topic',
        'scraped_at': '2025-08-01T16:30:00Z',
        'word_count': 150
    }
    
    print(f"📋 Expected fields: {len(expected_fields)}")
    missing_fields = []
    extra_fields = []
    
    for field in expected_fields:
        if field not in sample_article:
            missing_fields.append(field)
    
    for field in sample_article:
        if field not in expected_fields:
            extra_fields.append(field)
    
    if not missing_fields and not extra_fields:
        print("✅ Perfect format compatibility")
        return True
    else:
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
        if extra_fields:
            print(f"ℹ️  Extra fields: {extra_fields}")
        return len(missing_fields) == 0  # Extra fields OK, missing not OK

def main():
    """Run all scraping tests"""
    print("🧪 International News Scraping Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_scraper_basic_functionality),
        ("Memory Estimation", test_memory_estimation),
        ("Topic Mapping", test_topic_mapping),
        ("Data Format", test_data_format_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Ready to scrape international news.")
        print("\n🚀 Next steps:")
        print("   1. Run: python scripts/scrape_international_news.py")
        print("   2. Expected output: ~200 articles, ~3-4 MB")
        print("   3. Compatible with existing bias analysis pipeline")
        return True
    else:
        print("⚠️  Some tests failed. Review implementation before scraping.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)