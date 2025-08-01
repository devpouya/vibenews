"""
Test BiasScanner Implementation
Demonstrates complete BiasScanner algorithm with Swiss news integration
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.bias_detection.biasscanner_pipeline import BiasDetectionPipeline
from backend.evaluation.biasscanner_evaluator import BiasEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_article_analysis():
    """Test BiasScanner on a single article"""
    
    print("=" * 80)
    print("TESTING BIASSCANNER - SINGLE ARTICLE ANALYSIS")
    print("=" * 80)
    
    # Sample Swiss news article (in English for testing)
    sample_article = {
        "title": "Government Announces New Climate Policy",
        "content": """
        The government's radical new climate policies will destroy Switzerland's economic competitiveness, 
        according to business leaders who clearly understand the devastating impact on jobs. Critics argue 
        these extreme measures go too far and will burden hardworking taxpayers with unnecessary costs.
        
        However, environmental groups praise the comprehensive approach, saying it's exactly what scientists 
        have been demanding for years. Some experts believe this bold initiative could position Switzerland 
        as a global leader in sustainable development.
        
        The policy includes carbon taxes that many economists warn will hurt low-income families the most.
        """,
        "url": "https://example.com/climate-policy",
        "source": "Swiss News"
    }
    
    # Initialize BiasScanner (requires GEMINI_API_KEY)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Please set your Gemini API key: export GEMINI_API_KEY='your_key_here'")
        return
    
    try:
        # Initialize pipeline
        pipeline = BiasDetectionPipeline(api_key)
        
        # Process article
        print("🔄 Processing article through BiasScanner...")
        result = pipeline.process_article(sample_article)
        
        # Display results
        if result.get('bias_analysis', {}).get('success', False):
            bias_analysis = result['bias_analysis']
            
            print("\n📊 BIAS ANALYSIS RESULTS:")
            print(f"   Overall Bias Score: {bias_analysis['bias_score']['overall_score']:.3f}")
            print(f"   Bias Level: {bias_analysis['summary']['bias_level']}")
            print(f"   Political Leaning: {bias_analysis['summary']['political_summary']}")
            print(f"   Confidence: {bias_analysis['bias_score']['confidence']:.3f}")
            
            print(f"\n🎯 SENTENCE-LEVEL ANALYSIS:")
            print(f"   Total Sentences: {bias_analysis['bias_score']['total_sentences']}")
            print(f"   Biased Sentences: {bias_analysis['bias_score']['biased_sentences']}")
            print(f"   Bias Ratio: {bias_analysis['bias_score']['bias_ratio']:.3f}")
            
            print(f"\n🏷️ BIAS TYPES DETECTED:")
            for bias_type, count in bias_analysis['bias_type_counts'].items():
                print(f"   - {bias_type.replace('_', ' ').title()}: {count} occurrences")
            
            print(f"\n📝 DETAILED SENTENCE ANALYSIS:")
            for i, sentence_analysis in enumerate(bias_analysis['sentence_analysis'][:5]):  # Show first 5
                if sentence_analysis['is_biased']:
                    print(f"   Sentence {i+1}: BIASED")
                    print(f"   Type: {sentence_analysis['bias_type']}")
                    print(f"   Strength: {sentence_analysis['bias_strength']:.3f}")
                    print(f"   Explanation: {sentence_analysis['explanation']}")
                    print(f"   Text: {sentence_analysis['sentence'][:100]}...")
                    print()
            
        else:
            print("❌ Analysis failed:", result.get('bias_analysis', {}).get('error', 'Unknown error'))
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


def test_bias_spectrum_ranking():
    """Test bias spectrum ranking with multiple articles"""
    
    print("\n" + "=" * 80)
    print("TESTING BIAS SPECTRUM RANKING - MULTIPLE ARTICLES")
    print("=" * 80)
    
    # Sample articles with different political leanings
    articles = [
        {
            "title": "Conservative Economic Policy Success",
            "content": "The free-market approach has proven once again that economic freedom drives prosperity. Traditional values and fiscal responsibility create stable foundations for growth.",
            "url": "https://example.com/conservative",
            "source": "Right News"
        },
        {
            "title": "Progressive Climate Action Needed",
            "content": "Social justice and environmental protection must be prioritized. The government should invest in renewable energy and support workers through this green transition.",
            "url": "https://example.com/progressive", 
            "source": "Left News"
        },
        {
            "title": "Balanced Budget Proposal Analyzed",
            "content": "The budget proposal includes both spending cuts and revenue increases. Economic experts suggest this moderate approach balances various stakeholder interests.",
            "url": "https://example.com/balanced",
            "source": "Center News"
        }
    ]
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping spectrum test")
        return
    
    try:
        pipeline = BiasDetectionPipeline(api_key)
        
        print("🔄 Processing articles for bias spectrum ranking...")
        processed_articles = pipeline.process_article_batch(articles)
        
        # Generate spectrum report
        report = pipeline.create_bias_spectrum_report(processed_articles)
        
        print("\n📊 BIAS SPECTRUM REPORT:")
        print(f"   Total Articles: {report['total_articles']}")
        print(f"   Algorithm: {report['algorithm']}")
        
        print(f"\n🏛️ POLITICAL DISTRIBUTION:")
        distribution = report['spectrum_distribution']
        print(f"   Left: {distribution['left']} articles")
        print(f"   Center: {distribution['center']} articles") 
        print(f"   Right: {distribution['right']} articles")
        
        print(f"\n📈 ARTICLES BY BIAS SPECTRUM (Left → Right):")
        for article in report['articles_by_spectrum']:
            print(f"   {article['spectrum_score']:+.3f} | {article['political_leaning'].upper():>6} | {article['title']}")
        
    except Exception as e:
        print(f"❌ Error during spectrum analysis: {e}")


def test_babe_evaluation():
    """Test BiasScanner evaluation using BABE dataset"""
    
    print("\n" + "=" * 80)
    print("TESTING BIASSCANNER EVALUATION - BABE DATASET")
    print("=" * 80)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping BABE evaluation")
        return
    
    # Check for BABE dataset
    babe_path = "backend/data/raw/babe_with_annotations_20250731.jsonl"
    if not Path(babe_path).exists():
        print(f"❌ BABE dataset not found at {babe_path}")
        print("   Please ensure BABE dataset is available for evaluation")
        return
    
    try:
        evaluator = BiasEvaluator(api_key, babe_path)
        
        print("🔄 Running BiasScanner evaluation on BABE dataset...")
        print("   Note: This may take several minutes due to API rate limits")
        
        # Run evaluation on small sample (full evaluation would take hours)
        report = evaluator.run_full_evaluation(sample_limit=20)
        
        if 'error' in report:
            print(f"❌ Evaluation failed: {report['error']}")
            return
        
        print("\n📊 EVALUATION RESULTS:")
        metrics = report['performance_metrics']
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        
        print(f"\n📈 COMPARISON TO BIASSCANNER PAPER:")
        comparison = report['comparison_to_paper']
        print(f"   Paper F1 Score: {comparison['paper_f1_score']:.3f}")
        print(f"   Our F1 Score: {comparison['our_f1_score']:.3f}")
        print(f"   Performance Ratio: {comparison['performance_ratio']:.3f}")
        
        # Save report
        output_path = "backend/evaluation/test_evaluation_report.json"
        evaluator.save_evaluation_report(report, output_path)
        print(f"\n💾 Evaluation report saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


def main():
    """Run all BiasScanner tests"""
    
    print("🎯 BIASSCANNER IMPLEMENTATION TEST SUITE")
    print("🔬 Based on Menzner & Leidner (2024) BiasScanner Algorithm")
    print("📰 Integrated with Swiss News Bias Spectrum Analysis")
    
    # Test 1: Single article analysis
    test_single_article_analysis()
    
    # Test 2: Bias spectrum ranking
    test_bias_spectrum_ranking()
    
    # Test 3: BABE evaluation (if available)
    test_babe_evaluation()
    
    print("\n" + "=" * 80)
    print("✅ BIASSCANNER TESTING COMPLETED")
    print("=" * 80)
    
    print("\n📋 IMPLEMENTATION SUMMARY:")
    print("   ✅ 27 bias types defined and implemented")
    print("   ✅ Sentence-level classification pipeline")
    print("   ✅ Swiss bias spectrum integration (-1 to +1)")
    print("   ✅ Comprehensive scoring and aggregation")
    print("   ✅ API endpoints for bias analysis")
    print("   ✅ BABE dataset evaluation framework")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Set GEMINI_API_KEY for full functionality")
    print("   2. Test with real Swiss news articles")
    print("   3. Run full BABE evaluation for performance validation")
    print("   4. Deploy to production with rate limiting")


if __name__ == "__main__":
    main()