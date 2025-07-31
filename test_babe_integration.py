"""
Test BABE Dataset Integration
Complete test of BABE dataset integration once downloaded
"""

import logging
from backend.datasets.babe_loader import BABEDatasetLoader
from backend.datasets.babe_utils import BABEValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_integration():
    """Test complete BABE integration workflow"""
    
    print("="*60)
    print("BABE DATASET INTEGRATION TEST")
    print("="*60)
    
    # Step 1: Initialize components
    loader = BABEDatasetLoader()
    validator = BABEValidator()
    
    # Step 2: Load dataset (assumes already extracted)
    logger.info("=== Loading BABE Dataset ===")
    datasets = loader.load_dataset()
    
    if not datasets:
        print("❌ No dataset found. Please download and extract BABE dataset first.")
        print(loader.download_instructions())
        return False
    
    # Step 3: Explore dataset
    logger.info("=== Exploring Dataset Structure ===")
    exploration = loader.explore_dataset(datasets)
    
    # Step 4: Check scalability
    logger.info("=== Scalability Assessment ===")
    stats = loader.get_dataset_stats(datasets)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Files: {stats['total_files']}")
    print(f"   Rows: {stats['total_rows']:,}")
    print(f"   Size: {stats['total_size_mb']} MB")
    print(f"   JSON Lines feasible: {stats['scalability_assessment']['json_lines_feasible']}")
    print(f"   Recommendation: {stats['scalability_assessment']['recommendation']}")
    
    # Step 5: Convert to JSON Lines
    if stats['scalability_assessment']['json_lines_feasible']:
        logger.info("=== Converting to JSON Lines ===")
        jsonl_path = loader.convert_to_jsonl(datasets)
        print(f"✅ Converted to: {jsonl_path}")
        
        # Step 6: Load and validate converted data
        logger.info("=== Validating Converted Data ===")
        import os
        filename = os.path.basename(jsonl_path)
        babe_df = validator.load_babe_data(filename)
        
        print(f"📋 Loaded {len(babe_df)} BABE samples")
        
        # Step 7: Analyze bias distribution
        logger.info("=== Analyzing Bias Distribution ===")
        bias_analysis = validator.analyze_bias_distribution(babe_df)
        print(f"📈 Bias Analysis Summary:")
        print(f"   Total samples: {bias_analysis['total_samples']}")
        print(f"   Text avg length: {bias_analysis.get('text_statistics', {}).get('avg_length', 'N/A')}")
        
        for label_type, distribution in bias_analysis['label_distribution'].items():
            print(f"   {label_type}: {distribution}")
        
        # Step 8: Create validation split
        logger.info("=== Creating Validation Split ===")
        train_df, val_df = validator.create_validation_split(babe_df)
        print(f"📝 Validation split: {len(train_df)} train, {len(val_df)} validation")
        
        # Step 9: Prepare for pretraining
        logger.info("=== Preparing Pretraining Data ===")
        training_examples = validator.prepare_for_pretraining(babe_df[:10])  # Sample
        print(f"🚀 Sample training examples: {len(training_examples)}")
        
        if training_examples:
            print(f"   Example format: {list(training_examples[0].keys())}")
        
        # Step 10: Export validation set
        logger.info("=== Exporting Validation Set ===")
        val_path = validator.export_for_validation(val_df, "babe_validation.jsonl")
        print(f"💾 Validation set exported to: {val_path}")
        
        print(f"\n✅ INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print(f"   Dataset ready for:")
        print(f"   - Model validation")
        print(f"   - Gemini LoRA pretraining")
        print(f"   - Bias analysis research")
        
        return True
    
    else:
        print("❌ Dataset too large for JSON Lines integration")
        return False


def compare_with_swiss_data():
    """Compare BABE with Swiss articles if both available"""
    
    validator = BABEValidator()
    
    try:
        # Try to load both datasets
        babe_df = validator.load_babe_data("babe_dataset_20250731.jsonl")  # Update filename
        
        # Find latest Swiss articles file
        from backend.storage.json_storage import JSONLStorage
        storage = JSONLStorage()
        swiss_files = storage.list_files('raw')
        swiss_files = [f for f in swiss_files if f.startswith('articles_')]
        
        if not swiss_files:
            print("❌ No Swiss articles found for comparison")
            return
        
        swiss_file = swiss_files[-1]  # Latest file
        
        print(f"\n🔄 Comparing BABE with Swiss articles...")
        comparison = validator.compare_with_swiss_articles(babe_df, swiss_file)
        
        print(f"📊 Comparison Results:")
        print(f"   BABE samples: {comparison['dataset_sizes']['babe']}")
        print(f"   Swiss samples: {comparison['dataset_sizes']['swiss']}")
        
        if 'text_lengths' in comparison:
            print(f"   BABE avg length: {comparison['text_lengths']['babe_avg']:.0f} chars")
            print(f"   Swiss avg length: {comparison['text_lengths']['swiss_avg']:.0f} chars")
        
        if 'language_analysis' in comparison:
            lang_analysis = comparison['language_analysis']
            print(f"   Language mismatch: {lang_analysis.get('language_mismatch', 'Unknown')}")
            print(f"   BABE likely English: {lang_analysis.get('babe_likely_english', 'Unknown')}")
            print(f"   Swiss likely German: {lang_analysis.get('swiss_likely_german', 'Unknown')}")
        
        print(f"✅ Comparison completed")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


if __name__ == "__main__":
    success = test_full_integration()
    
    if success:
        print(f"\n" + "="*60)
        compare_with_swiss_data()
        print("="*60)