"""
BABE Dataset Explorer
Script to download, explore, and integrate the BABE dataset
"""

import asyncio
import logging
from backend.datasets.babe_loader import BABEDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Explore the BABE dataset"""
    
    logger.info("=== BABE Dataset Integration ===")
    
    # Initialize loader
    loader = BABEDatasetLoader()
    
    # Step 1: Check for downloaded dataset
    logger.info("\n=== STEP 1: Dataset Download Check ===")
    print(loader.download_instructions())
    
    # Step 2: Try to extract if ZIP exists
    logger.info("\n=== STEP 2: Dataset Extraction ===")
    extraction_success = loader.extract_dataset()
    
    if not extraction_success:
        logger.warning("Dataset extraction failed. Please download the dataset manually.")
        return
    
    # Step 3: Load dataset
    logger.info("\n=== STEP 3: Loading Dataset ===")
    datasets = loader.load_dataset()
    
    if not datasets:
        logger.error("Failed to load dataset")
        return
    
    # Step 4: Explore dataset
    logger.info("\n=== STEP 4: Dataset Exploration ===")
    exploration = loader.explore_dataset(datasets)
    
    # Step 5: Assess scalability
    logger.info("\n=== STEP 5: Scalability Assessment ===")
    stats = loader.get_dataset_stats(datasets)
    
    print("\n" + "="*60)
    print("SCALABILITY ASSESSMENT")
    print("="*60)
    print(f"Total files: {stats['total_files']}")
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"JSON Lines feasible: {stats['scalability_assessment']['json_lines_feasible']}")
    print(f"Memory efficient: {stats['scalability_assessment']['memory_efficient']}")
    print(f"Recommendation: {stats['scalability_assessment']['recommendation']}")
    
    # Step 6: Convert to JSON Lines if feasible
    if stats['scalability_assessment']['json_lines_feasible']:
        logger.info("\n=== STEP 6: Converting to JSON Lines ===")
        jsonl_path = loader.convert_to_jsonl(datasets)
        print(f"✅ BABE dataset converted to: {jsonl_path}")
        
        # Show bias label distribution
        print("\nBias Label Distribution:")
        for label_type, distribution in exploration['bias_labels'].items():
            print(f"  {label_type}: {distribution}")
    else:
        logger.warning("Dataset too large for JSON Lines integration")
    
    print("\n" + "="*60)
    print("✅ BABE DATASET EXPLORATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()