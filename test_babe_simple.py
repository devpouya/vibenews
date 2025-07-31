#!/usr/bin/env python3
"""
Simple test to load BABE dataset without Gemini API
Tests basic functionality of BABE utilities
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_babe_loading():
    """Test loading BABE dataset without Gemini"""
    print("🧪 Testing BABE dataset loading (no Gemini required)...")
    
    try:
        from datasets.babe_utils import BABEValidator
        print("✅ Imported BABEValidator successfully")
        
        # Initialize validator
        validator = BABEValidator()
        print("✅ Created BABEValidator instance")
        
        # Check if BABE file exists
        babe_file = "backend/data/raw/babe_with_annotations_20250731.jsonl"
        if not os.path.exists(babe_file):
            print(f"❌ BABE file not found: {babe_file}")
            print("Available files in data/raw:")
            raw_dir = Path("backend/data/raw")
            if raw_dir.exists():
                for file in raw_dir.iterdir():
                    print(f"  - {file.name}")
            return False
        
        print(f"✅ Found BABE file: {babe_file}")
        
        # Load dataset
        print("📂 Loading BABE dataset...")
        df = validator.load_babe_data('babe_with_annotations_20250731.jsonl')
        print(f"✅ Loaded {len(df)} samples")
        
        # Analyze bias distribution
        print("📊 Analyzing bias distribution...")
        analysis = validator.analyze_bias_distribution(df)
        print(f"✅ Analysis complete:")
        print(f"  - Total samples: {analysis['total_samples']}")
        print(f"  - Text stats: {analysis.get('text_statistics', {})}")
        
        # Create validation split
        print("✂️ Creating validation split...")
        train_df, val_df = validator.create_validation_split(df, test_size=0.2)
        print(f"✅ Split created: {len(train_df)} train, {len(val_df)} validation")
        
        print("\n🎉 All tests passed! BABE dataset is ready for use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_babe_loading()
    sys.exit(0 if success else 1)