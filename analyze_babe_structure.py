"""
Analyze BABE dataset structure to understand the bias annotations
"""

import pandas as pd
import os
from pathlib import Path

def analyze_babe_files():
    """Analyze the structure of BABE dataset files"""
    
    data_dir = Path("backend/datasets/raw/babe_extracted/data")
    
    print("=" * 60)
    print("BABE DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Analyze key files
    key_files = [
        "news_headlines_usa_biased.csv",
        "news_headlines_usa_neutral.csv", 
        "final_labels_MBIC.csv",
        "final_labels_SG1.csv",
        "final_labels_SG2.csv"
    ]
    
    for filename in key_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"❌ {filename} - Not found")
            continue
            
        print(f"\n📄 {filename}")
        print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            # Try different CSV parsing approaches
            approaches = [
                {"sep": ",", "quoting": 0},  # Standard CSV
                {"sep": ",", "quoting": 1},  # Quote all
                {"sep": ",", "error_bad_lines": False},  # Skip bad lines
                {"sep": ";", "quoting": 0},  # Semicolon separated
                {"sep": "\t", "quoting": 0},  # Tab separated
            ]
            
            loaded = False
            for i, params in enumerate(approaches):
                try:
                    df = pd.read_csv(filepath, **params, nrows=5)  # Just first 5 rows
                    print(f"   ✅ Loaded with approach {i+1}: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    
                    # Show sample data
                    print("   Sample data:")
                    for idx, row in df.head(2).iterrows():
                        print(f"     Row {idx+1}:")
                        for col in df.columns[:5]:  # Show first 5 columns
                            val = str(row[col])[:50] + "..." if len(str(row[col])) > 50 else row[col]
                            print(f"       {col}: {val}")
                    
                    loaded = True
                    break
                    
                except Exception as e:
                    continue
            
            if not loaded:
                print(f"   ❌ Could not load with any approach")
                
                # Try to read first few lines manually
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [f.readline().strip() for _ in range(5)]
                    print("   Raw content (first 5 lines):")
                    for i, line in enumerate(lines):
                        preview = line[:100] + "..." if len(line) > 100 else line
                        print(f"     Line {i+1}: {preview}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Look for any README or description files
    print(f"\n📋 Other files in directory:")
    all_files = list(data_dir.glob("*"))
    for file in all_files:
        if file.name not in key_files:
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"   {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    analyze_babe_files()