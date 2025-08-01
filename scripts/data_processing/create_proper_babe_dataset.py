"""
Create properly integrated BABE dataset with bias annotations
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_babe_with_annotations():
    """Load BABE dataset with proper bias annotations"""
    
    data_dir = Path("backend/datasets/raw/babe_extracted/data")
    
    print("=" * 60)
    print("CREATING INTEGRATED BABE DATASET")
    print("=" * 60)
    
    # Load the two bias annotation files
    datasets = {}
    
    # Load SG1 (uses semicolon separator)
    sg1_path = data_dir / "final_labels_SG1.csv"
    if sg1_path.exists():
        try:
            df_sg1 = pd.read_csv(sg1_path, sep=";")
            datasets['SG1'] = df_sg1
            print(f"✅ Loaded SG1: {len(df_sg1)} samples")
            print(f"   Columns: {list(df_sg1.columns)}")
            print(f"   Bias labels: {df_sg1['label_bias'].value_counts().to_dict()}")
            print(f"   Opinion labels: {df_sg1['label_opinion'].value_counts().to_dict()}")
        except Exception as e:
            print(f"❌ Error loading SG1: {e}")
    
    # Load SG2 (uses semicolon separator) 
    sg2_path = data_dir / "final_labels_SG2.csv"
    if sg2_path.exists():
        try:
            df_sg2 = pd.read_csv(sg2_path, sep=";")
            datasets['SG2'] = df_sg2
            print(f"✅ Loaded SG2: {len(df_sg2)} samples")
            print(f"   Columns: {list(df_sg2.columns)}")
            print(f"   Bias labels: {df_sg2['label_bias'].value_counts().to_dict()}")
            print(f"   Opinion labels: {df_sg2['label_opinion'].value_counts().to_dict()}")
        except Exception as e:
            print(f"❌ Error loading SG2: {e}")
    
    if not datasets:
        print("❌ No datasets loaded successfully")
        return
    
    # Combine datasets
    print(f"\n🔄 Combining datasets...")
    combined_data = []
    
    for dataset_name, df in datasets.items():
        for idx, row in df.iterrows():
            # Create structured entry
            entry = {
                'id': f"babe_{dataset_name.lower()}_{idx}",
                'dataset': 'babe',
                'source_dataset': dataset_name,
                'created_at': datetime.now().isoformat(),
                
                # Main content
                'text': row.get('text', ''),
                'news_link': row.get('news_link', ''),
                'outlet': row.get('outlet', ''),
                'topic': row.get('topic', ''),
                
                # Bias annotations (preserve original BABE labels)
                'bias_labels': {
                    'label_bias': row.get('label_bias', ''),
                    'label_opinion': row.get('label_opinion', ''),
                    'outlet_type': row.get('type', ''),  # left/center/right
                    'biased_words': row.get('biased_words', ''),
                },
                
                # Original data for reference
                'original_data': row.to_dict()
            }
            
            combined_data.append(entry)
    
    print(f"✅ Combined {len(combined_data)} total samples")
    
    # Save to JSON Lines
    output_path = Path("backend/data/raw/babe_with_annotations_20250731.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved to: {output_path}")
    
    # Analysis
    print(f"\n📊 Dataset Analysis:")
    
    # Bias label distribution
    bias_labels = [entry['bias_labels']['label_bias'] for entry in combined_data]
    bias_dist = pd.Series(bias_labels).value_counts()
    print(f"   Bias distribution: {bias_dist.to_dict()}")
    
    # Opinion label distribution  
    opinion_labels = [entry['bias_labels']['label_opinion'] for entry in combined_data]
    opinion_dist = pd.Series(opinion_labels).value_counts()
    print(f"   Opinion distribution: {opinion_dist.to_dict()}")
    
    # Topic distribution
    topics = [entry['topic'] for entry in combined_data]
    topic_dist = pd.Series(topics).value_counts()
    print(f"   Topic distribution: {topic_dist.head().to_dict()}")
    
    # Outlet type distribution
    outlet_types = [entry['bias_labels']['outlet_type'] for entry in combined_data]
    outlet_dist = pd.Series(outlet_types).value_counts()
    print(f"   Outlet type distribution: {outlet_dist.to_dict()}")
    
    # Text length stats
    text_lengths = [len(entry['text']) for entry in combined_data if entry['text']]
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        print(f"   Average text length: {avg_length:.0f} characters")
        print(f"   Text length range: {min(text_lengths)} - {max(text_lengths)}")
    
    print(f"\n✅ BABE DATASET WITH ANNOTATIONS CREATED SUCCESSFULLY")
    return str(output_path)


if __name__ == "__main__":
    output_file = load_babe_with_annotations()