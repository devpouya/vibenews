"""
BABE Dataset Loader and Explorer
Loads the BABE (Media Bias Annotations by Experts) dataset for validation and pretraining
"""

import os
import zipfile
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from backend.config import settings
from backend.storage.json_storage import JSONLStorage

logger = logging.getLogger(__name__)


class BABEDatasetLoader:
    """Loader for BABE media bias dataset"""
    
    def __init__(self):
        self.datasets_dir = Path("backend/datasets/raw")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.json_storage = JSONLStorage()
        
        # Expected BABE files after extraction
        self.babe_files = {
            'sentences': 'babe_sentences.csv',
            'words': 'babe_words.csv',
            'metadata': 'babe_metadata.csv'
        }
    
    def download_instructions(self) -> str:
        """Return instructions for manual dataset download"""
        return """
To use the BABE dataset:

1. Go to: https://www.kaggle.com/datasets/timospinde/babe-media-bias-annotations-by-experts
2. Download the dataset ZIP file (requires Kaggle account)
3. Place the ZIP file in: backend/datasets/raw/
4. Run the load_dataset() method to extract and process

The dataset contains:
- 3,700 sentences with expert bias annotations
- Word-level and sentence-level bias labels
- English language news articles
- Size: ~62MB
"""
    
    def extract_dataset(self, zip_path: Optional[str] = None) -> bool:
        """
        Extract BABE dataset from ZIP file
        
        Args:
            zip_path: Path to ZIP file, or None to auto-detect
        
        Returns:
            True if extraction successful
        """
        if zip_path is None:
            # Look for ZIP files in datasets directory
            zip_files = list(self.datasets_dir.glob("*.zip"))
            if not zip_files:
                logger.error("No ZIP files found in datasets directory")
                logger.info(self.download_instructions())
                return False
            zip_path = zip_files[0]
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            return False
        
        try:
            extract_dir = self.datasets_dir / "babe_extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted BABE dataset to: {extract_dir}")
            
            # List extracted files
            extracted_files = list(extract_dir.rglob("*"))
            logger.info(f"Extracted {len(extracted_files)} files:")
            for file in extracted_files[:10]:  # Show first 10
                logger.info(f"  - {file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            return False
    
    def load_dataset(self, extracted_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load BABE dataset into pandas DataFrames
        
        Args:
            extracted_dir: Directory with extracted files, or None to auto-detect
        
        Returns:
            Dictionary with DataFrames for sentences, words, metadata
        """
        if extracted_dir is None:
            extracted_dir = self.datasets_dir / "babe_extracted"
        
        extracted_dir = Path(extracted_dir)
        if not extracted_dir.exists():
            logger.error(f"Extracted directory not found: {extracted_dir}")
            logger.info("Run extract_dataset() first")
            return {}
        
        # Find CSV files (they might be in subdirectories)
        csv_files = list(extracted_dir.rglob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in extracted directory")
            return {}
        
        datasets = {}
        
        logger.info(f"Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.name} ({csv_file.stat().st_size / 1024:.1f} KB)")
            
            try:
                df = pd.read_csv(csv_file)
                datasets[csv_file.stem] = df
                logger.info(f"    Loaded: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
        
        return datasets
    
    def explore_dataset(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Explore and analyze the BABE dataset
        
        Args:
            datasets: Dictionary of DataFrames from load_dataset()
        
        Returns:
            Dictionary with exploration results
        """
        exploration = {
            'overview': {},
            'bias_labels': {},
            'sample_data': {}
        }
        
        for name, df in datasets.items():
            logger.info(f"\n=== Exploring {name} ===")
            
            # Basic info
            info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            exploration['overview'][name] = info
            
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Look for bias-related columns
            bias_columns = [col for col in df.columns if 'bias' in col.lower()]
            if bias_columns:
                logger.info(f"Bias columns: {bias_columns}")
                
                for col in bias_columns:
                    if col in df.columns:
                        value_counts = df[col].value_counts()
                        exploration['bias_labels'][f"{name}_{col}"] = value_counts.to_dict()
                        logger.info(f"  {col} distribution: {value_counts.to_dict()}")
            
            # Sample data
            if len(df) > 0:
                sample = df.head(3).to_dict('records')
                exploration['sample_data'][name] = sample
                
                # Show sample with truncated text
                logger.info("Sample data:")
                for i, row in enumerate(df.head(2).iterrows()):
                    logger.info(f"  Row {i+1}:")
                    for col, val in row[1].items():
                        if isinstance(val, str) and len(val) > 100:
                            val = val[:100] + "..."
                        logger.info(f"    {col}: {val}")
        
        return exploration
    
    def convert_to_jsonl(
        self, 
        datasets: Dict[str, pd.DataFrame], 
        output_filename: Optional[str] = None
    ) -> str:
        """
        Convert BABE dataset to JSON Lines format for integration
        
        Args:
            datasets: Dictionary of DataFrames from load_dataset()
            output_filename: Optional filename, defaults to babe_dataset_YYYYMMDD.jsonl
        
        Returns:
            Path to created JSON Lines file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_filename = f"babe_dataset_{timestamp}.jsonl"
        
        # Determine which dataset contains the main sentence data
        main_dataset = None
        for name, df in datasets.items():
            if 'sentence' in name.lower() or len(df) > 1000:  # Likely the main dataset
                main_dataset = df
                main_dataset_name = name
                break
        
        if main_dataset is None:
            # Use the largest dataset
            main_dataset_name = max(datasets.keys(), key=lambda k: len(datasets[k]))
            main_dataset = datasets[main_dataset_name]
        
        logger.info(f"Using {main_dataset_name} as main dataset with {len(main_dataset)} rows")
        
        # Convert to BABE format compatible with our system
        babe_articles = []
        
        for idx, row in main_dataset.iterrows():
            # Create a structured article format
            article_data = {
                'id': f"babe_{idx}",
                'dataset': 'babe',
                'source_dataset': main_dataset_name,
                'created_at': datetime.now().isoformat(),
                'original_data': row.to_dict()
            }
            
            # Extract key fields if available
            text_columns = [col for col in row.index if any(keyword in col.lower() 
                          for keyword in ['text', 'sentence', 'content', 'article'])]
            if text_columns:
                article_data['text'] = row[text_columns[0]]
            
            # Extract bias labels
            bias_columns = [col for col in row.index if 'bias' in col.lower()]
            if bias_columns:
                article_data['bias_labels'] = {col: row[col] for col in bias_columns}
            
            babe_articles.append(article_data)
        
        # Save to JSON Lines
        filepath = self.json_storage.raw_dir / output_filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in babe_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Converted {len(babe_articles)} BABE entries to {filepath}")
        return str(filepath)
    
    def get_dataset_stats(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get comprehensive statistics about the BABE dataset"""
        total_size_mb = sum(df.memory_usage(deep=True).sum() for df in datasets.values()) / 1024 / 1024
        total_rows = sum(len(df) for df in datasets.values())
        
        return {
            'total_files': len(datasets),
            'total_rows': total_rows,
            'total_size_mb': round(total_size_mb, 2),
            'files': {name: len(df) for name, df in datasets.items()},
            'scalability_assessment': {
                'json_lines_feasible': total_size_mb < 500,  # Threshold: 500MB
                'memory_efficient': total_size_mb < 100,
                'recommendation': 'integrate' if total_size_mb < 100 else 'separate_storage'
            }
        }