# BABE Dataset Integration - Complete ✅

## Summary

Successfully integrated the **BABE (Media Bias Annotations by Experts)** dataset into your Swiss news bias project. The dataset is now ready for validation and Gemini LoRA pretraining.

## Dataset Overview

### 📊 **Statistics**
- **Total Samples**: 5,374 expert-annotated news articles
- **File Size**: ~10MB (JSON Lines format)
- **Language**: English
- **Average Text Length**: 202 characters  
- **Text Range**: 3-606 characters

### 🏷️ **Bias Labels (Original BABE)**
- **Bias Classification**:
  - Non-biased: 2,663 (49.5%)
  - Biased: 2,556 (47.6%)
  - No agreement: 155 (2.9%)

- **Opinion Classification**:
  - Entirely factual: 2,239 (41.7%)
  - Somewhat factual but also opinionated: 1,453 (27.0%)
  - Expresses writer's opinion: 1,283 (23.9%)
  - No agreement: 399 (7.4%)

- **Outlet Types**:
  - Right: 1,684 (31.3%)
  - Left: 1,683 (31.3%)
  - Center: 1,007 (18.7%)

### 📰 **Top Topics**
- Marriage equality: 347 articles
- Vaccine: 299 articles  
- Black Lives Matter: 289 articles
- Environment: 270 articles
- White nationalism: 262 articles

## File Locations

### 📁 **Primary Dataset**
```
backend/data/raw/babe_with_annotations_20250731.jsonl
```

### 📁 **Validation Export**
```
backend/data/processed/babe_validation.jsonl  
```

## Data Structure

Each entry contains:

```json
{
  "id": "babe_sg1_0",
  "dataset": "babe", 
  "source_dataset": "SG1",
  "created_at": "2025-07-31T14:48:16.891227",
  "text": "The Republican president assumed he was helping...",
  "news_link": "http://www.msnbc.com/rachel-maddow-show/...",
  "outlet": "msnbc",
  "topic": "environment",
  "bias_labels": {
    "label_bias": "Biased",
    "label_opinion": "Expresses writer's opinion", 
    "outlet_type": "left",
    "biased_words": "[]"
  },
  "original_data": {...}
}
```

## Usage Examples

### 🔍 **Load Dataset**
```python
from backend.datasets.babe_utils import BABEValidator

validator = BABEValidator()
df = validator.load_babe_data('babe_with_annotations_20250731.jsonl')
print(f"Loaded {len(df)} samples")
```

### ✂️ **Create Validation Split**
```python
train_df, val_df = validator.create_validation_split(df, test_size=0.2)
```

### 🚀 **Prepare for Gemini LoRA**
```python
training_examples = validator.prepare_for_pretraining(df, output_format='gemini')
```

### 📈 **Compare with Swiss Articles**
```python
comparison = validator.compare_with_swiss_articles(
    babe_df, 
    'articles_20250731_140338.jsonl'
)
```

## Integration Benefits

### ✅ **For Validation**
- **5,374 expert-labeled examples** for model evaluation
- **Balanced bias distribution** (biased vs non-biased)
- **Multiple bias dimensions** (bias + opinion + outlet type)
- **Topic diversity** across political subjects

### ✅ **For Pretraining**
- **High-quality training data** from expert annotations
- **English text** for cross-language transfer learning
- **Structured format** ready for Gemini LoRA fine-tuning
- **Consistent JSON Lines** format with your Swiss articles

### ✅ **For Research**
- **Benchmark dataset** for bias detection research
- **Cross-dataset comparison** with Swiss news
- **Word-level bias indicators** (biased_words field)
- **Multi-source coverage** (left/center/right outlets)

## Next Steps

1. **Model Validation**: Use BABE as test set for your bias models
2. **Gemini LoRA**: Pretrain on BABE, then fine-tune on Swiss articles  
3. **Cross-lingual Transfer**: Leverage English BABE → German Swiss articles
4. **Bias Analysis**: Compare bias patterns between US (BABE) and Swiss news

## Storage Impact

- ✅ **Scalable**: Only 10MB additional storage
- ✅ **JSON Lines Compatible**: Fits existing ML pipeline
- ✅ **Memory Efficient**: Loads easily into pandas/training frameworks
- ✅ **Version Controlled**: Timestamped files for reproducibility

---

**Status**: 🎉 **COMPLETE** - BABE dataset fully integrated and ready for use!