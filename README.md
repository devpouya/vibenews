# VibeNews - Swiss News Bias Aggregator

A news bias aggregator that ranks Swiss news articles along a bias spectrum (-1 to +1) for specific topics. Users can search for topics (e.g., "Russia-Ukraine conflict") and view articles ranked from most pro-stance to most anti-stance.

## 🚀 Project Status

### ✅ Completed
- **BiasScanner Implementation**: Complete 27 bias types algorithm from Menzner & Leidner (2024)
- **BABE Dataset Integration**: 5,374 expert-annotated news articles loaded and validated
- **BERT Classification Pipeline**: Complete bias classification system with confidence scoring
- **Data Infrastructure**: ChromaDB vector storage, FastAPI endpoints, Swiss news scraper base

### 🔧 Current Architecture

#### BERT Bias Classifier
- **Model**: BERT-base-uncased + classification head
- **Classes**: 3-class bias detection (Non-biased, Biased, No agreement)
- **Data**: 5,374 BABE samples (4,299 train, 1,075 validation)
- **Features**: Raw text processing, confidence scoring, evaluation metrics
- **Status**: Pipeline ready, training pending

#### BiasScanner (Research Implementation)
- **27 Bias Types**: Complete classification system
- **Performance Target**: 76% F1-score (based on original paper)
- **Swiss Integration**: Political spectrum mapping (-1 to +1)
- **Status**: Fully implemented and tested

## 🏃‍♂️ Next Steps Brainstorm

### **Immediate Actions (Ready to Execute)**

1. **Train the BERT Model**
   - Run `trainer.train()` on BABE dataset
   - Monitor training metrics and validation performance
   - Save best model checkpoint

2. **Model Evaluation & Analysis**
   - Test on held-out BABE validation set
   - Generate confusion matrix for 3-class bias detection
   - Compare performance to BiasScanner (76% F1 target)
   - Analyze misclassified examples

3. **Swiss News Integration**
   - Apply trained BERT model to Swiss articles
   - Map BABE labels to Swiss political spectrum (-1 to +1)
   - Compare bias patterns: US (BABE) vs Swiss news

### **Model Enhancement**

4. **Cross-lingual Transfer Learning**
   - Fine-tune English BERT model on German Swiss articles
   - Use multilingual BERT (bert-base-multilingual-cased)
   - Implement translation-based data augmentation

5. **Advanced Architecture**
   - Try DistilBERT for faster inference
   - Implement ensemble methods (BERT + BiasScanner)
   - Add attention visualization for bias word detection

6. **Multi-task Learning**
   - Train jointly on bias + opinion + outlet_type prediction
   - Use BABE's biased_words for word-level supervision
   - Implement hierarchical classification

### **Data & Pipeline**

7. **Data Quality & Augmentation**
   - Handle "No agreement" samples (remove/reweight/separate model)
   - Active learning: identify uncertain predictions for manual labeling
   - Create Swiss-specific bias annotation guidelines

8. **Production Pipeline**
   - Real-time bias scoring API endpoint
   - Batch processing for news scraping pipeline
   - Model versioning and A/B testing framework

9. **Advanced Metrics**
   - Calibration analysis (confidence vs accuracy)
   - Fairness metrics across political topics
   - Temporal bias drift detection

### **System Integration**

10. **VibeNews Integration**
    - Replace/augment BiasScanner with BERT model
    - Update ChromaDB storage with BERT predictions
    - Enhance bias spectrum visualization

11. **Swiss News Scraping**
    - Implement scrapers for NZZ, SRF, Watson, 20min
    - Schedule automated bias analysis pipeline
    - Build topic clustering with bias-aware embeddings

12. **User Interface**
    - Interactive bias explanations (which words/phrases)
    - Confidence intervals on bias scores
    - Historical bias trend analysis

### **Research & Experiments**

13. **Comparative Studies**
    - BERT vs GPT vs Gemini for bias detection
    - English-trained vs German-trained models
    - Human annotation study on Swiss news

14. **Bias Taxonomy**
    - Extend beyond 3-class to 27 BiasScanner types
    - Swiss-specific bias categories (political parties, regions)
    - Multi-dimensional bias scoring

15. **Advanced ML**
    - Few-shot learning for new bias types
    - Prompt-based bias detection with LLMs
    - Graph neural networks for source credibility

### **Innovation Opportunities**

16. **Real-time Analysis**
    - Live bias monitoring during breaking news
    - Social media bias propagation tracking
    - Cross-platform bias correlation analysis

17. **Personalization**
    - User-specific bias sensitivity settings
    - Personalized bias spectrum calibration
    - Individual bias bubble detection

18. **Regulatory & Ethics**
    - GDPR-compliant bias analysis
    - Bias detection transparency reports
    - Media literacy educational tools

## 🛠️ Quick Start

### Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas scikit-learn transformers torch accelerate

# Set Gemini API key
echo 'GEMINI_API_KEY=your_api_key_here' > .env
```

### Test Components
```bash
# Test BABE dataset loading
python test_babe_simple.py

# Test BERT pipeline setup
python test_bert_pipeline.py

# Test BiasScanner (requires API key)
python test_biasscanner.py
```

### Start Training
```python
from backend.datasets.babe_utils import BABEValidator
from backend.ml.bias_classifier import BiasClassifier

# Load data and train
validator = BABEValidator()
babe_df = validator.load_babe_data('babe_with_annotations_20250731.jsonl')

classifier = BiasClassifier()
texts, labels = classifier.load_babe_data(babe_df)
train_dataset, val_dataset = classifier.create_datasets(texts, labels)

classifier.initialize_model(num_labels=3)
trainer = classifier.setup_training(train_dataset, val_dataset)

# Start training
trainer.train()
```

## 📁 Project Structure

```
vibenews/
├── backend/
│   ├── config.py                 # Settings with Gemini API
│   ├── models/article.py         # Article data models
│   ├── scraper/base.py          # News scraper base class
│   ├── bias_detection/          # BiasScanner implementation
│   ├── datasets/                # BABE dataset utilities
│   ├── ml/bias_classifier.py    # BERT bias classifier
│   └── api/                     # FastAPI endpoints
├── test_babe_simple.py          # BABE dataset validation
├── test_bert_pipeline.py        # BERT pipeline testing
├── .env                         # API keys (create this)
└── README.md                    # This file
```

## 🎯 Strategic Directions

**Which direction interests you most?**
- **Quick wins**: Train BERT → Evaluate → Integrate
- **Research depth**: Cross-lingual transfer → Swiss adaptation  
- **Production focus**: API → Scraping → Real-time pipeline
- **Innovation**: Multi-modal bias (text + images) → Social graphs

**Primary Goals:**
- Academic research paper
- Production news platform  
- Swiss media analysis tool
- Bias detection API service

## 📊 Dataset Information

### BABE Dataset
- **Total Samples**: 5,374 expert-annotated articles
- **Language**: English
- **Labels**: Bias classification, Opinion type, Outlet political leaning
- **File**: `backend/data/raw/babe_with_annotations_20250731.jsonl`

### Swiss News Sources (Planned)
- NZZ (Neue Zürcher Zeitung)
- SRF (Swiss Radio and Television)
- 20 Minuten  
- Watson

## 🔬 Research Background

This project implements bias detection algorithms from:
- **BiasScanner**: Menzner & Leidner (2024) - 27 bias types classification
- **BABE Dataset**: Media Bias Annotations by Experts
- **BERT Classification**: Fine-tuned transformer for bias detection

Target performance: 76% F1-score (BiasScanner baseline)