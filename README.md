# VibeNews - Swiss News Bias Aggregator

A RAG-powered news aggregator that ranks Swiss news articles by bias on various topics.

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate vibenews
```

2. Install additional dependencies:
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Development

- Backend: Python/FastAPI
- Frontend: React
- ML: PyTorch + Transformers  
- Vector DB: ChromaDB
- Sources: NZZ, SRF, 20 Minuten, Watson