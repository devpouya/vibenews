# VibeNews Development Commands

.PHONY: help test test-unit test-validation test-integration clean install frontend-install frontend-start build-container submit-job check-job

# Default target
help:
	@echo "VibeNews Development Commands"
	@echo "============================="
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-validation   Run validation tests only"
	@echo "  test-integration  Run integration tests only"
	@echo ""
	@echo "Training:"
	@echo "  build-container   Build Docker container for training"
	@echo "  submit-job        Submit training job to Vertex AI"
	@echo "  check-job         Check training job status"
	@echo ""
	@echo "Frontend:"
	@echo "  frontend-install  Install frontend dependencies"
	@echo "  frontend-start    Start frontend development server"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean temporary files"
	@echo "  install          Install Python dependencies"

# Testing targets
test:
	@echo "🧪 Running all tests..."
	python tests/run_all_tests.py

test-unit:
	@echo "🔧 Running unit tests..."
	@for test in tests/unit/*.py; do \
		echo "Running $$test..."; \
		python "$$test" || exit 1; \
	done

test-validation:
	@echo "✅ Running validation tests..."
	@for test in tests/validation/*.py; do \
		echo "Running $$test..."; \
		python "$$test" || exit 1; \
	done

test-integration:
	@echo "🔗 Running integration tests..."
	@for test in tests/integration/*.py; do \
		echo "Running $$test..."; \
		python "$$test" || exit 1; \
	done

# Training targets
build-container:
	@echo "🐳 Building Docker container..."
	gcloud builds submit --tag gcr.io/vibenews-bias-detection/bias-trainer:latest .

submit-job:
	@echo "🚀 Submitting training job..."
	source venv/bin/activate && \
	GOOGLE_CLOUD_PROJECT=vibenews-bias-detection \
	python scripts/submit_vertex_job.py --config vertex_configs/distilbert_cpu_only.yaml --submit-only

check-job:
	@echo "📊 Checking job status..."
	source venv/bin/activate && python debug/check_job_status.py

# Frontend targets
frontend-install:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install

frontend-start:
	@echo "🌐 Starting frontend development server..."
	cd frontend && npm start

# Utility targets
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf frontend/node_modules/.cache/ 2>/dev/null || true

install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

# Deployment pipeline
deploy-check: test-validation
	@echo "✅ Deployment checks passed"

# Development workflow
dev-check: test-unit test-validation
	@echo "✅ Development checks passed"