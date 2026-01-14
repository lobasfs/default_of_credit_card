.PHONY: help install test lint format clean data train experiments api docker-build docker-run monitor

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make data          - Prepare data"
	@echo "  make train         - Train model"
	@echo "  make experiments   - Run multiple experiments"
	@echo "  make api           - Run API server"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make monitor       - Run monitoring"
	@echo "  make dvc-repro     - Run DVC pipeline"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src tests --max-line-length=127 --exclude=__pycache__

format:
	black src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

data:
	python -m src.data.prepare

train:
	python scripts/train_model.py

experiments:
	python scripts/run_experiments.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t credit-card-api:latest .

docker-run:
	docker run -d --name credit-card-api -p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/data:/app/data \
		credit-card-api:latest

docker-stop:
	docker stop credit-card-api || true
	docker rm credit-card-api || true

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

monitor:
	python scripts/monitor.py

dvc-repro:
	dvc repro

mlflow-ui:
	mlflow ui --port 5000
