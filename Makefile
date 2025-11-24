.PHONY: help install dev test lint format build deploy clean

# Default target
help:
	@echo "Crash Prediction System - Available Commands"
	@echo "============================================="
	@echo "make install     - Install dependencies"
	@echo "make dev         - Start development environment"
	@echo "make test        - Run test suite"
	@echo "make lint        - Run linters"
	@echo "make format      - Format code"
	@echo "make build       - Build Docker images"
	@echo "make deploy-stg  - Deploy to staging"
	@echo "make deploy-prod - Deploy to production"
	@echo "make clean       - Clean up"

# Install dependencies
install:
	poetry install
	cd dashboard && npm install

# Start development environment
dev:
	docker-compose up -d postgres redis timescale minio
	sleep 5
	poetry run alembic upgrade head
	@echo "Starting API server..."
	poetry run uvicorn app.main:app --reload &
	@echo "Starting Celery worker..."
	poetry run celery -A app.celery_app worker --loglevel=info &
	@echo "Starting dashboard..."
	cd dashboard && npm run dev

# Run tests
test:
	docker-compose -f docker-compose.test.yml up -d
	sleep 5
	poetry run pytest tests/ -v --cov=app --cov-report=html
	docker-compose -f docker-compose.test.yml down

# Run unit tests only
test-unit:
	poetry run pytest tests/unit -v

# Run integration tests only
test-integration:
	docker-compose -f docker-compose.test.yml up -d
	sleep 5
	poetry run pytest tests/integration -v
	docker-compose -f docker-compose.test.yml down

# Lint code
lint:
	poetry run flake8 app tests
	poetry run mypy app
	poetry run bandit -r app -ll

# Format code
format:
	poetry run black app tests
	poetry run isort app tests

# Build Docker images
build:
	docker build -t crash-prediction/api:latest --target production .
	docker build -t crash-prediction/worker:latest --target worker .
	cd dashboard && docker build -t crash-prediction/dashboard:latest .

# Deploy to staging
deploy-stg:
	kubectl apply -f k8s/staging/
	kubectl rollout status deployment/crash-prediction-api -n staging

# Deploy to production
deploy-prod:
	kubectl apply -f k8s/production/
	kubectl rollout status deployment/crash-prediction-api -n production

# Run database migrations
migrate:
	poetry run alembic upgrade head

# Create new migration
migration:
	poetry run alembic revision --autogenerate -m "$(msg)"

# Start monitoring stack
monitoring:
	docker-compose -f docker-compose.monitoring.yml up -d

# Clean up
clean:
	docker-compose down -v
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Generate documentation
docs:
	poetry run mkdocs build

# Serve documentation locally
docs-serve:
	poetry run mkdocs serve
