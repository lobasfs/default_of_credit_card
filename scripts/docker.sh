#!/bin/bash

# Docker build and run script

set -e

echo "======================================"
echo "Docker Build and Run Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# Parse command line arguments
ACTION=${1:-build}

case $ACTION in
    build)
        print_status "Building Docker image..."
        docker build -t credit-card-default-api:latest .
        print_status "Docker image built successfully!"
        ;;
    
    run)
        print_status "Starting Docker container..."
        docker run -d \
            --name credit-card-api \
            -p 8000:8000 \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/data:/app/data \
            credit-card-default-api:latest
        
        print_status "Container started!"
        print_status "API available at: http://localhost:8000"
        print_status "API docs at: http://localhost:8000/docs"
        ;;
    
    stop)
        print_status "Stopping Docker container..."
        docker stop credit-card-api || print_warning "Container not running"
        docker rm credit-card-api || print_warning "Container not found"
        print_status "Container stopped and removed"
        ;;
    
    logs)
        print_status "Showing container logs..."
        docker logs -f credit-card-api
        ;;
    
    compose-up)
        print_status "Starting services with Docker Compose..."
        docker-compose up -d
        print_status "Services started!"
        print_status "API available at: http://localhost:8000"
        print_status "MLflow UI at: http://localhost:5000"
        ;;
    
    compose-down)
        print_status "Stopping Docker Compose services..."
        docker-compose down
        print_status "Services stopped"
        ;;
    
    test)
        print_status "Testing API..."
        
        # Wait for API to be ready
        sleep 5
        
        # Test health endpoint
        print_status "Testing /health endpoint..."
        curl -X GET http://localhost:8000/health | jq .
        
        # Test prediction endpoint
        print_status "Testing /predict endpoint..."
        curl -X POST http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{
                "LIMIT_BAL": 20000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913.0,
                "BILL_AMT2": 3102.0,
                "BILL_AMT3": 689.0,
                "BILL_AMT4": 0.0,
                "BILL_AMT5": 0.0,
                "BILL_AMT6": 0.0,
                "PAY_AMT1": 0.0,
                "PAY_AMT2": 689.0,
                "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0,
                "PAY_AMT5": 0.0,
                "PAY_AMT6": 0.0
            }' | jq .
        
        print_status "API tests completed!"
        ;;
    
    *)
        print_error "Unknown command: $ACTION"
        echo ""
        echo "Usage: $0 {build|run|stop|logs|compose-up|compose-down|test}"
        echo ""
        echo "Commands:"
        echo "  build        - Build Docker image"
        echo "  run          - Run Docker container"
        echo "  stop         - Stop and remove Docker container"
        echo "  logs         - Show container logs"
        echo "  compose-up   - Start services with Docker Compose"
        echo "  compose-down - Stop Docker Compose services"
        echo "  test         - Test API endpoints"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Done!"
echo "======================================"
