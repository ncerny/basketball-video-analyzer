# Basketball Video Analyzer - Backend

FastAPI-based backend server for the Basketball Video Analyzer application.

## Tech Stack

- **Framework**: FastAPI 0.109+
- **Python**: 3.11+
- **Database**: SQLite with SQLAlchemy ORM
- **Migrations**: Alembic
- **Video Processing**: OpenCV, FFmpeg
- **ML/CV**: YOLOv8, PyTorch (CPU mode)
- **Package Manager**: Poetry

## Setup

### Prerequisites

- Python 3.11 or higher
- Poetry ([installation guide](https://python-poetry.org/docs/#installation))

### Installation

```bash
cd backend
poetry install
```

This will install all dependencies including:
- Core dependencies (FastAPI, SQLAlchemy, etc.)
- Development tools (pytest, black, ruff, mypy)
- Video processing libraries (OpenCV, FFmpeg)
- ML libraries (PyTorch CPU, YOLOv8, EasyOCR)

### Running the Development Server

```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app

# Run specific test file
poetry run pytest tests/test_main.py
```

### Code Formatting

```bash
# Format code with black
poetry run black app tests

# Lint with ruff
poetry run ruff check app tests

# Type check with mypy
poetry run mypy app
```

### Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application entry point
│   ├── config.py         # Application configuration
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic schemas for API validation
│   ├── api/              # API route handlers
│   ├── services/         # Business logic layer
│   └── ml/               # ML model inference
├── tests/                # Test files
├── pyproject.toml        # Poetry dependencies and configuration
└── README.md
```

## API Documentation

Once the server is running, visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.

## Database Migrations

Database migrations are managed with Alembic (to be configured in Phase 1).

```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1
```
