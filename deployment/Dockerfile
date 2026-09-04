FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install Python requirements
COPY vision/ /app/vision/
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY docs/ /app/docs/

RUN uv pip install --system \
    numpy \
    scipy \
    opencv-python-headless \
    pydantic \
    fastapi \
    uvicorn \
    requests \
    httpx \
    networkx \
    pytest

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
