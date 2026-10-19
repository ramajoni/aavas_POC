# Aavas pricing engine - API image.
#
# Models are trained at build time so the container starts serving immediately.
# For a real deployment, train in CI and mount or COPY the artefacts instead:
# a container that retrains on every build cannot be reproduced later.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY src/ ./src/
COPY data/synthetic/ ./data/synthetic/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Generate the synthetic dataset and train the champion models into the image.
RUN python data/synthetic/generate_dataset.py \
 && python scripts/train_all.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"status":"ok"' || exit 1

CMD ["uvicorn", "aavas.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
