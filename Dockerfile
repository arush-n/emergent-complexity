FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn emergent.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
