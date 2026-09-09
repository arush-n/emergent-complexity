FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
COPY frontend ./frontend

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -c constraints.txt .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn emergent.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
