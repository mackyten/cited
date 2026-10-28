FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# The index and cache live in ./data (mounted by docker-compose). Build it first with:
#   docker compose run --rm cited cited ingest
# git is only needed by `cited ingest` to fetch the docs.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1

# The API key is never baked in: pass GEMINI_API_KEY through the environment at run time.
CMD ["cited", "serve", "--host", "0.0.0.0", "--port", "8000"]
