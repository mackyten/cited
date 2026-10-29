# Demo image for Cloud Run. Unlike the root Dockerfile it bakes in the prebuilt index
# (data/index, staged by deploy/cloudrun.sh), so the container never calls the API to start.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY data/index ./data/index

# Cloud Run sets $PORT. The API key arrives as a secret env var, never in the image.
CMD ["sh", "-c", "exec cited serve --host 0.0.0.0 --port ${PORT:-8080}"]
