# CPU-only serving image: the CLI and the API answering queries against a prebuilt runtime_index/.
#
# It deliberately does NOT contain mteb, datasets or a CUDA build of torch. Evaluation and index
# building happen on a GPU notebook; this image only loads an index and encodes one query, which is the
# whole reason the index is exported as files.
#
#   docker build -t apps-retrieval .
#   docker run --rm -p 8000:8000 -v "$PWD/runtime_index:/app/runtime_index:ro" apps-retrieval
#   curl localhost:8000/health
#   curl -s localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"binary search","k":5}'
#
#   # one-off CLI query instead of the server
#   docker run --rm -v "$PWD/runtime_index:/app/runtime_index:ro" apps-retrieval \
#       python src/cli.py "count the primes below n" -k 5
#
# The index is mounted, not copied: it is model output, it changes independently of the code, and baking
# a 36 MB+ embedding matrix into the image would tie the two together for no benefit.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    INDEX_DIR=/app/runtime_index \
    SEARCH_DEVICE=cpu \
    HF_HOME=/app/hf_cache \
    HF_HUB_OFFLINE=0

WORKDIR /app

# CPU wheels only -- the default index would pull ~2.5 GB of CUDA libraries this image can never use.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-serve.txt

COPY src/ ./src/
COPY configs/ ./configs/

# Non-root: nothing here needs write access to the image, and the index is mounted read-only.
# /app/hf_cache is where the model weights land on the first search; it exists (and is owned by
# appuser) so that a named volume mounted there by docker-compose.yml inherits that ownership.
RUN mkdir -p /app/hf_cache && useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# /health answers before the model is loaded (it loads on the first search), so this reports the
# container as healthy during a multi-GB model load rather than killing it half way through.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
