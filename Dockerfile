FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt ./requirements.txt

# 👉 Use prebuilt CPU wheels for llama-cpp-python
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt

# Cache the multilingual embedder used by the router
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/multilingual-e5-small')
print("Cached intfloat/multilingual-e5-small")
PY

COPY . .

ENTRYPOINT ["python", "solution.py"]
