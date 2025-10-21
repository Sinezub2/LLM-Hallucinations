FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# 1) toolchain for building llama-cpp-python from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake && rm -rf /var/lib/apt/lists/*

# (optional) enable native CPU optimizations; safe on most machines
ENV CMAKE_ARGS="-DLLAMA_NATIVE=ON"

# 2) deps
COPY requirements.txt ./requirements.txt
# IMPORTANT: use the normal index so pip *builds* from source instead of pulling a musl wheel
RUN pip install --no-cache-dir -r requirements.txt

# 3) cache router embedder (offline runtime)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/multilingual-e5-small')
print("Cached intfloat/multilingual-e5-small")
PY

# 4) project files
COPY . .

ENTRYPOINT ["python", "solution.py"]
