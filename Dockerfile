# Step 3 — Dockerfile
FROM python:3.11-slim

# UTF-8 + quiet tokenizers
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# 1) Install Python deps first (use layer caching)
COPY requirements.txt ./requirements.txt

# Try the normal index first. If your build later fails on llama-cpp, switch to the next line.
RUN pip install --no-cache-dir -r requirements.txt
# Fallback (uncomment if the previous line fails to find a wheel for llama-cpp):
# RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt

# 2) Pre-cache the multilingual embedder used by the router (so runtime is offline)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/multilingual-e5-small')
print("Cached intfloat/multilingual-e5-small")
PY

# 3) Copy the project (includes src/, models/, solution.py, etc.)
COPY . .

# 4) The grader runs solution.py, which reads input.json and writes output.json
ENTRYPOINT ["python", "solution.py"]
