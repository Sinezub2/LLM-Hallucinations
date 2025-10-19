FROM python:3.11-slim

ENV TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Use caching: install deps first
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the sentence-transformers model so runtime is offline
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-small-v2')
print("Cached intfloat/e5-small-v2")
PY

# Copy project
COPY . .

# Contest runner calls solution.py; it reads input.json and writes output.json
ENTRYPOINT ["python", "solution.py"]
