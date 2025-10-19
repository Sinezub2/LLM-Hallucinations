FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Avoid tokenizers multi-proc warnings/noise
ENV TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

# Core deps for the router
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-cache the Sentence-Transformers model inside the image so runtime is offline
RUN python3 - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-small-v2')
print("Cached intfloat/e5-small-v2 into /root/.cache")
PY

# Now copy the rest of the repo (src/, models/, solution.py, data/, etc.)
COPY . .

# The contest runner expects: python solution.py will read input.json and write output.json
ENTRYPOINT ["python3", "solution.py"]
