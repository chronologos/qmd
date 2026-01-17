# QMD Embed/Rerank Server
# Build: docker build -t qmd-embed-rerank server/
# Run:   docker run -d --gpus all -p 8001:8001 qmd-embed-rerank

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY embed_rerank.py .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8001/health || exit 1

# Run server
CMD ["uvicorn", "embed_rerank:app", "--host", "0.0.0.0", "--port", "8001"]
