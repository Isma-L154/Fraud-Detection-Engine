
# This keeps the final image lean — no build tools, no cache, no unnecessary files.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements first 
# If requirements.txt doesn't change, Docker skips pip install on rebuild.

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user — running as root inside a container is a security
# risk. If an attacker exploits your app, they get root on the container.
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder — not from the internet again
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model artifact
COPY app/ ./app/
COPY models/ ./models/

# Switch to non-root before the app starts
USER appuser

# Document the port — doesn't actually publish it, just metadata
EXPOSE 8000

# Healthcheck lets Docker know if the container is actually healthy, not just running. Checks every 30s, fails after 3 consecutive failures.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]