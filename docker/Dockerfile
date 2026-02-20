# ─────────────────────────────────────────────
# Stage 1: dependency builder
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries required by LightGBM and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "gunicorn[gthread]==21.2.0"


# ─────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DRAI Offline Prototype"
LABEL org.opencontainers.image.description="DR AI offline simulation and inference server"

# libgomp1 is needed at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/gunicorn   /usr/local/bin/gunicorn

WORKDIR /app

# Copy application source (models + analysis baked in for offline use)
COPY app.py core_logic.py simulation.py model_loader.py \
     MDnC.py Pellet.py ProcessTags.py merging.py ./
COPY templates/ templates/
COPY static/    static/
COPY dashboard/ dashboard/
COPY models/    models/
COPY analysis/  analysis/

# Copy the container entrypoint
COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Unprivileged user
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/uploads \
 && chown -R appuser:appuser /app

USER appuser

# Internal port (mapped by Traefik / docker-compose)
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:5000/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
