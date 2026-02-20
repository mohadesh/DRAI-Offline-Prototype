#!/bin/sh
# Gunicorn entrypoint for the DRAI Offline container.
# Workers × threads should match available CPU cores on the host.
set -e

WORKERS=${GUNICORN_WORKERS:-2}
THREADS=${GUNICORN_THREADS:-4}
PORT=${APP_PORT:-5000}
TIMEOUT=${GUNICORN_TIMEOUT:-120}

echo "Starting DRAI Offline — workers=${WORKERS} threads=${THREADS} port=${PORT}"

exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WORKERS}" \
    --threads "${THREADS}" \
    --timeout "${TIMEOUT}" \
    --access-logfile - \
    --error-logfile - \
    --log-level "${LOG_LEVEL:-info}" \
    app:app
