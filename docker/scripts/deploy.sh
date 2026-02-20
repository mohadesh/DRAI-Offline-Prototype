#!/usr/bin/env bash
# Deploy DRAI Offline to the production server via SSH.
#
# Prerequisites:
#   1. SSH key-based access to the server.
#   2. Docker and docker-compose v2 installed on the server.
#   3. A .env file ready on the server at REMOTE_DEPLOY_DIR.
#
# Usage:
#   ./docker/scripts/deploy.sh [TAG] [SSH_HOST]
#
# Environment variables (can also be set in shell):
#   REMOTE_HOST       – server hostname or IP  (required)
#   REMOTE_USER       – SSH user               (default: deploy)
#   REMOTE_DEPLOY_DIR – path on server         (default: ~/drai-offline)
#   IMAGE_TAG         – image tag to deploy    (default: latest)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_TAG="${1:-${IMAGE_TAG:-latest}}"
REMOTE_HOST="${2:-${REMOTE_HOST:?Set REMOTE_HOST or pass as second argument}}"
REMOTE_USER="${REMOTE_USER:-deploy}"
REMOTE_DEPLOY_DIR="${REMOTE_DEPLOY_DIR:-~/drai-offline}"
IMAGE_NAME="drai-offline"
COMPOSE_FILE="docker/docker-compose.yml"

SSH="ssh -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST}"

echo "──────────────────────────────────────────────────────"
echo " Deploying : ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Target    : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEPLOY_DIR}"
echo "──────────────────────────────────────────────────────"

# ── 1. Save image to tar and copy to server ───────────────────────────────────
echo "[1/4] Exporting image …"
TARBALL="/tmp/${IMAGE_NAME}_${IMAGE_TAG}.tar"
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${TARBALL}"

echo "[2/4] Uploading image to server …"
scp "${TARBALL}" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/"
rm -f "${TARBALL}"

# ── 2. Upload compose file and env example ────────────────────────────────────
echo "[3/4] Syncing compose files …"
$SSH "mkdir -p ${REMOTE_DEPLOY_DIR}/docker/scripts"
scp "${COMPOSE_FILE}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEPLOY_DIR}/${COMPOSE_FILE}"
scp docker/.env.example "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEPLOY_DIR}/docker/.env.example"

# ── 3. Load image and restart service ────────────────────────────────────────
echo "[4/4] Loading image and restarting service on server …"
$SSH bash -s << EOF
set -e
echo "  Loading image …"
docker load -i /tmp/${IMAGE_NAME}_${IMAGE_TAG}.tar
rm -f /tmp/${IMAGE_NAME}_${IMAGE_TAG}.tar

cd ${REMOTE_DEPLOY_DIR}

# Ensure .env exists (first deploy: copy from example)
if [[ ! -f .env ]]; then
    cp docker/.env.example .env
    echo "⚠  Created .env from example — please update it with real values!"
fi

# Update IMAGE_TAG in .env
sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=${IMAGE_TAG}/" .env

echo "  Pulling up service …"
docker compose -f docker/docker-compose.yml up -d --no-build

echo "  Removing dangling images …"
docker image prune -f
EOF

echo ""
echo "✓ Deployment complete → ${IMAGE_NAME}:${IMAGE_TAG} on ${REMOTE_HOST}"
echo "  Check logs: ssh ${REMOTE_USER}@${REMOTE_HOST} 'docker logs -f drai-offline'"
