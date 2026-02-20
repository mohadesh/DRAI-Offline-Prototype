#!/usr/bin/env bash
# Build the DRAI Offline Docker image.
#
# Usage:
#   ./docker/scripts/build.sh [TAG]
#
# Examples:
#   ./docker/scripts/build.sh             # tags as "latest"
#   ./docker/scripts/build.sh 1.2.3       # tags as "1.2.3" AND "latest"
#   ./docker/scripts/build.sh 1.2.3 --no-cache

set -euo pipefail

# Always run from the project root regardless of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

IMAGE_NAME="drai-offline"
TAG="${1:-latest}"
shift || true   # consume TAG arg; remaining args forwarded to docker build

echo "──────────────────────────────────────────"
echo " Building: ${IMAGE_NAME}:${TAG}"
echo " Context : ${PROJECT_ROOT}"
echo "──────────────────────────────────────────"

docker build \
    -f docker/Dockerfile \
    -t "${IMAGE_NAME}:${TAG}" \
    "$@" \
    .

# Also tag as "latest" when a version is given
if [[ "${TAG}" != "latest" ]]; then
    docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"
    echo "✓ Tagged ${IMAGE_NAME}:${TAG} as ${IMAGE_NAME}:latest"
fi

echo ""
echo "✓ Build complete → ${IMAGE_NAME}:${TAG}"
echo ""
echo "Next steps:"
echo "  Local test : docker compose -f docker/docker-compose.dev.yml up"
echo "  Deploy     : ./docker/scripts/deploy.sh ${TAG}"
