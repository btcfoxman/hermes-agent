#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent}"
APP_USER="${APP_USER:-btcfoxman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"

log() {
  printf '[hermes-agent-deploy] %s\n' "$*"
}

retry() {
  local attempts="$1"
  local delay="$2"
  shift 2
  local i
  for i in $(seq 1 "${attempts}"); do
    if "$@"; then
      return 0
    fi
    if [ "${i}" -eq "${attempts}" ]; then
      return 1
    fi
    log "Command failed, retrying in ${delay}s (${i}/${attempts}): $*"
    sleep "${delay}"
  done
}

mkdir -p "${APP_DIR}"

log "Syncing deployment compose to ${COMPOSE_FILE}"
cp -f "${SCRIPT_DIR}/docker-compose.yml" "${COMPOSE_FILE}"

if [ ! -f "${APP_DIR}/.env" ]; then
  cp -f "${SCRIPT_DIR}/.env.example" "${APP_DIR}/.env"
  chmod 600 "${APP_DIR}/.env" || log "Could not chmod ${APP_DIR}/.env; continuing"
  log "Created ${APP_DIR}/.env from example. Fill secrets before deploying."
  exit 1
fi

chmod 600 "${APP_DIR}/.env" || log "Could not chmod existing ${APP_DIR}/.env; continuing"

if id "${APP_USER}" >/dev/null 2>&1; then
  chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}" || true
fi

if [ -n "${GHCR_TOKEN:-}" ]; then
  log "Logging in to GHCR"
  docker_login_ghcr() {
    printf '%s' "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME:-${GITHUB_ACTOR:-btcfoxman}}" --password-stdin >/dev/null
  }
  retry 5 10 docker_login_ghcr
fi

cd "${APP_DIR}"
export IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io}"
export IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-btcfoxman}"
export IMAGE_NAME="${IMAGE_NAME:-hermes-agent}"
export IMAGE_TAG="${IMAGE_TAG:-test-latest}"

log "Validating compose config"
docker compose config >/dev/null

log "Pulling image"
retry 5 10 docker compose pull hermes-agent

log "Starting service"
docker compose up -d --remove-orphans hermes-agent

log "Waiting for service health"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:8095/health" >/dev/null; then
    docker compose ps
    log "Deployment complete"
    exit 0
  fi
  sleep 5
done

log "Health check failed"
docker compose ps || true
docker compose logs --tail=200 hermes-agent || true
exit 1
