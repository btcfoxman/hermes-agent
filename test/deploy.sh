#!/usr/bin/env bash
set -euo pipefail
umask 077

APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent}"
APP_USER="${APP_USER:-btcfoxman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
TEMP_ENV_FILE=""

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

cleanup() {
  if [ -n "${TEMP_ENV_FILE}" ] && [ -f "${TEMP_ENV_FILE}" ]; then
    rm -f "${TEMP_ENV_FILE}"
  fi
}

trap cleanup EXIT

upsert_env_value() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  local escaped_value line

  if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    log "${key} must be a single-line value"
    return 1
  fi

  TEMP_ENV_FILE="$(mktemp "${env_file}.tmp.XXXXXX")"
  if [ -f "${env_file}" ]; then
    while IFS= read -r line || [ -n "${line}" ]; do
      case "${line}" in
        "${key}="*|"export ${key}="*) continue ;;
      esac
      printf '%s\n' "${line}" >> "${TEMP_ENV_FILE}"
    done < "${env_file}"
  fi

  # Compose treats single-quoted dotenv values literally. Escape only the
  # characters that can terminate or alter that representation; values such
  # as base64 tokens containing '=' remain byte-for-byte intact.
  escaped_value="${value//\\/\\\\}"
  escaped_value="${escaped_value//\'/\\\'}"
  printf "%s='%s'\n" "${key}" "${escaped_value}" >> "${TEMP_ENV_FILE}"
  chmod 600 "${TEMP_ENV_FILE}"
  mv -f "${TEMP_ENV_FILE}" "${env_file}"
  TEMP_ENV_FILE=""
}

authenticated_get() {
  local url="$1"
  printf 'Authorization: Bearer %s\n' "${HERMES_API_KEY}" \
    | curl --fail --silent --show-error --header @- "${url}" >/dev/null
}

if [ -z "${HERMES_API_KEY:-}" ]; then
  log "HERMES_API_KEY is required from the GitHub test Environment"
  exit 1
fi

mkdir -p "${APP_DIR}"

log "Syncing deployment compose to ${COMPOSE_FILE}"
cp -f "${SCRIPT_DIR}/docker-compose.yml" "${COMPOSE_FILE}"

if [ ! -f "${APP_DIR}/.env" ]; then
  cp -f "${SCRIPT_DIR}/.env.example" "${APP_DIR}/.env"
  log "Created ${APP_DIR}/.env from example"
fi

upsert_env_value "${APP_DIR}/.env" "HERMES_API_KEY" "${HERMES_API_KEY}"
chmod 600 "${APP_DIR}/.env"
log "Updated service authentication configuration"

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
if [ -z "${IMAGE_TAG:-}" ] || [ "${IMAGE_TAG}" = "test-latest" ]; then
  log "IMAGE_TAG must be the immutable test-<git-sha> tag"
  exit 1
fi
export IMAGE_TAG

log "Validating compose config"
docker compose config >/dev/null

log "Pulling image"
retry 5 10 docker compose pull hermes-agent

log "Starting service"
docker compose up -d --remove-orphans hermes-agent

log "Waiting for service health"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:8095/health" >/dev/null \
    && authenticated_get "http://127.0.0.1:8095/api/v1/operators/health"; then
    probes_ok=true
    for role in commercial industry personal_ip; do
      if ! authenticated_get "http://127.0.0.1:8095/api/v1/operators/${role}/probe"; then
        probes_ok=false
        break
      fi
    done
    if [ "${probes_ok}" = true ]; then
      docker compose ps
      log "Deployment complete; all operator profiles are ready"
      exit 0
    fi
  fi
  sleep 5
done

log "Health check failed"
docker compose ps || true
docker compose logs --tail=200 hermes-agent || true
exit 1
