#!/usr/bin/env bash
set -euo pipefail
umask 077

APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent-test}"
APP_USER="${APP_USER:-btcfoxman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hermes-agent-test}"
HERMES_HOST_PORT="${HERMES_HOST_PORT:-18095}"
TEMP_ENV_FILE=""
TEMP_DOCKER_CONFIG=""

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
  if [ -n "${TEMP_DOCKER_CONFIG}" ] && [ -d "${TEMP_DOCKER_CONFIG}" ]; then
    rm -rf -- "${TEMP_DOCKER_CONFIG}"
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
  curl --fail --silent --show-error \
    --header "Authorization: Bearer ${HERMES_API_KEY}" \
    "${url}" >/dev/null
}

probe_compose_contracts() {
  compose exec -T hermes-agent-test python -c '
import json
import os
import urllib.request

payload = {
    "objective": "deployment contract probe",
    "topic": "",
    "audience": "test",
    "channels": ["wechat_mp"],
    "constraints": [],
    "authorized_context": [],
    "as_of": "2026-07-17T00:00:00Z",
    "approved_proposal": {
        "title": "Deployment contract probe",
        "angle": "Validate contract only",
        "audience_value": "Contract availability",
        "key_points": [],
        "suggested_formats": ["wechat_mp"],
        "cta": None,
        "first_person": False,
    },
    "claims": [],
}
body = json.dumps(payload).encode("utf-8")
token = os.environ["HERMES_API_KEY"]
expected_status = {
    "commercial": "needs_input",
    "industry": "evidence_insufficient",
    "personal_ip": "needs_input",
}
for role, status in expected_status.items():
    request = urllib.request.Request(
        f"http://127.0.0.1:8095/api/v1/operators/{role}/compose",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.load(response)
    assert result["schema_version"] == "operator.content.v1", result
    assert result["role_id"] == role, result
    assert result["action"] == "compose", result
    assert result["status"] == status, result
    assert result["master_content"] is None, result
    assert result["platform_variants"] == [], result
    assert result["critic"]["passed"] is False, result
    assert result["requires_human_review"] is True, result
'
}

if [ -z "${HERMES_API_KEY:-}" ]; then
  log "HERMES_API_KEY is required from the GitHub test Environment"
  exit 1
fi

resolved_app_dir="$(realpath -m "${APP_DIR}")"
if [[ "${resolved_app_dir,,}" != *test* ]]; then
  log "Refusing non-test APP_DIR: ${resolved_app_dir}"
  exit 1
fi
if [[ "${COMPOSE_PROJECT_NAME,,}" != *test* ]]; then
  log "Refusing non-test COMPOSE_PROJECT_NAME: ${COMPOSE_PROJECT_NAME}"
  exit 1
fi
if [[ ! "${HERMES_HOST_PORT}" =~ ^[0-9]+$ ]] \
  || [ "${HERMES_HOST_PORT}" -lt 1024 ] \
  || [ "${HERMES_HOST_PORT}" -gt 65535 ] \
  || [ "${HERMES_HOST_PORT}" = "8095" ]; then
  log "Refusing invalid or production HERMES_HOST_PORT: ${HERMES_HOST_PORT}"
  exit 1
fi

APP_DIR="${resolved_app_dir}"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
mkdir -p "${APP_DIR}/logs/hermes-agent"

log "Syncing deployment compose to ${COMPOSE_FILE}"
cp -f "${SCRIPT_DIR}/docker-compose.yml" "${COMPOSE_FILE}"

if [ ! -f "${APP_DIR}/.env" ]; then
  cp -f "${SCRIPT_DIR}/.env.example" "${APP_DIR}/.env"
  log "Created ${APP_DIR}/.env from example"
fi

upsert_env_value "${APP_DIR}/.env" "COMPOSE_PROJECT_NAME" "${COMPOSE_PROJECT_NAME}"
upsert_env_value "${APP_DIR}/.env" "HERMES_HOST_PORT" "${HERMES_HOST_PORT}"
upsert_env_value "${APP_DIR}/.env" "HERMES_API_KEY" "${HERMES_API_KEY}"
upsert_env_value "${APP_DIR}/.env" "HERMES_OPENAI_BASE_URL" "${HERMES_OPENAI_BASE_URL:-}"
upsert_env_value "${APP_DIR}/.env" "HERMES_OPENAI_API_KEY" "${HERMES_OPENAI_API_KEY:-}"
upsert_env_value "${APP_DIR}/.env" "HERMES_MODEL" "${HERMES_MODEL:-gpt-4.1-mini}"
chmod 600 "${APP_DIR}/.env"
log "Updated service authentication configuration"

if id "${APP_USER}" >/dev/null 2>&1; then
  chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}" || true
fi

if [ -n "${GHCR_TOKEN:-}" ]; then
  TEMP_DOCKER_CONFIG="$(mktemp -d "${TMPDIR:-/tmp}/hermes-agent-test-docker.XXXXXX")"
  chmod 700 "${TEMP_DOCKER_CONFIG}"
  export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
  log "Logging in to GHCR with an ephemeral Docker config"
  docker_login_ghcr() {
    printf '%s' "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME:-${GITHUB_ACTOR:-btcfoxman}}" --password-stdin >/dev/null
  }
  retry 5 10 docker_login_ghcr
fi

cd "${APP_DIR}"
export IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io}"
export IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-btcfoxman}"
export IMAGE_NAME="${IMAGE_NAME:-hermes-agent}"
if [[ ! "${IMAGE_TAG:-}" =~ ^test-[0-9a-f]{40}$ ]]; then
  log "IMAGE_TAG must be the immutable test-<40-char-git-sha> tag"
  exit 1
fi
export IMAGE_TAG

compose() {
  docker compose --project-name "${COMPOSE_PROJECT_NAME}" --file "${COMPOSE_FILE}" "$@"
}

log "Validating compose config"
compose config >/dev/null

log "Pulling image"
retry 5 10 compose pull hermes-agent-test

log "Starting service"
compose up -d --remove-orphans hermes-agent-test

log "Waiting for service health"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${HERMES_HOST_PORT}/health" >/dev/null \
    && authenticated_get "http://127.0.0.1:${HERMES_HOST_PORT}/api/v1/operators/health"; then
    probes_ok=true
    for role in commercial industry personal_ip; do
      if ! authenticated_get "http://127.0.0.1:${HERMES_HOST_PORT}/api/v1/operators/${role}/probe"; then
        probes_ok=false
        break
      fi
    done
    if [ "${probes_ok}" = true ] && probe_compose_contracts; then
      compose ps
      log "Deployment complete; all operator profiles and compose contracts are ready"
      exit 0
    fi
  fi
  sleep 5
done

log "Health check failed"
compose ps || true
compose logs --tail=200 hermes-agent-test || true
exit 1
