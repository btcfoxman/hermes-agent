#!/usr/bin/env bash
set -euo pipefail
umask 077

APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hermes-agent}"
HERMES_HOST_PORT="${HERMES_HOST_PORT:-8095}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
TEMP_DOCKER_CONFIG=""

log() {
  printf '[hermes-agent-deploy] %s\n' "$*"
}

acquire_shared_deploy_lock() {
  # Every repository deploying to this small LAN host uses the same inode.
  # Never unlink the lock: waiters must not accidentally lock different files.
  readonly SHARED_DEPLOY_LOCK="/home/btcfoxman/docker/.ai-marketing-test-deploy.lock"
  if ! command -v flock >/dev/null 2>&1; then
    log "flock is required for serialized LAN deployment"
    return 1
  fi
  if [ -L "${SHARED_DEPLOY_LOCK}" ] || { [ -e "${SHARED_DEPLOY_LOCK}" ] && [ ! -f "${SHARED_DEPLOY_LOCK}" ]; }; then
    log "Refusing a symbolic-link or non-regular shared deployment lock"
    return 1
  fi
  exec 200>>"${SHARED_DEPLOY_LOCK}"
  if [ -L "${SHARED_DEPLOY_LOCK}" ] || [ ! -f "${SHARED_DEPLOY_LOCK}" ] || [ ! "/proc/$$/fd/200" -ef "${SHARED_DEPLOY_LOCK}" ]; then
    log "Shared deployment lock changed while opening it; refusing deployment"
    return 1
  fi
  log "Waiting for shared LAN deployment lock (up to 1200 seconds)"
  if ! flock -w 1200 200; then
    log "Timed out waiting for shared LAN deployment lock; no deployment changes were made"
    return 1
  fi
  if [ -L "${SHARED_DEPLOY_LOCK}" ] || [ ! "/proc/$$/fd/200" -ef "${SHARED_DEPLOY_LOCK}" ]; then
    log "Shared deployment lock changed while waiting; refusing deployment"
    return 1
  fi
  log "Acquired shared LAN deployment lock; protecting preflight, pull, startup and health checks"
  # FD 200 stays open through all deployment and EXIT cleanup operations.
  # The operating system releases the lock when this process exits.
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
  if [ -n "${TEMP_DOCKER_CONFIG}" ] && [ -d "${TEMP_DOCKER_CONFIG}" ]; then
    rm -rf -- "${TEMP_DOCKER_CONFIG}"
  fi
}

trap cleanup EXIT

compose() {
  docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    "$@"
}

validate_hermes_receipt_home() {
  local env_file="$1" line value
  local assignment='^[[:space:]]*(export[[:space:]]+)?HERMES_HOME([[:space:]]*=|[[:space:]]*(#.*)?$)'
  local canonical_bare='^/data/hermes([[:space:]]+(#.*)?)?$'
  local canonical_double='^"/data/hermes"[[:space:]]*(#.*)?$'
  local canonical_single="^'/data/hermes'[[:space:]]*(#.*)?$"
  if [ "${HERMES_HOME+x}" = x ] && [ "${HERMES_HOME}" != "/data/hermes" ]; then
    log "Hermes HERMES_HOME must be /data/hermes for durable editorial receipts"
    return 1
  fi
  [ -f "${env_file}" ] || return 0
  # Never source credentials. Check every declaration, including duplicate or
  # explicit empty values, rather than accepting only the first occurrence.
  while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%$'\r'}"
    [[ "${line}" =~ ${assignment} ]] || continue
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    if [[ ! "${value}" =~ ${canonical_bare} && ! "${value}" =~ ${canonical_double} && ! "${value}" =~ ${canonical_single} ]]; then
      log "Hermes HERMES_HOME must be /data/hermes for durable editorial receipts"
      return 1
    fi
  done < "${env_file}"
}

verify_hermes_receipt_mount() {
  local container_name="$1" receipt_mount canonical_home
  receipt_mount="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/data/hermes/marketing_api"}}{{.Type}}|{{.Name}}{{end}}{{end}}' "${container_name}" 2>/dev/null)" || return 1
  [ "${receipt_mount}" = "volume|hermes-editorial-receipts" ] || {
    log "Hermes durable editorial receipt volume is missing or unexpected"
    return 1
  }
  canonical_home="$(docker inspect --format '{{range .Config.Env}}{{if eq . "HERMES_HOME=/data/hermes"}}canonical{{end}}{{end}}' "${container_name}" 2>/dev/null)" || return 1
  [ "${canonical_home}" = canonical ] || {
    log "Hermes durable editorial receipt home is unexpected"
    return 1
  }
}

probe_service_health() {
  compose exec -T hermes-agent python -c '
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8095/health", timeout=10) as response:
    assert response.status == 200
'
}

probe_operator_endpoints() {
  compose exec -T hermes-agent python -c '
import json
import os
import urllib.request

token = os.environ["HERMES_API_KEY"]
for path in (
    "/api/v1/operators/health",
    "/api/v1/operators/commercial/probe",
    "/api/v1/operators/industry/probe",
    "/api/v1/operators/personal_ip/probe",
):
    request = urllib.request.Request(
        "http://127.0.0.1:8095" + path,
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        assert response.status == 200
        result = json.load(response)
    assert result["status"] == "ok"
    if path.endswith("/health"):
        assert all(role["ready"] for role in result["roles"].values())
    else:
        assert result["role_id"] == path.split("/")[-2]
        assert result["knowledge_source"] == "request_authorized_context_only"
        assert result["schema_versions"] == {"proposal": "operator.proposal.v1", "content": "operator.content.v1"}
'
}

probe_schema_contract() {
  compose exec -T hermes-agent python -c '
import json
import os
import urllib.error
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8095/openapi.json", timeout=20) as response:
    schema = json.load(response)
paths = schema["paths"]
for path in (
    "/api/v1/operators/{role_id}/compose",
    "/api/v1/operators/{role_id}/propose",
    "/api/v1/operators/{role_id}/revise",
    "/api/v1/operators/editorial/plan",
    "/api/v1/operators/{role_id}/editorial-steps/{step}",
):
    assert "post" in paths[path], path
receipt_path = "/api/v1/operators/{role_id}/editorial-steps/{step}/receipts/{reservation_id}"
assert "get" in paths[receipt_path]
request = urllib.request.Request(
    "http://127.0.0.1:8095/api/v1/operators/industry/editorial-steps/review_editorial/receipts/deployment_probe",
    headers={"Authorization": "Bearer " + os.environ["HERMES_API_KEY"], "X-Hermes-Request-SHA256": "0" * 64},
)
try:
    with urllib.request.urlopen(request, timeout=20):
        raise AssertionError("A deployment probe must not match an existing paid receipt")
except urllib.error.HTTPError as error:
    assert error.code == 404
definitions = schema["components"]["schemas"]
assert {"genre", "style_references", "editorial_feedback"} <= set(definitions["OperatorComposeRequest"]["properties"])
assert {"candidate", "candidate_signature", "budget_reservation_id"} <= set(definitions["EditorialStepRequest"]["properties"])
assert {"surfaces", "content_sha256"} <= set(definitions["QualityScorecard"]["properties"])
'
}

acquire_shared_deploy_lock

resolved_app_dir="$(realpath -m "${APP_DIR}")"
[ "${resolved_app_dir}" = "/home/btcfoxman/docker/hermes-agent" ] || {
  log "Refusing unexpected APP_DIR: ${resolved_app_dir}"
  exit 1
}
[ "${COMPOSE_PROJECT_NAME}" = "hermes-agent" ] || {
  log "Refusing unexpected COMPOSE_PROJECT_NAME: ${COMPOSE_PROJECT_NAME}"
  exit 1
}
[ "${HERMES_HOST_PORT}" = "8095" ] || {
  log "Refusing unexpected HERMES_HOST_PORT: ${HERMES_HOST_PORT}"
  exit 1
}

APP_DIR="${resolved_app_dir}"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
mkdir -p "${APP_DIR}/logs/hermes-agent"

if [ ! -f "${APP_DIR}/.env" ]; then
  log "Canonical ${APP_DIR}/.env is required; deployment does not create or replace model credentials"
  exit 1
fi
validate_hermes_receipt_home "${APP_DIR}/.env" || exit 1

log "Syncing canonical deployment compose while preserving the existing .env"
cp -f "${SCRIPT_DIR}/docker-compose.yml" "${COMPOSE_FILE}"
chmod 644 "${COMPOSE_FILE}"
chmod 600 "${APP_DIR}/.env"

if [ -n "${GHCR_TOKEN:-}" ]; then
  TEMP_DOCKER_CONFIG="$(mktemp -d "${TMPDIR:-/tmp}/hermes-agent-docker.XXXXXX")"
  chmod 700 "${TEMP_DOCKER_CONFIG}"
  export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
  log "Logging in to GHCR with an ephemeral Docker config"
  docker_login_ghcr() {
    printf '%s' "${GHCR_TOKEN}" |
      docker login ghcr.io \
        -u "${GHCR_USERNAME:-${GITHUB_ACTOR:-btcfoxman}}" \
        --password-stdin >/dev/null
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

log "Validating compose config"
compose config >/dev/null

log "Pulling immutable image ${IMAGE_TAG}"
retry 5 10 compose pull hermes-agent

log "Starting canonical shared Hermes service"
compose up -d --remove-orphans hermes-agent
verify_hermes_receipt_mount hermes-agent || exit 1

log "Waiting for canonical service and operator contracts"
for i in $(seq 1 30); do
  if probe_service_health \
    && probe_operator_endpoints \
    && probe_schema_contract; then
    compose ps
    log "Deployment complete; canonical Hermes health, role and API schema contracts are ready"
    log "Paid model smoke was not executed; use orchestration budget reservations for controlled generation acceptance"
    exit 0
  fi
  sleep 5
done

log "Health or contract verification failed"
compose ps || true
compose logs --tail=200 hermes-agent || true
exit 1
