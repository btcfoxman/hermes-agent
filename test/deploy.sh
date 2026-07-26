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
        json.load(response)
'
}

probe_compose_contracts() {
  compose exec -T hermes-agent python -c '
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
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
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

probe_industry_claim_contract() {
  compose exec -T hermes-agent python -c '
import json
import os
import urllib.request

source_text = "监管机构发布了可逐字核验的官方事实。"
payload = {
    "objective": "验证行业操盘手证据边界",
    "topic": "官方事实验收",
    "audience": "AI product teams",
    "channels": ["wechat_mp"],
    "constraints": ["Only use facts present in authorized_context."],
    "authorized_context": [{
        "record_id": "canonical-official-probe",
        "space": "industry",
        "record_type": "source_item",
        "title": "Canonical official probe",
        "content": source_text,
        "structured_data": {},
        "status": "approved",
        "authorized_roles": ["industry"],
        "source_uri": "https://example.com/canonical-official-probe",
        "source_tier": "official",
    }],
    "as_of": "2026-07-27T00:00:00Z",
}
request = urllib.request.Request(
    "http://127.0.0.1:8095/api/v1/operators/industry/propose",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + os.environ["HERMES_API_KEY"],
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(request, timeout=240) as response:
    result = json.load(response)
assert result["schema_version"] == "operator.proposal.v1", result
assert str(result.get("model") or "").lower() != "fallback", result
assert result["status"] == "proposal", result
assert [
    (claim["text"], claim["kind"], claim["evidence_ids"])
    for claim in result["claims"]
] == [(source_text, "fact", ["canonical-official-probe"])], result
assert not any(
    risk.get("code") == "unsupported_fact"
    for risk in result.get("risk_flags") or []
), result
'
}

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

log "Waiting for canonical service and operator contracts"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${HERMES_HOST_PORT}/health" >/dev/null \
    && probe_operator_endpoints \
    && probe_compose_contracts; then
    if retry 3 10 probe_industry_claim_contract; then
      compose ps
      log "Deployment complete; canonical Hermes and all operator contracts are ready"
      exit 0
    fi
    log "Canonical industry model contract remained unavailable after retries"
    break
  fi
  sleep 5
done

log "Health or contract verification failed"
compose ps || true
compose logs --tail=200 hermes-agent || true
exit 1
