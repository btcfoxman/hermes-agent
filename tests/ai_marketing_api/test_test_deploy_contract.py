from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_test_workflow_targets_the_canonical_lan_service_with_an_immutable_tag():
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(
        encoding="utf-8"
    )

    assert "environment: test" in workflow
    assert "IMAGE_TAG: test-${{ github.sha }}" in workflow
    assert 'echo "image_sha_tag=test-${GITHUB_SHA}"' in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.count("if: github.ref == 'refs/heads/test'") == 3
    assert "cleanup_auth" in workflow
    assert "trap cleanup_auth EXIT" in workflow
    assert "APP_DIR: /home/btcfoxman/docker/hermes-agent" in workflow
    assert "COMPOSE_PROJECT_NAME: hermes-agent" in workflow
    assert "HERMES_HOST_PORT: '8095'" in workflow
    assert "hermes-agent-test" not in workflow
    assert "HERMES_API_KEY: ${{ secrets.HERMES_API_KEY }}" not in workflow
    assert "HERMES_OPENAI_API_KEY:" not in workflow
    assert "python -m venv .venv" in workflow
    assert ".venv/bin/python -m pip install" in workflow
    assert "bash scripts/run_tests.sh tests/ai_marketing_api" in workflow
    assert "python -m pytest" not in workflow


def test_followup_push_queues_without_canceling_an_active_canonical_deployment():
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(
        encoding="utf-8"
    )
    concurrency = workflow.split("\nconcurrency:\n", 1)[1].split("\n\n", 1)[0]
    assert concurrency.splitlines() == [
        "  group: hermes-agent-canonical-test-deploy",
        "  cancel-in-progress: false",
    ]
    assert "cancel-in-progress: true" not in workflow


def test_deploy_script_preserves_server_credentials_and_updates_only_canonical_hermes():
    deploy_script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")

    assert 'APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent}"' in deploy_script
    assert (
        'COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hermes-agent}"' in deploy_script
    )
    assert 'HERMES_HOST_PORT="${HERMES_HOST_PORT:-8095}"' in deploy_script
    assert (
        '[ "${resolved_app_dir}" = "/home/btcfoxman/docker/hermes-agent" ]'
        in deploy_script
    )
    assert "deployment does not create or replace model credentials" in deploy_script
    assert "upsert_env_value" not in deploy_script
    assert '[[ ! "${IMAGE_TAG:-}" =~ ^test-[0-9a-f]{40}$ ]]' in deploy_script
    assert "TEMP_DOCKER_CONFIG" in deploy_script
    assert 'export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"' in deploy_script
    assert "probe_operator_endpoints" in deploy_script
    assert "probe_compose_contracts" in deploy_script
    assert "probe_industry_claim_contract" in deploy_script
    assert "probe_service_health" in deploy_script
    assert 'result["model"]' not in deploy_script
    assert 'str(result.get("model") or "").lower() != "fallback"' in deploy_script
    assert '[(source_text, "fact", ["canonical-official-probe"])]' in deploy_script
    assert "retry 3 10 probe_industry_claim_contract" in deploy_script
    assert (
        'docker compose \\\n    --project-name "${COMPOSE_PROJECT_NAME}"'
        in deploy_script
    )
    assert '"http://127.0.0.1:8095/health"' in deploy_script
    assert "http://127.0.0.1:${HERMES_HOST_PORT}/health" not in deploy_script
    assert "compose pull hermes-agent" in deploy_script
    assert "compose up -d --remove-orphans hermes-agent" in deploy_script
    assert "compose exec -T hermes-agent" in deploy_script
    assert "hermes-agent-test" not in deploy_script


def test_container_healthcheck_authenticates_canonical_operator_readiness():
    compose = (REPOSITORY_ROOT / "test/docker-compose.yml").read_text(encoding="utf-8")

    assert "${IMAGE_TAG:?IMAGE_TAG must be an immutable test git SHA tag}" in compose
    assert "'Authorization': 'Bearer ' + os.environ['HERMES_API_KEY']" in compose
    assert "/api/v1/operators/health" in compose
    assert "name: hermes-agent" in compose
    assert "container_name: hermes-agent" in compose
    assert "hermes-agent-test" not in compose
    assert '"192.168.3.6:8095:8095"' in compose
    assert "./logs/hermes-agent:/app/logs" in compose


def test_env_example_documents_only_runtime_configuration():
    env_example = (REPOSITORY_ROOT / "test/.env.example").read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME" not in env_example
    assert "HERMES_HOST_PORT" not in env_example
    assert "HERMES_OPENAI_BASE_URL=" in env_example
    assert "HERMES_OPENAI_API_KEY=" in env_example
    assert "HERMES_API_KEY=" in env_example


def test_host_deployment_is_locked_before_all_preflight_and_runtime_changes():
    script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")
    assert (
        'readonly SHARED_DEPLOY_LOCK="/home/btcfoxman/docker/.ai-marketing-test-deploy.lock"'
        in script
    )
    assert "umask 077" in script
    assert 'exec 200>>"${SHARED_DEPLOY_LOCK}"' in script
    assert "flock -w 1200 200" in script
    assert "command -v flock" in script
    assert '[ -L "${SHARED_DEPLOY_LOCK}" ]' in script
    assert '[ ! "/proc/$$/fd/200" -ef "${SHARED_DEPLOY_LOCK}" ]' in script
    assert "Waiting for shared LAN deployment lock" in script
    assert "Acquired shared LAN deployment lock" in script
    acquired = script.index("\nacquire_shared_deploy_lock\n")
    for change in (
        "resolved_app_dir=",
        'mkdir -p "${APP_DIR}',
        'if [ ! -f "${APP_DIR}/.env" ]',
        "cp -f ",
        "retry 5 10 docker_login_ghcr",
        "compose config >/dev/null",
        "retry 5 10 compose pull",
        "compose up -d",
        "for i in $(seq 1 30)",
    ):
        assert acquired < script.index(change)
    assert "flock -u" not in script
    assert "exec 200>&-" not in script
    assert not any(
        "rm " in line and "SHARED_DEPLOY_LOCK" in line for line in script.splitlines()
    )
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(
        encoding="utf-8"
    )
    assert "flock" not in workflow  # Hosted quality/image builds remain parallel.


def _lock_fragment(path: Path) -> str:
    script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")
    function = script.split("acquire_shared_deploy_lock() {", 1)[1].split("\n}\n", 1)[0]
    return (
        "\n".join([
            "set -euo pipefail",
            "umask 077",
            'log() { printf "%s\\n" "$*"; }',
            "acquire_shared_deploy_lock() {" + function + "\n}",
            "acquire_shared_deploy_lock",
        ])
        .replace("/home/btcfoxman/docker/.ai-marketing-test-deploy.lock", str(path))
        .replace("flock -w 1200 200", "flock -w 1 200")
    )


@pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("flock")),
    reason="Linux Bash/flock integration",
)
def test_shared_lock_rejects_symlinks_and_keeps_the_target_untouched(tmp_path):
    target = tmp_path / "target"
    target.write_text("preserve", encoding="utf-8")
    lock = tmp_path / "lock"
    lock.symlink_to(target)
    result = subprocess.run(
        ["bash", "-c", _lock_fragment(lock)], capture_output=True, text=True, timeout=5
    )
    assert result.returncode != 0
    assert "Refusing a symbolic-link" in result.stdout
    assert target.read_text(encoding="utf-8") == "preserve"
    assert lock.is_symlink()


@pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("flock")),
    reason="Linux Bash/flock integration",
)
def test_shared_lock_serializes_waiters_and_persists_after_exit(tmp_path):
    lock = tmp_path / "lock"
    fragment = _lock_fragment(lock)
    with subprocess.Popen(
        ["bash", "-c", fragment + '\nprintf "holder-ready\\n"\nread -r release'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as holder:
        try:
            assert "Waiting" in holder.stdout.readline()
            assert "Acquired" in holder.stdout.readline()
            assert holder.stdout.readline().strip() == "holder-ready"
            inode = lock.stat().st_ino
            assert lock.stat().st_mode & 0o777 == 0o600
            waiting = subprocess.run(
                ["bash", "-c", fragment], capture_output=True, text=True, timeout=5
            )
            assert waiting.returncode != 0
            assert "Timed out waiting" in waiting.stdout
            assert "Acquired" not in waiting.stdout
        finally:
            holder.communicate("release\n", timeout=5)
    after = subprocess.run(
        ["bash", "-c", fragment], capture_output=True, text=True, timeout=5
    )
    assert after.returncode == 0, after.stderr
    assert "Acquired" in after.stdout
    assert lock.exists() and lock.stat().st_ino == inode
