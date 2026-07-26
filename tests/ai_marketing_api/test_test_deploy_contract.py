from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_test_workflow_targets_the_canonical_lan_service_with_an_immutable_tag():
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(
        encoding="utf-8"
    )

    assert "environment: test" in workflow
    assert "IMAGE_TAG: test-${{ github.sha }}" in workflow
    assert 'echo "image_sha_tag=test-${GITHUB_SHA}"' in workflow
    assert "cancel-in-progress: true" in workflow
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


def test_deploy_script_preserves_server_credentials_and_updates_only_canonical_hermes():
    deploy_script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")

    assert 'APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent}"' in deploy_script
    assert 'COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hermes-agent}"' in deploy_script
    assert 'HERMES_HOST_PORT="${HERMES_HOST_PORT:-8095}"' in deploy_script
    assert '[ "${resolved_app_dir}" = "/home/btcfoxman/docker/hermes-agent" ]' in deploy_script
    assert "deployment does not create or replace model credentials" in deploy_script
    assert "upsert_env_value" not in deploy_script
    assert '[[ ! "${IMAGE_TAG:-}" =~ ^test-[0-9a-f]{40}$ ]]' in deploy_script
    assert "TEMP_DOCKER_CONFIG" in deploy_script
    assert 'export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"' in deploy_script
    assert "probe_operator_endpoints" in deploy_script
    assert "probe_compose_contracts" in deploy_script
    assert "probe_industry_claim_contract" in deploy_script
    assert 'result["model"]' not in deploy_script
    assert 'str(result.get("model") or "").lower() != "fallback"' in deploy_script
    assert '[(source_text, "fact", ["canonical-official-probe"])]' in deploy_script
    assert "retry 3 10 probe_industry_claim_contract" in deploy_script
    assert 'docker compose \\\n    --project-name "${COMPOSE_PROJECT_NAME}"' in deploy_script
    assert 'http://127.0.0.1:${HERMES_HOST_PORT}/health' in deploy_script
    assert "compose pull hermes-agent" in deploy_script
    assert "compose up -d --remove-orphans hermes-agent" in deploy_script
    assert "compose exec -T hermes-agent" in deploy_script
    assert "hermes-agent-test" not in deploy_script


def test_container_healthcheck_authenticates_canonical_operator_readiness():
    compose = (REPOSITORY_ROOT / "test/docker-compose.yml").read_text(
        encoding="utf-8"
    )

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
