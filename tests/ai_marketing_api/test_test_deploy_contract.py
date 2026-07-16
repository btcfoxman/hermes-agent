from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_test_workflow_uses_environment_secret_and_immutable_commit_tag():
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(encoding="utf-8")

    assert "environment: test" in workflow
    assert "HERMES_API_KEY: ${{ secrets.HERMES_API_KEY }}" in workflow
    assert "IMAGE_TAG: test-${{ github.sha }}" in workflow
    assert 'echo "image_sha_tag=test-${GITHUB_SHA}"' in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.count("if: github.ref == 'refs/heads/test'") == 3
    assert "cleanup_auth" in workflow
    assert "trap cleanup_auth EXIT" in workflow
    assert "/home/btcfoxman/docker/hermes-agent-test" in workflow
    assert "HERMES_HOST_PORT: '18095'" in workflow
    assert "HERMES_OPENAI_API_KEY: ${{ secrets.HERMES_OPENAI_API_KEY }}" not in workflow


def test_deploy_script_persists_secret_without_using_a_mutable_deploy_tag():
    deploy_script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")

    assert "HERMES_API_KEY is required from the GitHub test Environment" in deploy_script
    assert 'upsert_env_value "${APP_DIR}/.env" "HERMES_API_KEY" "${HERMES_API_KEY}"' in deploy_script
    assert '[[ ! "${IMAGE_TAG:-}" =~ ^test-[0-9a-f]{40}$ ]]' in deploy_script
    assert 'upsert_env_value "${APP_DIR}/.env" "HERMES_OPENAI_API_KEY" "${HERMES_OPENAI_API_KEY:-}"' in deploy_script
    assert 'if [ -n "${HERMES_OPENAI_API_KEY:-}" ]' not in deploy_script
    assert "TEMP_DOCKER_CONFIG" in deploy_script
    assert 'export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"' in deploy_script
    assert 'authenticated_get "http://127.0.0.1:${HERMES_HOST_PORT}/api/v1/operators/${role}/probe"' in deploy_script
    assert "probe_compose_contracts" in deploy_script
    assert 'f"http://127.0.0.1:8095/api/v1/operators/{role}/compose"' in deploy_script
    assert 'result["schema_version"] == "operator.content.v1"' in deploy_script
    assert 'result["master_content"] is None' in deploy_script
    assert 'result["requires_human_review"] is True' in deploy_script
    assert '--header "Authorization: Bearer ${HERMES_API_KEY}"' in deploy_script
    assert "--header @-" not in deploy_script
    assert 'APP_DIR="${APP_DIR:-/home/btcfoxman/docker/hermes-agent-test}"' in deploy_script
    assert 'docker compose --project-name "${COMPOSE_PROJECT_NAME}"' in deploy_script
    assert '"${HERMES_HOST_PORT}" = "8095"' in deploy_script
    assert 'http://127.0.0.1:${HERMES_HOST_PORT}/health' in deploy_script
    assert "compose pull hermes-agent-test" in deploy_script
    assert "compose up -d --remove-orphans hermes-agent-test" in deploy_script
    assert "compose exec -T hermes-agent-test" in deploy_script


def test_container_healthcheck_authenticates_operator_readiness():
    compose = (REPOSITORY_ROOT / "test/docker-compose.yml").read_text(encoding="utf-8")

    assert "${IMAGE_TAG:?IMAGE_TAG must be an immutable test git SHA tag}" in compose
    assert "'Authorization': 'Bearer ' + os.environ['HERMES_API_KEY']" in compose
    assert "/api/v1/operators/health" in compose
    assert "name: ${COMPOSE_PROJECT_NAME:-hermes-agent-test}" in compose
    assert "container_name: hermes-agent-test" in compose
    assert "container_name: hermes-agent\n" not in compose
    assert '"8095:8095"' not in compose
    assert '"${HERMES_HOST_PORT:-18095}:8095"' in compose
    assert "./logs/hermes-agent:/app/logs" in compose


def test_test_env_example_has_only_test_project_and_host_port_defaults():
    env_example = (REPOSITORY_ROOT / "test/.env.example").read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME=hermes-agent-test" in env_example
    assert "HERMES_HOST_PORT=18095" in env_example
