from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_test_workflow_uses_environment_secret_and_immutable_commit_tag():
    workflow = (REPOSITORY_ROOT / ".github/workflows/deploy-test.yml").read_text(encoding="utf-8")

    assert "environment: test" in workflow
    assert "HERMES_API_KEY: ${{ secrets.HERMES_API_KEY }}" in workflow
    assert "IMAGE_TAG: test-${{ github.sha }}" in workflow
    assert 'echo "image_sha_tag=test-${GITHUB_SHA}"' in workflow
    assert "cancel-in-progress: false" in workflow


def test_deploy_script_persists_secret_without_using_a_mutable_deploy_tag():
    deploy_script = (REPOSITORY_ROOT / "test/deploy.sh").read_text(encoding="utf-8")

    assert "HERMES_API_KEY is required from the GitHub test Environment" in deploy_script
    assert 'upsert_env_value "${APP_DIR}/.env" "HERMES_API_KEY" "${HERMES_API_KEY}"' in deploy_script
    assert 'if [ -z "${IMAGE_TAG:-}" ] || [ "${IMAGE_TAG}" = "test-latest" ]' in deploy_script
    assert 'authenticated_get "http://127.0.0.1:8095/api/v1/operators/${role}/probe"' in deploy_script
    assert '--header "Authorization: Bearer ${HERMES_API_KEY}"' in deploy_script
    assert "--header @-" not in deploy_script


def test_container_healthcheck_authenticates_operator_readiness():
    compose = (REPOSITORY_ROOT / "test/docker-compose.yml").read_text(encoding="utf-8")

    assert "${IMAGE_TAG:?IMAGE_TAG must be an immutable test git SHA tag}" in compose
    assert "'Authorization': 'Bearer ' + os.environ['HERMES_API_KEY']" in compose
    assert "/api/v1/operators/health" in compose
