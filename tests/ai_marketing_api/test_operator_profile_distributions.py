from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_marketing_api.operator_runtime import PROFILE_ROOT, OperatorRole
from hermes_cli.config import load_config
from hermes_cli.profile_distribution import install_distribution, read_manifest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def test_bundled_operator_assets_install_as_three_isolated_profiles(profile_env, monkeypatch):
    installed = {}
    for role in OperatorRole:
        source = PROFILE_ROOT / role.value
        manifest = read_manifest(source)
        assert manifest is not None
        assert manifest.name == role.value

        plan = install_distribution(str(source))
        installed[role.value] = plan.target_dir

    assert len(set(installed.values())) == len(installed)
    assert len({(path / "SOUL.md").read_bytes() for path in installed.values()}) == len(installed)
    for role, path in installed.items():
        assert path == profile_env / "profiles" / role
        assert (path / "SOUL.md").is_file()
        assert (path / "config.yaml").is_file()
        permissions = json.loads((path / "operator.json").read_text(encoding="utf-8"))
        assert permissions["role_id"] == role
        assert permissions["version"] == read_manifest(path).version
        assert permissions["allowed_tools"] == []
        assert "cross_profile_memory" in permissions["denied_capabilities"]
        assert not (path / ".env").exists()

        monkeypatch.setenv("HERMES_HOME", str(path))
        runtime_config = load_config()
        assert runtime_config["toolsets"] == []
        assert runtime_config["memory"]["memory_enabled"] is False


def test_profile_configs_have_no_runtime_tool_or_memory_access():
    for role in OperatorRole:
        config = (PROFILE_ROOT / role.value / "config.yaml").read_text(encoding="utf-8")
        assert "toolsets: []" in config
        assert "memory_enabled: false" in config
        permissions = json.loads(
            (PROFILE_ROOT / role.value / "operator.json").read_text(encoding="utf-8")
        )
        assert permissions["allowed_tools"] == []
        assert {"direct_database", "direct_cache", "filesystem_knowledge"}.issubset(
            permissions["denied_capabilities"]
        )
