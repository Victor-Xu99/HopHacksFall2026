"""The .env loader, which is the only thing standing between a teammate and a
hard-coded instance name in a tracked file."""

import os

import pytest

from src.config import load_env_file


@pytest.fixture
def env_file(tmp_path):
    def write(text: str):
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    return write


class TestLoadEnvFile:
    def test_sets_values_from_the_file(self, env_file, monkeypatch):
        monkeypatch.delenv("SAFETYNET_SQL_SERVER", raising=False)
        load_env_file(env_file("SAFETYNET_SQL_SERVER=BOXNAME\n"))
        assert os.environ["SAFETYNET_SQL_SERVER"] == "BOXNAME"

    def test_a_real_environment_variable_wins(self, env_file, monkeypatch):
        monkeypatch.setenv("SAFETYNET_SQL_SERVER", "FROM_SHELL")
        load_env_file(env_file("SAFETYNET_SQL_SERVER=FROM_FILE\n"))
        assert os.environ["SAFETYNET_SQL_SERVER"] == "FROM_SHELL"

    def test_ignores_comments_and_blank_lines(self, env_file, monkeypatch):
        monkeypatch.delenv("SAFETYNET_SQL_DATABASE", raising=False)
        load_env_file(
            env_file("# a comment\n\n   \nSAFETYNET_SQL_DATABASE=CoreDb\n")
        )
        assert os.environ["SAFETYNET_SQL_DATABASE"] == "CoreDb"

    def test_strips_surrounding_quotes(self, env_file, monkeypatch):
        monkeypatch.delenv("SAFETYNET_SQL_SERVER", raising=False)
        load_env_file(env_file('SAFETYNET_SQL_SERVER="  .\\SQLEXPRESS  "\n'))
        assert os.environ["SAFETYNET_SQL_SERVER"] == ".\\SQLEXPRESS"

    def test_keeps_backslashes_in_named_instances(self, env_file, monkeypatch):
        """A named instance is .\\NAME and must survive parsing intact."""
        monkeypatch.delenv("SAFETYNET_SQL_SERVER", raising=False)
        load_env_file(env_file("SAFETYNET_SQL_SERVER=.\\SQLEXPRESS\n"))
        assert os.environ["SAFETYNET_SQL_SERVER"] == ".\\SQLEXPRESS"

    def test_values_containing_equals_are_kept_whole(self, env_file, monkeypatch):
        monkeypatch.delenv("SAFETYNET_ODD", raising=False)
        load_env_file(env_file("SAFETYNET_ODD=a=b=c\n"))
        assert os.environ["SAFETYNET_ODD"] == "a=b=c"

    def test_missing_file_is_not_an_error(self, tmp_path):
        load_env_file(tmp_path / "nope.env")

    def test_malformed_lines_are_skipped(self, env_file, monkeypatch):
        monkeypatch.delenv("SAFETYNET_SQL_SERVER", raising=False)
        load_env_file(env_file("no equals sign here\nSAFETYNET_SQL_SERVER=OK\n"))
        assert os.environ["SAFETYNET_SQL_SERVER"] == "OK"


def test_example_file_documents_every_key_config_reads():
    """`.env.example` is the only instruction a teammate gets; keep it complete."""
    from pathlib import Path

    from src.config import REPO_ROOT

    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "SAFETYNET_SQL_SERVER",
        "SAFETYNET_SQL_DATABASE",
        "SAFETYHOPS_SQL_DATABASE",
        "SAFETYNET_SQL_DRIVER",
    ):
        assert key in example, f"{key} is read by src/config.py but undocumented"
