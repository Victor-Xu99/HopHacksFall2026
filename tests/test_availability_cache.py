"""The availability probes are memoized, and the memo has to be escapable.

A cached probe is what stops one unreachable database from costing a connect
timeout on every request and repeating the same warning until it buries the
log. The risk it introduces is the opposite one: a server that comes up later
staying marked as down forever. Both directions are covered here.
"""

import pyodbc
import pytest

from src.data import safetyhops, sql_store


class TestSqlStoreAvailability:
    def test_probe_runs_once_per_server_and_database(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sql_store, "_probe", lambda s, d: calls.append((s, d)) or True)
        sql_store.reset_availability_cache()

        for _ in range(5):
            assert sql_store.available("SRV", "DB") is True

        assert calls == [("SRV", "DB")]

    def test_distinct_targets_are_probed_separately(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sql_store, "_probe", lambda s, d: calls.append((s, d)) or True)
        sql_store.reset_availability_cache()

        sql_store.available("SRV", "one")
        sql_store.available("SRV", "two")

        assert calls == [("SRV", "one"), ("SRV", "two")]

    def test_negative_results_are_cached_too(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sql_store, "_probe", lambda s, d: calls.append(1) or False)
        sql_store.reset_availability_cache()

        assert sql_store.available("SRV", "DB") is False
        assert sql_store.available("SRV", "DB") is False
        assert len(calls) == 1

    def test_reset_lets_a_recovered_server_be_noticed(self, monkeypatch):
        answers = iter([False, True])
        monkeypatch.setattr(sql_store, "_probe", lambda s, d: next(answers))
        sql_store.reset_availability_cache()

        assert sql_store.available("SRV", "DB") is False
        sql_store.reset_availability_cache()
        assert sql_store.available("SRV", "DB") is True


class TestSafetyHopsAvailability:
    def test_probe_runs_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(safetyhops, "_probe", lambda s, d: calls.append(1) or True)
        safetyhops.reset_availability_cache()

        for _ in range(4):
            safetyhops.hops_available("SRV", "DB")

        assert len(calls) == 1

    def test_reset_clears_the_memo(self, monkeypatch):
        answers = iter([False, True])
        monkeypatch.setattr(safetyhops, "_probe", lambda s, d: next(answers))
        safetyhops.reset_availability_cache()

        assert safetyhops.hops_available("SRV", "DB") is False
        safetyhops.reset_availability_cache()
        assert safetyhops.hops_available("SRV", "DB") is True


class TestDiagnosis:
    """A login failure reported as an unreachable server sends you hunting for
    the wrong problem, which is exactly what happened on this machine."""

    def test_missing_server_points_at_the_instance_name(self):
        exc = pyodbc.Error("08001", "[08001] Error Locating Server/Instance Specified")
        message = sql_store._diagnose("HOST", "SafetyNet", exc)
        assert "cannot reach server HOST" in message
        assert "SAFETYNET_SQL_SERVER" in message

    def test_login_failure_points_at_the_missing_database(self):
        exc = pyodbc.Error("28000", "[28000] Login failed for user 'x'. (18456)")
        message = sql_store._diagnose("HOST", "SafetyNet", exc)
        assert "could not open database SafetyNet" in message
        assert "build_safetynet_db" in message
        assert "cannot reach" not in message

    def test_unrecognized_sqlstate_still_names_both_ends(self):
        exc = pyodbc.Error("HY000", "[HY000] something else entirely")
        message = sql_store._diagnose("HOST", "SafetyNet", exc)
        assert "HOST/SafetyNet" in message


@pytest.mark.parametrize("module", (sql_store, safetyhops))
def test_reset_is_safe_to_call_when_nothing_was_probed(module):
    module.reset_availability_cache()
    module.reset_availability_cache()
