"""Offline model load + `booster start` waiting for the daemon's real outcome."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from booster import daemon, embed


@pytest.fixture
def fake_st(monkeypatch):
    state = types.SimpleNamespace(calls=[], cached=True)

    class FakeST:
        def __init__(self, name, local_files_only=False):
            state.calls.append(local_files_only)
            if local_files_only and not state.cached:
                raise OSError("not in cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setattr(embed, "_embed_model", None)
    return state


def test_cached_model_never_touches_the_hub(fake_st):
    embed.get_embed_model()
    assert fake_st.calls == [True]


def test_uncached_model_falls_back_to_download(fake_st):
    fake_st.cached = False
    embed.get_embed_model()
    assert fake_st.calls == [True, False]


class _Proc:
    def __init__(self, exit_after: int | None, code: int = 1):
        self.polls, self.exit_after, self.returncode = 0, exit_after, None
        self._code = code

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls >= self.exit_after:
            self.returncode = self._code
        return self.returncode


@pytest.fixture
def start(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)

    def run(proc, ping_after=None, log=""):
        pings = {"n": 0}

        def ping(root):
            pings["n"] += 1
            return {"pid": 1} if ping_after is not None and pings["n"] > ping_after else None

        def popen(*a, stdout=None, **k):
            stdout.write(log)
            stdout.flush()
            return proc

        monkeypatch.setattr(daemon, "daemon_ping", ping)
        monkeypatch.setattr("subprocess.Popen", popen)
        return daemon.start_daemon(tmp_path)
    return run


def test_start_reports_crash_log(start):
    assert start(_Proc(exit_after=3), log="ImportError: no torch") == "ImportError: no torch"


def test_start_reports_exit_code_when_log_empty(start):
    assert start(_Proc(exit_after=2, code=3)) == "daemon exited with code 3"


def test_start_waits_past_old_5s_limit(start):
    # 40 polls x 0.25s = 10s of warm-up: the old 20-poll loop gave up here.
    assert start(_Proc(exit_after=None), ping_after=41) is True


def test_start_slow_but_alive_is_not_an_error(start, monkeypatch):
    monkeypatch.setattr(daemon, "_START_TIMEOUT_S", 0)
    assert start(_Proc(exit_after=None)) is True


def test_socket_path_short_root_stays_in_project():
    root = Path("/tmp/p")  # pytest's tmp_path alone can exceed the limit on macOS
    assert daemon.socket_path(root) == root / ".booster" / "daemon.sock"


def test_socket_path_deep_root_moves_to_tempdir(tmp_path):
    deep = tmp_path / ("x" * 120)
    p = daemon.socket_path(deep)
    assert len(str(p)) <= daemon._MAX_SOCK_PATH and p.name.startswith("booster-")
    assert daemon.socket_path(deep) == p  # stable per repo
    assert daemon.socket_path(tmp_path / ("y" * 120)) != p  # distinct per repo
