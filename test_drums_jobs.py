#!/usr/bin/env python3
"""The default worker count of `drums.drum_roles`. No audio; a file of its own so that it runs in the fast set
(test_drums.py is marked slow as a whole file in conftest.py)."""

import subprocess

import pytest

from roadie_analysis import drums


@pytest.fixture(autouse=True)
def fresh():
    drums.default_jobs.cache_clear()
    yield
    drums.default_jobs.cache_clear()


def _sysctl(monkeypatch, out=None, error=None):
    def run(cmd, **kw):
        assert cmd == ["sysctl", "-n", "hw.perflevel0.logicalcpu"]
        if error:
            raise error
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(drums.subprocess, "run", run)
    monkeypatch.setattr(drums.sys, "platform", "darwin")
    monkeypatch.setattr(drums.os, "cpu_count", lambda: 12)


def test_the_default_worker_count_is_the_fast_cores(monkeypatch):
    _sysctl(monkeypatch, out="6\n")
    assert drums.default_jobs() == 6


@pytest.mark.parametrize("error,out", [(subprocess.CalledProcessError(1, "sysctl"), None), (FileNotFoundError(), None),
                                       (subprocess.TimeoutExpired("sysctl", 5), None), (None, "oops\n"), (None, "0\n")])
def test_a_failing_sysctl_falls_back_to_every_core(monkeypatch, error, out):
    _sysctl(monkeypatch, out=out, error=error)
    assert drums.default_jobs() == 12


def test_other_platforms_use_every_core(monkeypatch):
    _sysctl(monkeypatch, out="6\n")
    monkeypatch.setattr(drums.sys, "platform", "linux")
    assert drums.default_jobs() == 12
