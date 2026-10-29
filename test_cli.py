#!/usr/bin/env python3
"""Tests for the `roadie` dispatcher — subprocess only, nothing sent to Live.

The .py scripts are run through their `#!/usr/bin/env python3` shebang, exactly
as the dispatcher runs them, so PATH is pointed at the interpreter running the
suite and VIRTUAL_ENV is set to keep their .venv re-exec guard out of the way.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROADIE = os.path.join(HERE, "roadie")
SUBCOMMANDS = ("split", "load", "open", "sections", "scenes", "als", "stats")


def _env():
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = env.get("VIRTUAL_ENV") or "1"
    return env


def run(*argv):
    return subprocess.run(list(argv), capture_output=True, text=True,
                          timeout=120, env=_env(), check=False)


@pytest.fixture(autouse=True)
def _no_real_log(tmp_path, monkeypatch):
    """Keep every test run out of the user's real usage log."""
    monkeypatch.setenv("ROADIE_LOG", str(tmp_path / "usage.log"))
    monkeypatch.setenv("ROADIE_CONFIG", str(tmp_path / "no-config"))


@pytest.mark.parametrize("args", [(), ("-h",), ("--help",)])
def test_usage_lists_every_subcommand(args):
    r = run(ROADIE, *args)
    assert r.returncode == 0, r.stderr
    for sub in SUBCOMMANDS:
        assert sum(line.lstrip().startswith("roadie %s " % sub)
                   for line in r.stdout.splitlines()) == 1, (sub, r.stdout)


def test_unknown_subcommand_exits_2_with_usage():
    r = run(ROADIE, "bogus")
    assert r.returncode == 2
    assert "unknown command" in r.stderr and "roadie load" in r.stderr


@pytest.mark.parametrize("sub, script", [("load", "stems2live.py"),
                                         ("sections", "sections.py")])
def test_subcommand_help_is_the_scripts_own_help(sub, script):
    via = run(ROADIE, sub, "--help")
    direct = run(os.path.join(HERE, script), "--help")
    assert direct.returncode == 0, direct.stderr
    assert via.returncode == 0, via.stderr
    assert via.stdout == direct.stdout


def test_split_help_succeeds():
    r = run(ROADIE, "split", "--help")
    assert r.returncode == 0, r.stderr
    assert "yt2stems" in r.stdout


def test_works_through_a_symlink_elsewhere(tmp_path):
    """install.sh links `roadie` into ~/.local/bin; it must still find the repo."""
    (tmp_path / "bin").mkdir()
    link = tmp_path / "bin" / "roadie"
    # Relative link through a second hop, the case a naive dirname gets wrong.
    os.symlink(ROADIE, str(tmp_path / "hop"))
    os.symlink(os.path.join("..", "hop"), str(link))
    via = run(str(link), "load", "--help")
    direct = run(os.path.join(HERE, "stems2live.py"), "--help")
    assert via.returncode == 0, via.stderr
    assert via.stdout == direct.stdout


def test_stems2als_picks_the_highest_live_version_not_the_last_by_name(tmp_path):
    """A text sort puts "Ableton Live 9" after "Ableton Live 12"."""
    for name in ("Ableton Live 9 Suite.app", "Ableton Live 12 Suite.app",
                 "Ableton Live 11 Lite.app"):
        (tmp_path / name).mkdir()
    script = os.path.join(HERE, "stems2als.sh")
    snippet = ("set -euo pipefail; source <(sed -n '/^newest_live ()/,/^}/p' %s); "
               "newest_live \"$1\"" % script)
    r = run("bash", "-c", snippet, "_", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout == str(tmp_path / "Ableton Live 12 Suite.app")
    r = run("bash", "-c", snippet, "_", str(tmp_path / "nothing"))
    assert r.returncode == 0 and r.stdout == ""


def _fake_tools(tmp_path):
    """ffmpeg and yt-dlp stand-ins: yt-dlp logs its argv and fails, so a URL
    run stops at the metadata call without touching the network."""
    bin_ = tmp_path / "fakebin"
    bin_.mkdir()
    log = tmp_path / "ytdlp.log"
    (bin_ / "ffmpeg").write_text("#!/bin/sh\nexit 0\n")
    (bin_ / "yt-dlp").write_text('#!/bin/sh\necho "$@" >> "%s"\nexit 1\n' % log)
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = _env()
    env["PATH"] = str(bin_) + os.pathsep + env["PATH"]
    return env, log


@pytest.mark.parametrize("src, want", [
    ("youtu.be/abc", "https://youtu.be/abc"),
    ("www.youtube.com/watch?v=abc", "https://www.youtube.com/watch?v=abc"),
    ("https://youtu.be/abc", "https://youtu.be/abc"),
])
def test_split_treats_a_url_shaped_argument_as_a_url(tmp_path, src, want):
    env, log = _fake_tools(tmp_path)
    subprocess.run([ROADIE, "split", src, "-o", str(tmp_path / "out")],
                   capture_output=True, text=True, timeout=60, env=env,
                   cwd=str(tmp_path), check=False)
    assert log.exists(), "yt-dlp was never asked: the URL became --file"
    assert want in log.read_text().split()


@pytest.mark.parametrize("argv", [["--", "missing.wav"], ["missing.wav"],
                                  ["-o", "out", "--", "missing.wav"]])
def test_split_treats_a_plain_name_as_a_file_and_accepts_double_dash(tmp_path, argv):
    env, log = _fake_tools(tmp_path)
    r = subprocess.run([ROADIE, "split"] + argv, capture_output=True, text=True,
                       timeout=60, env=env, cwd=str(tmp_path), check=False)
    assert "no such file: missing.wav" in r.stderr, r.stderr
    assert "unknown flag" not in r.stderr
    assert not log.exists()


def test_split_passes_an_existing_file_as_file(tmp_path):
    env, log = _fake_tools(tmp_path)
    (tmp_path / "youtu.be").mkdir()                 # a real path wins over URL shape
    (tmp_path / "youtu.be" / "x.wav").write_bytes(b"not audio")
    r = subprocess.run([ROADIE, "split", "--", "youtu.be/x.wav", "-o", "out"],
                       capture_output=True, text=True, timeout=60, env=env,
                       cwd=str(tmp_path), check=False)
    assert "local source: youtu.be/x.wav" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not log.exists()


# --- usage log ---------------------------------------------------------------

def _logged(tmp_path, *argv, **env_over):
    env = _env()
    env.update(env_over)
    return subprocess.run([ROADIE, *argv], capture_output=True, text=True,
                          timeout=120, env=env, cwd=str(tmp_path), check=False)


def _rows(log):
    return [line.split("\t") for line in log.read_text().splitlines()]


def test_a_run_on_a_stems_dir_is_logged_with_its_exit_status(tmp_path):
    stems = tmp_path / "x_120bpm_stems"
    stems.mkdir()
    log = tmp_path / "usage.log"
    r = _logged(tmp_path, "sections", "x_120bpm_stems")      # relative path, no stems inside
    rows = _rows(log)
    assert len(rows) == 1, (r.stdout, r.stderr)
    ts, cmd, rc, secs, target = rows[0]
    assert cmd == "sections" and target == str(stems.resolve())
    assert rc == str(r.returncode) and secs.isdigit() and ts.endswith("Z")


def test_the_log_does_not_change_the_exit_status(tmp_path):
    stems = tmp_path / "s"
    stems.mkdir()
    on = _logged(tmp_path, "sections", "s")
    off = _logged(tmp_path, "sections", "s", ROADIE_LOG="")
    assert on.returncode == off.returncode
    assert on.stdout == off.stdout


def test_help_and_unknown_commands_are_not_logged(tmp_path):
    log = tmp_path / "usage.log"
    _logged(tmp_path, "load", "--help")
    _logged(tmp_path, "bogus")
    assert not log.exists()


def test_a_missing_dir_is_not_logged_as_a_track(tmp_path):
    _logged(tmp_path, "load", "no-such-dir")
    assert not (tmp_path / "usage.log").exists()


def test_empty_roadie_log_turns_logging_off(tmp_path):
    (tmp_path / "s").mkdir()
    _logged(tmp_path, "sections", "s", ROADIE_LOG="")
    assert not (tmp_path / "usage.log").exists()


def test_an_unwritable_log_never_fails_the_command(tmp_path):
    (tmp_path / "s").mkdir()
    blocked = tmp_path / "file"
    blocked.write_text("")
    a = _logged(tmp_path, "sections", "s", ROADIE_LOG=str(blocked / "sub" / "usage.log"))
    b = _logged(tmp_path, "sections", "s", ROADIE_LOG="")
    assert a.returncode == b.returncode and a.stdout == b.stdout


def test_split_logs_the_source_as_an_absolute_path_or_a_url(tmp_path):
    env, _ = _fake_tools(tmp_path)
    env["ROADIE_LOG"] = str(tmp_path / "usage.log")
    (tmp_path / "set.wav").write_bytes(b"not audio")
    for src in ("set.wav", "youtu.be/abc"):
        subprocess.run([ROADIE, "split", src, "-o", "out"], capture_output=True,
                       text=True, timeout=60, env=env, cwd=str(tmp_path), check=False)
    targets = [row[4] for row in _rows(tmp_path / "usage.log")]
    assert targets == [str((tmp_path / "set.wav").resolve()), "https://youtu.be/abc"]


def test_stats_counts_runs_and_distinct_targets(tmp_path):
    log = tmp_path / "usage.log"
    log.write_text("2026-01-01T00:00:00Z\tload\t0\t5\t/a\n"
                   "2026-01-02T00:00:00Z\tload\t0\t5\t/a\n"
                   "2026-01-03T00:00:00Z\tload\t1\t5\t/b\n"
                   "2026-01-04T00:00:00Z\tsplit\t0\t90\t/c.wav\n")
    out = _logged(tmp_path, "stats").stdout
    assert "4 runs, 2026-01-01T00:00:00Z to 2026-01-04T00:00:00Z" in out
    assert "load" in out and "3 runs, 1 failed — 1 distinct stems dirs" in out
    assert "1 runs, 0 failed — 1 distinct sources" in out


def test_stats_with_no_log_says_so(tmp_path):
    assert "no usage logged yet" in _logged(tmp_path, "stats").stdout
    assert "off" in _logged(tmp_path, "stats", ROADIE_LOG="").stdout


# --- config file: default output dir ----------------------------------------

def _split_out(tmp_path, *argv, config=None, home=None):
    """Run `roadie split` against the fake yt-dlp and return what it printed."""
    env, _ = _fake_tools(tmp_path)
    env["ROADIE_LOG"] = ""
    env["ROADIE_CONFIG"] = str(tmp_path / "config")
    if home:
        env["HOME"] = home
    if config is not None:
        (tmp_path / "config").write_text(config)
    r = subprocess.run([ROADIE, "split", "youtu.be/abc", *argv], capture_output=True,
                       text=True, timeout=60, env=env, cwd=str(tmp_path), check=False)
    return r.stdout + r.stderr


def test_split_uses_out_from_the_config_when_o_is_not_given(tmp_path):
    out = tmp_path / "studies"
    _split_out(tmp_path, config="# where stems go\nout = %s\n" % out)
    assert out.is_dir()


def test_an_explicit_o_beats_the_config(tmp_path):
    cfg, flag = tmp_path / "from-config", tmp_path / "from-flag"
    _split_out(tmp_path, "-o", str(flag), config="out=%s\n" % cfg)
    assert flag.is_dir() and not cfg.exists()


def test_config_tilde_expands_to_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    _split_out(tmp_path, config="out = ~/Studies\n", home=str(home))
    assert (home / "Studies").is_dir()
