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
SUBCOMMANDS = ("split", "load", "open", "sections", "scenes")


def _env():
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = env.get("VIRTUAL_ENV") or "1"
    return env


def run(*argv):
    return subprocess.run(list(argv), capture_output=True, text=True,
                          timeout=120, env=_env(), check=False)


@pytest.mark.parametrize("args", [(), ("-h",), ("--help",)])
def test_usage_lists_the_five_subcommands(args):
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
