"""The `.venv` re-exec preamble at the top of five scripts hops at most once.

als_writer, beat_aligner, sections, session_view and stems2live each begin by
re-executing under `<script dir>/.venv/bin/python`. Two ways that went wrong:

  * a `.venv` whose python never looks like the venv to the check (a symlinked
    interpreter) made every exec land back in the guard's condition and loop
    forever, at 100% CPU with no output;
  * with ANOTHER venv active, an `VIRTUAL_ENV` escape hatch skipped the hop, so
    `roadie load` ran in the wrong environment and failed on `import sideman`.

The preamble is lifted out of each real script and run in a scratch directory,
so what is tested is the text that ships.
"""

import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["als_writer.py", "beat_aligner.py", "sections.py", "session_view.py",
           "stems2live.py"]
SHOW = 'print("RAN", os.path.realpath(sys.prefix), os.environ.get("ROADIE_REEXEC"))\n'


def _preamble(name):
    with open(os.path.join(HERE, name)) as f:
        src = f.read()
    m = re.search(r"^_real_script = .*?^    os\.execv\(.*?\n", src, re.S | re.M)
    assert m, "%s: preamble not found" % name
    return m.group(0)


def _script(tmp, name):
    path = os.path.join(str(tmp), name)
    with open(path, "w") as f:
        f.write("import os\nimport sys\n" + _preamble(name) + SHOW)
    return path


def _run(python, script, **env):
    base = {k: v for k, v in os.environ.items()
            if k not in ("ROADIE_REEXEC", "VIRTUAL_ENV")}
    base.update(env)
    r = subprocess.run([python, script], capture_output=True, text=True,
                       timeout=60, check=False, env=base)
    assert r.returncode == 0, r.stderr[-800:]
    return [ln.split(" ", 2)[1:] for ln in r.stdout.splitlines() if ln.startswith("RAN")]


@pytest.fixture(scope="module")
def real_venv(tmp_path_factory):
    d = tmp_path_factory.mktemp("realvenv") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d)],
                   check=True, capture_output=True)
    return str(d)


@pytest.mark.parametrize("name", SCRIPTS)
def test_an_interpreter_that_never_looks_like_the_venv_hops_once(name, tmp_path):
    """The loop: `.venv/bin/python` is a bare symlink to the base python, so
    its prefix is never the venv. Two hops would be a loop; one is the limit."""
    os.makedirs(tmp_path / ".venv" / "bin")
    os.symlink(os.path.realpath(sys.executable), tmp_path / ".venv" / "bin" / "python")
    ran = _run(sys.executable, _script(tmp_path, name))
    assert len(ran) == 1
    assert ran[0][1] == "1"          # it did hop, and the guard held


@pytest.mark.parametrize("name", SCRIPTS)
def test_another_active_venv_still_hops_into_the_repos_venv(name, tmp_path, real_venv):
    os.symlink(real_venv, tmp_path / ".venv")     # a symlinked .venv, as well
    ran = _run(sys.executable, _script(tmp_path, name),
               VIRTUAL_ENV=os.path.dirname(os.path.dirname(sys.executable)))
    assert len(ran) == 1
    assert ran[0][0] == os.path.realpath(real_venv)
    assert ran[0][1] == "None"       # arrived: later scripts may hop again


@pytest.mark.parametrize("name", SCRIPTS)
def test_already_in_the_venv_does_not_hop(name, tmp_path, real_venv):
    os.symlink(real_venv, tmp_path / ".venv")
    ran = _run(os.path.join(real_venv, "bin", "python"), _script(tmp_path, name))
    assert ran == [[os.path.realpath(real_venv), "None"]]
