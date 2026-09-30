#!/usr/bin/env bash
# Install Roadie: check dependencies, build .venv, symlink the commands onto PATH.
#
#   ./install.sh [--bin DIR] [--check]
#
#   --bin DIR   where to put the symlinks. Default ~/.local/bin
#   --check     report only; create nothing
#
# .venv is built with `uv venv --python 3.13` from requirements.txt when it does
# not exist yet, and repaired in place when it cannot import them (a half-built
# venv from an interrupted install). --check reports and changes nothing.
#
# Symlinks rather than copies, so editing a script takes effect immediately and
# the repo stays the single source of truth. They also strip the .sh/.py
# extensions, so you type `stems2als`, not `stems2als.sh`.
#
# This installs the CLI side only. The Ableton bridge (needed for roadie
# load/open/scenes) is Sideman, installed on its own — see README.md, "Ableton
# side: Sideman" (https://github.com/slegroux/sideman). --check probes it.

set -euo pipefail
trap 'rc=$?; echo "install: FAILED (exit $rc) at line $LINENO: $BASH_COMMAND" >&2' ERR

usage () { awk 'NR>1 && /^#/{print; next} NR>1{exit}' "$0"; }

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HOME/.local/bin"
CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { echo "--bin needs a directory" >&2; usage >&2; exit 2; }
      BIN="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# command -> what breaks without it
declare -a REQ=(
  "ffmpeg|everything — conversion and all measurement"
  "yt-dlp|fetching from a URL (--file mode works without it)"
  "demucs|stem separation (-s)"
  "audio-separator|drum split (-d)"
  "aubiotrack|BPM detection (filenames just go untagged without it)"
  "deno|yt-dlp's JS runtime; without it YouTube extraction is deprecated and some formats vanish"
)
# how to get each one, where it isn't obvious
declare -a FIX=(
  "ffmpeg|brew install ffmpeg"
  "yt-dlp|uv tool install yt-dlp"
  "demucs|uv tool install demucs --with numpy   # metadata omits numpy; a plain install dies on import"
  "audio-separator|uv tool install \"audio-separator[cpu]\"   # base package lacks onnxruntime and exits 0 on failure"
  "aubiotrack|brew install aubio"
  "deno|brew install deno"
)

echo "== dependencies"
missing=0
for entry in "${REQ[@]}"; do
  cmd="${entry%%|*}"; why="${entry#*|}"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf "  \033[32mok\033[0m    %-17s\n" "$cmd"
  else
    printf "  \033[31mMISS\033[0m  %-17s needed for: %s\n" "$cmd" "$why"
    for f in "${FIX[@]}"; do
      [[ "${f%%|*}" == "$cmd" ]] && printf "        %s\n" "${f#*|}"
    done
    missing=$((missing + 1))
  fi
done

# Python packages inside .venv, which the command check above cannot see. Only
# madmom is reported: it is beat_aligner's OPTIONAL trained detector. The
# default is the heuristic, which needs nothing extra; asking for madmom
# without it installed degrades silently to the heuristic, so say whether it
# is there.
echo
echo "== python environment (.venv)"
VENV_PY="$HERE/.venv/bin/python"
# Probe the packages, not just the interpreter: a `uv pip install` that died
# half way leaves a python with nothing importable, and checking only for the
# binary would call that installed forever.
venv_ok () { [[ -x "$VENV_PY" ]] && PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" -c "import librosa, numpy, scipy, soundfile" >/dev/null 2>&1; }
if venv_ok; then
  printf "  \033[32mok\033[0m    .venv has the requirements (%s)\n" "$("$VENV_PY" -V 2>&1)"
elif [[ $CHECK -eq 1 ]]; then
  if [[ -x "$VENV_PY" ]]; then
    printf "  BAD   .venv is missing packages (would install requirements.txt)\n"
  else
    printf "  none  .venv (would create with uv from requirements.txt)\n"
  fi
else
  command -v uv >/dev/null 2>&1 || {
    echo "  uv is required to build .venv: https://docs.astral.sh/uv/ (brew install uv)" >&2
    exit 1
  }
  [[ -x "$VENV_PY" ]] || uv venv --python 3.13 "$HERE/.venv"
  uv pip install --python "$VENV_PY" -r "$HERE/requirements.txt"
  venv_ok || { echo "  .venv still cannot import the requirements — see the uv output above" >&2; exit 1; }
  printf "  \033[32mok\033[0m    .venv built from requirements.txt\n"
fi

echo
echo "== beat_aligner backends (.venv)"
if [[ ! -x "$VENV_PY" ]]; then
  printf "  \033[33m--\033[0m    no .venv — beat_aligner will use the system python\n"
elif PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" -c "import madmom" >/dev/null 2>&1; then
  printf "  \033[32mok\033[0m    madmom            optional trained detector (--detector madmom)\n"
else
  printf "  \033[33m--\033[0m    madmom            not installed (optional; the default heuristic\n"
  printf "                          needs nothing). --detector madmom would fall back to it.\n"
  printf "        The PyPI release is from 2018 and needs Python < 3.10; git main does not:\n"
  printf "          uv pip install \"setuptools<81\" cython\n"
  printf "          uv pip install --no-build-isolation \\\\\n"
  printf "            \"madmom @ git+https://github.com/CPJKU/madmom\"\n"
fi

echo
echo "== commands"
# script -> command name
# `roadie` is the front door; the per-script names stay linked so existing
# habits and scripts keep working.
declare -a LINKS=(
  "roadie|roadie"
  "yt2stems.sh|yt2stems"
  "stems2live.py|stems2live"
  "stems2als.sh|stems2als"
  "beat_aligner.py|beat_aligner"
)

for entry in "${LINKS[@]}"; do
  src="$HERE/${entry%%|*}"; name="${entry#*|}"; dst="$BIN/$name"
  if [[ ! -s "$src" ]]; then
    printf "  \033[31mMISS\033[0m  %-12s (no %s)\n" "$name" "${entry%%|*}"
    missing=$((missing + 1)); continue
  fi
  if [[ $CHECK -eq 1 ]]; then
    cur=$(readlink "$dst" 2>/dev/null || true)
    if [[ "$cur" == "$src" ]]; then printf "  ok    %-12s -> %s\n" "$name" "$src"
    elif [[ -e "$dst" ]];    then printf "  DIFF  %-12s -> %s\n" "$name" "${cur:-<real file>}"
    else                          printf "  none  %-12s (would link)\n" "$name"; fi
    continue
  fi
  mkdir -p "$BIN"
  chmod +x "$src"
  ln -sfn "$src" "$dst"
  printf "  linked %-12s -> %s\n" "$name" "$src"
done

# A symlink in a directory that isn't on PATH is invisible; say so rather than
# leaving the user to wonder why the command isn't found.
echo
case ":$PATH:" in
  *":$BIN:"*) echo "== $BIN is on PATH" ;;
  *) echo "== WARNING: $BIN is NOT on PATH."
     echo "   Add to ~/.zshrc:  export PATH=\"$BIN:\$PATH\"" ;;
esac

if [[ $missing -gt 0 ]]; then
  echo
  echo "== $missing item(s) missing — the commands will still install, but the"
  echo "   stages needing those tools will fail with a clear message at runtime."
fi

echo
echo "== Ableton bridge: Sideman (only needed for roadie load / open / scenes)"
if lsof -nP -iTCP:9878 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  ok    Live is running and the Sideman socket is up on :9878"
  # An open port proves nothing: the socket answers normally whatever version
  # of the handlers Live has loaded. So ask for warp_markers_set, which Sideman
  # ships and older AbletonLOM handlers lack. Empty params fail INSIDE the
  # handler (KeyError 'path') when the op exists and at dispatch ("unknown op")
  # when it does not, so this distinguishes them without touching the session.
  if command -v python3 >/dev/null 2>&1; then
    probe=$(python3 - <<'PROBE' 2>/dev/null || true
import json, socket
try:
    s = socket.create_connection(("localhost", 9878), timeout=10)
    s.sendall((json.dumps({"id": 1, "op": "warp_markers_set",
                           "params": {}}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        c = s.recv(65536)
        if not c:
            break
        buf += c
    r = json.loads(buf.decode().split("\n")[0])
    print((r.get("error") or {}).get("message", "ok"))
except Exception:
    pass
PROBE
)
    case "$probe" in
      *"unknown op"*)
        echo "  --    but these handlers predate warp_markers_set: warp maps will fail"
        echo "        on a drifting set. Update Sideman (https://github.com/slegroux/sideman)"
        echo "        and send {\"op\": \"reload\"} to :9878 — the handlers hot-reload,"
        echo "        so this needs NO Live restart." ;;
      "")
        echo "  --    no reply to the capability probe; cannot tell which version of"
        echo "        the handlers Live has loaded." ;;
      *)
        echo "  ok    warp_markers_set answers — Sideman's warp support is loaded" ;;
    esac
  fi
else
  echo "  --    not running. Install Sideman (https://github.com/slegroux/sideman),"
  echo "        then in Live set Preferences -> Link, Tempo & MIDI -> Control Surface"
  echo "        = AbletonLOM (Sideman's remote script). See README.md \"Ableton side\"."
  if lsof -nP -iTCP:9877 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  --    (AbletonMCP IS up on :9877, but roadie no longer uses it."
    echo "        The two coexist by design; enable Sideman as well.)"
  fi
fi
