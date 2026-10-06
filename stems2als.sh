#!/usr/bin/env bash
# Make a new Ableton project from a folder of stems, in one command.
#
#   stems2als.sh <stems_dir> [-n NAME] [-o DIR] [--tempo BPM] [--detector D] [--template PATH]
#                [--analyze] [--locators] [--energy-map] [--session] [--keep-locators]
#                [--start-bar N] [--dry-run]
#
#   -n NAME       project name. Default: the stems folder's name minus "_stems".
#   -o DIR        where the .als goes. Default: the stems folder's parent, so
#                 the project sits beside the stems rather than inside them.
#   --tempo BPM   session tempo. Default: MEASURED from the audio by stems2live.
#                 The _126bpm in a filename is a rounded label and is never read
#                 back as a tempo — it is 0.17-0.44 BPM off in practice, enough
#                 to walk the grid ~0.75s off the audio across a 9-minute set.
#   --detector D  downbeat backend: heuristic (default, no extra deps) or
#                 madmom (optional install, see README).
#                 Watch the "phase confidence" line stems2live prints — at
#                 chance level the bar phase is a guess and the set can land a
#                 half-bar out. That is when a trained detector earns its keep.
#   --template P  .als to copy. Default: ~/Music/Ableton/User Library/Templates/empty.als
#
#   --analyze     the whole reading in one flag: --locators --energy-map --session.
#                 The labels (Intro/Build/Drop/Break/Outro, Main/Quiet/Mid when no
#                 rule fits) are RULES over loudness, position and contrast, not
#                 recognition. The boundaries are the trustworthy part.
#   --locators    a named locator per detected section, snapped to 8-bar phrases.
#   --energy-map  a silent MIDI track whose piano roll maps who plays where.
#   --session     a Session View grid, one scene per section, on the same beat grid.
#   --keep-locators  with --locators, add to existing locators instead of clearing.
#   --start-bar N where the audio starts in the arrangement. Default: stems2live's (9).
#   These five are forwarded to stems2live unchanged, and a flag given both
#   directly and through --analyze is passed once.
#   --dry-run     print the stems2live command and stop: no template copy, no .als,
#                 no Live, no socket wait. Exists so the forwarding can be checked
#                 without Live running. Values are validated first and an existing
#                 project of the same name is reported (exit 1), as a real run would.
#                 A value flag followed by anything starting with "-" is an error
#                 (exit 2), never a value, so --dry-run can't be swallowed into a
#                 real run. The cost: a name or folder can't start with "-".
#
# Live is found as the newest /Applications/Ableton Live*.app; set ABLETON_APP
# to an app name or path to pick another.
#
# WHY COPY A TEMPLATE rather than have Live make a new set: the copy already has
# a file path, so your ⌘S saves silently in place. An untitled set throws a
# Save-As dialog every time.
#
# THIS SCRIPT CANNOT SAVE FOR YOU. Live's Object Model exposes no save command —
# not to this bridge, not to any of them (there is no save symbol anywhere in
# Live's own bundled remote scripts). Automating ⌘S needs Accessibility
# permission granted to your terminal, which only you can do in
# System Settings -> Privacy & Security -> Accessibility. Until then the last
# step is one keystroke by hand.
#
# Needs Live installed with the Sideman (formerly AbletonLOM) control surface
# active on port 9878 (Preferences -> Link, Tempo & MIDI -> Control Surface).

set -euo pipefail
trap 'rc=$?; echo "stems2als: FAILED (exit $rc) at line $LINENO: $BASH_COMMAND" >&2' ERR

usage () { awk 'NR>1 && /^#/{print; next} NR>1{exit}' "$0"; }

# The repo is wherever this script really lives. It is installed as a symlink
# in ~/.local/bin, so follow the link chain first — macOS has no `readlink -f`
# before 12.3, hence the loop rather than one call.
SELF="${BASH_SOURCE[0]}"
while [[ -L "$SELF" ]]; do
  LINKDIR="$(cd -P "$(dirname "$SELF")" && pwd)"
  SELF="$(readlink "$SELF")"
  [[ "$SELF" == /* ]] || SELF="$LINKDIR/$SELF"
done
REPO="$(cd -P "$(dirname "$SELF")" && pwd)"

PORT=9878
TEMPLATE="$HOME/Music/Ableton/User Library/Templates/empty.als"
PY="python3"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
fi
LOADER="$REPO/stems2live.py"

STEMS="" NAME="" OUTDIR="" TEMPO="" DETECTOR="" START_BAR=""
LOCATORS="" ENERGY_MAP="" SESSION="" KEEP_LOCATORS="" DRY_RUN=""

# A value flag must not swallow the next flag: `--start-bar --dry-run` used to
# take "--dry-run" as the bar number, turn the dry run OFF, and go on to open a
# real set in whatever Live was running. So a value flag refuses a missing value
# or one that looks like a flag, and --dry-run is also found by a pre-scan, so
# no parsing slip can turn "touch nothing" into a real run.
need_val () {
  [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { echo "$1 needs a value" >&2; exit 2; }
}
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; fi
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) need_val "$@"; NAME="$2"; shift 2 ;;
    -o) need_val "$@"; OUTDIR="$2"; shift 2 ;;
    --tempo) need_val "$@"; TEMPO="$2"; shift 2 ;;
    --detector) need_val "$@"; DETECTOR="$2"; shift 2 ;;
    --template) need_val "$@"; TEMPLATE="$2"; shift 2 ;;
    --analyze) LOCATORS=1; ENERGY_MAP=1; SESSION=1; shift ;;
    --locators) LOCATORS=1; shift ;;
    --energy-map) ENERGY_MAP=1; shift ;;
    --session) SESSION=1; shift ;;
    --keep-locators) KEEP_LOCATORS=1; shift ;;
    --start-bar) need_val "$@"; START_BAR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) STEMS="$1"; shift ;;
  esac
done

[[ -n "$STEMS" ]] || { usage; exit 2; }
[[ -d "$STEMS" ]] || { echo "no such stems dir: $STEMS" >&2; exit 1; }

STEMS=$(cd "$STEMS" && pwd)

# Checked before anything is touched. stems2live rejects these too, but only
# after the .als is copied and Live has opened it, which leaves a stray project
# behind and a "project already exists" on the retry.
if [[ -n "$START_BAR" && ! "$START_BAR" =~ ^[1-9][0-9]*$ ]]; then
  echo "--start-bar must be a bar number >= 1, got: $START_BAR" >&2; exit 2
fi
case "$DETECTOR" in
  ""|heuristic|madmom) ;;
  *) echo "--detector must be heuristic or madmom, got: $DETECTOR" >&2; exit 2 ;;
esac
if [[ -n "$TEMPO" ]] && ! [[ "$TEMPO" =~ ^[0-9]+(\.[0-9]+)?$ && "$TEMPO" =~ [1-9] ]]; then
  echo "--tempo must be a number > 0, got: $TEMPO" >&2; exit 2
fi

# Built here, not at the end, so --dry-run can print exactly what a real run
# would hand to stems2live. Flags are booleans above, so --analyze plus an
# explicit --locators still yields one --locators.
LOADER_ARGS=("$STEMS")
[[ -n "$TEMPO" ]] && LOADER_ARGS+=(--tempo "$TEMPO")
[[ -n "$DETECTOR" ]] && LOADER_ARGS+=(--detector "$DETECTOR")
[[ -n "$LOCATORS" ]] && LOADER_ARGS+=(--locators)
[[ -n "$ENERGY_MAP" ]] && LOADER_ARGS+=(--energy-map)
[[ -n "$SESSION" ]] && LOADER_ARGS+=(--session)
[[ -n "$KEEP_LOCATORS" ]] && LOADER_ARGS+=(--keep-locators)
[[ -n "$START_BAR" ]] && LOADER_ARGS+=(--start-bar "$START_BAR")

BASE=$(basename "$STEMS"); BASE="${BASE%_stems}"
[[ -n "$NAME" ]] || NAME="$BASE"

# Carry the BPM into the project TITLE. This reads the rounded tag out of the
# stems folder name, which is fine HERE because it is only ever a label.
# It must NOT be used as the session tempo — that is measured from the audio by
# stems2live, and the rounded value is 0.17-0.44 BPM off in practice. Keep those
# two uses separate; conflating them is the bug this script shipped with.
if [[ "$NAME" != *bpm && "$BASE" =~ _([0-9]{2,3})bpm ]]; then
  NAME="${NAME}_${BASH_REMATCH[1]}bpm"
fi

# Default beside the stems folder, NOT $PWD. Running `stems2als .` from inside
# the stems dir would otherwise drop the .als among the audio files, where Live
# then treats the folder as the project root.
OUT="${OUTDIR:-$(dirname "$STEMS")}"
[[ ! -d "$OUT" ]] || OUT=$(cd "$OUT" && pwd)   # no mkdir yet: a dry run creates nothing
ALS="$OUT/$NAME.als"

if [[ -e "$ALS" ]]; then
  echo "a project already exists: $ALS" >&2
  echo "  open it:      open \"$ALS\"" >&2
  echo "  or rename:    stems2als \"$STEMS\" -n <other-name>" >&2
  echo "  or replace:   rm \"$ALS\" && stems2als \"$STEMS\"" >&2
  exit 1
fi

# Before the template check on purpose: nothing below here is needed to show the
# command, and none of it should run (template copy, Live, socket). The project-
# exists check above needs no template, so a dry run reports it too.
if [[ -n "$DRY_RUN" ]]; then
  printf '%q' "$PY"; printf ' %q' "$LOADER" "${LOADER_ARGS[@]}"; echo
  exit 0
fi

[[ -s "$TEMPLATE" ]] || { echo "no template at: $TEMPLATE" >&2; exit 1; }
command -v "$PY" >/dev/null || { echo "missing: $PY" >&2; exit 1; }
[[ -s "$LOADER" ]] || { echo "missing loader: $LOADER" >&2; exit 1; }

# Whichever Live is installed — Suite, Standard, Intro, Lite, any version.
# Compared by the NUMBER after "Ableton Live", not by name: a glob sorts as
# text, and "Ableton Live 9" sorts after "Ableton Live 12".
newest_live () {
  local a v best="" bestv=-1
  for a in "$1"/Ableton\ Live*.app; do
    [[ -d "$a" ]] || continue
    v=$(basename "$a" | sed -nE 's/^Ableton Live ([0-9]+).*/\1/p')
    [[ -n "$v" ]] || v=0
    if (( v > bestv )); then best="$a"; bestv=$v; fi
  done
  printf '%s' "$best"
}
APP="${ABLETON_APP:-}"
[[ -n "$APP" ]] || APP="$(newest_live /Applications)"
[[ -n "$APP" ]] || { echo "no Ableton Live found in /Applications; set ABLETON_APP" >&2; exit 1; }

# NOT read from the folder name, deliberately. yt2stems tags filenames with a
# ROUNDED integer BPM (_126bpm) because that reads better; the real tempo of the
# set it was built against is 125.829. Feeding the rounded value to Live is
# 0.65 ms/beat out, which accumulates to ~0.75s — about 1.6 beats — over nine
# minutes, and the grid slowly walks off the audio. stems2live measures the
# tempo from the kick stem itself, so we stay quiet unless the caller insists.
if [[ -n "$TEMPO" ]]; then
  echo "==> tempo $TEMPO (explicit; overrides the measured value)"
fi

# Fixed above, before the dry-run exit; only the directory is made here.
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); ALS="$OUT/$NAME.als"

cp "$TEMPLATE" "$ALS"
echo "==> $ALS  (from $(basename "$TEMPLATE"))"

# Open the copy. If Live is already running with another set open it will just
# open this one alongside — the bridge talks to whichever set has focus, so a
# running instance is a real hazard here.
if pgrep -f "$APP" >/dev/null 2>&1; then
  echo "!!! Live is already running. This will open a second set, and the bridge"
  echo "!!! talks to the focused one — stems may land in the wrong project."
  echo "!!! Quit Live first if that's not what you want.  (5s to Ctrl-C)"
  sleep 5
fi

open -a "$APP" "$ALS"
printf "==> waiting for the Sideman socket on :%s" "$PORT"
for _ in $(seq 1 60); do
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then ok=1; break; fi
  printf "."; sleep 2
done
echo
[[ "${ok:-}" == "1" ]] || {
  echo "socket never came up. Check Preferences -> Link, Tempo & MIDI ->" >&2
  echo "Control Surface = the Sideman remote script, AbletonLOM (Input/Output = None)." >&2
  exit 1
}

# Live binds the socket slightly before the set is fully loaded; giving it a
# moment avoids tracks landing in the outgoing set.
sleep 3

"$PY" "$LOADER" "${LOADER_ARGS[@]}"

echo
echo "==> $ALS is populated but NOT saved."
echo "==> Tempo and bar alignment were measured from the audio, not the filename."
echo "==> Press Cmd-S in Live now. It will save in place, with no dialog,"
echo "    because the file already exists on disk."
