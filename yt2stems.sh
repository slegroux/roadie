#!/usr/bin/env bash
# Pull one of your own sets/mixes off YouTube, convert to Ableton-ready WAV,
# optionally split into stems. Code lives in this repo; audio never does.
#
#   yt2stems.sh <url>          [-o DIR] [-a] [-s] [-D] [-m MODEL] [-f]
#   yt2stems.sh --file <path>  [-o DIR] [-a] [-s] [-D] [-m MODEL] [-f]
#
#   --file P   process a local audio file instead of fetching a URL. Everything
#              downstream is identical: merge, BPM tag, drum split, flat layout.
#              Prefer this for anything you own losslessly — separation artifacts
#              compound on top of codec artifacts, so a purchased WAV/FLAC gives
#              cleaner stems than a 133kbps YouTube transcode ever will.
#   -o DIR     write here (created if missing). Default: current directory.
#   -a         studies layout: one folder per track, grouped by artist —
#              DIR/<Artist>/<Title>/ holds the WAV, the _stems folder and (via
#              `roadie als`/`open`, which write beside the stems) the .als.
#              Artist and title split at the first " - ". An existing folder
#              named for the artist, or for their last name, is reused: "Marc
#              Holstege - Mangata" lands in an existing Holstege/Mangata/. A name
#              with no " - " gets DIR/<Title>/.
#   -s         also split into stems (BS-Roformer-SW through audio-separator),
#              then split the drums stem into kick/snare/toms/hh/ride/crash
#              (MDX23C-DrumSep, also audio-separator). The drum pass is on by
#              default and SLOW — see below.
#   -D         --no-drums: stop at the four stems, skip the drum pass.
#   -d         the drum pass, implying -s. Kept for scripts that pass it.
#   -m MODEL   stem model. Default bs-roformer-sw — see the note below.
#   -f         force: redo every stage even if its output is already there.
#
#   Models: bs-roformer-sw (default, 6 stems, audio-separator)
#           | htdemucs_6s | htdemucs_ft | mdx_extra | htdemucs (demucs)
#
#   ROADIE_MODEL_DIR   where audio-separator keeps its models. Default
#              ~/.cache/audio-separator-models (its own default is /tmp, which
#              a reboot empties and which would re-download 2.9 GB). A model
#              not in it is downloaded first: BS-Roformer-SW is ~700 MB.
#
# Re-running is cheap: each of the three stages (download, separate, drumsep) is
# skipped when its output already exists, so re-running -s on an earlier -D grab
# only runs the drum pass. The title is resolved from metadata BEFORE downloading so
# we know what to look for on disk — one cheap API call.
# Granularity is per STAGE, not per stem: a model emits all its stems in a single
# forward pass, so one missing stem re-runs that whole separation.
#
# Why BS-Roformer-SW. Scored against the true stems of one remix pack (Rodriguez
# Jr. "Synthwave #1": 30 stems summed to a mix; studies/06-separation-benchmark),
# SI-SDR in dB, BS-Roformer-SW against htdemucs_6s:
#   kick 19.3 vs 17.0   snare 7.1 vs 6.4   bass 5.7 vs 3.7   synths -0.4 vs -1.7
#   drums 18.8 vs 16.4  hats 1.6 vs 1.8 (even)
# and it wins 86-99% of the active bars on kick, snare, bass and synths. The
# synths stem carries half the bass it did (15.0% against 29.9% of its energy).
# That is ONE track, a remix pack with no master bus, so easier than a rip: read
# the margins as a direction, not a constant. Cost, timed back to back on an
# M3 Pro (a 5:43 mix, whole `-s` run): 5:26 against 3:35 for htdemucs_6s. The
# separation is 2:25 against 0:34, on the GPU (MPS); the drum pass (3:00) is the
# same. Peak memory 2.3 GB; an 11:25 mix took 10:32 and 3.3 GB, no CPU fallback.
# Report: ~/.omc/reports/studies-separation-benchmark/default-switch-*.md
#
# History of the old default, kept because -m still selects it: htdemucs_6s was
# chosen for deep/melodic house on a bass-deficit measure. Across 5 windows of one
# set all four demucs models had medians within 0.5 dB (3.7-4.2 dB); only in
# bass-SPARSE passages did htdemucs lose (19-29 dB down against 9-12 for 6s).
# Window choice alone swings results by up to 26 dB, so re-measure with
# benchmark.py --samples 5 (or --truth), never on one excerpt.
#
# Both 6-stem models emit `piano`/`guitar`, but on synth-based house those
# measured as the SAME spectral contour as `other` at lower level —
# fragmentation, not separation. So they are merged back with `other` into a
# single `synths.wav` (float32, see the note at the merge) and deleted. Net
# output is vocals/drums/bass/synths — what the music actually contains.
#
# Stems are FLAC (lossless, ~57% smaller, Live imports them fine); synths.wav is
# the one exception and must stay float32 WAV, since the three merged inputs sum
# past full scale and FLAC has no float format. The mixed extensions are
# deliberate. BS-Roformer-SW stems come out at the working copy's 24 bits;
# demucs's are 16-bit. --overlap is left at demucs's 0.25 default.
#
# -d runs at ~1.3x realtime (vs ~14.5x for demucs), so a 9-minute set costs ~7
# minutes for the drum pass alone. It is on by default anyway (-D skips it):
# studies want the parts, and the beat aligner fits its grid to drums_kick first,
# so a grab without it gets a looser tempo. Its stems ARE real: kick peaks
# at 60-250Hz, hh at 4-10kHz, snare at 1-4kHz, i.e. genuinely distinct sources.
# ride/crash come out near-silent on electronic material, which is correct
# (no acoustic cymbals) rather than a failure.
#
# To drop a grab straight into an Ableton project, point -o at its
# Samples/Imported — Live's own convention, so "Collect All and Save"
# bundles the audio with the set:
#
#   yt2stems.sh URL -s -o "<your set> Project/Samples/Imported"
#
# Install:  uv tool install "audio-separator[cpu]"      # always needed
#           uv tool install demucs --with numpy         # only for -m htdemucs*, mdx_extra
#   ^ neither declares its deps fully: audio-separator needs the [cpu] extra
#     for onnxruntime (the extra still uses the Mac's GPU: its log says "setting
#     Torch device to MPS"), and demucs 4.1.0 omits numpy. Both install
#     "successfully" and then die on import without these.

set -euo pipefail

# `set -e` aborts with no message at all, which made a glob-returns-nothing bug
# look like "the tool produced an empty directory". Always say where we died.
trap 'rc=$?; echo "yt2stems: FAILED (exit $rc) at line $LINENO: $BASH_COMMAND" >&2' ERR

# Print the comment header as usage. Derived from the file, not a hardcoded line
# range — the range version silently truncated the help every time the header grew.
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

# Tempo from beat intervals (aubiotrack). Prints a rounded integer BPM, or
# nothing if the estimate isn't steady enough to be worth putting in a filename.
# Feed it the KICK stem when you have one: measured on a real set, kick-only
# detection gave 7.0% interval jitter vs 18.6% on the full mix — same ~126 BPM,
# but four times tighter. A wrong BPM baked into a filename is worse than none,
# hence the jitter gate.
#
# NEVER FAILS: prints a BPM or nothing and always returns 0, so a missing
# aligner or aubio leaves the file untagged instead of tripping `set -e`.
detect_bpm () {
  local py="$REPO/.venv/bin/python"
  local aligner="$REPO/beat_aligner.py"
  if [[ -x "$py" && -s "$aligner" ]]; then
    local bpm_val
    # Prefer the stems DIRECTORY over any single stem. Same tempo either way —
    # the aligner picks the kick itself — but a directory run also resolves the
    # bar phase off the summed mix and CACHES the whole result in
    # alignment.json, which stems2live then reads instead of repeating this
    # analysis. Handed a lone kick stem the phase is near chance and the result
    # is not reusable, so the second pass would have to redo everything.
    bpm_val=$("$py" "$aligner" "$1" --json 2>/dev/null | grep '"bpm":' | awk -F': ' '{printf "%.0f\n", $2}') || bpm_val=""
    if [[ -n "$bpm_val" ]]; then
      echo "$bpm_val"
      return 0
    fi
  fi
  # aubiotrack reads one audio FILE; handed the stems directory it just errors.
  [[ -f "$1" ]] || return 0
  command -v aubiotrack >/dev/null || return 0
  aubiotrack -i "$1" 2>/dev/null \
    | awk 'NR>1{d=$1-p; if(d>0.15&&d<2) print d} {p=$1}' \
    | sort -n \
    | awk '{v[NR]=$1} END{
        if(NR<32) exit                                  # too few beats to trust
        med=v[int(NR/2)]; q1=v[int(NR*0.25)+1]; q3=v[int(NR*0.75)]
        if(med<=0) exit
        if(100*(q3-q1)/med > 10) exit
        printf "%.0f\n", 60/med
      }' || true
}

DRUMSEP_MODEL="MDX23C-DrumSep-aufr33-jarredou.ckpt"
BSR_CKPT="BS-Roformer-SW.ckpt"
MODEL_DIR="${ROADIE_MODEL_DIR:-$HOME/.cache/audio-separator-models}"
DRUM_PARTS=(kick snare toms hh ride crash)
URL="" SRCFILE="" OUTDIR="" BY_ARTIST=0 STEMS=0 DRUMS=1 FORCE=0 MODEL="bs-roformer-sw"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file|-i) SRCFILE="${2:-}"; shift 2 ;;
    -o) OUTDIR="${2:-}"; shift 2 ;;
    -m) MODEL="${2:-}"; shift 2 ;;
    -a) BY_ARTIST=1; shift ;;
    -s) STEMS=1; shift ;;
    -d) DRUMS=1; STEMS=1; shift ;;          # -d is useless without the drums stem
    -D|--no-drums) DRUMS=0; shift ;;
    -f) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  URL="$1"; shift ;;
  esac
done
# The drum pass splits the drums stem, so with no stems there is nothing to split.
[[ $STEMS -eq 1 ]] || DRUMS=0

[[ -n "$URL" || -n "$SRCFILE" ]] || { usage; exit 2; }
[[ -z "$URL" || -z "$SRCFILE" ]] || { echo "give a URL or --file, not both" >&2; exit 2; }
[[ -z "$SRCFILE" || -s "$SRCFILE" ]] || { echo "no such file: $SRCFILE" >&2; exit 1; }

command -v ffmpeg >/dev/null || { echo "missing: ffmpeg" >&2; exit 1; }
if [[ -z "$SRCFILE" ]]; then
  command -v yt-dlp >/dev/null || { echo "missing: yt-dlp" >&2; exit 1; }
fi
# Check up front, not after a 9-minute download. demucs only when a demucs
# model is chosen; audio-separator for the default model and for the drum pass.
USE_DEMUCS=0; [[ "$MODEL" == "bs-roformer-sw" ]] || USE_DEMUCS=1
if [[ $STEMS -eq 1 && $USE_DEMUCS -eq 1 ]] && ! command -v demucs >/dev/null; then
  echo "missing: demucs — uv tool install demucs --with numpy, or drop -m for the default model" >&2; exit 1
fi
if [[ $STEMS -eq 1 && $USE_DEMUCS -eq 0 ]] && ! command -v audio-separator >/dev/null; then
  echo "missing: audio-separator — uv tool install \"audio-separator[cpu]\"" >&2; exit 1
fi
if [[ $DRUMS -eq 1 ]] && ! command -v audio-separator >/dev/null; then
  echo "missing: audio-separator — uv tool install \"audio-separator[cpu]\", or -D to skip the drum pass" >&2; exit 1
fi

OUT="${OUTDIR:-$PWD}"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)          # absolute, so demucs/ffmpeg can't be confused by cwd

# Resolve the name first so every later stage can check what's already on disk
# before doing work. For a URL that means one cheap metadata call, no bytes.
if [[ -n "$SRCFILE" ]]; then
  NAME=$(basename "$SRCFILE"); NAME="${NAME%.*}"
  # Match yt-dlp --restrict-filenames so both paths produce the same shape of
  # name, and nothing downstream has to quote around spaces.
  NAME=$(printf '%s' "$NAME" | tr ' ' '_' | tr -cd 'A-Za-z0-9._@-')
else
  NAME=$(yt-dlp -f bestaudio --no-playlist --restrict-filenames \
         --print filename -o "%(title)s.%(ext)s" "$URL" 2>/dev/null | head -1)
  NAME="${NAME%.*}"
fi
[[ -n "$NAME" ]] || { echo "couldn't resolve a name from the input" >&2; exit 1; }

# -a: before WAV/DEST are built, so the resume checks below look in the track
# folder. Names are --restrict-filenames shaped here, so " - " is "_-_".
if [[ $BY_ARTIST -eq 1 ]]; then
  title="$NAME"
  if [[ "$NAME" == *_-_* ]]; then
    artist="${NAME%%_-_*}"; artist="${artist//_/ }"; title="${NAME#*_-_}"
    lc() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
    want=$(lc "$artist") last=$(lc "${artist##* }") pick="$artist"
    for d in "$OUT"/*/; do
      [[ -d "$d" ]] || continue
      b=$(basename "$d"); lb=$(lc "$b")
      if [[ "$lb" == "$want" ]]; then pick="$b"; break; fi
      [[ "$lb" == "$last" ]] && pick="$b"
    done
    OUT="$OUT/$pick"
  else
    echo "no ' - ' in the title, so no artist folder: writing under $OUT (-o picks one)" >&2
  fi
  OUT="$OUT/${title//_/ }"; mkdir -p "$OUT"
fi

WAV="$OUT/$NAME.wav"
DEST="$OUT/${NAME}_stems"

# A previous run may have tagged the filename with a detected BPM. Adopt that
# name so the skip-checks below find the existing work instead of redoing it.
if [[ ! -s "$WAV" ]]; then
  # Plain glob, not `ls ... | head`: under `set -euo pipefail` a glob that
  # matches nothing makes ls exit 1, pipefail propagates it, and set -e kills
  # the script with NO message. That bug only fired on a fresh output dir.
  prev=""
  for f in "$OUT/${NAME}_"[0-9]*bpm.wav; do
    if [[ -e "$f" ]]; then prev="$f"; break; fi
  done
  if [[ -n "$prev" ]]; then
    NAME=$(basename "$prev" .wav); WAV="$prev"; DEST="$OUT/${NAME}_stems"
  fi
fi

echo "==> $OUT"
echo "==> $NAME"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# ---- stage 1: fetch + convert -------------------------------------------
if [[ $FORCE -eq 0 && -s "$WAV" ]]; then
  echo "--- have $NAME.wav, skipping download"
else
  if [[ -n "$SRCFILE" ]]; then
    SRC="$SRCFILE"
    echo "==> local source: $SRCFILE"
  else
    # --no-playlist so a URL carrying a list= param doesn't pull the whole
    # playlist. Needs a JS runtime (brew install deno) or YouTube extraction is
    # deprecated and some formats go missing.
    yt-dlp -f bestaudio --no-playlist \
           -o "$TMP/%(title)s.%(ext)s" \
           --restrict-filenames \
           "$URL"
    SRC=$(find "$TMP" -type f -maxdepth 1 | head -1)
    [[ -n "$SRC" ]] || { echo "download produced nothing" >&2; exit 1; }
  fi
  # 44.1k / 24-bit working copy: Live's default rate, and the depth costs nothing
  # here since this is the one file we keep as WAV. (Stems are FLAC.)
  ffmpeg -nostdin -loglevel warning -y -i "$SRC" \
         -ar 44100 -c:a pcm_s24le "$WAV"
  echo "==> $WAV"
fi

# ---- stage 2: stem separation -------------------------------------------
if [[ $STEMS -eq 1 ]]; then
  # What ends up on disk AFTER the 6-stem merge below — not what the model emits.
  case "$MODEL" in
    bs-roformer-sw|htdemucs_6s) WANT=(bass drums synths vocals) ;;
    *)           WANT=(bass drums other vocals) ;;
  esac
  # Stems are FLAC except the merged synths, which must stay float32 WAV — so
  # the presence check has to accept either extension or caching never hits.
  need=0
  for s in "${WANT[@]}"; do
    [[ -s "$DEST/$s.flac" || -s "$DEST/$s.wav" ]] || need=1
  done
  if [[ $FORCE -eq 0 && $need -eq 0 ]]; then
    echo "--- have ${#WANT[@]} $MODEL stems, skipping separation"
  else
    # demucs: --flac without --int24, so the stems are 16-bit FLAC (PCM_16), NOT the
    # same samples as a 24-bit WAV run: they are rounded to 16 bits. That loses
    # nothing measurable on a ~133kbps opus source off YouTube, whose noise floor
    # sits far above 16-bit's ~96 dB. ~57% smaller than WAV (a 9-min bass stem:
    # 149MB -> 64MB, whole set ~1.6GB -> ~700MB), and Live imports .flac from an
    # absolute path exactly like .wav — tested, not assumed.
    #
    # --overlap left at demucs's own 0.25 default. The earlier --overlap 0.5
    # recommendation came from a single window; across 5 windows 0.25 and 0.50
    # differ by 0.1 dB. --shifts skipped for the same reason.
    #
    # demucs insists on writing <out>/<model>/<trackname>/*.<ext>. Send it to a
    # temp tree and lift the stems up, so they land directly in <name>_stems/ —
    # two fewer levels to dig through when dragging into Live.
    mkdir -p "$TMP/stems"
    if [[ $USE_DEMUCS -eq 1 ]]; then
      echo "==> separating with $MODEL"
      demucs -n "$MODEL" --flac -o "$TMP/stems" "$WAV"
    else
      # audio-separator downloads a model that is not in --model_file_dir; say
      # so before a 700 MB fetch looks like a hang.
      [[ -s "$MODEL_DIR/$BSR_CKPT" ]] ||
        echo "--- $BSR_CKPT is not in $MODEL_DIR: audio-separator downloads it first (~700 MB)"
      echo "==> separating with BS-Roformer-SW (~2x realtime on an Apple GPU)"
      # It exits 0 even when it dies on an import, so the stems are checked
      # below rather than its status. Its log goes to stdout; errors stay on
      # stderr.
      audio-separator "$WAV" -m "$BSR_CKPT" --model_file_dir "$MODEL_DIR" \
        --output_dir="$TMP/stems" --output_format=FLAC >/dev/null
      # It names files <input>_(bass)_BS-Roformer-SW.flac. Pull the stem out of
      # the parens, as the drum pass does, and land it as bass.flac etc.
      for f in "$TMP/stems"/*.flac; do
        [[ -e "$f" ]] || continue
        stem=$(basename "$f" | sed -n 's/.*(\([a-z]*\)).*/\1/p')
        [[ -n "$stem" ]] && mv "$f" "$TMP/stems/$stem.flac"
      done
      for stem in bass drums other vocals piano guitar; do
        [[ -s "$TMP/stems/$stem.flac" ]] ||
          { echo "audio-separator wrote no $stem stem (see its output above)" >&2; exit 1; }
      done
    fi
    mkdir -p "$DEST"
    find "$TMP/stems" -type f \( -name '*.flac' -o -name '*.wav' \) -exec mv {} "$DEST/" \;

    # The 6-stem models emit piano/guitar, but on synth-based house they measured as
    # the SAME spectral contour as `other` at lower level — fragmentation, not
    # separation. Fold all three into one honest `synths` stem. For htdemucs_6s
    # their sum lands at -19.6 dB, i.e. what the plain 4-stem model's `other`
    # already was (-19.4 dB), so nothing is lost.
    #
    # synths.wav is float32 WAV while everything else is FLAC. That mix is
    # DELIBERATE, not an oversight: the three inputs can sum past full scale
    # (htdemucs_6s: true peak 1.0018, 0.016 dB over), and FLAC has no float format — so FLAC or int
    # would clip it. float32 keeps the sum exact, so synths+bass+drums+vocals
    # still reconstructs the mix.
    if [[ ( "$MODEL" == "htdemucs_6s" || "$MODEL" == "bs-roformer-sw" ) && -s "$DEST/other.flac" ]]; then
      echo "==> merging other+piano+guitar -> synths.wav (float32)"
      ffmpeg -nostdin -loglevel error -y \
        -i "$DEST/other.flac" -i "$DEST/piano.flac" -i "$DEST/guitar.flac" \
        -filter_complex "[0][1][2]amix=inputs=3:normalize=0[s]" \
        -map "[s]" -c:a pcm_f32le "$DEST/synths.wav"
      rm -f "$DEST/other.flac" "$DEST/piano.flac" "$DEST/guitar.flac"
    fi
  fi
fi

# ---- stage 3: drum split ------------------------------------------------
if [[ $DRUMS -eq 1 ]]; then
  # drums is FLAC now; resolve whichever extension exists.
  DRUMSRC=""
  for e in flac wav; do [[ -s "$DEST/drums.$e" ]] && { DRUMSRC="$DEST/drums.$e"; break; }; done
  [[ -n "$DRUMSRC" ]] || { echo "no drums stem to split" >&2; exit 1; }
  need=0
  for p in "${DRUM_PARTS[@]}"; do
    [[ -s "$DEST/drums_$p.flac" || -s "$DEST/drums_$p.wav" ]] || need=1
  done
  if [[ $FORCE -eq 0 && $need -eq 0 ]]; then
    echo "--- have ${#DRUM_PARTS[@]} drum parts, skipping drumsep"
  else
    echo "==> splitting drums (slow: ~1.3x realtime)"
    # FLAC, matching the demucs stems: six WAV drum parts were 568MB of the
    # 830MB output — two thirds of the total for the least-used stems.
    audio-separator "$DRUMSRC" -m "$DRUMSEP_MODEL" --model_file_dir "$MODEL_DIR" \
      --output_dir="$TMP/drumsep" --output_format=FLAC >/dev/null
    # audio-separator names files drums_(kick)_MDX23C-....wav — pull the part
    # out of the parens and land it as drums_kick.<ext> beside the other stems.
    for f in "$TMP/drumsep"/*.flac "$TMP/drumsep"/*.wav; do
      [[ -e "$f" ]] || continue
      part=$(basename "$f" | sed -n 's/.*(\([a-z]*\)).*/\1/p')
      ext="${f##*.}"
      [[ -n "$part" ]] && mv "$f" "$DEST/drums_$part.$ext"
    done
  fi
fi

# ---- stage 4: tag the filename with the detected tempo -------------------
# Runs last so the kick stem exists when -d was used: on a real set the kick
# gave IQR 3.9% vs 4.6% for the full mix — same answer, but the cleanest signal,
# since there are no pads or vocals to confuse onset detection. Falls back
# through drums to the full mix. Silent when nothing passes the gate; a wrong
# BPM baked into a filename is worse than no BPM.
if [[ "$NAME" != *bpm ]]; then
  BPM=""
  # The stems DIRECTORY first: one run there tags the filename AND leaves a
  # reusable alignment.json, so stems2live does not repeat the analysis. The
  # per-file candidates remain for a run with no stems (-s omitted), where the
  # only thing on disk is the merged wav.
  for cand in "$DEST" "$DEST/drums_kick.flac" "$DEST/drums_kick.wav" \
              "$DEST/drums.flac" "$DEST/drums.wav" "$WAV"; do
    [[ -s "$cand" || -d "$cand" ]] || continue
    BPM=$(detect_bpm "$cand") || BPM=""
    if [[ -n "$BPM" ]]; then break; fi
  done
  if [[ -n "${BPM:-}" ]]; then
    mv "$WAV" "$OUT/${NAME}_${BPM}bpm.wav"
    [[ -d "$DEST" ]] && mv "$DEST" "$OUT/${NAME}_${BPM}bpm_stems"
    NAME="${NAME}_${BPM}bpm"; WAV="$OUT/$NAME.wav"; DEST="$OUT/${NAME}_stems"
    echo "==> ${BPM} BPM — tagged"
  else
    echo "--- no steady BPM detected (or no aligner/aubiotrack available) — filename left untagged"
  fi
fi

echo "==> $WAV"
[[ -d "$DEST" ]] && { echo "==> $DEST/"; ls "$DEST"; } || true
