#!/usr/bin/env python3
"""yt2stems.sh's stem stage: BS-Roformer-SW is the default, Demucs stays selectable.

Subprocess only, like test_cli.py. `audio-separator` and `demucs` are stub scripts
first on PATH that write the files the real ones write (names included) and log
their argv, so the test shows which tool ran with which flags without a model.
ffmpeg is the real one: the stub stems are real FLACs and the synths merge runs.
PATH holds only the stubs, ffmpeg/ffprobe and the system dirs, so a real demucs
or audio-separator on the machine is never reached.
"""

import os
import shutil
import subprocess
import wave

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
YT2STEMS = os.path.join(HERE, "yt2stems.sh")
DRUM_PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

# Writes <input>_(<stem>)_<model>.flac into --output_dir, the way audio-separator
# does: six stems for BS-Roformer-SW, the drum parts for the DrumSep model.
# STUB_STEMS (when set, even empty) replaces the BS-Roformer-SW stem list, to play
# the real tool leaving out a silent stem, or writing nothing.
AUDIO_SEPARATOR = r"""#!/bin/bash
echo "audio-separator $*" >> "$STUB_LOG"
in="$1"; shift
model="" outdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) model="$2"; shift 2 ;;
    --output_dir=*) outdir="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
base=$(basename "${in%.*}")
case "$model" in
  BS-Roformer-SW.ckpt) stems="${STUB_STEMS-bass drums other vocals piano guitar}"; tag=BS-Roformer-SW ;;
  *DrumSep*) stems="kick snare toms hh ride crash"; tag=MDX23C-DrumSep ;;
  *) echo "stub: unknown model $model" >&2; exit 1 ;;
esac
mkdir -p "$outdir"
for s in $stems; do cp "$STUB_FLAC" "$outdir/${base}_(${s})_${tag}.flac"; done
"""

# Writes <out>/<model>/<track>/<stem>.flac, the way demucs does.
DEMUCS = r"""#!/bin/bash
echo "demucs $*" >> "$STUB_LOG"
model="" out="" wav=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) model="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    --flac) shift ;;
    *) wav="$1"; shift ;;
  esac
done
case "$model" in
  htdemucs_6s) stems="bass drums other vocals piano guitar" ;;
  *) stems="bass drums other vocals" ;;
esac
dir="$out/$model/$(basename "${wav%.*}")"
mkdir -p "$dir"
for s in $stems; do cp "$STUB_FLAC" "$dir/$s.flac"; done
"""


@pytest.fixture
def rig(tmp_path):
    """A tiny wav, stub tools, and an env that sees only the stubs."""
    bin_ = tmp_path / "fakebin"
    bin_.mkdir()
    for name, body in (("audio-separator", AUDIO_SEPARATOR), ("demucs", DEMUCS)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    for tool in ("ffmpeg", "ffprobe"):
        (bin_ / tool).symlink_to(shutil.which(tool))
    flac = tmp_path / "stub.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=220:duration=0.5", str(flac)], check=True)
    # A name already ending in "bpm" skips stage 4's tempo detection (a slow
    # python start-up per run, and not what these tests are about).
    src = tmp_path / "tiny_120bpm.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00\x00" * 22050)
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.pathsep.join([str(bin_), "/usr/bin", "/bin"]),
           "HOME": str(home),
           "STUB_LOG": str(tmp_path / "calls.log"),
           "STUB_FLAC": str(flac)}
    return env, src, tmp_path


def split(rig, *flags):
    env, src, tmp = rig
    return subprocess.run([YT2STEMS, "-i", str(src), "-o", str(tmp / "out"), *flags],
                          capture_output=True, text=True, timeout=120, env=env,
                          cwd=str(tmp), check=False)


def calls(rig):
    log = rig[2] / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def stems_dir(rig):
    return rig[2] / "out" / "tiny_120bpm_stems"


def test_default_is_bs_roformer_sw_with_the_four_stems_and_synths(rig):
    r = split(rig, "-s")
    assert r.returncode == 0, r.stdout + r.stderr
    d = stems_dir(rig)
    assert sorted(p.name for p in d.iterdir() if not p.name.endswith(".json")) == sorted(
        ["bass.flac", "drums.flac", "vocals.flac", "synths.wav", ".stem-model"]
        + ["drums_%s.flac" % p for p in DRUM_PARTS])
    assert (d / ".stem-model").read_text().strip() == "bs-roformer-sw"
    # other + piano + guitar are folded into synths.wav, float32, and deleted
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(d / "synths.wav")],
                           capture_output=True, text=True, check=True, env=rig[0])
    assert probe.stdout.strip() == "pcm_f32le"
    log = calls(rig)
    assert not any(c.startswith("demucs") for c in log), log
    assert sum("-m BS-Roformer-SW.ckpt" in c for c in log) == 1, log


def test_both_audio_separator_calls_get_the_model_dir(rig):
    split(rig, "-s")
    default_dir = rig[0]["HOME"] + "/.cache/audio-separator-models"
    separator = [c for c in calls(rig) if c.startswith("audio-separator")]
    assert len(separator) == 2, separator           # the stems, then the drum pass
    for c in separator:
        assert "--model_file_dir %s" % default_dir in c, c


def test_roadie_model_dir_overrides_the_default(rig, tmp_path):
    rig[0]["ROADIE_MODEL_DIR"] = str(tmp_path / "models")
    split(rig, "-s")
    separator = calls(rig)
    assert len(separator) == 2
    for c in separator:
        assert "--model_file_dir %s" % (tmp_path / "models") in c, c


def test_a_missing_checkpoint_is_announced_once_before_the_download(rig, tmp_path):
    r = split(rig, "-s")
    assert r.stdout.count("BS-Roformer-SW.ckpt is not in") == 1, r.stdout
    models = tmp_path / "models"
    models.mkdir()
    (models / "BS-Roformer-SW.ckpt").write_bytes(b"x")
    rig[0]["ROADIE_MODEL_DIR"] = str(models)
    r = split(rig, "-s", "-f")
    assert "BS-Roformer-SW.ckpt is not in" not in r.stdout, r.stdout


def test_a_rerun_skips_both_stages(rig):
    assert split(rig, "-s").returncode == 0
    first = calls(rig)
    assert len(first) == 2
    r = split(rig, "-s")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls(rig) == first
    assert "skipping separation" in r.stdout and "skipping drumsep" in r.stdout, r.stdout


def test_force_reruns_both_stages(rig):
    split(rig, "-s")
    split(rig, "-s", "-f")
    assert len(calls(rig)) == 4


def test_no_drums_runs_only_the_stem_model(rig):
    r = split(rig, "-s", "-D")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls(rig)) == 1
    assert not list(stems_dir(rig).glob("drums_*"))


def test_htdemucs_6s_still_calls_demucs_and_merges_synths(rig):
    r = split(rig, "-s", "-m", "htdemucs_6s")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(rig)
    assert [c.split()[0] for c in log] == ["demucs", "audio-separator"], log
    assert "-n htdemucs_6s" in log[0]
    # the drum pass is audio-separator either way, with the model dir
    assert "--model_file_dir" in log[1] and "MDX23C-DrumSep" in log[1], log[1]
    d = stems_dir(rig)
    assert (d / "synths.wav").is_file() and not (d / "other.flac").exists()


@pytest.mark.parametrize("model", ["htdemucs_ft", "mdx_extra", "htdemucs"])
def test_the_other_demucs_models_keep_other_and_skip_the_merge(rig, model):
    r = split(rig, "-s", "-D", "-m", model)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "demucs -n %s" % model in calls(rig)[0]
    d = stems_dir(rig)
    assert (d / "other.flac").is_file() and not (d / "synths.wav").exists()


def test_demucs_is_only_required_when_a_demucs_model_is_chosen(rig):
    os.remove(os.path.join(rig[0]["PATH"].split(os.pathsep)[0], "demucs"))
    assert split(rig, "-s").returncode == 0
    r = split(rig, "-s", "-f", "-m", "htdemucs_6s")
    assert r.returncode != 0 and "missing: demucs" in r.stderr, r.stderr


# --- the stem model's flags ---------------------------------------------------

def test_the_stems_are_written_24_bit_and_unscaled(rig):
    """Without --use_soundfile audio-separator rounds to int16 inside a 24-bit
    container; at its default --normalization 0.9 the stems stop adding up to
    the mix. Measured on a real excerpt (README), asserted here as flags."""
    split(rig, "-s", "-D")
    c = calls(rig)[0]
    assert "--use_soundfile" in c and "--normalization 1.0" in c, c


def test_a_model_name_is_case_insensitive_and_an_unknown_one_is_refused(rig):
    r = split(rig, "-s", "-D", "-m", "BS-Roformer-SW")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(c.startswith("demucs") for c in calls(rig))
    r = split(rig, "-s", "-D", "-m", "htdemucs_7s")
    assert r.returncode == 2 and "unknown model" in r.stderr, r.stderr


# --- re-running over stems another model wrote -------------------------------

def test_a_model_switch_over_an_old_folder_reseparates_and_redoes_the_drum_parts(rig):
    """A folder without .stem-model predates it, so it is Demucs. Both models
    leave the same file names, so the file check alone would skip."""
    assert split(rig, "-s", "-m", "htdemucs_6s").returncode == 0
    (stems_dir(rig) / ".stem-model").unlink()
    before = len(calls(rig))
    r = split(rig, "-s")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "are from demucs; re-separating with bs-roformer-sw" in r.stdout, r.stdout
    assert "skipping separation" not in r.stdout
    new = calls(rig)[before:]
    assert [c.split()[0] for c in new] == ["audio-separator", "audio-separator"], new
    assert "BS-Roformer-SW.ckpt" in new[0] and "DrumSep" in new[1], new
    assert (stems_dir(rig) / ".stem-model").read_text().strip() == "bs-roformer-sw"


def test_a_marker_naming_another_model_reseparates_and_drops_the_old_synths(rig):
    assert split(rig, "-s", "-D").returncode == 0
    assert (stems_dir(rig) / "synths.wav").is_file()
    r = split(rig, "-s", "-D", "-m", "htdemucs_ft")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipping separation" not in r.stdout, r.stdout
    d = stems_dir(rig)
    assert (d / "other.flac").is_file() and not (d / "synths.wav").exists()
    assert (d / ".stem-model").read_text().strip() == "htdemucs_ft"


def test_the_same_model_still_skips(rig):
    split(rig, "-s", "-D")
    r = split(rig, "-s", "-D")
    assert "re-separating" not in r.stdout and "skipping separation" in r.stdout, r.stdout
    assert len(calls(rig)) == 1


def test_the_drum_parts_are_redone_when_the_stems_were(rig):
    split(rig, "-s")
    assert len(calls(rig)) == 2
    r = split(rig, "-s", "-m", "htdemucs_6s")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipping drumsep" not in r.stdout, r.stdout
    log = calls(rig)
    assert [c.split()[0] for c in log[2:]] == ["demucs", "audio-separator"], log
    assert "DrumSep" in log[3], log


# --- what audio-separator leaves out -----------------------------------------

def test_a_silent_vocals_stem_is_filled_with_silence_and_the_merge_runs(rig):
    """audio-separator writes no file for a stem whose peak is under 1e-6; on
    Synthwave BS-Roformer's vocals sit at -116 dB."""
    rig[0]["STUB_STEMS"] = "bass drums other"
    r = split(rig, "-s", "-D")
    assert r.returncode == 0, r.stdout + r.stderr
    for stem in ("vocals", "piano", "guitar"):
        assert "no %s stem (silent)" % stem in r.stderr, r.stderr
    d = stems_dir(rig)
    assert (d / "synths.wav").is_file()
    assert not (d / "piano.flac").exists()
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(d / "vocals.flac")],
                         capture_output=True, text=True, check=True, env=rig[0])
    assert abs(float(dur.stdout) - 0.5) < 0.01, dur.stdout
    assert (d / ".stem-model").is_file()


def test_a_missing_bass_stem_fails_and_lists_what_was_written(rig):
    rig[0]["STUB_STEMS"] = "drums other vocals"
    r = split(rig, "-s", "-D")
    assert r.returncode != 0
    assert "wrote no bass stem" in r.stderr and "drums.flac" in r.stderr, r.stderr
    assert not (stems_dir(rig) / ".stem-model").exists()


def test_a_stub_that_writes_nothing_fails_cleanly(rig):
    rig[0]["STUB_STEMS"] = ""
    r = split(rig, "-s", "-D")
    assert r.returncode != 0 and "wrote no bass stem" in r.stderr, r.stderr
    assert not stems_dir(rig).exists()


# --- tools and model dir ------------------------------------------------------

def test_no_audio_separator_is_refused_up_front_and_points_at_demucs(rig):
    os.remove(os.path.join(rig[0]["PATH"].split(os.pathsep)[0], "audio-separator"))
    r = split(rig, "-s", "-D")
    assert r.returncode != 0
    assert "missing: audio-separator" in r.stderr and "-m htdemucs_6s -D" in r.stderr, r.stderr
    assert calls(rig) == []
    # and the pointer is true: demucs without the drum pass needs no audio-separator
    assert split(rig, "-s", "-D", "-m", "htdemucs_6s").returncode == 0


def test_model_dir_falls_back_to_audio_separators_own_env_var(rig, tmp_path):
    rig[0]["AUDIO_SEPARATOR_MODEL_DIR"] = str(tmp_path / "theirs")
    split(rig, "-s")
    for c in calls(rig):
        assert "--model_file_dir %s" % (tmp_path / "theirs") in c, c
    rig[0]["ROADIE_MODEL_DIR"] = str(tmp_path / "ours")
    split(rig, "-s", "-f")
    for c in calls(rig)[2:]:
        assert "--model_file_dir %s" % (tmp_path / "ours") in c, c


def test_a_missing_drumsep_model_is_announced_too(rig, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "BS-Roformer-SW.ckpt").write_bytes(b"x")
    rig[0]["ROADIE_MODEL_DIR"] = str(models)
    r = split(rig, "-s")
    assert "MDX23C-DrumSep-aufr33-jarredou.ckpt is not in" in r.stdout, r.stdout
    assert "BS-Roformer-SW.ckpt is not in" not in r.stdout
