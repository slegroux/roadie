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
  BS-Roformer-SW.ckpt) stems="bass drums other vocals piano guitar"; tag=BS-Roformer-SW ;;
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
        ["bass.flac", "drums.flac", "vocals.flac", "synths.wav"]
        + ["drums_%s.flac" % p for p in DRUM_PARTS])
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
    assert "is not in" not in r.stdout, r.stdout


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
