"""roadie_analysis gives the same bytes as Sideman main for the same inputs.

The analysis half that moved here (oneshot, reference_profile, the element and drum layers of a mix_report
made against a profile) was copied from Sideman main bc9ba08 with its behaviour unchanged. `produce` builds a
fixed reference track with stems and a capture of it (seeded noise, a kick tuned two semitones down, a bass
4 dB up, a hat panned wide, a synth absent in one section) and returns every output as text; the files in
analysis_golden/ are what Sideman main wrote from the same fixtures. The tests run the same function on
roadie_analysis and compare the text, byte for byte.

Why two tests: the report's `presence` table is built by iterating a set, so the order of its keys (not
its content) depends on PYTHONHASHSEED, in Sideman main as in the copy. One test compares the parsed values
in any process; the other runs roadie_analysis with PYTHONHASHSEED=0, the seed the goldens were written
with, and compares bytes.

Regenerate from a checkout (or `git archive`) of Sideman main, whose own `listen` and `oneshot` do the
work, with the numpy and soundfile pinned in requirements.txt:

    PYTHONHASHSEED=0 PYTHONPATH=<sideman main checkout> .venv/bin/python test_analysis_equivalence.py --write analysis_golden
"""
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "analysis_golden"
SR = 48000
TEMPO = 120.0                      # 1 bar = 2 s
T0 = 0.25
MTIME = 1_700_000_000              # inputs get this mtime, so a profile's `inputs` do not depend on the run
SECTIONS = [{"name": "A", "start_bar": 0, "bars": 4}, {"name": "B", "start_bar": 4, "bars": 4},
            {"name": "C", "start_bar": 8, "bars": 4}]
BARS = 12


def _decay(freq, tau, seconds):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)


def _place(hits, length_s, fn):
    x = np.zeros(int(length_s * SR))
    for t in hits:
        h = fn()
        s0 = int(t * SR)
        h = h[:len(x) - s0]
        x[s0:s0 + len(h)] += h
    return x


def _write(d, name, x):
    p = d / name
    sf.write(str(p), x.astype(np.float32), SR, subtype="FLOAT")
    os.utime(p, (MTIME, MTIME))
    return str(p)


def _stems(d, kick_hz, bass_db, hat_pan, synth):
    """Stems of one rendering of the track: four on the floor, a bass line, off-beat hats, snare on 2 and 4."""
    rng = np.random.default_rng(7)
    total = T0 + BARS * 2.0
    beats = [T0 + 0.5 * k for k in range(BARS * 4)]
    kick = _place(beats, total, lambda: 0.5 * _decay(kick_hz, 0.05, 0.5))
    snare_noise = rng.standard_normal(int(0.2 * SR))
    snare = _place(beats[1::2], total, lambda: 0.3 * snare_noise * np.exp(-np.arange(len(snare_noise)) / SR / 0.04))
    hat_noise = rng.standard_normal(int(0.1 * SR))
    hat = _place([t + 0.25 for t in beats], total, lambda: 0.1 * hat_noise * np.exp(-np.arange(len(hat_noise)) / SR / 0.02))
    t = np.arange(len(kick)) / SR
    bass = 0.4 * 10 ** (bass_db / 20) * np.sin(2 * np.pi * 55 * t) * (np.mod(t - T0, 0.5) < 0.4)
    lead = 0.1 * np.sin(2 * np.pi * 880 * t) * np.exp(-np.mod(t - T0, 0.25) / 0.1)
    lead[int((T0 + 4 * 2.0) * SR):int((T0 + 8 * 2.0) * SR)] = 0.0     # B has no lead
    mono = {"kick": kick, "bass": bass, "snare": snare}
    out = {k: _write(d, k + ".wav", np.stack([v, v], 1)) for k, v in mono.items()}
    out["hh"] = _write(d, "hh.wav", np.stack([hat * (1 - hat_pan), hat * (1 + hat_pan)], 1))
    out["synth"] = _write(d, "synth.wav", np.stack([lead * synth, lead * synth], 1))
    drums = np.stack([snare + hat * (1 - hat_pan), snare + hat * (1 + hat_pan)], 1)
    mix = np.stack([kick + bass + snare + lead * synth + hat * (1 - hat_pan),
                    kick + bass + snare + lead * synth + hat * (1 + hat_pan)], 1)
    out["drums"] = _write(d, "drums.wav", drums)
    out["mix"] = _write(d, "mix.wav", mix)
    return out


def _norm(text, d):
    for p in sorted({str(d), str(d.resolve())}, key=len, reverse=True):   # /private/var before /var
        text = text.replace(p, "DIR")
    return text


def produce(profile_mod, oneshot_mod, d):
    """Every output of the moved half on the fixed fixtures, as {name: text}. `profile_mod` has
    reference_profile and mix_report, `oneshot_mod` profile and compare: roadie_analysis's, or Sideman main's
    listen and oneshot."""
    d = pathlib.Path(d)
    (d / "ref").mkdir()
    (d / "ours").mkdir()
    ref = _stems(d / "ref", 80.0, -6.0, 0.0, 1.0)
    ours = _stems(d / "ours", 80.0 * 2 ** (-2 / 12), -2.0, 0.6, 1.0)
    out = {}
    profile_mod.reference_profile(
        ref["mix"], {"kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"], "hh": ref["hh"],
                     "snare": ref["snare"], "drums": [ref["hh"], ref["snare"]]},
        TEMPO, T0, SECTIONS, out=d / "ref.profile.json")
    out["profile.json"] = (d / "ref.profile.json").read_text()
    manifest = {"tempo": TEMPO, "offset_s": T0, "sections": SECTIONS,
                "sources": {"master": ours["mix"], "kick": ours["kick"], "bass": ours["bass"],
                            "synth": ours["synth"], "hh": ours["hh"], "snare": ours["snare"]}}
    r = profile_mod.mix_report(manifest, reference=d / "ref.profile.json", out_dir=d / "out")
    out["mix_report.json"] = pathlib.Path(r["json_path"]).read_text()
    out["mix_report.md"] = pathlib.Path(r["md_path"]).read_text()
    take = np.concatenate([_decay(100.0, 0.08, 0.7), _decay(100.0, 0.08, 0.7) * 0.7, _decay(120.0, 0.05, 0.7)])
    sf.write(str(d / "take.wav"), np.stack([take, take], 1).astype(np.float32), SR, subtype="FLOAT")
    a, b = oneshot_mod.profile(str(d / "take.wav"), 0), oneshot_mod.profile(str(d / "take.wav"), 2)
    out["oneshot_profile.json"] = json.dumps(a, indent=2)
    out["oneshot_compare.json"] = json.dumps(oneshot_mod.compare(a, b), indent=2)
    return {k: _norm(v, d) for k, v in out.items()}


def test_the_fixture_exercises_every_layer(tmp_path):
    """Equal bytes mean little if both sides were empty: the report has gaps, width gaps, a pitch reading."""
    from roadie_analysis import oneshot, profile
    out = produce(profile, oneshot, tmp_path)
    rep = json.loads(out["mix_report.json"])
    sec = rep["sections"]
    assert all(s["elements"]["paired"] for s in sec) and rep["reference_kind"] == "profile"
    assert any(s["elements"]["gaps"] for s in sec) and any(s["elements"]["width_gaps"] for s in sec)
    assert any(d["readings"] for s in sec for d in s["drums"]["roles"])
    assert "Reference profile" in out["mix_report.md"] and json.loads(out["oneshot_compare.json"])["readings"]


def test_outputs_are_the_same_values_as_sideman_main(tmp_path):
    """Whatever the process's hash seed: the report's `presence` table is built by iterating a set, so the
    order of its keys (not its content) varies from one run to the next, in Sideman main as here."""
    from roadie_analysis import oneshot, profile
    got = produce(profile, oneshot, tmp_path)
    assert sorted(got) == sorted(p.name for p in GOLDEN.iterdir())
    for name, text in got.items():
        want = (GOLDEN / name).read_text()
        assert (json.loads(text) == json.loads(want)) if name.endswith(".json") else text == want, name


def test_outputs_are_byte_identical_to_sideman_main_under_one_hash_seed(tmp_path):
    """The goldens were written with PYTHONHASHSEED=0; roadie_analysis run the same way gives the same bytes."""
    import subprocess
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(HERE / "test_analysis_equivalence.py"), "--write", str(out), "--roadie"],
                   check=True, env={**os.environ, "PYTHONHASHSEED": "0"}, cwd=str(tmp_path))
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in GOLDEN.iterdir())
    for p in GOLDEN.iterdir():
        assert (out / p.name).read_bytes() == p.read_bytes(), p.name


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[1] != "--write" or (len(sys.argv) == 4 and sys.argv[3] != "--roadie"):
        sys.exit(__doc__)
    if len(sys.argv) == 4:
        sys.path.insert(0, str(HERE))
        from roadie_analysis import oneshot, profile as listen
    else:
        from sideman import listen, oneshot        # Sideman main, found through PYTHONPATH
        assert "site-packages" not in listen.__file__, listen.__file__
    assert os.environ.get("PYTHONHASHSEED") == "0", "set PYTHONHASHSEED=0"
    dest = pathlib.Path(sys.argv[2]).resolve()
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in produce(listen, oneshot, tmp).items():
            (dest / name).write_text(text)
    print("wrote", dest, "from", listen.__file__)
