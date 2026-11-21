#!/usr/bin/env python3
"""Tests for roadie_analysis/key.py: the key table and parser, and detection on synthetic tones in known keys."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roadie_analysis import key as K

HERE = os.path.dirname(os.path.abspath(__file__))

# the Camelot wheel as printed by DJ software: every key, short form and code
WHEEL = {
    "B major": ("B", "1B"), "G# minor": ("G#m", "1A"), "F# major": ("F#", "2B"), "Eb minor": ("Ebm", "2A"),
    "Db major": ("Db", "3B"), "Bb minor": ("Bbm", "3A"), "Ab major": ("Ab", "4B"), "F minor": ("Fm", "4A"),
    "Eb major": ("Eb", "5B"), "C minor": ("Cm", "5A"), "Bb major": ("Bb", "6B"), "G minor": ("Gm", "6A"),
    "F major": ("F", "7B"), "D minor": ("Dm", "7A"), "C major": ("C", "8B"), "A minor": ("Am", "8A"),
    "G major": ("G", "9B"), "E minor": ("Em", "9A"), "D major": ("D", "10B"), "B minor": ("Bm", "10A"),
    "A major": ("A", "11B"), "F# minor": ("F#m", "11A"), "E major": ("E", "12B"), "C# minor": ("C#m", "12A"),
}


def test_the_table_covers_all_24_keys_and_round_trips():
    seen = set()
    for pc in range(12):
        for mode in (K.MAJOR, K.MINOR):
            name = K.key_name(pc, mode)
            assert (K.short_name(pc, mode), K.camelot(pc, mode)) == WHEEL[name]
            assert K.parse_key(name) == (pc, mode) == K.parse_key(K.camelot(pc, mode))
            assert K.parse_key(K.short_name(pc, mode)) == (pc, mode)
            seen.add(name)
    assert seen == set(WHEEL)


def test_parse_reads_the_forms_the_tags_use():
    assert K.parse_key("Em") == (4, "minor") and K.parse_key("EMaj") == (4, "major")
    assert K.parse_key("F#Maj") == (6, "major") and K.parse_key("GMaj") == (7, "major")
    assert K.parse_key("C") == (0, "major") and K.parse_key("6A") == (7, "minor")
    assert K.parse_key("11B") == (9, "major") and K.parse_key("Abm") == (8, "minor")
    assert K.parse_key("a minor") == (9, "minor") and K.parse_key("Bb") == (10, "major")
    for bad in ("", "H", "13A", "0B", "Xm", "Am7"):
        with pytest.raises(ValueError):
            K.parse_key(bad)


SR = 22050


def tone(freqs, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return sum(np.sin(2 * np.pi * f * t) / len(freqs) for f in freqs)


def hz(pc, octave):
    return 440.0 * 2 ** ((pc - 9) / 12 + octave - 4)


def progression(pc, mode):
    """Bass roots and chords for I-IV-V-I (i-iv-V-i in minor), one second each."""
    third = 3 if mode == K.MINOR else 4
    degrees = [(0, third), (5, 3 if mode == K.MINOR else 4), (7, 4), (0, third)]
    bass = np.concatenate([tone([hz(pc + d, 2)]) for d, _ in degrees])
    chords = np.concatenate([tone([hz(pc + d + o, 4) for o in (0, t, 7)]) for d, t in degrees])
    return bass, chords


@pytest.mark.parametrize("pc,mode", [(9, K.MINOR), (3, K.MAJOR), (6, K.MINOR)])
def test_detects_synthetic_progressions_in_known_keys(tmp_path, pc, mode):
    bass, chords = progression(pc, mode)
    sf.write(tmp_path / "bass.wav", bass, SR)
    sf.write(tmp_path / "synths.wav", chords, SR)
    got = K.detect_key([tmp_path / "bass.wav", tmp_path / "synths.wav"])
    assert got["key"] == K.key_name(pc, mode) and got["camelot"] == K.camelot(pc, mode)
    assert got["confidence"] > 0 and got["runner_up"]["key"] != got["key"]
    assert got["stems"] == ["bass.wav", "synths.wav"]


def test_cli_reads_a_stems_folder_as_json(tmp_path):
    bass, chords = progression(9, K.MINOR)
    sf.write(tmp_path / "bass.wav", bass, SR)
    sf.write(tmp_path / "synths.wav", chords, SR)
    r = subprocess.run([sys.executable, os.path.join(HERE, "roadie_analysis", "key.py"), str(tmp_path), "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert (got["key"], got["short"], got["camelot"]) == ("A minor", "Am", "8A")
    bad = subprocess.run([sys.executable, os.path.join(HERE, "roadie_analysis", "key.py"), str(tmp_path / "nope")],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and bad.stderr.startswith("key:")


def test_files_on_the_pcloud_mount_map_to_remote_paths():
    mount = os.path.expanduser("~/pCloud Drive")
    assert K.pcloud_remote(mount + "/Music/Studies/Ben Böhmer/Title/s/bass.flac") == "pcloud:Music/Studies/Ben Böhmer/Title/s/bass.flac"
    assert K.pcloud_remote("/tmp/x/bass.flac") is None


def test_a_study_missing_a_stem_or_with_silent_stems_gives_no_key(tmp_path):
    bass, chords = progression(9, K.MINOR)
    sf.write(tmp_path / "bass.wav", bass, SR)
    cli = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "roadie_analysis", "key.py"), str(tmp_path), *a],
                                    capture_output=True, text=True)
    r = cli("--json")
    assert r.returncode == 1 and "no synths stem" in r.stderr and "--allow-partial" in r.stderr
    r = cli("--json", "--allow-partial")
    got = json.loads(r.stdout)
    assert r.returncode == 0 and got["complete"] is False and got["stems_used"] == ["bass"]
    sf.write(tmp_path / "synths.wav", chords, SR)
    got = json.loads(cli("--json").stdout)
    assert got["complete"] is True and got["stems_used"] == ["bass", "synths"]
    with pytest.raises(ValueError, match="no pitch content"):
        K.key_from_chroma([0.0] * 12)
    sf.write(tmp_path / "silent.wav", np.zeros(SR * 2), SR)
    with pytest.raises(ValueError, match="silent"):
        K.detect_key([tmp_path / "silent.wav"])
