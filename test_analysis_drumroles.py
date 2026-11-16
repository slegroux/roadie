"""roadie_analysis.drumroles.hit_groups on synthetic drum stems at 124 BPM whose answers follow from how they are built.

Copied from Sideman's tests/test_drumroles.py (branch feat/sideman-drum-roles, f877340); only the imports and this docstring differ.
Tests that take over 6 s are marked `slow`: `-m "not slow"` leaves them out, a plain run includes them.

Conventions used below: bar 0 starts at T0; a 16th is 60 / 124 / 4 s; step 0 is the downbeat, 4 / 8 / 12 are the other
beats, and the offbeats ("and" of each beat) are steps 2, 6, 10, 14, so `s % 4 == 2`. Every hit has its own noise seed
and a gain within +-2 dB.
"""
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from roadie_analysis import drumroles, oneshot  # noqa: E402

SR = 48000
BPM = 124.0
T0 = 0.25
STEP = 60.0 / BPM / 4.0
OFFBEAT = lambda s: s % 4 == 2  # noqa: E731


def noise_hit(seed, tau, db=0.0, seconds=0.6):
    """High-passed white noise (a hat) decaying with time constant `tau`, `db` dB from nominal."""
    t = np.arange(int(seconds * SR)) / SR
    return np.diff(np.random.default_rng(seed).standard_normal(len(t) + 1)) * np.exp(-t / tau) * 10 ** (db / 20)


def tom_hit(db=0.0, f=150.0, tau=0.08, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t / tau) * 10 ** (db / 20)


def snare_hit(seed, db=0.0, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    w = np.random.default_rng(seed).standard_normal(len(t))
    return (w * np.exp(-t / 0.08) * 0.4 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)) * 10 ** (db / 20)


def clap_hit(seed, db=0.0, seconds=0.6):
    """Three noise bursts 10 ms apart (each a little softer than the one before, so the hit is found at the first),
    then a tail, all band-limited by a 4-sample mean (duller than a snare's noise)."""
    t = np.arange(int(seconds * SR)) / SR
    w = np.convolve(np.random.default_rng(seed).standard_normal(len(t)), np.ones(4) / 4, mode="same")
    env = sum(a * np.exp(-np.maximum(t - b, 0) / 0.004) * (t >= b) for a, b in ((1.0, 0.0), (0.7, 0.010), (0.5, 0.020)))
    env = env + 0.3 * np.exp(-np.maximum(t - 0.02, 0) / 0.1) * (t >= 0.02)
    return w * env * 10 ** (db / 20)


def stem(events, bars, bed_db=None):
    """Mono stem [n]: `events` is [(time s, audio)]; the stem runs a second past the last bar. With `bed_db` a white
    bed `bed_db` dB under the loudest millisecond of a nominal closed hat runs under the whole stem."""
    x = np.zeros(int((T0 + bars * 16 * STEP + 1.0) * SR))
    for t, h in events:
        o = int(round(t * SR))
        m = min(len(h), len(x) - o)
        x[o:o + m] += h[:m]
    if bed_db is not None:
        nominal = oneshot._loudest_ms(noise_hit(1, 0.005)[:, None], SR)
        x += np.random.default_rng(5).standard_normal(len(x)) * nominal * 10 ** (-bed_db / 20)
    return x


def at(index, ms=0.0):
    """Time of the 16th `index` counted from bar 0 (bar * 16 + step), `ms` late."""
    return T0 + index * STEP + ms / 1000.0


def hat_events(bars, open_tau=0.025, closed_tau=0.005, rng_seed=7):
    """A longer hat (`open_tau`) on the offbeats (steps 2, 6, 10, 14), a shorter one (`closed_tau`) on every other
    16th. With the defaults the longer hat is not an open hat but a short one: at the next 16th (121 ms) it is 42 dB
    down, so the hat after it starts out of a tail well under the -30 dB the grouping accepts (the +-2 dB gains and the
    10 ms window the tail is read over cost it about 5 dB). A real open hat (tau 120 ms and more) is the H1 tests."""
    rng = np.random.default_rng(rng_seed)
    return [(at(i), noise_hit(i, open_tau if OFFBEAT(i % 16) else closed_tau, rng.uniform(-2, 2)))
            for i in range(bars * 16)]


def run(x, **kw):
    kw.setdefault("stability_runs", 0)                       # the regroupings triple the time; the stability tests ask for them
    return drumroles.hit_groups(x, BPM, T0, sr=SR, **kw)


def by_count(r):
    return sorted(r["groups"], key=lambda g: -g["count"])


def truth_label(h):
    return "open" if OFFBEAT(h["step"]) else "closed"


# ---------------------------------------------------------------- the hat stem

@pytest.fixture(scope="module")
def hat_stem():
    return stem(hat_events(8), 8)


@pytest.fixture(scope="module")
def hats(hat_stem):
    return run(hat_stem)


def test_a_short_hat_on_the_offbeats_and_a_shorter_one_on_the_rest_are_two_groups(hats):
    r = hats
    assert len(r["hits"]) == 8 * 16 and all(h["status"] == "grouped" for h in r["hits"]), r["unmeasured"]
    closed, open_ = r["groups"]                      # ordered by count: 96 closed, 32 open
    assert (closed["count"], open_["count"]) == (96, 32)
    assert closed["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert open_["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    assert closed["bars_with"] == open_["bars_with"] == 8 and closed["off_grid"] == open_["off_grid"] == 0
    for h in r["hits"]:
        assert (h["bar"], h["step"]) == divmod(h["id"], 16) and abs(h["offset_ms"]) < 1, h
        assert h["group"] == (open_ if OFFBEAT(h["step"]) else closed)["id"]
    assert r["unmeasured"] == [] and r["linkage"] == "average"


def test_hats_group_spread_is_the_noise_realisation_and_under_the_threshold(hats, hat_stem):
    # members are within the threshold of their medoid by construction of the grouping; two takes of one noise hat read
    # 0.5-0.8 (the reviews), so the spread is neither zero nor near the threshold
    for g in hats["groups"]:
        assert 0.2 < g["spread"]["median"] < g["spread"]["max"] <= drumroles.THRESHOLD, g["spread"]
    closed, open_ = (hats["hits"][i] for i in (0, 2))                  # step 0 (closed) and step 2 (open) of bar 0
    d = oneshot.spectral_distance(hat_stem[closed["start"]:closed["start"] + int(STEP * SR) - 100, None],
                                  hat_stem[open_["start"]:open_["start"] + int(STEP * SR) - 100, None], SR)
    assert d > 1.5 * drumroles.THRESHOLD, d           # closed against open: well apart (the reviews: 2.5 at the least)


def test_hats_levels_and_the_clean_hit(hats, hat_stem):
    closed, open_ = hats["groups"]
    # levels were built within +-2 dB of one another; the median of a group sits inside that band
    assert abs(closed["level_db"] - open_["level_db"]) < 4.0
    # a closed hat falls 40 dB (tau 5 ms: in 23 ms) inside its 121 ms: it has a clean hit, and it is a closed hat, cut as reported
    c = closed["clean_hit"]
    assert c is not None and c["hit"] in closed["members"]
    assert len(c["audio"]) == c["end"] - c["start"] <= int(drumroles.MAX_CUT_S * SR)
    assert np.array_equal(c["audio"][:, 0], hat_stem[c["start"]:c["end"]])
    assert abs(c["start"] / SR - hats["hits"][c["hit"]]["onset_s"]) < 1e-9
    assert open_["clean_hit"] is None or open_["clean_hit"]["hit"] in open_["members"]


def test_open_hats_with_a_rest_after_them_have_a_clean_hit_that_falls_40_db():
    # a hat of tau 100 ms falls 40 dB in 460 ms; the next hit is 4 sixteenths (484 ms) away
    r = run(stem([(at(i), noise_hit(i, 0.1)) for i in range(2, 64, 4)], 4))
    g, = r["groups"]
    c = g["clean_hit"]
    assert g["count"] == 16 and c is not None and c["hit"] in g["members"] and g["clean_note"] is None
    assert 0.46 * SR <= len(c["audio"]) <= drumroles.MAX_CUT_S * SR
    # its tail (-10.5, -21, -31.5 dB at the next three 16th lines, -42 at the fourth, where the next hit is) sounds through 3 lines
    assert all(r["hits"][i]["through"] == 3 for i in g["members"])
    # so the lines 1-3 after each hat (steps 3-5, 7-9, 11-13, 15-1) are inside a tail with no hit; the hats' own lines are not;
    # the steps 0 and 1 of bar 0 have no earlier hat (bars 1-3 of 4: 0.75)
    assert r["hidden_pattern"] == [0.75, 0.75, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------- one sound, and where it falls

def test_one_repeated_tom_is_one_group():
    rng = np.random.default_rng(3)
    x = stem([(at(i), tom_hit(rng.uniform(-2, 2))) for i in range(0, 4 * 16, 4)], 4)
    r = run(x)
    assert len(r["hits"]) == 16 and len(r["groups"]) == 1 and r["groups"][0]["count"] == 16
    assert r["groups"][0]["pattern"] == [1.0 if s % 4 == 0 else 0.0 for s in range(16)]
    assert r["groups"][0]["clean_hit"] is not None       # a tom falls 40 dB (tau 150 ms) inside its 484 ms to the next


def test_the_measurement_rules_need_at_least_three_complete_members_read():
    # a group with 1 or 2 complete members is not named from them, however well they agree; with 3 it is. (A result with no
    # `read_members` at all has only the clean hit to read, and the other tests name from that.)
    for n, want in ((1, "snare 1"), (2, "snare 1"), (3, "snare")):
        r = fake_result([{"steps": [4, 12], "audio": snare_hit(1), "read": [snare_hit(i) for i in range(n)]}])
        got, out = names(r, "snare")
        assert got == [want], (n, got)
        if n < 3:
            assert "too few complete members: %d read" % n in out["groups"][0]["reason"] and "at least 3" in out["groups"][0]["reason"]
    # the other kinds: 2 pitched members in a toms stem, 2 open hats in an hh stem that no pattern rule fits
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0), "read": [tom_hit(f=150.0)] * 2}]), "toms")[0] == ["toms 1"]
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0), "read": [tom_hit(f=150.0)] * 3}]), "toms")[0] == ["tom"]
    assert names(fake_result([{"steps": [0, 8], "audio": noise_hit(1, 0.1), "read": [noise_hit(1, 0.1)] * 2}]), "hh")[0] == ["hh 1"]
    assert names(fake_result([{"steps": [0, 8], "audio": noise_hit(1, 0.1), "read": [noise_hit(1, 0.1)] * 3}]), "hh")[0] == ["hh open"]


def test_each_group_keeps_up_to_eight_complete_members_nearest_the_medoid_for_the_role_rules(hats):
    # 16 offbeat hats and 48 closed ones, every one complete (each falls 40 dB well before the next 16th): 8 of each are kept,
    # members of the group, nearest its medoid first, the clean hit one of them
    for g in hats["groups"]:
        ids = [m["hit"] for m in g["read_members"]]
        d = [hats["hits"][i]["distance"] for i in ids]
        assert len(ids) == 8 == len(set(ids)) and set(ids) <= set(g["members"]) and d == sorted(d)
        assert g["clean_hit"]["hit"] in ids and all(len(m["audio"]) > 0 for m in g["read_members"])


def test_a_hit_8_ms_early_and_one_12_ms_late_keep_their_16th_and_say_how_far_off_they_are():
    events = [(at(i), tom_hit()) for i in range(0, 64, 4)]
    events[6] = (at(24, -8.0), tom_hit())                  # bar 1, beat 3 (step 8), 8 ms early
    events[10] = (at(40, 12.0), tom_hit())                 # bar 2, beat 3, 12 ms late
    events[14] = (at(56, 30.0), tom_hit())                 # bar 3, beat 3, 30 ms late: past the 20 ms the grid allows
    r = run(stem(events, 4))
    h = r["hits"]
    assert [(x["bar"], x["step"]) for x in h] == [(i // 4, 4 * (i % 4)) for i in range(16)]
    assert h[6]["offset_ms"] == pytest.approx(-8.0, abs=1.0) and h[10]["offset_ms"] == pytest.approx(12.0, abs=1.0)
    assert h[14]["offset_ms"] == pytest.approx(30.0, abs=1.0)
    assert h[6]["on_grid"] and h[10]["on_grid"] and not h[14]["on_grid"]
    assert all(abs(x["offset_ms"]) < 1.0 for i, x in enumerate(h) if i not in (6, 10, 14))
    # one sound in one group; the 30 ms hit is in it but counts in no pattern (3 of 4 bars at step 8)
    g, = r["groups"]
    assert g["count"] == 16 and g["off_grid"] == 1
    assert g["pattern"][8] == pytest.approx(0.75) and g["pattern"][0] == 1.0


def test_clap_and_snare_on_beats_2_and_4_are_two_groups():
    events = []
    for b in range(8):
        events += [(at(16 * b + 4), clap_hit(100 + b, (b % 3 - 1) * 1.5)), (at(16 * b + 12), snare_hit(200 + b, (b % 3 - 1) * 1.5))]
    r = run(stem(events, 8))
    assert len(r["groups"]) == 2 and all(g["count"] == 8 for g in r["groups"])
    by_step = {tuple(i for i, p in enumerate(g["pattern"]) if p): g for g in r["groups"]}
    assert set(by_step) == {(4,), (12,)}                  # clap on beat 2, snare on beat 4, nothing else
    assert all(g["pattern"][s] == 1.0 for (s,), g in by_step.items())
    assert all(g["clean_hit"] is not None for g in r["groups"])
    assert by_step[(4,)]["id"] != by_step[(12,)]["id"]


# ---------------------------------------------------------------- a bleed bed

def test_a_bed_40_db_under_the_hits_does_not_change_the_groups(hats):
    r = run(stem(hat_events(4), 4, bed_db=40.0))
    assert len(r["hits"]) == 64 and all(h["status"] == "grouped" for h in r["hits"]), r["unmeasured"]
    closed, open_ = r["groups"]
    assert (closed["count"], open_["count"]) == (48, 16)
    assert closed["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert open_["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    assert all(h["group"] == (open_ if OFFBEAT(h["step"]) else closed)["id"] for h in r["hits"])
    assert all(abs(h["offset_ms"]) < 1 for h in r["hits"])      # the first onset is not pulled onto the bed


def test_a_bed_30_db_under_the_hits_gives_fewer_groups_never_wrong_ones():
    # the hits start out of a bed only 30 dB under them (a lead-in of -30 dB or louder is "overlapped"): the hits that
    # are measured may still be grouped, and each one must be in a group of its own kind; the rest are reported
    r = run(stem(hat_events(4), 4, bed_db=30.0))
    assert {h["status"] for h in r["hits"]} <= {"grouped", "assigned", "overlapped", "faint", "no_lead"}
    assert sum(h["status"] in ("grouped", "assigned") for h in r["hits"]) + len(r["unmeasured"]) == len(r["hits"])
    for g in r["groups"]:
        assert len({truth_label(r["hits"][i]) for i in g["members"] + g["assigned_members"]}) == 1, g["pattern"]
    assert len(r["groups"]) <= 2
    assert all(u["reason"] and u["status"] in ("overlapped", "faint", "no_lead") for u in r["unmeasured"])


def test_a_bed_20_db_under_the_hits_measures_nothing_and_says_why():
    r = run(stem(hat_events(4), 4, bed_db=20.0))
    assert r["groups"] == [] and all(h["group"] is None for h in r["hits"])
    assert {h["status"] for h in r["hits"]} <= {"overlapped", "faint", "no_lead"}
    assert sum(h["status"] == "overlapped" for h in r["hits"]) >= 60
    assert all("dB" in h["reason"] or "start of the stem" in h["reason"] for h in r["hits"])


# ---------------------------------------------------------------- order, sections, cost

def test_the_order_the_hits_come_in_does_not_change_the_groups():
    ev = hat_events(2)
    cuts = [h[:int(STEP * SR)][:, None] for _, h in ev]      # each hat as far as the next 16th
    keys = [t for t, _ in ev]

    def partition(order, sample):
        groups, rest = drumroles.group_cuts([cuts[i] for i in order], [keys[i] for i in order], SR, sample=sample)
        assert rest == []
        return sorted(sorted(keys[order[i]] for i in g["members"]) for g in groups)
    base = partition(list(range(len(cuts))), 6)           # 32 hits against a sample of 6: sample, assign, repeat
    assert [len(g) for g in sorted(base, key=len)] == [8, 24]
    for seed in (1, 2, 3):
        order = list(np.random.default_rng(seed).permutation(len(cuts)))
        assert partition(order, 6) == base
    assert partition(list(range(len(cuts))), 32) == base  # and the sample does not matter here


def test_a_sound_the_sample_misses_still_gets_its_group():
    # 63 closed hats and one open hat among them (bar 2, step 6); a sample of 8 evenly spaced hits does not include the
    # open one, which is further than the threshold from the closed medoid and so goes round again as a group of its own
    ev = [(at(i), noise_hit(i, 0.025 if i == 38 else 0.005)) for i in range(64)]
    r = run(stem(ev, 4), sample=8)
    assert [g["count"] for g in r["groups"]] == [63, 1]
    assert r["groups"][1]["members"] == [38] and r["groups"][1]["pattern"][38 % 16] == 0.25


def plain(groups):
    """Groups without their audio (arrays do not compare)."""
    return [{k: v for k, v in g.items() if k not in ("clean_hit", "read_members")}
            | {"clean": g["clean_hit"] and g["clean_hit"]["hit"], "read": [m["hit"] for m in g["read_members"]]} for g in groups]


def test_sections_are_the_same_groups_restricted_and_their_order_does_not_matter():
    a, b = {"name": "a", "start_bar": 0, "bars": 2}, {"name": "b", "start_bar": 2, "bars": 2}
    # section a: closed hats only; section b: the full pattern
    ev = [(at(i), noise_hit(i, 0.005)) for i in range(32)] + hat_events(4)[32:]
    x = stem(ev, 4)
    r1, r2 = run(x, sections=[a, b]), run(x, sections=[b, a])
    s1, s2 = ({s["name"]: s for s in r["sections"]} for r in (r1, r2))
    assert plain(r1["groups"]) == plain(r2["groups"])
    for name in "ab":
        assert plain(s1[name]["groups"]) == plain(s2[name]["groups"]) and s1[name]["unmeasured"] == s2[name]["unmeasured"]
    assert [g["count"] for g in s1["a"]["groups"]] == [32]            # 2 bars of closed hats: one group
    assert s1["a"]["groups"][0]["pattern"] == [1.0] * 16
    assert [g["count"] for g in s1["b"]["groups"]] == [24, 8]
    assert s1["b"]["groups"][0]["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert s1["b"]["groups"][1]["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    ids = {g["id"] for g in r1["groups"]}
    assert ids == {0, 1} and {g["id"] for sec in r1["sections"] for g in sec["groups"]} == ids   # one id per sound in every section


# ---------------------------------------------------------------- what cannot be measured

def test_a_grace_note_55_ms_before_a_beat_is_too_short_and_the_hit_after_it_is_assigned():
    events = [(at(i), tom_hit()) for i in range(0, 32, 4)]
    events.append((at(16) - 0.055, tom_hit(-12)))          # a grace note 55 ms before the beat at bar 1
    r = run(stem(events, 2))
    grace, main = (x for x in r["hits"] if abs(x["onset_s"] - at(16)) < 0.1)
    assert abs(grace["onset_s"] - (at(16) - 0.055)) < 0.001 and grace["status"] == "too_short" and "55 ms" in grace["reason"]
    assert grace["group"] is None and grace["bar"] == 1 and grace["step"] == 0 and grace["offset_ms"] == pytest.approx(-55.0, abs=1)
    # the hit after it starts out of the grace note's tail (-12 dB, 55 ms later -6 dB more, so -18 dB under it: assigned over
    # its first 40 ms to the tom group, flagged as assigned, and not one of its measured members)
    assert main["status"] == "assigned" and "overlapped" in main["reason"] and main["group"] == r["groups"][0]["id"]
    g = r["groups"][0]
    assert g["count"] == 7 and g["assigned"] == 1 and main["id"] in g["assigned_members"] and main["id"] not in g["members"]
    assert g["assigned_pattern"] == [0.5 if i == 0 else 0.0 for i in range(16)] and g["pattern"][0] == 1.0
    assert r["unmeasured_pattern"] == [0.0] * 16           # the grace note is off the grid, the main hit has a group


def test_a_stem_read_from_a_file_and_the_inputs_that_are_refused(tmp_path):
    x = stem([(at(i), tom_hit()) for i in range(0, 32, 4)], 2)
    sf.write(str(tmp_path / "toms.wav"), x, SR, subtype="FLOAT")
    from_file = drumroles.hit_groups(tmp_path / "toms.wav", BPM, T0)
    from_array = run(x[:, None])
    assert from_file["sample_rate"] == SR and len(from_file["hits"]) == 8
    assert [g["pattern"] for g in from_file["groups"]] == [g["pattern"] for g in from_array["groups"]]
    assert drumroles.hit_groups(np.zeros(SR), BPM, T0, sr=SR)["hits"] == []
    for kw in ({"tempo": 0}, {"t0": float("nan")}, {"sections": [{"start_bar": 0}]}):
        with pytest.raises(ValueError):
            drumroles.hit_groups(x, **{"tempo": BPM, "t0": T0, "sr": SR, **kw})
    with pytest.raises(ValueError, match="sample rate"):
        drumroles.hit_groups(x, BPM, T0)


def test_average_linkage_does_not_chain():
    # points 0, 1.4 and 2.8 on a line: single linkage would chain all three at a threshold of 1.5; average linkage
    # merges the first two (1.4) and then finds the third 2.1 away
    p = np.array([0.0, 1.4, 2.8, 10.0, 10.5])
    clusters = drumroles._agglomerate(np.abs(p[:, None] - p[None, :]), 1.5)
    assert sorted(sorted(c) for c in clusters) == [[0, 1], [2], [3, 4]]


# ---------------------------------------------------------------- H1: 16th hats under a real open hat

def kinds_of(r, played, tol=0.015):
    """({hit id: kind of the played hat it is}, hits that are none of them) for `played` = [(time s, kind)]."""
    on = np.array([h["onset_s"] for h in r["hits"]])
    out = {}
    for t, kind in played:
        k = int(np.argmin(np.abs(on - t)))
        if abs(on[k] - t) <= tol:
            out[k] = kind
    return out, len(r["hits"]) - len(out)


@pytest.mark.parametrize("tau,grid,least_closed", [(0.12, False, 45), (0.12, True, 61), (0.2, False, 14), (0.2, True, 29)])
def test_16th_hats_under_a_real_open_hat_are_placed_in_the_hat_group_or_reported(tau, grid, least_closed):
    # Open hats (tau 120 and 200 ms) on the offbeats, closed hats (tau 12 ms) on the 96 other 16ths, its own seed and +-2 dB
    # each. The closed hat after an open one starts out of a tail 9 dB (tau 120) or 14 dB (200) under it, which the detector
    # does not see as a hit (`grid_onsets` looks for it at the 16th line), and where it is found its tail makes it
    # "overlapped": it is assigned to the nearest existing group, over a window that shrinks with the tail. The review
    # measured, with both: 61-63 closed hats placed (tau 120) and 29-33 (tau 200), where there were 2 before; the numbers
    # asserted for A off are what this code placed on these stems (51 and 16), less a hat or two. No hat in the wrong group.
    ev = hat_events(8, open_tau=tau, closed_tau=0.012)
    played = [(t, "open" if OFFBEAT(i % 16) else "closed") for i, (t, _) in enumerate(ev)]
    r = run(stem(ev, 8), grid_onsets=grid)
    kinds, extra = kinds_of(r, played)
    placed = {k: sum(r["hits"][i]["status"] in ("grouped", "assigned") for i, kd in kinds.items() if kd == k)
              for k in ("open", "closed")}
    assert placed["open"] == 32 and placed["closed"] >= least_closed, placed
    assert extra == 0                                       # no hit that was not played, with `grid_onsets` or without
    for g in r["groups"]:                                   # nothing in a group of the other kind
        assert len({kinds[i] for i in g["members"] + g["assigned_members"]}) == 1, g
    g, = [g for g in r["groups"] if kinds[(g["members"] + g["assigned_members"])[0]] == "open"]
    assert g["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)] and g["bars_with"] == 8
    assert g["count"] + g["assigned"] == 32
    assert all(h["status"] == "assigned" and h["distance"] <= drumroles.THRESHOLD and "overlapped" in h["reason"]
               for h in r["hits"] if h["group"] is not None and h["status"] != "grouped")
    assert g["clean_hit"] is None or g["clean_hit"]["hit"] in g["members"]       # an assigned hit is never the clean hit
    # the lines inside a tail where no hit was found: the open hats' own lines never; the line right after an open hat
    # (steps 3, 7, 11, 15: its tail is 9 dB down there) always without the search, less so with it
    assert [r["hidden_pattern"][s] for s in (2, 6, 10, 14)] == [0.0] * 4
    after = [r["hidden_pattern"][s] for s in (3, 7, 11, 15)]
    assert (max(after) < 1.0) if grid else (after == [1.0] * 4), after
    assert r["grid_hits"] == sum(h["source"] == "grid" for h in r["hits"]) == (len(r["hits"]) - sum(
        h["source"] == "segment" for h in r["hits"]))
    assert (r["grid_hits"] > 0) == grid


# ---------------------------------------------------------------- the grid-line detector on stems that are not hats

def bleed_stem(kind):
    """2 bars of a tom (every beat) or a kick (every beat), with a hat (every 16th) or a snare-like burst (the offbeats)
    bleeding in 45 dB under it: under the 40 dB `_segments` reads, over what the grid detector's 6 dB rise asks."""
    if kind == "tom":
        main = [(at(i), tom_hit(f=150.0, tau=0.08)) for i in range(0, 32, 4)]
        bleed = [(at(i), noise_hit(500 + i, 0.01, -45.0, 0.1)) for i in range(32)]
        lines = [i for i in range(32) if i % 4 != 0]             # 16th lines with no tom within 50 ms
    else:
        main = [(at(i), tom_hit(f=45.0, tau=0.10)) for i in range(0, 32, 4)]
        bleed = [(at(i), noise_hit(500 + i, 0.05, -45.0, 0.3)) for i in range(2, 32, 4)]
        lines = [i for i in range(32) if i % 4 == 2]
    return stem(main + bleed, 2), lines


@pytest.mark.parametrize("kind", ["tom", "kick"])
def test_grid_onsets_on_a_tom_or_kick_stem_count_the_bleed_as_hits_and_say_so(kind):
    x, lines = bleed_stem(kind)
    off, on = run(x), run(x, grid_onsets=True)
    assert len(off["hits"]) == 8 and off["grid_hits"] == 0 and not off["grid_onsets"]       # the beats of the drum
    # with the search every bleed hit at a free line is a hit: 24 hat bleeds (12 of the 16 lines a bar) on the tom stem,
    # 8 burst bleeds (the offbeats) on the kick stem. The kick stem gives 8 more: the line after each kick, where
    # the kick's own 45 Hz tail is found as a hit, because the first-difference power of a low tone swings with its
    # phase by more than the 6 dB the search asks for (its 22 ms period against the 3 and 5 ms windows).
    grid = [h for h in on["hits"] if h["source"] == "grid"]
    expected = lines if kind == "tom" else sorted(lines + [i - 1 for i in lines])
    assert on["grid_onsets"] and on["grid_hits"] == len(grid) == len(expected) == len(on["hits"]) - 8, (on["grid_hits"], len(expected))
    assert sorted(h["bar"] * 16 + h["step"] for h in grid) == expected and all(abs(h["offset_ms"]) < 20 for h in grid)
    # and the stem is not hat-like, which is what the option is for: nearly all of its energy lies under 4 kHz
    assert off["hat_like"] < 0.01 and on["hat_like"] == off["hat_like"]
    assert any("meant for hat-like stems" in n and "hat_like 0.00" in n for n in on["notes"])
    assert not any("meant for hat-like stems" in n for n in off["notes"])


def test_grid_onsets_on_the_hat_stem_find_only_hats_and_the_stem_reads_hat_like(hat_stem):
    # the same stem as the first test, the search on: nothing is added (every 16th has a hit), and the stem is hat-like
    r = run(hat_stem, grid_onsets=True)
    assert r["grid_hits"] == 0 and len(r["hits"]) == 128 and r["hat_like"] > 0.9
    assert not any("meant for hat-like stems" in n for n in r["notes"])


# ---------------------------------------------------------------- pitch: toms an octave or a fifth apart

@pytest.mark.parametrize("semitones", [12.0, 7.0])
def test_two_toms_an_octave_or_a_fifth_apart_are_two_groups(semitones):
    # beats 1 and 3 a tom at 150 Hz, beats 2 and 4 the same drum `semitones` up: the spectral distance alone puts toms an
    # octave apart in one group (review: 0.65 / 1.06 / 1.55 for 7 / 12 / 19 semitones on a 100 Hz tom); their body pitches
    # are clusters 7 and 12 semitones apart
    hi = 150.0 * 2 ** (semitones / 12)
    rng = np.random.default_rng(4)
    ev = [(at(i), tom_hit(rng.uniform(-2, 2), f=150.0 if (i // 4) % 2 == 0 else hi)) for i in range(0, 48, 4)]
    r = run(stem(ev, 3))
    lo_g, hi_g = sorted(r["groups"], key=lambda g: g["pattern"].index(1.0))
    assert len(r["groups"]) == 2 and lo_g["count"] == hi_g["count"] == 6
    assert [s for s, p in enumerate(lo_g["pattern"]) if p] == [0, 8] and [s for s, p in enumerate(hi_g["pattern"]) if p] == [4, 12]
    assert lo_g["pattern"][0] == lo_g["pattern"][8] == hi_g["pattern"][4] == hi_g["pattern"][12] == 1.0


def test_one_tom_with_a_third_of_a_semitone_of_jitter_stays_one_group():
    rng = np.random.default_rng(5)
    ev = [(at(i), tom_hit(rng.uniform(-2, 2), f=150.0 * 2 ** (rng.uniform(-0.3, 0.3) / 12))) for i in range(0, 48, 4)]
    r = run(stem(ev, 3))
    assert len(r["groups"]) == 1 and r["groups"][0]["count"] == 12


# ---------------------------------------------------------------- stability, tentative groups

@pytest.mark.slow
def test_well_separated_sounds_are_stable_and_a_decay_continuum_is_not():
    r = run(stem(hat_events(4), 4), stability_runs=2)
    assert [g["stability"] for g in r["groups"]] == [1.0, 1.0] and not any(g["tentative"] for g in r["groups"])
    # hats of tau 6 to 30 ms in 2 ms steps in a shuffled order: distances run on from 0.5 to 3 with no gap, so the
    # cut lies where the sample puts it and a regrouping with the sample moved does not find the same groups
    taus = np.random.default_rng(6).permutation(np.tile(np.arange(6, 31, 2) * 1e-3, 5))[:64]
    ev = [(at(i), noise_hit(i, float(tau))) for i, tau in enumerate(taus)]
    c = run(stem(ev, 4), stability_runs=2)
    measured = [g for g in c["groups"] if not g["tentative"]]
    assert measured and min(g["stability"] for g in measured) < 0.8, [(g["count"], g["stability"]) for g in c["groups"]]
    assert all(0.0 <= g["stability"] <= 1.0 for g in measured)
    assert all(g["stability"] is None for g in c["groups"] if g["tentative"])            # a tentative group has none
    # separation: the mean distance to the nearest other group's members. Separated sounds lie well over the threshold
    # from each other (closed against open 2.9 here, each the other's nearest); a continuum has groups that touch
    assert [g["nearest_group"] for g in r["groups"]] == [1, 0]
    assert all(g["separation"] > 1.4 * drumroles.THRESHOLD for g in r["groups"])
    assert min(g["separation"] for g in measured) < 1.4 * drumroles.THRESHOLD, [(g["count"], g["separation"]) for g in measured]
    assert all(g["nearest_group"] != g["id"] for g in c["groups"] if g["nearest_group"] is not None)
    # with no more measured hits than the sample the sample cannot move: no figure, a note, and no regrouping
    short = run(stem([(at(i), noise_hit(i, float(tau))) for i, tau in enumerate(taus[:32])], 2), stability_runs=2)
    assert all(g["stability"] is None for g in short["groups"]) and any("cannot move" in n for n in short["notes"])
    assert run(stem(hat_events(2), 2))["groups"][0]["stability"] is None             # not measured when not asked for


def test_a_tentative_satellite_is_not_the_neighbour_separation_is_measured_against():
    # 16 closed hats (tau 5 ms) on the beats, 16 longer ones (25 ms) on the offbeats, and 2 hats of tau 12.5 ms (1.63 from the
    # closed ones, 1.69 from the longer ones) at steps 1 and 9 of bar 0. The pair is a group of 2 (tentative): the nearest group
    # of the closed hats by medoid, but not a neighbour for `separation`, which is measured against the longer hats (2.9 away)
    ev = [(at(i), noise_hit(i, 0.005)) for i in range(0, 64, 4)] + [(at(i), noise_hit(100 + i, 0.025)) for i in range(2, 64, 4)]
    ev += [(at(1), noise_hit(200, 0.0125)), (at(9), noise_hit(201, 0.0125))]
    r = run(stem(ev, 4))
    closed, = [g for g in r["groups"] if g["pattern"][0] > 0]                 # the beats
    longer, = [g for g in r["groups"] if g["pattern"][2] > 0]                 # the offbeats
    sat, = [g for g in r["groups"] if g["tentative"]]
    assert (closed["count"], longer["count"]) == (16, 16) and len(r["groups"]) == 3
    assert sat["tentative"] and sat["count"] == 2 and not closed["tentative"] and not longer["tentative"]
    assert closed["nearest_any"] == sat["id"] and closed["separation_any"] < 2.0           # the satellite is the nearest of all
    assert closed["nearest_group"] == longer["id"] and closed["separation"] > 2.5          # the nearest that counts is the other big group
    assert longer["nearest_group"] == closed["id"] and longer["separation"] > 2.5
    assert sat["nearest_group"] in (closed["id"], longer["id"]) and sat["separation"] is not None


def test_a_group_of_under_4_hits_is_tentative_and_has_no_stability():
    # 30 short hats and 2 longer ones, a sample of 8 (it can move, 32 hits against 8): the pair is tentative, with no figure
    ev = [(at(i), noise_hit(i, 0.025 if i in (10, 26) else 0.005)) for i in range(32)]
    r = run(stem(ev, 2), sample=8, stability_runs=2)
    assert [(g["count"], g["tentative"]) for g in r["groups"]] == [(30, False), (2, True)]
    assert r["groups"][0]["stability"] is not None and r["groups"][1]["stability"] is None


# ---------------------------------------------------------------- review round 1: tests that pin the clustering

def test_a_decay_continuum_shuffled_three_times_gives_one_partition():
    # hats of tau 6 to 30 ms (distances 0.5 to 3, no gap): the groups depend on where the sample falls, so they only agree
    # when the hits are put in time order first and the same hits are sampled whatever order they came in
    taus = np.random.default_rng(8).permutation(np.tile(np.arange(6, 31, 2) * 1e-3, 3))[:32]
    cuts = [noise_hit(i, float(tau))[:int(STEP * SR), None] for i, tau in enumerate(taus)]

    def partition(order):
        groups, rest = drumroles.group_cuts([cuts[i] for i in order], [order[i] for i in range(len(order))], SR, sample=8)
        return sorted(sorted(order[i] for i in g["members"]) for g in groups), sorted(order[i] for i in rest)
    base = partition(list(range(32)))
    assert 1 < len(base[0]) < 8                              # the continuum is cut into a few groups, not one and not all
    for seed in (1, 2, 3):
        order = [int(i) for i in np.random.default_rng(seed).permutation(32)]
        assert partition(order) == base


def test_the_medoid_is_the_most_central_hit_not_the_first_and_the_outlier_is_the_furthest():
    # nine hats of one kind in one group, the earliest of them a little different (tau 20 ms against 12: about 1 away,
    # inside the threshold): the medoid is one of the others and the outlier is the member furthest from it
    cuts = [noise_hit(i, 0.02 if i == 0 else 0.012)[:int(STEP * SR), None] for i in range(9)]
    groups, rest = drumroles.group_cuts(cuts, list(range(9)), SR)
    g, = groups
    assert rest == [] and sorted(g["members"]) == list(range(9))
    assert g["medoid"] != 0 and g["distance"][0] == max(g["distance"].values()) and g["distance"][0] > 0.8


def test_a_hit_25_db_under_its_neighbours_is_faint_and_is_counted_where_it_fell():
    # a ghost note on a hat stem: 31 hats and one 25 dB under them (bar 0, step 13). It is not grouped (it may be a
    # bed or bleed), it is not dropped: the stem says where it fell (`unmeasured_pattern`) and why
    ev = [(at(i), noise_hit(i, 0.005, -25.0 if i == 13 else 0.0)) for i in range(32)]
    r = run(stem(ev, 2))
    ghost = r["hits"][13]
    assert ghost["status"] == "faint" and ghost["group"] is None and "ghost" in ghost["reason"]
    assert all(h["status"] == "grouped" for h in r["hits"] if h["id"] != 13) and r["groups"][0]["count"] == 31
    assert r["unmeasured_pattern"] == [0.5 if s == 13 else 0.0 for s in range(16)]
    assert r["unmeasured"] == [{"hit": 13, "status": "faint", "reason": ghost["reason"]}]
    assert r["groups"][0]["pattern"][13] == 0.5


def test_more_sounds_than_max_groups_leave_the_rest_ungrouped_and_say_so():
    # a hat on beats 1 and 3, a tom on beat 2, a snare on beat 4, 3 bars: three sounds, two groups allowed
    ev = []
    for b in range(3):
        ev += [(at(16 * b), noise_hit(b, 0.005)), (at(16 * b + 4), tom_hit()), (at(16 * b + 8), noise_hit(10 + b, 0.005)),
               (at(16 * b + 12), snare_hit(20 + b))]
    x = stem(ev, 3)
    r3 = run(x)
    assert len(r3["groups"]) == 3
    # separation: the tom (beat 2) and the snare (beat 4) are each other's nearest group (1.7 apart), the hats are 5.7 from
    # the tom: a group's nearest is the group with the nearest medoid, not the farthest. That is `nearest_any` / `separation_any`.
    # The tom and the snare have 3 hits each (tentative), so a neighbour that counts for `separation` is the hat group (6
    # hits), 5.7 away; the hats have no neighbour of 4 hits or more: separation None
    by_step = {[st for st, p in enumerate(g["pattern"]) if p][0]: g for g in r3["groups"]}
    assert by_step[4]["nearest_any"] == by_step[12]["id"] and by_step[12]["nearest_any"] == by_step[4]["id"]
    assert by_step[0]["nearest_any"] in (by_step[4]["id"], by_step[12]["id"]) and by_step[0]["separation_any"] > 3 * by_step[4]["separation_any"]
    assert by_step[4]["separation_any"] < 2.5
    assert by_step[4]["nearest_group"] == by_step[12]["nearest_group"] == by_step[0]["id"] and by_step[4]["separation"] > 2.5
    assert by_step[0]["nearest_group"] is None and by_step[0]["separation"] is None
    r = run(x, max_groups=2)
    left = [h for h in r["hits"] if h["status"] == "ungrouped"]
    assert len(r["groups"]) == 2 and left and "not forced" in left[0]["reason"]
    assert len({h["step"] for h in left}) <= 2 and {h["id"] for h in left}.isdisjoint(i for g in r["groups"] for i in g["members"])
    assert sum(g["count"] for g in r["groups"]) + len(left) == len(r["hits"]) == 12


@pytest.mark.slow
def test_bars_0_to_3_alone_and_inside_16_bars_give_the_same_groups_on_separated_sounds():
    big = run(stem(hat_events(16), 16), sections=[{"name": "a", "start_bar": 0, "bars": 4}])
    small = run(stem(hat_events(4), 4))
    keep = lambda groups: [(g["count"], g["pattern"], g["bars_with"], g["off_grid"], g["tentative"]) for g in groups]  # noqa: E731
    assert keep(big["sections"][0]["groups"]) == keep(small["groups"])
    assert [g["count"] for g in big["groups"]] == [192, 64] and [g["count"] for g in small["groups"]] == [48, 16]


def test_an_overlapped_hit_between_two_alike_groups_is_not_assigned_and_one_next_to_a_group_is():
    # _assign_overlapped on hand-made hits: group A's medoid a hat of tau 5 ms, group B's of tau 7 ms (about 0.9 apart), the
    # overlapped hit one of tau 6 ms: it is within the threshold of both and not 15 % nearer to either, so it stays
    # overlapped, with the numbers in its reason; against a group of tau 40 ms (far) it goes to the group of tau 5 ms
    def hat(i, tau):
        return noise_hit(i, tau)[:int(STEP * SR)]
    parts = [hat(1, 0.005), hat(2, 0.007), hat(3, 0.006), hat(4, 0.04)]
    x = np.concatenate([np.concatenate([p, np.zeros(SR // 10)]) for p in parts])[:, None]
    starts = np.cumsum([0] + [len(p) + SR // 10 for p in parts])[:4]
    hits = [{"id": i, "start": int(a), "end": int(a) + len(p), "status": "grouped", "reason": None, "lead_db": None,
             "group": None, "distance": None} for i, (a, p) in enumerate(zip(starts, parts))]
    hits[2].update(status="overlapped", reason="it starts out of a tail or bed only 20 dB under its own loudest ms (it needs 30)",
                   lead_db=-20.0)

    def assign(medoids):
        hits[2].update(group=None, distance=None)
        hits[2]["status"], hits[2]["reason"] = "overlapped", "it starts out of a tail or bed only 20 dB under its own loudest ms (it needs 30)"
        found = [(gid, {"medoid": m, "members": [m] * drumroles.TENTATIVE_N, "assigned": []}) for gid, m in enumerate(medoids)]
        drumroles._assign_overlapped(x, SR, hits, found, drumroles.THRESHOLD)
        return hits[2]["status"], hits[2]["reason"], found
    status, reason, _ = assign([0, 1])
    assert status == "overlapped" and "not assigned" in reason and "groups 0 and 1" in reason, reason
    status, reason, found = assign([0, 3])
    assert status == "assigned" and hits[2]["group"] == 0 and found[0][1]["assigned"] == [2] and "first 40 ms" in reason, reason


def test_a_lone_hit_on_a_tail_does_not_attract_the_hits_of_a_real_group():
    # a real group (6 clean hats, medoid 0, rose out of silence) and a lone hit heard over a noise tail 36 dB under it;
    # two overlapped hats over the same kind of tail, 29 dB under them. Their tail makes them nearer the lone hit than
    # the clean group, which is the tail's doing: a group of fewer than 4 hits takes assigned hits only when its medoid
    # rose out of near silence (-40 dB or under), so the real group gets them; with a clean medoid the lone one takes them
    n = int(STEP * SR)
    tail = noise_hit(77, 0.12)[:n]

    def over(seed, db):
        h = noise_hit(seed, 0.012)[:n]
        g = oneshot._loudest_ms(h[:, None], SR) * 10 ** (db / 20) / np.sqrt((tail[:int(0.01 * SR)] ** 2).mean())
        return h + g * tail
    parts = [noise_hit(i, 0.012)[:n] for i in range(1, 7)] + [over(10, -36.0), over(11, -29.0), over(12, -29.0)]
    x = np.concatenate([np.concatenate([p, np.zeros(SR // 10)]) for p in parts])[:, None]
    starts = np.cumsum([0] + [len(p) + SR // 10 for p in parts])[:len(parts)]
    d_lone = drumroles._pair_distance(x[starts[7]:starts[7] + n], x[starts[6]:starts[6] + n], SR)
    d_real = drumroles._pair_distance(x[starts[7]:starts[7] + n], x[starts[0]:starts[0] + n], SR)
    assert d_lone < d_real < drumroles.THRESHOLD, (d_lone, d_real)          # the premise: nearer the tail, still within reach

    def assign(lone_lead):
        hits = [{"id": i, "start": int(a), "end": int(a) + n, "status": "grouped", "reason": None, "lead_db": -90.0,
                 "group": None, "distance": None} for i, a in enumerate(starts)]
        hits[6]["lead_db"] = lone_lead
        for i in (7, 8):
            hits[i].update(status="overlapped", lead_db=-29.0,
                           reason="it starts out of a tail or bed only 29 dB under its own loudest ms (it needs 30)")
        found = [(0, {"members": list(range(6)), "medoid": 0, "assigned": []}), (1, {"members": [6], "medoid": 6, "assigned": []})]
        drumroles._assign_overlapped(x, SR, hits, found, drumroles.THRESHOLD)
        return [g["assigned"] for _, g in found], [hits[i]["status"] for i in (7, 8)]
    assert assign(-36.0) == ([[7, 8], []], ["assigned", "assigned"])        # the lone hit sits on a tail: the real group
    assert assign(-45.0) == ([[], [7, 8]], ["assigned", "assigned"])        # a lone hit from near silence is a candidate


def test_a_hat_of_tau_60_ms_with_a_rest_sounds_through_two_lines():
    # -17.5 and -35 dB at the next two 16th lines, -52 at the third: two lines inside 40 dB (a tau 30 ms hat has one,
    # -35 then -70)
    r = run(stem([(at(i), noise_hit(i, 0.06)) for i in range(2, 32, 4)], 2))
    assert [h["through"] for h in r["hits"]] == [2] * 8
    r = run(stem([(at(i), noise_hit(i, 0.03)) for i in range(2, 32, 4)], 2))
    assert [h["through"] for h in r["hits"]] == [1] * 8


# ---------------------------------------------------------------- role names: groups built by hand, to each rule

def click_hit(f=400.0, tau=0.003, seconds=0.3):
    """A rim-like click: a short ringing tone and an upper partial."""
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t / tau) + 0.3 * np.sin(2 * np.pi * 2.75 * f * t) * np.exp(-t / (0.7 * tau))


def fake_result(specs, n_bars=8, threshold=drumroles.THRESHOLD):
    """A hit_groups-shaped result built by hand: each spec is {steps, bars, audio (the clean hit, [n] or None), level (dB),
    stability, separation, nearest, assigned (steps whose hits count as assigned, not measured), hidden / unmeasured
    ((bar, step) cells with no hit that are listed as hidden under a tail / as holding an unmeasured hit), read (the audio of the
    members the measurement rules read, as `hit_groups` keeps them; without it the clean hit is the one member read), cells (the
    (bar, step) pairs the group plays at, instead of steps x bars), tail_db (the level of the tails over the `hidden` cells, dBFS; a
    group's `level` is -10 unless given; without it the result has no tail levels and every hidden cell hides)}. Group i has id i; its hits sit exactly on the 16th of every
    bar in `bars` at each step."""
    hits, groups, hidden, unmeasured, tails = [], [], [], [], {}
    for gid, s in enumerate(specs):
        members, assigned = [], []
        hidden += [list(c) for c in s.get("hidden", [])]              # (bar, step) cells the group is not seen in: under a tail
        tails.update({tuple(c): s["tail_db"] for c in s.get("hidden", []) if "tail_db" in s})   # (the level of that tail, if given)
        unmeasured += [list(c) for c in s.get("unmeasured", [])]      # ...or held an unmeasured hit
        for b, st in s.get("cells") or [(b, st) for b in s.get("bars", range(n_bars)) for st in s["steps"]]:
            if (b, st) in s.get("hidden", ()) or (b, st) in s.get("unmeasured", ()):
                continue
            h = {"id": len(hits), "bar": b, "step": st, "on_grid": True, "onset_s": at(b * 16 + st),
                 "level_db": s.get("level", -10.0), "status": "grouped", "group": gid}
            (assigned if st in s.get("assigned", ()) else members).append(h["id"])
            if st in s.get("assigned", ()):
                h["status"] = "assigned"
            hits.append(h)
        bars_with = len({hits[i]["bar"] for i in members + assigned})
        pat = [len({hits[i]["bar"] for i in members + assigned if hits[i]["step"] == st}) / n_bars for st in range(16)]
        audio = s.get("audio")
        read = [{"hit": 1000 * (gid + 1) + k, "audio": np.asarray(a)[:, None]} for k, a in enumerate(s["read"])] if "read" in s else None
        groups.append({
            "id": gid, "count": len(members), "assigned": len(assigned), "tentative": len(members) < drumroles.TENTATIVE_N,
            "stability": s.get("stability", 0.9), "separation": s.get("separation", 3.0), "nearest_group": s.get("nearest"),
            "bars_with": bars_with, "pattern": pat, "level_db": s.get("level", -10.0), "members": members,
            "assigned_members": assigned, "off_grid": 0,
            "clean_hit": None if audio is None else {"hit": members[0], "audio": np.asarray(audio)[:, None]},
            "clean_note": "none of the members tried is complete" if audio is None else None,
            **({} if read is None else {"read_members": read})})
    return {"sample_rate": SR, "tempo": BPM, "t0": T0, "threshold": threshold, "hits": hits, "groups": groups,
            "bars": n_bars, "first_bar": 0, "hidden_cells": sorted(hidden), "unmeasured_cells": sorted(unmeasured),
            **({"hidden_tail_db": [tails.get(tuple(c), 0.0) for c in sorted(hidden)]} if tails else {})}


OFFBEATS, ODDS = [2, 6, 10, 14], [1, 3, 5, 7, 9, 11, 13, 15]


def names(r, kind, **kw):
    out = drumroles.name_roles(r, kind, **kw)
    return [g["name"] for g in out["groups"]], out


def test_names_hh_offbeat_16th_and_open_each_by_its_rule_and_a_group_no_rule_fits_stays_neutral():
    r = fake_result([
        {"steps": OFFBEATS, "audio": noise_hit(1, 0.025)},
        {"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(2, 0.005)},
        {"steps": [0, 8], "audio": noise_hit(3, 0.1)},              # on the beats, falls 20 dB in 230 ms
        {"steps": [0, 8], "audio": noise_hit(4, 0.005)},            # the same 16ths, falls in 13 ms: neither open nor a pattern
    ])
    got, out = names(r, "hh")
    assert got == ["hh offbeat", "hh 16th", "hh open", "hh 4"]
    assert [g["named"] for g in out["groups"]] == [True, True, True, False]
    assert "no rule of the hh stem fits" in out["groups"][3]["reason"] and out["groups"][3]["rule"] is None
    assert out["groups"][0]["measures"]["offbeat_share"] == 1.0 and out["groups"][2]["measures"]["fall_20_ms"] > 200
    assert [role["name"] for role in out["roles"]] == ["hh 16th", "hh offbeat", "hh open", "hh 4"]       # roles by count: 96, 32, 16, 16
    assert [g["role"] for g in out["groups"]] == [1, 0, 2, 3]


def test_names_a_hat_that_plays_one_16th_of_each_beat_is_not_a_16th_hat():
    # 1, 5, 9, 13 (the "e" of each beat) are 4 of the 8 odd 16ths; a hat on 6 of them is a 16th hat, a hat on 4 is not
    on_e = fake_result([{"steps": [1, 5, 9, 13], "audio": noise_hit(1, 0.005)}])
    assert names(on_e, "hh")[0] == ["hh 1"] and "4 of 8 known odd 16ths" in names(on_e, "hh")[1]["groups"][0]["reason"]
    six = fake_result([{"steps": [1, 3, 5, 7, 9, 11], "audio": noise_hit(1, 0.005)}])
    assert names(six, "hh")[0] == ["hh 16th"] and "6 of 8 known odd 16ths" in names(six, "hh")[1]["groups"][0]["rule"]


def odd_steps(r, gid=0):
    return {row["step"]: row for row in drumroles.name_roles(r, "hh")["groups"][gid]["measures"]["steps"] if row["step"] % 2 == 1}


def test_names_a_hat_that_plays_two_hits_a_bar_over_the_odd_16ths_or_one_over_the_offbeats_is_not_named_from_the_pattern():
    # 8 bars. Bar b plays two odd 16ths, rotating: each of the 8 odd 16ths is played in 2 of the 8 bars (25 %, in use), but the
    # group plays 2 hits a bar where a 16th hat plays 4 or more. And one offbeat a bar, rotating: each offbeat in 2 of 8 bars
    two = fake_result([{"cells": [(b, ODDS[(2 * b + k) % 8]) for b in range(8) for k in (0, 1)], "audio": noise_hit(1, 0.005)}])
    rows = odd_steps(two)
    assert all(rows[st]["played_bars"] == 2 and rows[st]["state"] == "in use" for st in ODDS)
    got, out = names(two, "hh")
    assert got == ["hh 1"] and out["groups"][0]["measures"]["density"]["odd_density"] == 2.0     # every step observable: 2 either way
    assert "0.0 hits a bar on the offbeats and 2.0 on the odd 16ths" in out["groups"][0]["reason"]
    one = fake_result([{"cells": [(b, OFFBEATS[b % 4]) for b in range(8)], "audio": noise_hit(1, 0.005)}])
    got, out = names(one, "hh")
    assert got == ["hh 1"] and "1.0 hits a bar on the offbeats and 0.0 on the odd 16ths" in out["groups"][0]["reason"]
    assert [r["state"] for r in out["groups"][0]["measures"]["steps"] if OFFBEAT(r["step"])] == ["in use"] * 4
    # exactly at the limit (4 odd 16ths a bar, 2 offbeats a bar, each step in 4 of 8 bars) they are named
    four = fake_result([{"cells": [(b, ODDS[(4 * b + k) % 8]) for b in range(8) for k in range(4)], "audio": noise_hit(1, 0.005)}])
    got, out = names(four, "hh")
    assert got == ["hh 16th"] and "4.0 hits a bar" in out["groups"][0]["rule"] and out["roles"][0]["density"]["odd_density"] == 4.0
    # 3.5 a bar (4 hits in the even bars, 3 in the odd ones, every 16th observable in all 8 bars) is under the limit
    under = fake_result([{"cells": [(b, ODDS[(4 * b + k) % 8]) for b in range(8) for k in range(4 if b % 2 == 0 else 3)],
                          "audio": noise_hit(1, 0.005)}])
    got, out = names(under, "hh")
    assert got == ["hh 1"] and out["groups"][0]["measures"]["density"]["odd_density"] == 3.5
    off2 = fake_result([{"cells": [(b, OFFBEATS[(2 * b + k) % 4]) for b in range(8) for k in (0, 1)], "audio": noise_hit(1, 0.025)}])
    got, out = names(off2, "hh")
    assert got == ["hh offbeat"] and "2.0 hits a bar" in out["groups"][0]["rule"]


def test_the_density_is_read_over_the_bars_a_16th_could_be_seen_in_not_over_all_the_groups_bars():
    # 40 bars of closed hats on every odd 16th. An open hat's tail hides 6 of the 8 odd 16ths (1, 3, 5, 7, 9, 13) in 36 of the 40
    # bars, where no hat could be seen; they are seen in 4 bars (every one played). Hits a bar over all 40 bars: 2 (steps 11 and 15)
    # + 6 x 4/40 = 2.6, under 4. Over the bars each could be seen in: 8 steps at 100 % = 8.0
    hide = [(b, st) for b in range(4, 40) for st in (1, 3, 5, 7, 9, 13)]
    r = fake_result([{"steps": ODDS, "audio": noise_hit(1, 0.005), "hidden": hide}], n_bars=40)
    got, out = names(r, "hh")
    d = out["groups"][0]["measures"]["density"]
    assert d["odd_per_bar"] == pytest.approx(2.6) and d["odd_density"] == pytest.approx(8.0) and d["off_density"] == 0.0
    assert got == ["hh 16th"] and out["roles"][0]["density"] == d and "8.0 hits a bar on them over the bars they could be seen in (2.6 over all its bars)" in out["groups"][0]["rule"]
    # a step observable in under 4 bars is unknown and adds nothing: hide 3 and 7 in 38 bars (seen in 2): 6 known at 100 % = 6.0
    few = [(b, st) for b in range(2, 40) for st in (3, 7)]
    out = drumroles.name_roles(fake_result([{"steps": ODDS, "audio": noise_hit(1, 0.005), "hidden": few}], n_bars=40), "hh")
    assert out["groups"][0]["measures"]["density"]["odd_density"] == pytest.approx(6.0) and out["groups"][0]["named"]
    # the same hits with nothing hidden over a group that plays at 2.6 a bar everywhere is not named
    flat = fake_result([{"cells": [(b, ODDS[(k + 3 * b) % 8]) for b in range(40) for k in range(2)] + [(b, 11) for b in range(40)],
                         "audio": noise_hit(1, 0.005)}], n_bars=40)
    assert names(flat, "hh")[0] == ["hh 1"]


def test_a_cell_under_a_quiet_tail_is_not_hidden_for_a_group_whose_hit_would_have_risen_out_of_it():
    # a hat with 2 hits a bar rotating over the odd 16ths, 16 bars: each odd 16th played in 4 bars. Every empty odd line lies under a
    # tail. Under a tail 30 dB under the group's own level (-10 dBFS) a hit of the group would have been found: the cells are seen and
    # empty, the density is 2.0 and it is not a 16th hat. Under a tail within 15 dB of its level they are hidden: 4 of 4 bars at every
    # odd 16th, density 8.0, named (the rule counts what could be seen)
    cells = [(b, ODDS[(2 * b + k) % 8]) for b in range(16) for k in (0, 1)]
    empty = [(b, st) for b in range(16) for st in ODDS if (b, st) not in set(cells)]

    def read(**kw):
        r = fake_result([{"cells": cells, "audio": noise_hit(1, 0.005), "hidden": empty, **kw}], n_bars=16)
        out = drumroles.name_roles(r, "hh")
        return out["groups"][0]["name"], out["groups"][0]["measures"]["density"]["odd_density"], out
    name, density, out = read(tail_db=-40.0)
    assert (name, density) == ("hh 1", 2.0) and out["groups"][0]["measures"]["unknown_steps"] == []
    assert all(row["observable_bars"] == 16 for row in out["groups"][0]["measures"]["steps"] if row["step"] % 2)
    name, density, _ = read(tail_db=-20.0)
    assert (name, density) == ("hh 16th", 8.0)
    # the exact limit: a hit rises RISE_DB (15 dB) out of a tail 15 dB under it and is found; 14.9 dB under, it is not
    assert read(tail_db=-25.0)[:2] == ("hh 1", 2.0) and read(tail_db=-24.9)[:2] == ("hh 16th", 8.0)
    # a result with no tail levels hides every cell, as before
    r = fake_result([{"cells": cells, "audio": noise_hit(1, 0.005), "hidden": empty}], n_bars=16)
    assert "hidden_tail_db" not in r and drumroles.name_roles(r, "hh")["groups"][0]["name"] == "hh 16th"
    # with grid_onsets on, a hit is found out of a 6 dB rise (the line is looked at): 5.9 dB under hides it, 6.0 does not
    for tail, want in ((-16.0, "hh 1"), (-15.9, "hh 16th")):
        r = fake_result([{"cells": cells, "audio": noise_hit(1, 0.005), "hidden": empty, "tail_db": tail}], n_bars=16)
        r["grid_onsets"] = True
        assert drumroles.name_roles(r, "hh")["groups"][0]["name"] == want, tail


def test_names_judge_a_step_over_the_bars_it_could_be_seen_in_closed_hats_under_an_open_hats_tail():
    # 20 bars of closed hats on every 16th but the offbeats (the beats and all 8 odd 16ths, 12 steps a bar); an open hat on the
    # offbeats whose tail hides the 16th after it (3, 7, 11, 15, the "a" of each beat) in 16 of the 20 bars, where no hat can
    # be seen at that line. The hat is seen at those 4 steps in 4 bars only: 4 of 20 (20 %) over all bars, 4 of 4 over the
    # bars they were observable in
    hide = [(b, s) for b in range(4, 20) for s in (3, 7, 11, 15)]
    r = fake_result([{"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}], n_bars=20)
    rows = odd_steps(r)
    for s in (3, 7, 11, 15):
        assert (rows[s]["played_bars"], rows[s]["observable_bars"], rows[s]["share_all"], rows[s]["share_observable"]) == (4, 4, 0.2, 1.0)
        assert rows[s]["state"] == "in use"
    for s in (1, 5, 9, 13):
        assert (rows[s]["played_bars"], rows[s]["observable_bars"], rows[s]["share_all"], rows[s]["share_observable"]) == (20, 20, 1.0, 1.0)
    # judged over all bars (what the rule read before) it is 4 of 8 odd 16ths: no name; over the observable bars it is 8 of 8
    assert sum(rows[s]["share_all"] >= drumroles.STEP_MIN for s in rows) == 4
    got, out = names(r, "hh")
    assert got == ["hh 16th"] and "8 of 8 known odd 16ths in use" in out["groups"][0]["rule"] and out["groups"][0]["note"] is None
    assert out["roles"][0]["steps"] == out["groups"][0]["measures"]["steps"] and out["groups"][0]["measures"]["unknown_steps"] == []


def test_names_a_group_that_plays_only_the_a_steps_with_every_step_observable_is_not_a_16th_hat():
    # 3, 7, 11, 15 and the beats, in every bar, nothing hidden: the "e" steps (1, 5, 9, 13) were observable in all 20 bars and
    # not played in any of them, so they count against: 4 of 8 odd 16ths
    r = fake_result([{"steps": [3, 7, 11, 15, 0, 4, 8, 12], "audio": noise_hit(1, 0.012)}], n_bars=20)
    rows = odd_steps(r)
    assert all(rows[s]["observable_bars"] == 20 and rows[s]["state"] == "not in use" for s in (1, 5, 9, 13))
    got, out = names(r, "hh")
    assert got == ["hh 1"] and "4 of 8 known odd 16ths" in out["groups"][0]["reason"]


def test_names_a_step_observable_in_too_few_bars_is_unknown_counts_for_nothing_and_the_note_says_so():
    # the "e" steps 1 and 5 are under a tail in 6 of 8 bars: seen in 2 bars (unknown, under 4); the other six odd 16ths are
    # played in every bar: 6 of the 6 known are in use, so it is a 16th hat, named without 2 of 8, and says which
    hide = [(b, s) for b in range(2, 8) for s in (1, 5)]
    r = fake_result([{"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}])
    rows = odd_steps(r)
    assert [rows[s]["state"] for s in (1, 5)] == ["unknown", "unknown"] and rows[1]["observable_bars"] == 2 and rows[1]["played_bars"] == 2
    got, out = names(r, "hh")
    e = out["groups"][0]
    assert got == ["hh 16th"] and e["measures"]["unknown_steps"] == [1, 5]
    assert "named without 2 of the 8 odd 16ths: [1, 5]" in e["note"] and "under 4 bars" in e["note"] and e["note"] == out["roles"][0]["note"]
    assert "6 of 6 known odd 16ths in use (2 of the odd 16ths unknown" in e["rule"]
    # an unknown step counts neither for nor against: a group that plays only 3 of its 6 known odd 16ths is not named
    # with 2 unknown, 5 of the 6 known in use leaves 1 empty: named; 4 of 6 leaves 2 empty, the 2 a 16th hat may have (8 - 6),
    # so it is named too (it was not, when the rule read the same share of the known ones: 4 of 6 is under 3/4; that rule took a
    # name away from a group when one of its in-use steps was hidden, see the monotone test); 3 of 6 leaves 3 empty and is not
    five = fake_result([{"steps": [7, 9, 11, 13, 15, 0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}])
    assert names(five, "hh")[0] == ["hh 16th"] and "5 of 6 known odd 16ths in use" in names(five, "hh")[1]["groups"][0]["rule"]
    four = fake_result([{"steps": [7, 9, 11, 13, 0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}])
    assert names(four, "hh")[0] == ["hh 16th"] and "4 of 6 known odd 16ths in use" in names(four, "hh")[1]["groups"][0]["rule"]
    few = fake_result([{"steps": [3, 7, 11, 0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}])
    assert names(few, "hh")[0] == ["hh 1"]
    # too few known (3 of 8: the other 5 unknown) is not enough to name it, whatever the three show: a 16th hat needs at least
    # 6 of the 8 odd 16ths known
    hid5 = [(b, s) for b in range(2, 8) for s in (1, 3, 5, 7, 9)]
    scarce = fake_result([{"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hid5}])
    got, out = names(scarce, "hh")
    assert got == ["hh 1"] and "only 3 of 8 odd 16ths could be observed (6 needed to name a 16th hat)" in out["groups"][0]["reason"] and len(out["groups"][0]["measures"]["unknown_steps"]) == 5


def test_hiding_a_16th_that_is_in_use_never_takes_a_name_away():
    # a 16th hat may leave 2 of the 8 odd 16ths empty (an offbeat hat 1 of the 4); a step that is in use and becomes unknown
    # (hidden) leaves the empty ones as they were. The old rule (the share asked for, of the known ones, rounded up) named 8 known
    # with 6 in use and not 7 known with 5 in use
    assert drumroles._enough(6, 8, 8, 6, 6) and drumroles._enough(5, 7, 8, 6, 6) and drumroles._enough(4, 6, 8, 6, 6)
    assert not drumroles._enough(3, 6, 8, 6, 6) and not drumroles._enough(5, 5, 8, 6, 6)       # 3 empty; only 5 known
    assert drumroles._enough(3, 4, 4, 3, 3) and drumroles._enough(2, 3, 4, 3, 3) and not drumroles._enough(1, 3, 4, 3, 3)
    for total, needed, known_min in ((8, 6, 6), (4, 3, 3)):
        for known in range(known_min + 1, total + 1):
            for used in range(1, known + 1):
                if drumroles._enough(used, known, total, needed, known_min):
                    assert drumroles._enough(used - 1, known - 1, total, needed, known_min)
    # the same through the names: 6 of 8 in use is named; with one of those 6 hidden in 6 of 8 bars it is still named
    six = [1, 3, 5, 7, 9, 11, 0, 4, 8, 12]
    assert names(fake_result([{"steps": six, "audio": noise_hit(1, 0.012)}]), "hh")[0] == ["hh 16th"]
    hidden = fake_result([{"steps": six, "audio": noise_hit(1, 0.012), "hidden": [(b, 1) for b in range(2, 8)]}])
    got, out = names(hidden, "hh")
    assert got == ["hh 16th"] and out["groups"][0]["measures"]["unknown_steps"] == [1] and "5 of 7 known odd 16ths in use" in out["groups"][0]["rule"]


def test_names_a_16th_hat_needs_six_of_the_eight_odd_16ths_known_not_only_the_ones_it_was_seen_at():
    # the "a" steps (3, 7, 11, 15) are under a tail in 6 of 8 bars, so unknown; the group is seen at its four "e" steps, in use at
    # all four: 4 of 4 known in use is not 6 of 8 known, so it stays neutral and says how many could be observed. With 3 unknown
    # (5 known, all in use) it is also neutral; with 2 unknown (6 known) it is named
    def group(unknown):
        hide = [(b, s) for b in range(2, 8) for s in unknown]
        return fake_result([{"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(1, 0.012), "hidden": hide}])
    got, out = names(group([3, 7, 11, 15]), "hh")
    assert got == ["hh 1"] and "only 4 of 8 odd 16ths could be observed (6 needed to name a 16th hat)" in out["groups"][0]["reason"]
    assert names(group([3, 7, 11]), "hh")[0] == ["hh 1"]
    assert names(group([3, 7]), "hh")[0] == ["hh 16th"]


def test_names_an_offbeat_hat_needs_three_of_the_four_offbeats_known_a_group_seen_at_two_is_not_named():
    # every hit on the offbeats (100 %); steps 10 and 14 are under a tail in 6 of 8 bars. Seen at 2 and 6 only (2 of 4 known, both in
    # use) it plays two offbeats of the bar as far as anyone can tell: neutral. With only step 14 hidden (3 known, all in use) it
    # is an offbeat hat, named without one offbeat, and says so
    two = fake_result([{"steps": OFFBEATS, "audio": noise_hit(1, 0.025), "hidden": [(b, s) for b in range(2, 8) for s in (10, 14)]}])
    got, out = names(two, "hh")
    assert got == ["hh 1"] and "only 2 of 4 offbeats could be observed (3 needed to name an offbeat hat)" in out["groups"][0]["reason"]
    three = fake_result([{"steps": OFFBEATS, "audio": noise_hit(1, 0.025), "hidden": [(b, 14) for b in range(2, 8)]}])
    got, out = names(three, "hh")
    assert got == ["hh offbeat"] and "3 of 3 known offbeats in use" in out["groups"][0]["rule"]
    assert "named without 1 of the 4 offbeats: [14]" in out["groups"][0]["note"]


def test_names_an_unmeasured_hit_at_a_16th_makes_that_bar_unobservable_there_like_a_hidden_one():
    # the "e" step 1 held an unmeasured hit (overlapped, say) in 4 of 8 bars: it may have been this group's, so those bars do not
    # count against: played in 4, observable in 4, in use
    r = fake_result([{"steps": ODDS + [0, 4, 8, 12], "audio": noise_hit(1, 0.012), "unmeasured": [(b, 1) for b in range(4, 8)]}])
    row = odd_steps(r)[1]
    assert (row["played_bars"], row["observable_bars"], row["share_all"], row["share_observable"]) == (4, 4, 0.5, 1.0)


@pytest.mark.slow
def test_observable_cells_of_a_real_stem_are_the_hidden_and_unmeasured_lines():
    # open hats (tau 120 ms) on the offbeats, closed hats (12 ms) on the 96 other 16ths of 8 bars: the 16th after each open hat
    # lies in its tail with no hit found (without the search every such line, steps 3, 7, 11, 15, in all 8 bars), and with the
    # search the hits it finds there are overlapped hits: unmeasured_cells are exactly their cells
    ev = hat_events(8, open_tau=0.12, closed_tau=0.012)
    off = run(stem(ev, 8))
    assert off["bars"] == 8 and off["first_bar"] == 0 and off["unmeasured_cells"] == []
    assert {(b, s) for b, s in off["hidden_cells"] if s in (3, 7, 11, 15)} == {(b, s) for b in range(8) for s in (3, 7, 11, 15)}
    assert all(off["hidden_pattern"][s] == len({b for b, c in off["hidden_cells"] if c == s}) / 8 for s in range(16))
    # every hidden cell has the level of the tail there (an open hat of tau 120 ms, 121 ms on: about -9 dB under its own peak, which is
    # a few dB under the closed hats' level: inside the 15 dB that hides them)
    assert len(off["hidden_tail_db"]) == len(off["hidden_cells"]) and all(isinstance(v, float) for v in off["hidden_tail_db"])
    on = run(stem(ev, 8), grid_onsets=True)
    lost = {(h["bar"], h["step"]) for h in on["hits"] if h["on_grid"] and h["status"] in ("overlapped", "too_short", "no_lead", "ungrouped", "faint")}
    assert lost and {tuple(c) for c in on["unmeasured_cells"]} == lost and len(on["unmeasured_tail_db"]) == len(on["unmeasured_cells"])
    assert all(on["unmeasured_pattern"][s] == len({b for b, c in on["unmeasured_cells"] if c == s}) / 8 for s in range(16))
    # a section carries its own
    sec = run(stem(ev, 8), sections=[{"name": "b", "start_bar": 4, "bars": 4}])["sections"][0]
    assert sec["first_bar"] == 4 and sec["bars"] == 4 and all(4 <= b < 8 for b, _ in sec["hidden_cells"]) and sec["hidden_cells"]


def test_names_count_assigned_hits_in_the_pattern_and_say_how_many():
    # offbeats 2 and 6 measured, 10 and 14 only assigned (heard under another hit's tail): two offbeats in use without the
    # assigned hits (needs 3 of 4), four with them
    r = fake_result([{"steps": OFFBEATS, "assigned": (10, 14), "bars": range(4), "audio": noise_hit(1, 0.025)}], n_bars=4)
    got, out = names(r, "hh")
    g, = out["groups"]
    assert got == ["hh offbeat"] and (g["measures"]["measured"], g["measures"]["assigned"], g["measures"]["hits"]) == (8, 8, 16)
    r["groups"][0]["assigned_members"] = []                       # without them the pattern is two offbeats: no rule
    assert names(r, "hh")[0] == ["hh 1"]


def test_names_snare_stem_rim_snare_and_a_noise_hit_that_stays_neutral():
    r = fake_result([
        {"steps": [4, 12], "audio": snare_hit(1)},                  # a 190 Hz tone under noise
        {"steps": [3], "audio": click_hit()},                       # falls 20 dB in 8 ms
        {"steps": [2, 14], "audio": noise_hit(2, 0.08)},            # noise only: a clap, a noise snare? no rule says
    ])
    got, out = names(r, "snare")
    assert got == ["snare", "rim", "snare 3"]
    assert "tone under the noise" in out["groups"][0]["rule"] and "falls 20 dB in" in out["groups"][1]["rule"]
    assert "no rule of the snare stem fits" in out["groups"][2]["reason"] and out["groups"][2]["measures"]["noise_body_db"] == 0.0


def reading(fall=100.0, share=0.0, pitched=False, hz=None):
    """The readings of one clean hit that the rules use, as `_clean_measures` gives them."""
    return {"fall_20_ms": fall, "fall_40_ms": None, "pitched": pitched, "pitch_hz": hz, "noise_body_db": share}


def rules(kind, reads):
    """(name, rule text) that the rules of stem `kind` give a group whose members read `reads` (a list of `reading`s)."""
    r = fake_result([{"steps": [0], "audio": None}])
    return drumroles._apply_rules(kind, drumroles._use(r, r["groups"][0]), reads)


def rule(kind, **m):
    """The same for one member, with the readings `m`."""
    return rules(kind, [reading(**m)])


def test_the_snare_rule_is_a_band_a_tom_or_a_kick_reading_is_not_a_snare_and_the_edges_are_exact():
    # a tom (150 Hz, body noise share -31 dB) and a kick (45 Hz, -51 dB) in a snare stem stay neutral; a snare (-3 dB) is named
    r = fake_result([{"steps": [4], "audio": tom_hit(f=150.0)}, {"steps": [8], "audio": tom_hit(f=45.0, tau=0.07)},
                     {"steps": [12], "audio": snare_hit(1)}])
    got, out = names(r, "snare")
    assert got == ["snare 1", "snare 2", "snare"] and out["groups"][0]["measures"]["noise_body_db"] < -20
    assert "no rule of the snare stem fits" in out["groups"][0]["reason"]
    # the edges: over -12.0 and at most -1.0 (a fall of 100 ms: not a rim)
    assert [rule("snare", share=v)[0] for v in (-12.1, -12.0, -11.9, -5.0, -1.1, -1.0, -0.99, -0.3, 0.0)] == [
        None, None, "snare", "snare", "snare", "snare", None, None, None]
    assert rule("snare", share=None)[0] is None                       # no noise reading: no snare


def test_a_snare_stem_group_falling_20_db_in_exactly_30_ms_is_a_snare_in_29_9_ms_a_rim():
    assert rule("snare", fall=30.0, share=-5.0)[0] == "snare"
    name, why = rule("snare", fall=29.9, share=-5.0)
    assert name == "rim" and "29.9 ms (under 30)" in why                  # not rounded to "30 ms (under 30)"
    assert rule("snare", fall=29.9, share=0.0)[0] == "rim" and rule("snare", fall=30.0, share=0.0)[0] is None


def test_tom_roles_at_the_same_pitch_share_a_rank_they_are_not_split_low_and_high():
    # two toms at 150 Hz: neither is lower than the other, so they are not ranked and are numbered, with a note
    same = fake_result([{"steps": [4], "audio": tom_hit(f=150.0)}, {"steps": [8], "audio": tom_hit(f=150.0)}])
    got, out = names(same, "toms")
    assert got == ["tom", "tom 2"] and any("not ranked low, mid or high" in n for n in out["notes"])
    # 100, 150 and 150 Hz: two ranks, the two at 150 Hz are both "high"
    three = fake_result([{"steps": [4], "audio": tom_hit(f=100.0)}, {"steps": [8], "audio": tom_hit(f=150.0)},
                         {"steps": [12], "audio": tom_hit(f=150.0)}])
    got, out = names(three, "toms")
    assert sorted(got) == ["tom high", "tom high 2", "tom low"] and got[0] == "tom low"


@pytest.mark.slow
def test_the_measurement_rules_read_up_to_eight_members_and_name_only_when_two_thirds_agree():
    # a snare stem group whose clean hit reads a rim and whose 8 members read 3 snares (a tone under the noise), 3 rims and 2
    # noise hits: none reaches 6 of 8 (two thirds of 8 is 5.33), so it stays neutral and the reason has the counts
    mixed = [snare_hit(i) for i in range(3)] + [click_hit()] * 3 + [noise_hit(9, 0.08), noise_hit(10, 0.08)]
    r = fake_result([{"steps": [4, 12], "audio": click_hit(), "read": mixed}])
    got, out = names(r, "snare")
    assert got == ["snare 1"] and out["groups"][0]["measures"]["members_read"] == 8
    why = out["groups"][0]["reason"]
    assert "3 of 8 rim, 3 snare, 2 neither" in why and "a name needs 67 %" in why and not out["groups"][0]["named"]
    # 5 of 8 snares is under two thirds, 6 of 8 is over: named, and the rule says how many
    five = [snare_hit(i) for i in range(5)] + [click_hit()] * 2 + [noise_hit(9, 0.08)]
    assert names(fake_result([{"steps": [4, 12], "audio": snare_hit(1), "read": five}]), "snare")[0] == ["snare 1"]   # the clean hit is a snare
    six = [snare_hit(i) for i in range(6)] + [click_hit()] + [noise_hit(9, 0.08)]
    got, out = names(fake_result([{"steps": [4, 12], "audio": click_hit(), "read": six}]), "snare")
    assert got == ["snare"] and out["groups"][0]["rule"].startswith("6 of 8 members: a tone under the noise")
    # the clean hit alone says nothing once members are read: a snare clean hit among 8 rims is a rim
    rims = [click_hit()] * 8
    assert names(fake_result([{"steps": [4, 12], "audio": snare_hit(1), "read": rims}]), "snare")[0] == ["rim"]
    # the same agreement for the other rules: toms (pitched), kick (pitched, 70 Hz at most), hh open (falls 140 ms or more)
    toms = [tom_hit(f=150.0)] * 6 + [noise_hit(9, 0.08)] * 2
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0), "read": toms}]), "toms")[0] == ["tom"]
    # exactly two thirds (4 of the 6 that could be read) is enough; 3 of 5 is not
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0), "read": toms[2:]}]), "toms")[0] == ["tom"]
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0), "read": toms[3:]}]), "toms")[0] == ["toms 1"]
    kicks = [tom_hit(f=45.0, tau=0.07)] * 6 + [tom_hit(f=81.0)] * 2
    assert names(fake_result([{"steps": [0], "audio": tom_hit(f=45.0), "read": kicks}]), "kick")[0] == ["kick"]
    split = [tom_hit(f=45.0, tau=0.07)] * 4 + [tom_hit(f=81.0)] * 4
    got, out = names(fake_result([{"steps": [0], "audio": tom_hit(f=45.0), "read": split}]), "kick")
    assert got == ["kick 1"] and "4 of 8 kick, 4 neither" in out["groups"][0]["reason"]
    opens = [noise_hit(i, 0.1) for i in range(6)] + [noise_hit(9, 0.005)] * 2
    r = fake_result([{"steps": [0, 8], "audio": noise_hit(1, 0.005), "read": opens}])        # no pattern rule fits [0, 8]
    assert names(r, "hh")[0] == ["hh open"]
    r = fake_result([{"steps": [0, 8], "audio": noise_hit(1, 0.1), "read": opens[2:] + [noise_hit(9, 0.005)] * 2}])
    assert names(r, "hh")[0] == ["hh 1"]                                                       # 4 open of 8 read


def test_names_toms_by_pitch_low_mid_high_when_several_and_kick_by_a_low_body():
    r = fake_result([{"steps": [4], "audio": tom_hit(f=150.0)}, {"steps": [8], "audio": tom_hit(f=100.0)},
                     {"steps": [12], "audio": tom_hit(f=220.0)}, {"steps": [0], "audio": noise_hit(1, 0.1)}])
    got, out = names(r, "toms")
    assert got == ["tom mid", "tom low", "tom high", "toms 4"]                 # 150, 100, 220 Hz; the noise has no pitch
    assert any("ranked by body pitch" in n for n in out["notes"])
    assert names(fake_result([{"steps": [4], "audio": tom_hit(f=150.0)}]), "toms")[0] == ["tom"]     # one tom: no rank
    k = fake_result([{"steps": [0, 4, 8, 12], "audio": tom_hit(f=45.0, tau=0.1)},
                     {"steps": [6], "audio": tom_hit(f=81.0), "bars": range(0, 8, 2)}])
    got, out = names(k, "kick")
    assert got == ["kick", "kick 2"] and "81 Hz" in out["groups"][1]["reason"]                # 81 Hz is over the 70 Hz the rule allows


@pytest.mark.slow
def test_a_stem_with_no_more_hits_than_the_sample_is_stable_by_construction_and_named_not_asking_for_stability_is_not():
    # 8 bars of a 45 Hz kick on every beat: 32 hits, the size of the sample, so the first round's sample takes every hit and no
    # regrouping can differ. With the regroupings asked for (the default 2) that is "stable by construction" and the group is
    # named, with a note; with stability_runs=0 it was not asked for, and the group stays neutral and says so
    x = stem([(at(16 * b + s), tom_hit(f=45.0, tau=0.07)) for b in range(8) for s in (0, 4, 8, 12)], 8)
    r = run(x, stability_runs=2)
    g, = r["groups"]
    assert g["count"] == drumroles.SAMPLE_HITS == 32 and g["stability"] is None and g["stability_status"] == "stable by construction"
    assert any("stable by construction" in n for n in r["notes"])
    got, out = names(r, "kick")
    assert got == ["kick"] and out["groups"][0]["named"] and "stable by construction" in out["groups"][0]["note"]
    assert out["roles"][0]["note"] == out["groups"][0]["note"] and out["groups"][0]["measures"]["stability_status"] == "stable by construction"
    quiet = run(x, stability_runs=0)
    assert quiet["groups"][0]["stability"] is None and quiet["groups"][0]["stability_status"] == "not asked for"
    got, out = names(quiet, "kick")
    assert got == ["kick 1"] and "stability is not measured: it was not asked for" in out["groups"][0]["reason"]


def test_names_stay_neutral_for_a_stem_of_unknown_kind_and_a_group_without_a_clean_hit():
    r = fake_result([{"steps": OFFBEATS, "audio": noise_hit(1, 0.025)}, {"steps": [4], "audio": tom_hit()}])
    got, out = names(r, "perc")
    assert got == ["perc 1", "perc 2"] and "has no rules" in out["groups"][0]["reason"]
    r = fake_result([{"steps": [4], "audio": None}])
    got, out = names(r, "toms")
    assert got == ["toms 1"] and "no clean hit" in out["groups"][0]["reason"]
    # a pattern rule does not need the clean hit
    assert names(fake_result([{"steps": OFFBEATS, "audio": None}]), "hh")[0] == ["hh offbeat"]


def test_names_need_enough_hits_a_measured_stability_and_separation_and_the_reason_is_said():
    ok = {"steps": OFFBEATS, "audio": noise_hit(1, 0.025)}
    cases = [({"bars": range(1), "steps": [2, 6, 10]}, "only 3 measured hits (a name needs 4)"),
             ({"stability": 0.6}, "stability 0.60 is under 0.75"), ({"stability": None}, "stability is not measured"),
             ({"separation": 1.2, "nearest": 5}, "separation 1.20 is not over the grouping threshold 1.50")]
    for change, reason in cases:
        r = fake_result([{**ok, **change}])
        got, out = names(r, "hh")
        assert got == ["hh 1"] and not out["groups"][0]["named"] and reason in out["groups"][0]["reason"], out["groups"][0]
    # a group that is alone (no neighbour, separation None) is eligible; exactly at the stability limit too
    assert names(fake_result([{**ok, "separation": None}]), "hh")[0] == ["hh offbeat"]
    assert names(fake_result([{**ok, "stability": 0.75}]), "hh")[0] == ["hh offbeat"]
    # several reasons are all said
    r = fake_result([{**ok, "stability": 0.5, "separation": 1.0, "nearest": 3}])
    assert "stability 0.50" in names(r, "hh")[1]["groups"][0]["reason"] and "separation 1.00" in names(r, "hh")[1]["groups"][0]["reason"]


def test_names_two_offbeat_groups_that_touch_are_one_role_with_variants_and_far_apart_two_roles():
    short, long_ = noise_hit(1, 0.025), noise_hit(2, 0.06)         # falls 20 dB in 60 ms and in 137 ms: two decays
    near = fake_result([{"steps": OFFBEATS, "audio": short, "separation": 1.54, "nearest": 1},
                        {"steps": OFFBEATS, "bars": range(0, 8, 2), "audio": long_, "separation": 1.54, "nearest": 0, "level": -20.0}])
    got, out = names(near, "hh")
    assert got == ["hh offbeat", "hh offbeat"] and len(out["roles"]) == 1
    role, = out["roles"]
    assert role["groups"] == [0, 1] and role["count"] == 32 + 16 and [v["group"] for v in role["variants"]] == [0, 1]
    assert role["variants"][0]["fall_20_ms"] < 80 < 120 < role["variants"][1]["fall_20_ms"]
    assert any("may be one sound" in n for n in out["notes"])
    far = fake_result([{"steps": OFFBEATS, "audio": short, "separation": 3.0, "nearest": 1},
                       {"steps": OFFBEATS, "bars": range(0, 8, 2), "audio": long_, "separation": 3.0, "nearest": 0}])
    got, out = names(far, "hh")
    assert got == ["hh offbeat", "hh offbeat 2"] and len(out["roles"]) == 2 and all(r["variants"] is None for r in out["roles"])
    # near but not on the same 16ths: not one sound
    other = fake_result([{"steps": OFFBEATS, "audio": short, "separation": 1.54, "nearest": 1},
                         {"steps": ODDS + [0, 4, 8, 12], "audio": long_, "separation": 1.54, "nearest": 0}])
    assert names(other, "hh")[0] == ["hh offbeat", "hh 16th"] and len(drumroles.name_roles(other, "hh")["roles"]) == 2
    # near and alike, but the second one is not eligible: it stays neutral, and says it may be the first
    weak = fake_result([{"steps": OFFBEATS, "audio": short, "separation": 1.54, "nearest": 1},
                        {"steps": OFFBEATS, "bars": range(0, 8, 2), "audio": long_, "separation": 1.54, "nearest": 0,
                         "stability": 0.64}])
    got, out = names(weak, "hh")
    assert got == ["hh offbeat", "hh 2"] and "stability 0.64" in out["groups"][1]["reason"]
    assert out["roles"][0]["possibly_also"] == [1] and "possibly the same sound as group 0 (hh offbeat)" in out["groups"][1]["note"]


def test_names_leave_out_a_group_marked_as_bleed_and_it_stays_in_the_output():
    r = fake_result([{"steps": OFFBEATS, "audio": noise_hit(1, 0.025)}, {"steps": [0, 8], "audio": noise_hit(2, 0.1)}])
    got, out = names(r, "hh", bleed={0: {"bleed": False, "why": "-"}, 1: {"bleed": True, "why": "it is a copy"}})
    assert got == ["hh offbeat", "hh 2"] and len(out["roles"]) == 1
    g = out["groups"][1]
    assert g["bleed"] and g["role"] is None and not g["named"] and "marked as bleed: it is a copy" in g["reason"]


@pytest.mark.slow
def test_names_on_a_real_hat_stem_closed_16ths_and_a_short_hat_on_the_offbeats():
    # 4 bars of hat_events (the least a 16th can be known over: OBSERVE_MIN): a tau 25 ms hat on the offbeats, tau 5 ms on the other 16ths (64 measured hits, so the sample
    # moves and stability is measured); two sounds 2.9 apart. The offbeat hat falls 20 dB in 60 ms: not "open" (140 ms)
    r = run(stem(hat_events(4), 4), stability_runs=2)
    got, out = names(r, "hh")
    by_count = sorted(r["groups"], key=lambda g: -g["count"])
    assert [g["count"] for g in by_count] == [48, 16] and all(g["stability"] == 1.0 for g in by_count)
    assert {g["id"]: n for g, n in zip(r["groups"], got)} == {by_count[0]["id"]: "hh 16th", by_count[1]["id"]: "hh offbeat"}
    assert [role["name"] for role in out["roles"]] == ["hh 16th", "hh offbeat"] and all(g["named"] for g in out["groups"])
    assert all(g["stability_status"] == "measured" for g in r["groups"])         # more hits than the sample: the regroupings ran
    # the same groups as a result that was not asked for regroupings (stability_runs=0) read: neither is named, with the reason
    # (the real stability_runs=0 run is in the stable-by-construction test)
    quiet = {**r, "groups": [{**g, "stability": None, "stability_status": "not asked for"} for g in r["groups"]]}
    got, out = names(quiet, "hh")
    assert got == ["hh 1", "hh 2"] and all("stability is not measured" in g["reason"] for g in out["groups"])


# ---------------------------------------------------------------- bleed

@pytest.mark.slow
def test_a_tom_fill_copied_20_db_down_into_a_kick_stem_is_marked_bleed_and_the_kick_stem_keeps_one_role():
    # 9 bars. Kick stem: a 45 Hz kick on steps 0 and 8 of every bar (18 hits). Toms stem: a 150 Hz tom on steps 4 and 12 of
    # bars 1, 3, 5 and 7 (8 hits). The kick stem also holds the same toms 20 dB down: copies of those hits
    kick = lambda: tom_hit(f=45.0, tau=0.1) * 0.2     # noqa: E731
    fill = [(at(16 * b + s), tom_hit()) for b in (1, 3, 5, 7) for s in (4, 12)]
    toms = run(stem(fill, 9))
    kicks = run(stem([(at(16 * b + s), kick()) for b in range(9) for s in (0, 8)]
                     + [(t, h * 10 ** (-20 / 20)) for t, h in fill], 9))
    assert sorted(g["count"] for g in kicks["groups"]) == [8, 18] and len(toms["groups"]) == 1
    marks = drumroles.mark_bleed({"kick": kicks, "toms": toms})
    copy, = [g for g in kicks["groups"] if g["count"] == 8]
    real, = [g for g in kicks["groups"] if g["count"] == 18]
    ev = marks["kick"][copy["id"]]
    assert ev["bleed"] and ev["evidence"]["source_stem"] == "toms" and ev["evidence"]["coincide"] == 8
    # built to coincide in every hit, the tom stem plays those 16ths in 4 of 9 bars: chance 4/9 each; and 20 dB louder
    assert ev["evidence"]["rate"] == 1.0 and ev["evidence"]["expected"] == pytest.approx(4 / 9, abs=0.01)
    assert ev["evidence"]["louder_db"] == pytest.approx(20.0, abs=3.0) and ev["evidence"]["distance"] < 1.0
    assert not marks["kick"][real["id"]]["bleed"] and not marks["toms"][toms["groups"][0]["id"]]["bleed"]
    out = drumroles.name_roles(kicks, "kick", bleed=marks["kick"])
    assert len(out["roles"]) == 1 and out["roles"][0]["groups"] == [real["id"]]
    bled = [g for g in out["groups"] if g["id"] == copy["id"]][0]
    assert bled["bleed"] and bled["role"] is None and bled["reason"].startswith("marked as bleed")
    assert len(out["groups"]) == 2                              # the bled group stays in the output


def bleed_pair(a_audio, a_cells, a_level, b_audio, b_cells, b_level, n_bars=8):
    """Two stems of one hit each: group 0 of "x" plays the cells (bar, step) of `a_cells` and group 0 of "y" those of `b_cells`."""
    def result(audio, cells, level):
        r = fake_result([{"steps": [0], "audio": audio, "level": level}], n_bars=n_bars)
        steps = sorted(cells)
        hits = [{"id": i, "bar": b, "step": s, "on_grid": True, "onset_s": at(b * 16 + s), "level_db": level,
                 "status": "grouped", "group": 0} for i, (b, s) in enumerate(steps)]
        g = r["groups"][0]
        g.update(members=list(range(len(hits))), count=len(hits), assigned_members=[], assigned=0)
        r["hits"] = hits
        return r
    return {"x": result(a_audio, a_cells, a_level), "y": result(b_audio, b_cells, b_level)}


CELLS = [(b, 2 if b % 2 == 0 else 10) for b in range(8)]      # 8 hits, one a bar: step 2 in the even bars, 10 in the odd; chance 1/2 each


def test_a_sound_that_only_coincides_with_a_louder_different_sounding_hit_is_not_bleed():
    # x: a hat on 8 cells; y: a tom 15 dB louder on the same 8. The coincidence is far over chance (1.0 against 0.5, p 0.004)
    # and y is louder, but a hat is not a tom: the clean hits are over 3 apart (a hat against a tom 3.9 and more)
    marks = drumroles.mark_bleed(bleed_pair(noise_hit(1, 0.01), CELLS, -30.0, tom_hit(), CELLS, -15.0))
    ev = marks["x"][0]
    assert not ev["bleed"] and ev["evidence"]["failed"] == "likeness" and ev["evidence"]["distance"] > 1.5 * drumroles.BLEED_DISTANCE
    assert ev["evidence"]["rate"] == 1.0 and ev["evidence"]["louder_db"] == pytest.approx(15.0) and "do not sound alike" in ev["why"]
    assert not marks["y"][0]["bleed"]                           # and the tom is louder than the hat: nothing to mark there


def test_bleed_not_marked_when_it_coincides_and_sounds_alike_but_is_louder_in_this_stem_the_other_is():
    quiet = tom_hit() * 0.1
    marks = drumroles.mark_bleed(bleed_pair(tom_hit(), CELLS, -10.0, quiet, CELLS, -30.0))
    assert not marks["x"][0]["bleed"] and marks["x"][0]["evidence"]["failed"] == "level"
    assert marks["x"][0]["evidence"]["louder_db"] == pytest.approx(-20.0) and "it is not under that group" in marks["x"][0]["why"]
    assert marks["y"][0]["bleed"] and marks["y"][0]["evidence"]["source_stem"] == "x" and marks["y"][0]["evidence"]["distance"] < 0.1


def test_a_group_that_coincides_only_at_the_rate_the_patterns_give_by_chance_is_not_bleed():
    # x plays the 4 beats of every bar, 20 dB down, the same tom as y; y plays every 16th of every bar. Every hit of x
    # coincides, but y is there at every 16th: the chance rate is 100 %, which is the base rate, not bleed
    # (a hat against the tom: the coincidence cannot tell, so the groups are judged on level and likeness, and a hat is not a tom;
    # the same tom 20 dB down is a constant leak and is marked, see the co-located tests)
    every = [(b, s) for b in range(8) for s in range(16)]
    beats = [(b, s) for b in range(8) for s in (0, 4, 8, 12)]
    marks = drumroles.mark_bleed(bleed_pair(noise_hit(1, 0.01), beats, -30.0, tom_hit(), every, -10.0))
    ev = marks["x"][0]
    assert not ev["bleed"] and ev["evidence"]["failed"] == "likeness" and ev["evidence"]["co_located"]
    assert ev["evidence"]["rate"] == 1.0 and ev["evidence"]["expected"] == 1.0 and "100 %" in ev["why"]
    # coincidence at half the hits, with y playing on half the bars: 50 % observed against 50 % expected
    even = [(b, s) for b in range(0, 8, 2) for s in (0, 4, 8, 12)]
    marks = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, beats, -30.0, tom_hit(), even, -10.0))
    ev = marks["x"][0]["evidence"]
    assert not marks["x"][0]["bleed"] and ev["failed"] == "coincidence" and ev["rate"] == 0.5 and ev["expected"] == 0.5


# ---- a leak of a constant pattern: the patterns give the coincidence away, so level and likeness decide

KICK = lambda: tom_hit(f=45.0, tau=0.07)      # noqa: E731
EVERY_BEAT = (0, 4, 8, 12)


def leak_stems(own, own_steps, copy, copy_db, bars=8):
    """Real hit_groups on two short stems. "loud": the kick on every beat. "mixed": `own(b, s)` on `own_steps` of every bar and,
    on every beat, `copy` (an audio array) `copy_db` dB down. Returns (loud result, mixed result, mark_bleed's answer)."""
    loud = run(stem([(at(16 * b + s), KICK()) for b in range(bars) for s in EVERY_BEAT], bars))
    mixed = run(stem([(at(16 * b + s), own(b, s)) for b in range(bars) for s in own_steps]
                     + [(at(16 * b + s), copy * 10 ** (copy_db / 20)) for b in range(bars) for s in EVERY_BEAT], bars))
    return loud, mixed, drumroles.mark_bleed({"kick": loud, "mixed": mixed})


def group_at(r, step):
    """The one group of `r` that has hits at 16th `step`."""
    got = [g for g in r["groups"] if g["pattern"][step] > 0]
    assert len(got) == 1, [(g["id"], g["count"]) for g in r["groups"]]
    return got[0]


@pytest.mark.slow
def test_a_kick_leaking_20_db_down_on_every_beat_into_a_snare_stem_is_marked_and_is_not_a_role():
    # kick stem: a kick on every beat of 8 bars. Snare stem: a snare on 2 and 4 (steps 4, 12) and the same kick 20 dB down on
    # every beat. The kick copies alone (steps 0 and 8) are a group of 16; the kick stem plays those 16ths in every bar, so the
    # chance of a coincidence is 100 % and the coincidence cannot tell: it is the level (20 dB) and the likeness (the same
    # sound) that mark it
    loud, mixed, marks = leak_stems(lambda b, s: snare_hit(100 * b + s), (4, 12), KICK(), -20.0)
    leak, snare = group_at(mixed, 0), group_at(mixed, 4)
    assert leak["id"] != snare["id"] and leak["count"] == 16 and group_at(mixed, 8)["id"] == leak["id"]
    ev = marks["mixed"][leak["id"]]
    assert ev["bleed"] and ev["evidence"]["source_stem"] == "kick" and ev["evidence"]["co_located"]
    assert ev["evidence"]["rate"] == 1.0 and ev["evidence"]["expected"] == 1.0 and ev["evidence"]["louder_db"] == pytest.approx(20.0, abs=2.0)
    assert ev["evidence"]["distance"] < drumroles.CO_DISTANCE and "co-located: chance cannot tell" in ev["why"]
    assert not marks["mixed"][snare["id"]]["bleed"] and not any(m["bleed"] for m in marks["kick"].values())
    out = drumroles.name_roles(mixed, "snare", bleed=marks["mixed"])
    bled = [e for e in out["groups"] if e["id"] == leak["id"]][0]
    assert bled["bleed"] and bled["role"] is None and not bled["named"]
    assert all(leak["id"] not in role["groups"] for role in out["roles"]) and [r["groups"] for r in out["roles"]] == [[snare["id"]]]


def constant_leak(x_audio, x_level, y_level=-10.0, bars=8, x_steps=(0, 8)):
    """The leak tests on fake results (the real-audio one is above): y plays a kick on every beat of every bar at `y_level`; x plays
    `x_audio` at `x_level` on `x_steps` of every bar. Expected rate 100 %: the coincidence cannot tell."""
    every = [(b, s) for b in range(bars) for s in EVERY_BEAT]
    x = [(b, s) for b in range(bars) for s in x_steps]
    return drumroles.mark_bleed(bleed_pair(x_audio, x, x_level, KICK(), every, y_level, n_bars=bars))["x"][0]


def test_a_constant_leak_of_either_stem_is_marked_a_different_sound_or_the_same_one_8_db_down_is_not():
    # (fake results; what the pipeline does with real audio is the test above)
    ev = constant_leak(KICK() * 0.1, -30.0)                                  # the kick 20 dB down on beats 1 and 3 of the other stem
    assert ev["bleed"] and ev["evidence"]["co_located"] and ev["evidence"]["expected"] == 1.0 and ev["evidence"]["louder_db"] == pytest.approx(20.0)
    assert "co-located: chance cannot tell" in ev["why"]
    # the other way round: a snare 20 dB down on steps 4 and 12 of a stem against a snare on every beat
    every = [(b, s) for b in range(8) for s in EVERY_BEAT]
    snare = snare_hit(1)
    ev = drumroles.mark_bleed(bleed_pair(snare * 0.1, [(b, s) for b in range(8) for s in (4, 12)], -30.0, snare, every, -10.0))["x"][0]
    assert ev["bleed"] and ev["evidence"]["co_located"] and ev["evidence"]["distance"] < drumroles.CO_DISTANCE
    # a hat 20 dB under the kick on the same 16ths: co-located, 20 dB under, but a hat is not a kick: likeness fails
    ev = constant_leak(noise_hit(1, 0.012) * 0.1, -30.0)
    assert not ev["bleed"] and ev["evidence"]["co_located"] and ev["evidence"]["failed"] == "likeness"
    assert ev["evidence"]["distance"] > 2 * drumroles.CO_DISTANCE and "do not sound alike" in ev["why"]
    # the same kick only 8 dB down: it would pass the 6 dB of the ordinary limits, not the 12 dB of these
    ev = constant_leak(KICK() * 10 ** (-8 / 20), -18.0)
    assert not ev["bleed"] and ev["evidence"]["co_located"] and ev["evidence"]["failed"] == "level"
    assert ev["evidence"]["louder_db"] == pytest.approx(8.0) and "it needs +12 dB" in ev["why"]


def test_the_chance_is_reported_over_the_groups_own_bars_too_and_p_is_not_evidence_when_the_two_agree():
    # x plays the 4 beats in bars 4-7 (16 hits), 20 dB under y, the same tom; y plays them in those bars and a 16th it never shares
    # elsewhere in bars 0-3 and 8-19, so the track runs over 20 bars. Over the track y is at each beat in 4 of 20 bars (20 %); over
    # the 4 bars x plays in, in all 4 (100 %), which is the rate observed: p (tiny, over the track) says only that both live in
    # bars 4-7. The test keeps the track-wide chance, so x is marked, and the evidence says what p is worth
    beats = [(b, s) for b in range(4, 8) for s in (0, 4, 8, 12)]
    y = beats + [(b, 14) for b in list(range(4)) + list(range(8, 20))]
    ev = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, beats, -30.0, tom_hit(), y, -10.0, n_bars=20))["x"][0]
    e = ev["evidence"]
    assert ev["bleed"] and e["rate"] == 1.0 and e["expected"] == pytest.approx(4 / 20) and e["own_expected"] == 1.0 and e["own_bars"] == 4
    assert e["p"] < drumroles.BLEED_P and e["p_evidence"] is False
    assert "20 % by chance over the track's bars and 100 % over the 4 bars this group plays in" in ev["why"]
    assert "not evidence of leakage here" in ev["why"]
    # a group that plays in every bar of the track has the same chance over its bars as over the track, and 8 hits that all
    # coincide against 50 % are evidence (p 0.004)
    ev = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, CELLS, -30.0, tom_hit(), CELLS, -10.0))["x"][0]
    assert ev["bleed"] and ev["evidence"]["own_expected"] == ev["evidence"]["expected"] == 0.5 and ev["evidence"]["p_evidence"] is True
    assert "not evidence of leakage" not in ev["why"]


def test_a_hit_at_a_16th_the_source_never_plays_still_has_a_chance_of_coinciding():
    # y's hits sit 21 ms off the 16th line (not on the grid: in no cell), so y never "plays" at x's 16th; each of x's 8 hits still
    # coincides with one. The chance of each is the floor (2 %), so the chance of all 8 is 0.02 ** 8, small and not 0
    pair = bleed_pair(tom_hit() * 0.1, CELLS, -30.0, tom_hit(), CELLS, -10.0)
    for h in pair["y"]["hits"]:
        h["on_grid"] = False
        h["onset_s"] += 0.021
    ev = drumroles.mark_bleed(pair)["x"][0]["evidence"]
    assert ev["coincide"] == 8 and ev["expected"] == pytest.approx(drumroles.CHANCE_FLOOR)
    assert ev["p"] == pytest.approx(drumroles.CHANCE_FLOOR ** 8) and ev["p"] > 0


def test_the_level_of_the_other_group_is_the_median_of_the_pairs_not_the_mean():
    # three coincident pairs with the other group 20, 20 and 2 dB louder: median 20, mean 14; 2 of 3 are over the 6 dB asked
    pair = bleed_pair(tom_hit() * 0.1, CELLS[:3], -30.0, tom_hit(), CELLS[:3], -10.0)
    for h, level in zip(pair["y"]["hits"], (-10.0, -10.0, -28.0)):
        h["level_db"] = level
    ev = drumroles.mark_bleed(pair)["x"][0]["evidence"]
    assert ev["louder_db"] == pytest.approx(20.0) and ev["louder_share"] == pytest.approx(2 / 3)


def sectional_pair(x_audio, x_level, y_audio, y_level):
    """x plays the beats of bars 4-7 and y those beats and a 16th it never shares elsewhere in a 20-bar track: the chance of x's hits
    over the track is 20 %, over the 4 bars x plays in 100 %."""
    beats = [(b, s) for b in range(4, 8) for s in (0, 4, 8, 12)]
    y = beats + [(b, 14) for b in list(range(4)) + list(range(8, 20))]
    return drumroles.mark_bleed(bleed_pair(x_audio, beats, x_level, y_audio, y, y_level, n_bars=20))["x"][0]


def test_when_the_other_group_is_there_in_the_groups_own_bars_the_level_and_likeness_limits_are_the_strict_ones():
    # a sectional fill (x only in bars 4-7, where y plays too): over the track 20 %, over x's bars 100 %
    ev = sectional_pair(tom_hit() * 0.1, -30.0, tom_hit(), -10.0)               # the same tom 20 dB down: still marked
    assert ev["bleed"] and ev["evidence"]["strict"] and not ev["evidence"]["co_located"] and ev["evidence"]["distance"] < 1.0
    assert ev["evidence"]["louder_min_db"] == drumroles.CO_LOUDER_DB and "strict ones" in ev["why"]
    # the same sound only 8 dB under: it would pass the 6 dB of the ordinary limits, not the 12 dB of the strict ones
    ev = sectional_pair(tom_hit() * 0.1, -18.0, tom_hit(), -10.0)
    assert not ev["bleed"] and ev["evidence"]["failed"] == "level" and ev["evidence"]["louder_db"] == pytest.approx(8.0)
    # a sound 1.79 from it (a 500 Hz tom against a 150 Hz one) 20 dB under passes the 2.0 of the ordinary limits, not the 1.5
    ev = sectional_pair(tom_hit(f=500.0) * 0.1, -30.0, tom_hit(), -10.0)
    assert not ev["bleed"] and ev["evidence"]["failed"] == "likeness" and 1.5 < ev["evidence"]["distance"] < 2.0
    # the control: x in every bar of the track (own chance 50 %, track 50 %): the ordinary limits, 8 dB and 1.79 are enough
    for audio in (tom_hit() * 0.1, tom_hit(f=500.0) * 0.1):
        ev = drumroles.mark_bleed(bleed_pair(audio, CELLS, -18.0, tom_hit(), CELLS, -10.0))["x"][0]
        assert ev["bleed"] and not ev["evidence"]["strict"] and ev["evidence"]["louder_min_db"] == drumroles.BLEED_LOUDER_DB


def test_a_co_located_group_is_marked_at_an_expected_rate_of_95_percent_and_needs_90_percent_of_its_hits_to_coincide():
    # x plays step 0 in 20 bars, 20 dB under y's copy; y plays it in 19 of them (and elsewhere in the last): chance 95 %, and 19 of
    # 20 hits coincide (95 %). Over 90 % chance the coincidence cannot tell (rate and margin are not asked), so level and likeness mark it
    x = [(b, 0) for b in range(20)]
    y = [(b, 0) for b in range(19)] + [(19, 14)]
    ev = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, x, -30.0, tom_hit(), y, -10.0, n_bars=20))["x"][0]["evidence"]
    assert ev["co_located"] and ev["expected"] == pytest.approx(0.95) and ev["rate"] == pytest.approx(0.95) and ev["failed"] is None
    # y plays step 0 in every bar (chance 100 %), but 6 of x's 20 hits are 30 ms off y's (not coincident): 70 % coincide, under the 90 %
    pair = bleed_pair(tom_hit() * 0.1, x, -30.0, tom_hit(), [(b, 0) for b in range(20)], -10.0, n_bars=20)
    for h in pair["x"]["hits"][:6]:
        h["onset_s"] += 0.030
    marks = drumroles.mark_bleed(pair)["x"][0]
    assert not marks["bleed"] and marks["evidence"]["co_located"] and marks["evidence"]["rate"] == pytest.approx(0.7)
    assert marks["evidence"]["failed"] == "coincidence" and "too few of the hits coincide (it needs 90 %)" in marks["why"]


def test_an_offbeat_hat_needs_two_hits_a_bar_on_the_offbeats_1_75_is_not_enough():
    # 8 bars, 14 hits on the offbeats (2 a bar for six bars, 1 for the last two), every offbeat in use (3 or 4 bars of 8): every hit
    # on the offbeats, all 4 offbeats known and in use, 14 / 8 = 1.75 a bar. 2 a bar rotating (16 hits) is named
    cells = [(b, OFFBEATS[(2 * b + k) % 4]) for b in range(6) for k in (0, 1)] + [(6, OFFBEATS[0]), (7, OFFBEATS[2])]
    r = fake_result([{"cells": cells, "audio": noise_hit(1, 0.025)}])
    got, out = names(r, "hh")
    assert got == ["hh 1"] and out["groups"][0]["measures"]["density"]["off_density"] == 1.75
    assert [row["state"] for row in out["groups"][0]["measures"]["steps"] if OFFBEAT(row["step"])] == ["in use"] * 4
    assert "1.8 hits a bar on the offbeats" in out["groups"][0]["reason"]


def split_source_pair(second_audio):
    """Stem x (a kick on step 0 of 16 bars, and a tom copy 20 dB down on step 2 of bars 0-7) against stem y, where the tom
    fill that the copy comes from is split over two groups: step 2 in bars 0-3 and step 2 in bars 4-7."""
    tom = tom_hit()
    x = fake_result([{"steps": [0], "audio": tom_hit(f=45.0, tau=0.1), "level": -12.0, "bars": range(16)},
                     {"steps": [2], "audio": tom * 0.1, "level": -35.0, "bars": range(8)}], n_bars=16)
    y = fake_result([{"steps": [2], "audio": tom, "level": -15.0, "bars": range(4)},
                     {"steps": [2], "audio": second_audio, "level": -15.0, "bars": range(4, 8)}], n_bars=16)
    return {"x": x, "y": y}


def test_bleed_of_a_source_split_over_two_groups_is_marked_by_the_union_and_not_by_either_alone():
    marks = drumroles.mark_bleed(split_source_pair(tom_hit()))
    copy, kick = marks["x"][1], marks["x"][0]
    assert copy["bleed"] and copy["evidence"]["union"] and copy["evidence"]["source_groups"] == [0, 1]
    # the second test: each group alone has half the hits (50 % against 25 %: under the 70 % asked); together all 8 of 8,
    # the track runs over 16 bars and the groups together are at that 16th in 8 of them (chance 50 %, p 0.004), 20 dB louder
    ev = copy["evidence"]
    assert ev["coincide"] == ev["hits"] == 8 and ev["expected"] == pytest.approx(0.5) and ev["p"] == pytest.approx(0.5 ** 8)
    assert ev["louder_db"] == pytest.approx(20.0) and ev["distance"] < 0.1 and set(ev["distances"]) == {0, 1}
    assert "the union of groups [0, 1]" in copy["why"]
    assert not kick["bleed"] and not marks["y"][0]["bleed"] and not marks["y"][1]["bleed"]
    # with only the first group in the stem there is nothing to unite: 4 of 8 hits coincide, not marked, no second test
    pair = split_source_pair(tom_hit())
    pair["y"]["groups"] = pair["y"]["groups"][:1]
    alone = drumroles.mark_bleed(pair)["x"][1]
    assert not alone["bleed"] and alone["union"] == {} and alone["evidence"]["coincide"] == 4


def test_bleed_by_the_union_needs_every_group_a_pair_belongs_to_to_sound_alike_and_never_marks_on_coincidence_alone():
    # the second half of the fill is a hat, not a tom: the pairs with it are 3 or more away from the copy, so the union fails
    # on likeness although it coincides in all 8 hits and is louder
    marks = drumroles.mark_bleed(split_source_pair(noise_hit(1, 0.01)))
    copy = marks["x"][1]
    assert not copy["bleed"] and copy["union"]["y"]["evidence"]["failed"] == "likeness"
    assert copy["union"]["y"]["evidence"]["distances"][0] < 0.1 and copy["union"]["y"]["evidence"]["distances"][1] > drumroles.BLEED_DISTANCE
    assert "against the union of a stem's groups" in copy["why"] and "do not sound alike" in copy["union"]["y"]["why"]
    # the same split source, but quieter than the copy: coincidence alone does not mark
    pair = split_source_pair(tom_hit())
    for g in pair["y"]["groups"]:
        g["level_db"] = -45.0
    for h in pair["y"]["hits"]:
        h["level_db"] = -45.0
    quiet = drumroles.mark_bleed(pair)["x"][1]
    assert not quiet["bleed"] and quiet["union"]["y"]["evidence"]["failed"] == "level"


def test_a_small_excess_over_chance_in_a_long_stem_is_not_bleed():
    # x plays the 4 beats in bars 0-14 (60 hits), y plays them in bars 0-12 and then another 16th in bars 15-19, so the
    # track runs over 20 bars: y is at each beat 16th in 13 of 20 bars (chance 65 %), x coincides in 13 of 15 bars (87 %):
    # 22 points over chance, under the 25 asked for, though with 60 hits the chance of that from the patterns is under 1 %
    x = [(b, s) for b in range(15) for s in (0, 4, 8, 12)]
    y = [(b, s) for b in range(13) for s in (0, 4, 8, 12)] + [(b, 14) for b in range(15, 20)]
    ev = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, x, -30.0, tom_hit(), y, -10.0, n_bars=20))["x"][0]
    assert not ev["bleed"] and ev["evidence"]["failed"] == "coincidence"
    assert ev["evidence"]["rate"] == pytest.approx(13 / 15) and ev["evidence"]["expected"] == pytest.approx(13 / 20)
    assert ev["evidence"]["p"] < drumroles.BLEED_P and ev["evidence"]["rate"] - ev["evidence"]["expected"] < drumroles.BLEED_MARGIN


def test_a_partial_coincidence_far_over_chance_is_not_bleed():
    # x plays steps 2 and 10 in bars 0-9 (20 hits); y plays them in bars 0-5 (12 coincide) and another 16th in bars 10-39, so
    # the track runs over 40 bars and y is at those 16ths in 6 of 40 (chance 15 %): 60 % against 15 % is far over chance
    # (p tiny), y is louder and the same sound, but 4 hits in 10 have no source in y: this is not a copy
    x = [(b, s) for b in range(10) for s in (2, 10)]
    y = [(b, s) for b in range(6) for s in (2, 10)] + [(b, 14) for b in range(10, 40)]
    ev = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, x, -30.0, tom_hit(), y, -10.0, n_bars=40))["x"][0]["evidence"]
    assert ev["failed"] == "coincidence" and ev["rate"] == 0.6 and ev["expected"] == pytest.approx(0.15)
    assert ev["p"] < drumroles.BLEED_P and ev["rate"] - ev["expected"] >= drumroles.BLEED_MARGIN


def test_bleed_is_not_marked_without_enough_hits_or_a_clean_hit_and_says_so():
    # 2 hits that both coincide with y, which plays that 16th in 6 of the 8 bars: chance 3/4 each, p 0.56, over the 0.01 limit
    two = [(1, 2), (3, 2)]
    marks = drumroles.mark_bleed(bleed_pair(tom_hit() * 0.1, two, -30.0, tom_hit(), two + [(5, 2), (7, 2), (0, 2), (2, 2)], -10.0))
    assert not marks["x"][0]["bleed"] and "not well above chance" in marks["x"][0]["why"] and marks["x"][0]["evidence"]["p"] > 0.01
    # over its own bars (1 and 3) y plays there in both: the chance is 100 % too, but p is not small, so it is not said to mislead
    assert marks["x"][0]["evidence"]["p_evidence"] is False and "not evidence of leakage" not in marks["x"][0]["why"]
    # everything holds but x has no clean hit: not marked, with the reason
    pair = bleed_pair(None, CELLS, -30.0, tom_hit(), CELLS, -10.0)
    marks = drumroles.mark_bleed(pair)
    assert not marks["x"][0]["bleed"] and "likeness not tested" in marks["x"][0]["why"]
    assert drumroles.mark_bleed({"x": pair["x"]})["x"][0]["why"] == "there is no other stem to compare with"
    other = dict(pair["y"], tempo=120.0)
    with pytest.raises(ValueError, match="share a tempo"):
        drumroles.mark_bleed({"x": pair["x"], "y": other})
    assert drumroles._tail([0.5, 0.5], 2) == pytest.approx(0.25) and drumroles._tail([0.5, 0.5], 1) == pytest.approx(0.75)
    assert drumroles._tail([1.0, 1.0, 1.0], 3) == pytest.approx(1.0) and drumroles._tail([0.25] * 4, 4) == pytest.approx(0.25 ** 4)


def test_a_result_without_tail_levels_says_its_hidden_cells_were_not_judged():
    # a hat on the "a" 16ths whose "e" lines are hidden in every bar, from a result that carries no tail levels (built by
    # hand, or stored before they existed): the cells stay hidden, and the role says they could not be judged
    hits, cells = [], []
    for bar in range(8):
        for st in (3, 7, 11, 15):
            hits.append({"bar": bar, "step": st, "on_grid": True})
        cells += [[bar, st] for st in (1, 5, 9, 13)]
    g = {"id": 0, "members": list(range(len(hits))), "assigned_members": [], "pattern": [0.0] * 16, "level_db": -20.0}
    result = {"hits": hits, "hidden_cells": cells, "unmeasured_cells": []}
    assert drumroles._use(result, g)["hidden_unjudged"] is True
    result["hidden_tail_db"] = [-60.0] * len(cells)            # a quiet tail: a hit of -20 dB would have been found
    use = drumroles._use(result, g)
    assert use["hidden_unjudged"] is False
    assert [r["state"] for r in use["steps"] if r["step"] in (1, 5, 9, 13)] == ["not in use"] * 4
