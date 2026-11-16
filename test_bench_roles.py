"""bench_roles.py on synthetic drum stems and handmade results. No pack audio."""
import numpy as np
import pytest

import bench_roles as R
from roadie_analysis import drumroles

SR = 44100
BPM = 124.0
STEP = 60.0 / BPM / 4.0           # a 16th, seconds


# ---------------------------------------------------------------- matching and choosing, on numbers

def test_match_onsets_pairs_one_to_one_within_the_tolerance():
    assert R.match_onsets([1.0, 2.0, 3.0], [1.01, 2.02, 3.02]) == 3
    assert R.match_onsets([1.0, 2.0, 3.0], [1.03, 2.0]) == 1          # 30 ms is outside 25 ms
    assert R.match_onsets([1.0, 1.01], [1.005]) == 1                  # one hit pairs with one onset only
    assert R.match_onsets([1.0], [0.99, 1.01]) == 1
    assert R.match_onsets([], [1.0]) == 0 and R.match_onsets([1.0], []) == 0
    assert R.match_onsets([3.0, 1.0], [1.0, 3.0]) == 2                # unsorted input


def test_two_pointer_matching_is_maximal_where_greedy_by_nearest_is_not():
    # nearest-first would pair 1.02 with 1.0 and leave 1.04 unmatched; the maximum is two pairs
    assert R.match_onsets([1.0, 1.045], [1.02, 1.065]) == 2


def _role(name, groups, named=True):
    return {"name": name, "named": named, "groups": groups, "count": 0, "assigned": 0}


def _result(hits):
    """hits: (onset, group, status) -> a result with just the keys bench_roles reads."""
    return {"hits": [{"id": i, "onset_s": t, "group": g, "status": s} for i, (t, g, s) in enumerate(hits)],
            "groups": [{"id": g, "count": sum(1 for _, gg, s in hits if gg == g and s == "grouped"),
                        "clean_hit": {"audio": np.zeros(10)} if g != 9 else None, "members": []}
                       for g in sorted({g for _, g, _ in hits if g is not None})]}


def test_role_onsets_takes_the_grouped_and_assigned_hits_of_the_roles_groups_only():
    res = _result([(0.1, 0, "grouped"), (0.2, 0, "assigned"), (0.3, 0, "overlapped"), (0.4, 1, "grouped"), (0.5, None, "faint")])
    assert R.role_onsets(res, _role("a", [0])) == [0.1, 0.2]
    assert R.role_onsets(res, _role("b", [0, 1])) == [0.1, 0.2, 0.4]


def test_main_group_is_the_biggest_with_a_clean_hit():
    res = _result([(0.1, 9, "grouped"), (0.2, 9, "grouped"), (0.3, 9, "grouped"), (0.4, 1, "grouped")])
    assert R.main_group(res, _role("x", [9, 1]))["id"] == 1            # group 9 is bigger but has no clean hit
    assert R.main_group(res, _role("x", [9])) is None


def test_best_role_takes_the_best_f1_and_a_tie_goes_to_recall():
    true = [1.0, 2.0, 3.0, 4.0]
    cands = [("s", _role("all", [0]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),     # recall 1, precision .5: F1 .67
             ("s", _role("half", [1]), [1.0, 2.0]),                                   # recall .5, precision 1: F1 .67
             ("s", _role("most", [2]), [1.0, 2.0, 3.0, 9.0])]                         # recall .75, precision .75: F1 .75
    b = R.best_role(true, cands)
    assert b["role"]["name"] == "most" and b["recall"] == 0.75 and b["precision"] == 0.75 and b["tp"] == 3
    assert R.best_role(true, cands[:2])["role"]["name"] == "all"                    # equal F1: the higher recall
    assert R.best_role(true, []) is None
    assert R.best_role(true, [("s", _role("none", [0]), [])])["f1"] == 0.0


# ---------------------------------------------------------------- audio: onsets, groups, distances

def _hat(seed=1, tau=0.012, length=0.1):
    n = int(length * SR)
    return np.random.default_rng(seed).standard_normal(n) * np.exp(-np.arange(n) / (tau * SR)) * 0.3


def _dark(seed=2, tau=0.02, length=0.1):
    """A darker noise hit: the same kind of burst through an 8-point moving average, so a hat that sounds apart from _hat."""
    return np.convolve(_hat(seed, tau, length), np.ones(8) / 8, mode="same") * 2.5


def _kick(f=50.0, length=0.2):
    n = int(length * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.06) * 0.8


def _stem(sample, steps, bars=8, lead=0.02):
    """The sample on the 16th lines of `steps` (0-15) in `bars` bars, sample-accurate; t0 = lead seconds in."""
    x = np.zeros(int((bars * 16 * STEP + 1.0 + lead) * SR))
    for b in range(bars):
        for s in steps:
            i = int(round((lead + (b * 16 + s) * STEP) * SR))
            x[i:i + len(sample)] += sample
    return x, lead


def test_true_onsets_finds_each_hit_within_a_few_ms():
    x, t0 = _stem(_hat(), [0, 4, 8, 12], bars=4)
    ons = R.true_onsets(x, SR)
    want = [t0 + (b * 16 + s) * STEP for b in range(4) for s in (0, 4, 8, 12)]
    assert len(ons) == len(want) == 16
    assert max(abs(a - b) for a, b in zip(ons, want)) < 0.005
    assert R.true_onsets(np.zeros(SR), SR) == []


def test_reference_hit_is_a_clean_hit_of_the_source():
    sample = _hat(3)
    x, t0 = _stem(sample, [0, 2, 4, 6, 8, 10, 12, 14], bars=4)
    ons = R.true_onsets(x, SR)
    ref, info = R.reference_hit(x, SR, ons, BPM, t0)
    assert ref is not None and info["group_hits"] >= 4
    assert R.distance(ref, sample, SR) < 0.3                       # it is the sample (cut to its own length)
    assert R.reference_hit(x, SR, [], BPM, t0)[0] is None


@pytest.fixture(scope="module")
def drum_set():
    """A kick stem and a hat stem (a hat on the offbeats, another sound on the 16ths of the second half), grouped."""
    hat, hat2 = _hat(1), _dark()                                  # a hat and a darker one
    kick = _kick()
    xk, t0 = _stem(kick, [0, 4, 8, 12])
    xh, _ = _stem(hat, [2, 6, 10, 14])
    xo, _ = _stem(hat2, [1, 5, 9, 13])
    xh = xh + np.pad(xo, (0, max(0, len(xh) - len(xo))))[:len(xh)]
    results = {"drums_kick": drumroles.hit_groups(xk, BPM, t0, sr=SR), "drums_hh": drumroles.hit_groups(xh, BPM, t0, sr=SR)}
    named = {s: drumroles.name_roles(r, R.KIND.get(s)) for s, r in results.items()}
    return {"kick": kick, "hat": hat, "hat2": hat2, "xk": xk, "xh": xh, "t0": t0, "results": results, "named": named}


def test_a_synthetic_kick_is_found_as_one_role_with_every_hit_and_the_right_sound(drum_set):
    d = drum_set
    true = [d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    cands = R.candidates(d["results"], d["named"])
    b = R.best_role(true, cands)
    assert b["stem"] == "drums_kick" and b["recall"] == 1.0 and b["precision"] == 1.0
    g = R.main_group(d["results"]["drums_kick"], b["role"])
    assert R.distance(g["clean_hit"]["audio"], d["kick"], SR) < 0.5
    assert R.distance(g["clean_hit"]["audio"], d["hat"], SR) > 3.0      # and it is not the hat's sound


def test_the_two_hats_are_two_roles_and_each_source_finds_its_own(drum_set):
    d = drum_set
    for sample, steps, want_dist in ((d["hat"], (2, 6, 10, 14), 0.6), (d["hat2"], (1, 5, 9, 13), 0.6)):
        true = [d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in steps]
        b = R.best_role(true, R.candidates(d["results"], d["named"]))
        assert b["stem"] == "drums_hh" and b["recall"] >= 0.9 and b["precision"] >= 0.9
        g = R.main_group(d["results"]["drums_hh"], b["role"])
        assert R.distance(g["clean_hit"]["audio"], sample, SR) < want_dist
    r1 = R.best_role([d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (2, 6, 10, 14)], R.candidates(d["results"], d["named"]))
    r2 = R.best_role([d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (1, 5, 9, 13)], R.candidates(d["results"], d["named"]))
    assert r1["role"]["groups"] != r2["role"]["groups"]


def test_group_distances_find_the_sound_wherever_it_sits(drum_set):
    d = drum_set
    gd = R.group_distances(d["results"], d["kick"], SR)
    (stem, _), (dist, clean) = min(gd.items(), key=lambda kv: kv[1][0])
    assert stem == "drums_kick" and dist < 0.5 and clean
    gd = R.group_distances(d["results"], d["hat2"], SR)
    (stem, _), (dist, _) = min(gd.items(), key=lambda kv: kv[1][0])
    assert stem == "drums_hh" and dist < 0.6


def test_a_sound_condition_keeps_a_source_off_the_role_that_merely_plays_on_its_steps():
    true = [1.0, 2.0, 3.0, 4.0]
    kick_like = ("kick", _role("kick", [0]), [1.0, 2.0, 3.0, 4.0, 5.0])     # plays where the source plays, and elsewhere
    hat_like = ("hh", _role("hh", [1]), [1.0, 2.0, 3.0])                     # fewer of them, but the right sound
    assert R.best_role(true, [kick_like, hat_like])["role"]["name"] == "kick"          # timing alone
    dist = {"kick": 4.0, "hh": 1.4}
    b = R.best_role(true, [kick_like, hat_like], lambda s, r: dist[s])
    assert b["role"]["name"] == "hh" and b["sound"] is True
    assert b["timing_best"]["role"] == "kick"                                          # and what timing alone would have said
    b = R.best_role(true, [kick_like, hat_like], lambda s, r: 5.0)
    assert b["role"]["name"] == "kick" and b["sound"] is False                         # nothing sounds like it: plain best, flagged
    assert R.best_role(true, [kick_like, hat_like], lambda s, r: None)["sound"] is False


def test_score_sources_rows_carry_recall_precision_detected_and_distance(drum_set):
    d = drum_set
    true = [d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    rows = R.score_sources(d["results"], d["named"], {"kick": {"onsets": true}}, {"kick": d["kick"]}, SR)
    b = rows["kick"]["best"]
    assert (b["stem"], b["recall"], b["precision"], b["detected"], b["sound"]) == ("drums_kick", 1.0, 1.0, 1.0, True)
    assert b["dist"] < 0.5 and b["dist_main"] < 0.5 and rows["kick"]["closest"]["stem"] == "drums_kick" and rows["kick"]["true_hits"] == 32
    # a source with no onset in the stems at all: the best role has recall 0
    none = R.score_sources(d["results"], d["named"], {"ghost": {"onsets": [100.0, 101.0]}}, {"ghost": None}, SR)["ghost"]
    assert none["best"]["recall"] == 0.0 and none["closest"] is None and none["best"]["dist"] is None and none["best"]["sound"] is None


# ---------------------------------------------------------------- averaging

def test_averaged_hits_beat_a_single_hit_when_each_one_has_its_own_bleed():
    hat = _hat(5)
    x, t0 = _stem(hat, [0, 2, 4, 6, 8, 10, 12, 14], bars=8)
    rng = np.random.default_rng(11)
    for b in range(8):                                             # a different bleed under every hit: noise at about -3 dB re the hit
        for s in (0, 2, 4, 6, 8, 10, 12, 14):
            i = int(round((t0 + (b * 16 + s) * STEP) * SR))
            x[i:i + int(0.1 * SR)] += rng.standard_normal(int(0.1 * SR)) * 0.1
    res = drumroles.hit_groups(x, BPM, t0, sr=SR, stability_runs=0)
    g = max(res["groups"], key=lambda g: g["count"])
    assert g["count"] >= 32 and g["clean_hit"] is not None
    xu = drumroles.oneshot._undc(x[:, None], SR)                       # the stem as hit_groups read it
    got = R.average_hits(xu, SR, res, g, g["clean_hit"]["audio"], hat, 0.0)
    assert got["hits"] == g["count"]
    assert got["mean"] < got["single"] and got["median"] < got["single"]      # bleed that differs hit to hit averages out
    assert 0 < got["length_ms"] <= 1000 * R.AVG_MAX_S
    aligned = R.average_hits(xu, SR, res, g, g["clean_hit"]["audio"], hat)
    assert aligned["mean"] < aligned["single"]


def test_average_of_too_few_hits_is_none():
    hat = _hat(5)
    x, t0 = _stem(hat, [0, 8], bars=1)
    res = drumroles.hit_groups(x, BPM, t0, sr=SR, stability_runs=0)
    g = max(res["groups"], key=lambda g: g["count"])
    assert R.average_hits(drumroles.oneshot._undc(x[:, None], SR), SR, res, g, hat, hat) is None


# ---------------------------------------------------------------- a group with no complete member

def test_group_audio_falls_back_to_the_medoids_cut_and_says_so():
    x = np.arange(100, dtype=float)[:, None]
    res = {"hits": [{"id": 0, "start": 10, "end": 20, "distance": 0.4}, {"id": 1, "start": 30, "end": 45, "distance": 0.0}]}
    g = {"id": 0, "count": 2, "clean_hit": None, "members": [0, 1]}
    audio, clean = R.group_audio(res, g, lambda stem: x, "s")
    assert not clean and audio[:, 0].tolist() == list(range(30, 45))          # the member at distance 0 is the medoid
    assert R.group_audio(res, g) == (None, False)                              # nothing to read the stem with
    assert R.group_audio(res, {**g, "clean_hit": {"audio": "a"}})[1] is True
    assert R.main_group({"groups": [g]}, {"groups": [0]}) is None and R.main_group({"groups": [g]}, {"groups": [0]}, any_group=True) is g
