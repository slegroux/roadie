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


def _alone(true, cands, others=(), dist=None):
    """The best role of one source taken alone (role_scores, best_alone), its score dict with `role` the role dict; every role
    sounds like the source unless `dist` says otherwise."""
    return R.best_alone(R.role_scores(true, list(others), cands, dist or (lambda stem, role: 0.0))[0])


def test_best_alone_takes_the_best_f1_and_a_tie_goes_to_recall():
    true = [1.0, 2.0, 3.0, 4.0]
    cands = [("s", _role("all", [0]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),     # recall 1, precision .5: F1 .67
             ("s", _role("half", [1]), [1.0, 2.0]),                                   # recall .5, precision 1: F1 .67
             ("s", _role("most", [2]), [1.0, 2.0, 3.0, 9.0])]                         # recall .75, precision .75: F1 .75
    b = _alone(true, cands)
    assert b["role"]["name"] == "most" and b["recall"] == 0.75 and b["precision"] == 0.75 and b["tp"] == 3
    assert _alone(true, cands[:2])["role"]["name"] == "all"                         # equal F1: the higher recall
    assert R.best_alone([]) is None
    assert _alone(true, [("s", _role("none", [0]), [])]) is None               # a role with no hit at its onsets is not the source's


def test_exclusive_onsets_leave_out_the_positions_another_source_plays_at():
    a = [0.5 * i for i in range(40)]                                  # a hat on every half second
    b = [float(i) for i in range(20)]                                 # another on every second: half of a's positions
    ex, only = R.exclusive_onsets(a, b)
    assert only and len(ex) == 20 and all(abs(t - round(t)) > 0.1 for t in ex)
    assert R.exclusive_onsets(a, [t + 0.02 for t in b])[0] == ex      # 20 ms off is still the same position (25 ms)
    assert len(R.exclusive_onsets(a, [t + 0.03 for t in b])[0]) == 40         # 30 ms off is not
    ex, only = R.exclusive_onsets(a[:4], a)                           # no position of its own: all of them, flagged
    assert not only and ex == a[:4]
    assert R.exclusive_onsets(a, []) == (a, True)


def test_a_role_that_only_plays_where_another_source_plays_scores_nothing_on_exclusive_positions():
    # source A: a hat on the 8ths of bars 0-3; source B (same stem): a hat on every 16th of bars 0-7, which includes A's 8ths.
    # A role of B's hits on the 8ths read precision 1.00 against A (and recall 1.00) by timing alone; on A's exclusive positions
    # (none: B plays at every one of them) it has no hit to score.
    step = 0.1
    a = [i * 2 * step for i in range(32)]
    b = [i * step for i in range(128)]
    b_on_a = ("hh", _role("b on eighths", [0]), a)                    # B's hits, but only the ones at A's positions
    sc, info = R.role_scores(a, b, [b_on_a])
    assert info["only"] is False and sc[0]["precision"] == 1.0 and sc[0]["recall"] == 1.0   # A has no exclusive position: all of them, flagged
    # give A positions of its own (it plays 4 more 8ths in bars 8-9 that B does not): then the role has nothing there
    a2 = a + [12.8 + i * 2 * step for i in range(16)]
    sc, info = R.role_scores(a2, b, [b_on_a])
    assert info["only"] and info["exclusive"] == 16
    assert sc[0]["hits"] == 0 and sc[0]["recall"] == 0.0 and sc[0]["f1"] == 0.0          # all its hits sit where B plays


def test_assign_roles_gives_each_role_to_one_source_among_the_roles_that_sound_right():
    def sc(*f1s, alike=None):
        return [{"k": k, "f1": f, "recall": f, "alike": alike is None or k in alike} for k, f in enumerate(f1s)]
    got = R.assign_roles({"s1": sc(0.9, 0.5), "s2": sc(0.8, 0.7)})
    assert got["s1"]["k"] == 0 and got["s2"]["k"] == 1          # alone, s2 would take role 0 too: 0.9 + 0.7 beats 0.5 + 0.8
    assert R.assign_roles({"s1": sc(0.9, 0.3, alike=(1,))})["s1"]["k"] == 1       # only the role that sounds like it is eligible
    assert R.assign_roles({"s1": sc(0.9, 0.5, alike=())})["s1"] is None          # none does: no role (timing_best shows the rest)
    assert R.assign_roles({"s1": sc(0.0, 0.0)})["s1"] is None                     # nothing matches a single onset
    assert R.assign_roles({"s1": sc(0.0, 0.5, alike=(0,))})["s1"] is None         # sounds like it but never plays where it does
    assert R.assign_roles({"a": sc(0.5), "b": sc(0.4)}) == {"a": {"k": 0, "f1": 0.5, "recall": 0.5, "alike": True}, "b": None}   # one role, two
    assert R.assign_roles({}) == {} and R.assign_roles({"a": []}) == {"a": None}
    # a source with no role of its own does not take one from the source that has it: the kick keeps its role although the
    # hat, which does not sound like a kick, matches its timing better than the kick's other role would
    kick = [{"k": 0, "f1": 0.77, "recall": 0.6, "alike": True}, {"k": 1, "f1": 0.55, "recall": 0.4, "alike": True}]
    hat = [{"k": 0, "f1": 0.62, "recall": 0.5, "alike": False}, {"k": 1, "f1": 0.0, "recall": 0.0, "alike": False}]
    got = R.assign_roles({"kick": kick, "hat": hat})
    assert got["kick"]["k"] == 0 and got["hat"] is None


def test_timing_best_filters_on_precision_before_it_takes_the_best_f1():
    def sc(k, f, p):
        return {"k": k, "f1": f, "precision": p, "recall": f}
    scores = [sc(0, 0.80, 0.76), sc(1, 0.58, 0.99), sc(2, 0.90, 0.5)]
    assert R.timing_best(scores)["k"] == 1                                   # the P 0.76 role does not hide the P 0.99 one
    assert R.timing_best(scores, 0.0)["k"] == 2                              # the raw best, which score_sources keeps in the JSON
    assert R.timing_best([sc(0, 0.0, 1.0), sc(1, 0.8, 0.5)]) is None         # no F1, or no precision: none


def test_the_sound_condition_does_not_hide_a_role_with_precision_one():
    # the source plays on steps 0 (a kick-like sound) and 2, 6, 10, 14 (hats). The role that sounds like it is the kick's (the
    # reference is a kick): recall .2 at precision .25. The hat role matches its hits (recall .8, precision 1.0) and does not
    # sound like it. The pick is the kick's role; the hat role is shown as the timing best.
    hat, kick = _hat(1), _kick()
    xk, t0 = _stem(kick, [0, 4, 8, 12])
    xh, _ = _stem(hat, [2, 6, 10, 14])
    results = {"drums_kick": drumroles.hit_groups(xk, BPM, t0, sr=SR), "drums_hh": drumroles.hit_groups(xh, BPM, t0, sr=SR)}
    named = {s: drumroles.name_roles(r, R.KIND.get(s)) for s, r in results.items()}
    true = [t0 + (b * 16 + s) * STEP for b in range(8) for s in (0, 2, 6, 10, 14)]
    row = R.score_sources(results, named, {"src": {"onsets": true}}, {"src": kick}, SR)["src"]
    assert row["best"]["stem"] == "drums_kick" and row["best"]["sound"] is True and row["best"]["f1"] < 0.5
    tb = row["timing_shown"]
    assert row["timing_best"]["stem"] == "drums_hh" and row["timing_best"]["qualifies"] is True
    assert tb["stem"] == "drums_hh" and tb["precision"] >= 0.9 and tb["f1"] > row["best"]["f1"] and tb["alike"] is False
    # a source the pick already finds shows no timing best
    only_kick = [t0 + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    assert R.score_sources(results, named, {"k": {"onsets": only_kick}}, {"k": kick}, SR)["k"]["timing_shown"] is None


def test_the_number_of_sources_with_a_role_comes_before_the_f1_sum():
    a = [{"k": 0, "f1": 0.9, "alike": True}, {"k": 1, "f1": 0.0, "alike": False}]
    b = [{"k": 0, "f1": 0.5, "alike": True}, {"k": 1, "f1": 0.4, "alike": True}]
    got = R.assign_roles({"a": a, "b": b})
    assert got["a"]["k"] == 0 and got["b"]["k"] == 1
    c = [{"k": 0, "f1": 0.6, "alike": True}, {"k": 1, "f1": 0.0, "alike": False}]
    got = R.assign_roles({"a": a, "c": c})                               # one role, and both want it: the better F1 takes it
    assert got["a"]["k"] == 0 and got["c"] is None


def test_sources_with_the_same_timing_share_a_role_and_say_so():
    assert R.shares_timing([1.0, 2.0, 3.0], [1.01, 2.0, 3.02])
    assert not R.shares_timing([1.0, 2.0, 3.0], [1.0])                       # one of the three: not the other's
    assert not R.shares_timing([], [])
    kick = _kick()
    xk, t0 = _stem(kick, [0, 4, 8, 12])
    results = {"drums_kick": drumroles.hit_groups(xk, BPM, t0, sr=SR)}
    named = {"drums_kick": drumroles.name_roles(results["drums_kick"], "kick")}
    on = [t0 + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    srcs = {"a": {"onsets": on, "role": "drums_snare"}, "b": {"onsets": [t + 0.004 for t in on], "role": "drums_snare"}}
    rows = R.score_sources(results, named, srcs, {"a": kick, "b": kick}, SR)
    assert rows["a"]["shares_with"] == ["b"] and rows["b"]["shares_with"] == ["a"]
    assert rows["a"]["best"]["stem"] == rows["b"]["best"]["stem"] == "drums_kick" and rows["a"]["alone"] is None and rows["b"]["alone"] is None
    assert any("shares the role with: b" in l for l in R.table({"s": rows}).splitlines())
    srcs["b"]["role"] = "drums_hh"                                           # in another role stem it is another source: one role, one source
    rows = R.score_sources(results, named, srcs, {"a": kick, "b": kick}, SR)
    assert rows["a"]["shares_with"] == [] and (rows["a"]["best"] is None) != (rows["b"]["best"] is None)


def test_a_role_goes_to_one_source_and_the_other_reads_what_it_would_have_taken_alone():
    kick, hat = _kick(), _hat(1)
    xk, t0 = _stem(kick, [0, 4, 8, 12])
    results = {"drums_kick": drumroles.hit_groups(xk, BPM, t0, sr=SR)}
    named = {"drums_kick": drumroles.name_roles(results["drums_kick"], "kick")}
    on = [t0 + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    srcs = {"a": {"onsets": on, "role": "drums_kick"}, "b": {"onsets": on, "role": "bass"}}       # two sources that play the same hits
    rows = R.score_sources(results, named, srcs, {"a": kick, "b": kick}, SR)
    assert (rows["a"]["best"] is None) != (rows["b"]["best"] is None)            # one role, one source
    loser = rows["a"] if rows["a"]["best"] is None else rows["b"]
    assert loser["alone"]["stem"] == "drums_kick" and loser["alone"]["f1"] > 0.9


def test_two_sources_of_one_role_stem_are_scored_on_their_own_positions():
    hat, dark = _hat(1), _dark()
    xh, t0 = _stem(hat, [2, 6, 10, 14])
    xo, _ = _stem(dark, [1, 5, 9, 13])
    x = xh + np.pad(xo, (0, max(0, len(xh) - len(xo))))[:len(xh)]
    results = {"drums_hh": drumroles.hit_groups(x, BPM, t0, sr=SR)}
    named = {"drums_hh": drumroles.name_roles(results["drums_hh"], "hh")}
    on = lambda steps: [t0 + (b * 16 + s) * STEP for b in range(8) for s in steps]   # noqa: E731
    srcs = {"hat": {"onsets": on((2, 6, 10, 14)), "role": "drums_hh"}, "dark": {"onsets": on((1, 5, 9, 13)), "role": "drums_hh"}}
    rows = R.score_sources(results, named, srcs, {"hat": hat, "dark": dark}, SR)
    assert rows["hat"]["best"]["groups"] != rows["dark"]["best"]["groups"]               # one role each, one to one
    for r in rows.values():
        assert r["exclusive_only"] and r["exclusive"] == 32 and r["best"]["recall"] >= 0.9 and r["best"]["precision"] >= 0.9


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
    b = _alone(true, cands)
    assert b["stem"] == "drums_kick" and b["recall"] == 1.0 and b["precision"] == 1.0
    g = R.main_group(d["results"]["drums_kick"], b["role"])
    assert R.distance(g["clean_hit"]["audio"], d["kick"], SR) < 0.5
    assert R.distance(g["clean_hit"]["audio"], d["hat"], SR) > 3.0      # and it is not the hat's sound


def test_the_two_hats_are_two_roles_and_each_source_finds_its_own(drum_set):
    d = drum_set
    for sample, steps, want_dist in ((d["hat"], (2, 6, 10, 14), 0.6), (d["hat2"], (1, 5, 9, 13), 0.6)):
        true = [d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in steps]
        b = _alone(true, R.candidates(d["results"], d["named"]))
        assert b["stem"] == "drums_hh" and b["recall"] >= 0.9 and b["precision"] >= 0.9
        g = R.main_group(d["results"]["drums_hh"], b["role"])
        assert R.distance(g["clean_hit"]["audio"], sample, SR) < want_dist
    r1 = _alone([d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (2, 6, 10, 14)], R.candidates(d["results"], d["named"]))
    r2 = _alone([d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (1, 5, 9, 13)], R.candidates(d["results"], d["named"]))
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
    true = [float(i) for i in range(1, 11)]
    kick_like = ("kick", _role("kick", [0]), true + [11.0])                  # plays where the source plays (P 0.91), and elsewhere
    hat_like = ("hh", _role("hh", [1]), true[:7])                            # fewer of them, but the right sound
    assert _alone(true, [kick_like, hat_like])["role"]["name"] == "kick"                # timing alone
    dist = {"kick": 4.0, "hh": 1.4}
    sc, _ = R.role_scores(true, [], [kick_like, hat_like], lambda s, r: dist[s])
    assert R.best_alone(sc)["role"]["name"] == "hh" and R.best_alone(sc)["alike"] is True
    assert R.timing_best(sc)["role"]["name"] == "kick"                                  # and what timing alone would have said
    sc, _ = R.role_scores(true, [], [kick_like, hat_like], lambda s, r: 5.0)
    assert R.best_alone(sc) is None and R.timing_best(sc)["role"]["name"] == "kick"      # nothing sounds like it: no role
    sc, _ = R.role_scores(true, [], [kick_like, hat_like], lambda s, r: None)
    assert not any(c["alike"] for c in sc)


def test_score_sources_rows_carry_recall_precision_detected_and_distance(drum_set):
    d = drum_set
    true = [d["t0"] + (b * 16 + s) * STEP for b in range(8) for s in (0, 4, 8, 12)]
    rows = R.score_sources(d["results"], d["named"], {"kick": {"onsets": true}}, {"kick": d["kick"]}, SR)
    b = rows["kick"]["best"]
    assert (b["stem"], b["recall"], b["precision"], b["detected"], b["sound"]) == ("drums_kick", 1.0, 1.0, 1.0, True)
    assert rows["kick"]["exclusive"] == 32 and rows["kick"]["alone"] is None and rows["kick"]["timing_shown"] is None
    assert b["dist"] < 0.5 and b["dist_main"] < 0.5 and rows["kick"]["closest"]["stem"] == "drums_kick" and rows["kick"]["true_hits"] == 32
    # a source with no onset in the stems at all: the best role has recall 0
    none = R.score_sources(d["results"], d["named"], {"ghost": {"onsets": [100.0, 101.0]}}, {"ghost": None}, SR)["ghost"]
    assert none["best"] is None and none["closest"] is None and none["alike"]["roles"] == []        # no role matches any of its onsets


def test_one_table_lists_every_set_side_by_side_and_what_timing_alone_would_have_picked():
    hat, kick = _hat(1), _kick()
    xk, t0 = _stem(kick, [0, 4, 8, 12])
    xh, _ = _stem(hat, [2, 6, 10, 14])
    results = {"drums_kick": drumroles.hit_groups(xk, BPM, t0, sr=SR), "drums_hh": drumroles.hit_groups(xh, BPM, t0, sr=SR)}
    named = {s: drumroles.name_roles(r, R.KIND.get(s)) for s, r in results.items()}
    on = lambda steps: [t0 + (b * 16 + s) * STEP for b in range(8) for s in steps]            # noqa: E731
    srcs = {"mixed": {"onsets": on((0, 2, 6, 10, 14)), "role": "drums_hh"}}
    rows = R.score_sources(results, named, srcs, {"mixed": kick}, SR)
    out = R.table({"separated": rows, "true": rows}).splitlines()
    assert len(out[0].split()) >= 10 and out[0].startswith("source")
    data = [l.split() for l in out[1:] if l.split()[0] in ("mixed", "true")]
    assert [d[0] for d in data] == ["mixed", "true"] and data[0][1] == "separated"      # a row per set, the source named once
    assert any("timing best: hh:" in l for l in out)


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


def _shifted_group(shifts, sample, n_tail=0.25):
    """A stem with `sample` at known places, and a handmade hit_groups result whose hit starts are off by `shifts` (samples)
    from the real onsets: what a detector's onset error looks like to average_hits. Returns (x, result, group)."""
    gap = int(0.4 * SR)
    x = np.zeros((len(shifts) * gap + int(n_tail * SR), 1))
    hits = []
    for k, sh in enumerate(shifts):
        at = int(0.1 * SR) + k * gap
        x[at:at + len(sample), 0] += sample
        hits.append({"id": k, "start": at + sh, "end": at + sh + len(sample), "status": "grouped"})
    return x, {"hits": hits}, {"members": list(range(len(shifts)))}


def test_aligned_average_finds_a_known_shift_and_logs_it():
    sample = _hat(4, tau=0.03, length=0.2)
    ms = lambda m: int(round(m * SR / 1000))                                       # noqa: E731
    shifts = [ms(v) for v in (-8, -5, -3, 0, 2, 4, 6, 8, -6, 3)]                     # detector onsets off by up to 8 ms either way
    x, res, g = _shifted_group(shifts, sample)
    got = R.average_hits(x, SR, res, g, sample.reshape(-1, 1), sample)
    assert got["hits"] == 10 and got["complete"] == 10
    # every hit moved by minus its shift, to within 0.1 ms; the shifts are logged
    assert np.allclose(got["lags_ms"], [-1000.0 * v / SR for v in shifts], atol=0.1)
    assert got["lag_edge"] == 0.0 and abs(got["lag_median_ms"] - np.median([-1000.0 * v / SR for v in shifts])) < 0.1
    assert got["mean"] < 0.3                                                         # aligned, the average is the sample
    off = R.average_hits(x, SR, res, g, sample.reshape(-1, 1), sample, 0.0)          # not moved: the mean is smeared
    assert off["mean"] > got["mean"] + 0.3 and off["lags_ms"] == []
    # a hit that is exactly at the edge of the search range is counted at the edge
    x2, res2, g2 = _shifted_group([ms(10)] * 5, sample)
    edge = R.average_hits(x2, SR, res2, g2, sample.reshape(-1, 1), sample)
    assert edge["lag_edge"] == 1.0 and abs(edge["lag_median_ms"] + 10.0) < 0.1


def test_an_aligned_cut_ends_at_its_own_end():
    sample = _hat(4, tau=0.03, length=0.2)
    ms = lambda m: int(round(m * SR / 1000))                                       # noqa: E731
    x, res, g = _shifted_group([-ms(8)] * 6, sample)                               # onsets read 8 ms early: the move is 8 ms later
    got = R.average_hits(x, SR, res, g, sample.reshape(-1, 1), sample)
    assert abs(got["lag_median_ms"] - 8.0) < 0.1
    assert abs(got["length_ms"] - (200.0 - 8.0)) < 0.1                             # each cut stops at the hit's `end`, not 8 ms past it


def test_the_average_takes_complete_members_only():
    sample = _hat(4, tau=0.03, length=0.2)
    x, res, g = _shifted_group([0] * 8, sample)
    for h in res["hits"][:3]:                                                        # three members cut 10 ms after their onset
        h["end"] = h["start"] + int(0.010 * SR)
    got = R.average_hits(x, SR, res, g, sample.reshape(-1, 1), sample, 0.0)
    assert got["members"] == 8 and got["complete"] == got["hits"] == 5
    for h in res["hits"][3:6]:
        h["end"] = h["start"] + int(0.010 * SR)
    assert R.average_hits(x, SR, res, g, sample.reshape(-1, 1), sample, 0.0) is None    # 2 complete members: fewer than AVG_MIN_HITS


def test_align_lag_ignores_the_level_of_the_hit():
    sample = _hat(7, tau=0.03, length=0.2)
    x = np.zeros((SR, 1))
    x[4000:4000 + len(sample), 0] = 0.1 * sample                                     # 20 dB quieter than the clean hit
    assert R.align_lag(x, 4000 + int(0.004 * SR), sample.reshape(-1, 1), int(0.01 * SR), int(0.03 * SR)) == -int(0.004 * SR)
    assert R.align_lag(x[:500], 3, sample.reshape(-1, 1), 10, 1000) == 0             # a window that does not fit: no move


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


# ---------------------------------------------------------------- caches

def test_a_cached_result_is_read_only_with_the_key_it_was_written_with(tmp_path):
    import pickle
    p = str(tmp_path / "c.pkl")
    key = {"input": [1, 2], "code": "a", "tempo": 124}
    R.write_cache(p, key, {"x": 1})
    assert R.read_cache(p, key) == {"x": 1}
    for changed in ({**key, "code": "b"}, {**key, "input": [1, 3]}, {**key, "tempo": 125}):
        assert R.read_cache(p, changed) is None
    assert R.read_cache(str(tmp_path / "none.pkl"), key) is None
    with open(p, "wb") as fh:
        fh.write(b"not a pickle")
    assert R.read_cache(p, key) is None
    with open(p, "wb") as fh:
        pickle.dump({"hits": [], "groups": []}, fh)                        # a result cached before keys existed
    assert R.read_cache(p, key) is None


def test_file_key_follows_size_and_mtime_and_code_hash_follows_the_code(tmp_path, monkeypatch):
    import os
    f = tmp_path / "a.bin"
    f.write_bytes(b"1234")
    k = R.file_key(str(f))
    os.utime(f, ns=(1, 1))
    assert R.file_key(str(f)) != k and R.file_key(str(f))[1] == 4
    f.write_bytes(b"12345")
    assert R.file_key(str(f))[1] == 5
    h = R.code_hash()
    assert h == R.code_hash() and R.code_hash(R.true_onsets) != h and R.code_hash(R.true_onsets) != R.code_hash(R.reference_hit)
    fake = tmp_path / "drumroles.py"
    fake.write_text("x = 1\n")
    monkeypatch.setattr(drumroles, "__file__", str(fake))
    assert R.code_hash() != h                                                # a different drumroles.py is a different hash
    monkeypatch.undo()
    fake = tmp_path / "listen.py"
    fake.write_text("y = 1\n")
    assert R.code_hash(files=(str(fake),)) != h                              # another file listed is another hash
    assert R.code_hash(R.group_audio) != R.code_hash(R.medoid_cut)


class _Inline:
    """A stand-in for ProcessPoolExecutor that runs the jobs here, so a test can see what was run."""
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return [fn(i) for i in it]


def test_run_set_redoes_a_stem_only_when_its_file_the_code_or_the_parameters_change(tmp_path, monkeypatch):
    import soundfile as sf
    x, t0 = _stem(_hat(), [0, 4, 8, 12], bars=2)
    path = str(tmp_path / "drums_hh.flac")
    sf.write(path, x, SR, subtype="PCM_24", format="FLAC")
    runs = []
    real = R._run_stem
    monkeypatch.setattr(R, "ProcessPoolExecutor", _Inline)
    monkeypatch.setattr(R, "_run_stem", lambda a: (runs.append(a[1:3]), real(a))[1])
    cache = str(tmp_path / "roles")

    def go(tempo=BPM, rerun=False):
        R.run_set("s", {"drums_hh": path}, tempo, t0, cache, 1, rerun)
        return len(runs)
    assert go() == 1 and go() == 1                                           # the second call reads the cache
    assert go(tempo=BPM + 1) == 2                                            # another tempo
    assert go(tempo=BPM + 1) == 2
    sf.write(path, x[:-SR // 2], SR, subtype="PCM_24", format="FLAC")        # another file (its size and mtime)
    assert go(tempo=BPM + 1) == 3
    monkeypatch.setattr(R, "code_hash", lambda *f: "another drumroles.py")
    assert go(tempo=BPM + 1) == 4
    assert go(tempo=BPM + 1, rerun=True) == 5


def test_sources_stage_is_cached_and_follows_the_source_files(tmp_path, monkeypatch):
    import soundfile as sf
    (tmp_path / "truth_sources").mkdir()
    x, t0 = _stem(_hat(3), [0, 2, 4, 6, 8, 10, 12, 14], bars=4)
    f = tmp_path / "truth_sources" / "hat.flac"
    sf.write(str(f), x, SR, subtype="PCM_24", format="FLAC")
    manifest = {"bpm": BPM, "bar1_s": t0, "sources": [{"file": "HAT.aif", "source": "HAT", "role": "drums_hh", "drum_source": "hat"},
                                                       {"file": "BASS.aif", "source": "BASS", "role": "bass"}]}
    calls = []
    real = R.true_onsets

    def counting(x, sr):
        calls.append(1)
        return real(x, sr)
    monkeypatch.setattr(R, "true_onsets", counting)
    cache = str(tmp_path / "roles")
    first = R.sources_stage(manifest, str(tmp_path), cache)
    assert len(first["info"]["hat"]["onsets"]) == 32 and first["info"]["hat"]["role"] == "drums_hh" and first["refs"]["hat"] is not None
    assert "bass" not in first["info"] and first["sr"] == SR
    again = R.sources_stage(manifest, str(tmp_path), cache)
    assert len(calls) == 1 and again["info"]["hat"]["onsets"] == first["info"]["hat"]["onsets"]
    sf.write(str(f), x[:-SR], SR, subtype="PCM_24", format="FLAC")                          # a different file
    R.sources_stage(manifest, str(tmp_path), cache)
    assert len(calls) == 2
    R.sources_stage({**manifest, "bpm": BPM + 1}, str(tmp_path), cache)                        # a different tempo
    assert len(calls) == 3
    for field, value in (("role", "drums_snare"), ("file", "HAT2.aif")):                      # the manifest's role or file of a source
        changed = {**manifest, "sources": [{**manifest["sources"][0], field: value}, manifest["sources"][1]]}
        n = len(calls)
        R.sources_stage(changed, str(tmp_path), cache)
        assert len(calls) == n + 1, field
        R.sources_stage(changed, str(tmp_path), cache)
        assert len(calls) == n + 1
        R.sources_stage({**manifest, "bpm": BPM + 1}, str(tmp_path), cache)                    # back: the cache was rewritten
    from sideman import listen
    other = tmp_path / "listen.py"
    other.write_text("z = 1\n")
    monkeypatch.setattr(listen, "__file__", str(other))                                         # another sideman/listen.py
    n = len(calls)
    R.sources_stage({**manifest, "bpm": BPM + 1}, str(tmp_path), cache)
    assert len(calls) == n + 1
    monkeypatch.setattr(R, "group_audio", lambda *a, **k: (None, False))                        # the functions reference_hit calls
    n = len(calls)
    R.sources_stage({**manifest, "bpm": BPM + 1}, str(tmp_path), cache)
    assert len(calls) == n + 1
