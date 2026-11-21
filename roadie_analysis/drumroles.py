"""The hits inside one drum stem, each with its place on the grid and its group: hits that sound alike.

A drum stem from a separator holds several sounds (every hat of a track lands in `drums_hh`, a clap in
`drums_snare`). Electronic drums are a few samples repeated on a grid, so hits that fall at the same 16ths and sound
alike are one sound. `hit_groups` finds the hits (oneshot's hit detection), places each on the grid, groups them with
`spectral_distance` and reports each group's pattern, level and one clean hit. Local only; needs numpy and soundfile (requirements.txt).

Choices that are not obvious from the code:

- Where a hit sits: the grid line nearest its onset (bar from t0, 16th 0..15 from the bar's start). `offset_ms` is the
  onset minus that line, so a hit 8 ms early reads -8 and keeps its 16th. A hit more than GRID_LEAD_S (20 ms, the
  tolerance the reference-profile windows use) from every line is `on_grid: false`: it keeps its group but counts in no
  pattern. A pattern is the share of the scope's bars (the stem: first to last bar with a real hit; a section: its `bars`)
  in which the group plays at that 16th.
- Onsets: oneshot's, so hit detection is `oneshot._segments` (a new hit rises 15 dB within 10 ms, to within 6 dB of its
  peak, and 6 dB over the 50 ms before it, at least 50 ms after the last), and it is relative to the stem's loudest
  moment: a hit more than 40 dB under the loudest 5 ms of the stem is not found (a stem with one very loud hit loses
  its quiet ones). The first hit's onset is read with `_hit_onset` (12 dB over the bed before it), the others with
  `hit_cut`'s (the first rise out of the tail, which it reads as the loudest ms over 20 ms: one ms of a low tom's tail
  is loud or quiet with the phase it falls on, and read from 1 ms the onset came 5 ms early).
- What is compared: each hit's cut runs from its onset to 2 ms before the next hit's onset, or `max_cut_s` after it, and
  two cuts are compared over the shorter of the two (both ended with a 5 ms fade, so the cut itself is not a click). What
  follows a hit never decides its group, and a hit is only compared as far as it is alone: two hats a 16th apart are
  compared over that 16th.
- Hits that cannot be measured are listed with their reason, never dropped and never put in a group:
  `too_short` (under MIN_CUT_S before the next hit: a grace note or flam; the distance was measured down to 60 ms),
  `no_lead` (the stem starts just before it: nothing says what lies under it), `overlapped` (the 10 ms before its onset
  are less than OVERLAP_DB under its loudest ms: the tail of an earlier hit or a bleed bed), `faint` (FAINT_DB under
  the median of the hits around it: with a bleed bed under the whole stem `_segments` reports the bed before the first
  hit as a hit, and a ghost note reads the same, so faint hits are counted in `unmeasured_pattern`), `silent`,
  `ungrouped` (more than `max_groups` sounds: not forced into the nearest).
- Overlapped hits and groups: an overlapped hit never forms or moves a group. Once the groups exist it is compared with
  each group's medoid, over the whole cut when its tail is ASSIGN_FULL_DB (-24 dB) or more under it, over its first 40 ms
  when the tail is between -24 and -14 dB under it, not at all when louder; it goes to the nearest medoid when that is
  within the threshold and 15 % nearer than the second nearest, else it stays "overlapped" with the numbers in its
  reason. An assigned hit has status "assigned", counts in its group's `pattern` (and in `assigned_pattern`, the same for
  the assigned hits alone), `bars_with` and `assigned`, never in `count`, `level_db`, `spread`, the medoid or the clean hit.
  Measured on 24 stems of 8 bars (open hats of tau 80, 120, 160 and 200 ms on the offbeats, closed hats of tau 12 ms
  on the other 16ths, 3 seeds, the search off and on): no hat in a group of the other kind; the closed hats placed
  went from 2 to about 50 (tau 120) and 14 (200) without the search and 61-63 and 29-33 with it. The 25 ms window the
  review also tried (tail louder than -14 dB) placed 368 right and 69 wrong, so it is not used. OVERLAP_DB is measured: a closed noise hat (tau 12 ms) over the tail of an open
  one, against a clean closed hat, reads 0.8-0.9 with the tail 35 dB under it, 1.0-1.1 at 30, 1.2-1.3 at 27, 1.4-1.5 at
  24 and 1.6-1.7 at 21 (the threshold below is 1.5); a bed that every hit shares does not separate them until about
  20 dB under (within 0.8, between 2.5 at 40 dB, review round 2), but one rule cannot tell a shared bed from an earlier
  hit's tail, so a bed at -30 dB or louder under the hits makes them overlapped, and the report says so.
- Hits the detector cannot see: a hit that does not rise out of a tail as `_segments` asks (a closed hat on the 16th
  after an open hat of tau 120 ms, whose tail is only 9 dB down there) is part of the earlier hit. `through` lists the
  16th lines each hit's tail still sounds through (within 40 dB), and `hidden_pattern` (per 16th, the share of bars in
  which that line is inside a tail with no hit found at it) says where they could be, even without the search.
  `unmeasured_pattern` leaves them out, and every hit that was never found. `grid_onsets=True` looks at every 16th line
  with no onset within 50 ms for a rise of 6 dB or more (the first-difference power, 3 ms after against 5 ms before, the
  best point within 20 ms of the line, no more than 60 dB under the stem's peak) and adds it as a hit with
  `source: "grid"`. It is meant for hat-like stems (`hat_like`: the share of energy above 4 kHz; a hat stem reads
  0.9 and more, kick, snare and toms stems well under 0.2): on the others it finds bleed and the phase of a low tone's
  tail as hits (118-355 hits on the Flare kick, snare and toms stems, review round 1; a synthetic kick stem here reads
  the kick's own tail at every line after a kick), which `grid_hits` and `source` count and mark. It reads only a
  window of the stem around each line, never the whole stem at once.
- Grouping: agglomerative, average linkage, cut at `threshold` (1.5). Average linkage because single linkage chains (a
  velocity gradient or a bleed bed links closed to open through the hits between them: 0, 1.4 and 2.8 are one group
  at 1.5) and complete linkage splits one sound on its one worst pair, which for a noise hit is the noise realisation.
  The number is measured on noise hats (tau 12 and 120 ms, cuts of 60 to 500 ms): two realisations of one hat read
  0.55-0.74 (closed) and 0.65-0.82 (open); a hat with a 50 % longer decay 0.7-1.3; a closed hat against an open one
  2.8-4.9; and the review of the distance measured at least 2.5 between two sounds over a bleed bed. 1.5 is 1.8 times the
  worst pair of one sound and 1.7 times under the nearest of two.
  Material whose distances lie around the threshold (a continuum of decays, a velocity gradient: 48 of 247 hat member
  distances on Flare lie between 1.3 and 1.5) is cut where the sample puts the cut: a different sample, or the same
  bars inside a longer stem, gives other groups for it (the review: ARI 0.65-0.95 for the sample at 16 to 128 hits). Well
  separated sounds do not move. `stability` says which groups moved (see hit_groups; None when the sample cannot move,
  and for tentative groups), `separation` how far each group lies from the next, and groups under 4 measured hits are
  `tentative`.
- Pitch: the distance moves little with pitch on low drums, so a group whose members' body pitches (`profile_array`'s
  `pitch_body_hz` on each member's cut) fall in two clusters more than 3 semitones apart is split after grouping, each
  side keeping at least two hits; members with no pitch stay with the larger cluster. Measured as the same drum
  shifted up by 7 / 12 / 19 semitones, the distance reads 0.44 / 0.61 / 0.98 (45 Hz kick), 0.65 / 1.06 / 1.55 (100 Hz
  tom), 0.87 / 1.37 / 1.87 (150 Hz tom, review round 1): toms an octave apart and a kick a fifth to an octave up share a
  group without the split. A group is read only when at least half of 8 members spread over it have a pitch (a hat
  group costs 8 reads of 7 ms; a tom, 150 ms for each member, which is most of the run time of a kick or toms stem).
- Cost: every distance is one `spectral_distance` (about 2 ms on a 120 ms cut, 6 ms on 500 ms). The hits are not
  compared pairwise: a deterministic sample of SAMPLE_HITS hits (evenly spaced in time) is grouped (S^2 / 2 distances),
  every other hit goes to the nearest of the sample's medoids if within `threshold` of it (n * k distances for k
  groups), and what is left over is sampled again, until nothing is or `max_groups` groups exist. Overlapped hits cost
  one distance per group (a tentative group is a candidate only when its medoid rose out of near silence, lead -40 dB or
  less: a lone hit on a tail is the tail's sound as much as its own). `stability_runs` regroupings with a shifted sample reuse every distance already measured, and
  cost about as much again for each (Flare hh, bars 32-64, 374 hits: 41 s; the whole stem, 2522 hits: 441 s, 175 ms a
  hit, with the default two regroupings and the pitch reads of the groups that have one). The hits are put in time order first, so the order they come in never matters;
  ties go to the earlier hit. Memory: the stem as float64.
- Sections do not regroup: a section's groups are the stem's groups restricted to the hits whose bar lies in the section
  (the same ids in every section; the count, pattern, level, spread and clean hit are the section's own, `stability`
  is the stem's).
- Role names are rules over a group's stem, pattern and clean-hit measurements, never recognition (`name_roles`; the
  rules and where each number comes from are in docs/DRUMROLES.md). A group is named only when it
  passes the eligibility rule (NAME_MIN_HITS measured hits, a measured stability of at least NAME_STABILITY, a
  separation over the grouping threshold); a group no rule fits keeps `<stem> N`, with the reason. Bleed (`mark_bleed`)
  needs three things at once: hits that coincide with a group of another stem far more often than the two patterns
  would give by chance, the other group louder at those moments, and a clean hit that sounds like its clean hit.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

from roadie_analysis import oneshot
from sideman.listen import GRID_LEAD_S, check_channels, check_sections, load

THRESHOLD = 1.5          # average-linkage distance (spectral_distance, mel) at which groups stop merging
MAX_CUT_S = 0.5          # a hit's cut is at most this long
END_GUARD_S = 0.002      # a cut ends this long before the next hit's onset
MIN_CUT_S = 0.060        # a hit with less room than this before the next one is too short to compare
FADE_S = 0.005           # the shared cut ends with a half-Hann fade of this length (at most a quarter of it)
LEAD_WINDOW_S = 0.010    # what came before a hit is read over this, ending 1 ms before the onset
OVERLAP_DB = -30.0       # a hit is overlapped when the 10 ms before it are louder than this under its loudest ms
SAMPLE_HITS = 32         # hits grouped pairwise at a time
MAX_GROUPS = 24
MAX_CANDIDATES = 40      # members tried for the clean hit, nearest the medoid first
FAINT_DB = 20.0          # a hit this far under the median of its 12 neighbours is a bed or bleed, not a hit
NEIGHBOURS = 6
THROUGH_DB = 40.0        # a tail still sounds through a grid line within this of its hit's loudest ms
ASSIGN_FULL_DB = -24.0   # an overlapped hit whose tail is this far under it or more is compared over its whole cut...
ASSIGN_SHORT_DB = -14.0  # ...one with a tail between this and that over its first ASSIGN_SHORT_S, a louder tail: not at all
ASSIGN_SHORT_S = 0.040
ASSIGN_MARGIN = 1.15     # the nearest group must be this much nearer than the second nearest
GRID_RISE_DB = 6.0       # grid_onsets: a rise of this much (3 ms after against 5 ms before) within GRID_LEAD_S of a 16th line
GRID_FLOOR_DB = 60.0     # ...to a level no further than this under the stem's peak
GRID_QUIET_S = 0.050     # ...at a line with no onset within this
HAT_HZ = 4000.0          # `hat_like`: the share of the stem's energy above this
PITCH_SPLIT_ST = 3.0     # a group whose members' body pitches fall in two clusters further apart than this is split
PITCH_SCREEN = 8         # members whose pitch is read first; a group under half pitched is not read further
STABILITY_RUNS = 2       # regroupings with a shifted sample that `stability` is measured against
TENTATIVE_N = 4          # a group under this many hits is tentative
CLEAN_START_DB = -40.0   # a tentative group takes assigned hits only when its medoid rose out of a lead this far under it
SEP_MEMBERS = 16         # `separation`: members of a group read...
SEP_OTHERS = 12          # ...against this many of the nearest other group's

# Role names (name_roles). The reasons for each number are in docs/DRUMROLES.md.
NAME_MIN_HITS = TENTATIVE_N   # a group is named from at least this many measured hits (the reviewer's rule, round 2)
NAME_STABILITY = 0.75    # ...a measured stability of at least this (Flare hh: 0.79-0.91 named, 0.13-0.69 not)
STEP_MIN = 0.25          # a 16th is "in use" when the group plays there in this share of the own bars it was observable in (chosen: one in four)
OFFBEAT_SHARE = 0.9      # hh offbeat: this share of the group's on-grid hits fall on the offbeats (steps 2, 6, 10, 14)...
OFFBEAT_STEPS = 3        # ...and at least this many of the 4 offbeats are in use
ODD_STEPS = 6            # hh 16th: at least this many of the 8 odd 16ths (the "e" and "a" of each beat) are in use (4 let a hat
#                          that plays only the "e" of each beat, Flare hh group 2, take the name)
ODD_PER_BAR = 4.0        # hh 16th: and the group plays at least this many hits a bar on the odd 16ths, over the bars they can be seen in (half of the 8: with
#                          each of 6 odd 16ths in use in a quarter of its bars, 2 hits a bar rotating over them passed; a 16th-note hat plays most)
OFFBEAT_PER_BAR = 2.0    # hh offbeat: and at least this many hits a bar on the offbeats (half of the 4; one a bar rotating over them is no offbeat
#                          hat. Flare hh group 0 plays its offbeats in 56-89 % of the bars: 2.2 to 3.6 a bar)
OBSERVE_MIN = 4          # a 16th observable in fewer bars than this is unknown: it counts neither for nor against the pattern
OFFBEAT_KNOWN = 3        # the offbeat rule needs at least this many of the 4 offbeats known (3/4, as many as it asks to be in use)
ODD_KNOWN = 6            # the 16th rule needs at least this many of the 8 odd 16ths known (6/8 = the number it asks in use: with 4 known,
#                          Flare hh group 2, seen at its four "e" steps with the "a" steps unobservable, was named from half the evidence)
OPEN_FALL_20_MS = 140.0  # hh open: the clean hit takes this long to fall 20 dB (library: closed hats 18-130 ms, open 15-269)
RIM_FALL_20_MS = 30.0    # rim: it falls 20 dB in under this (library: 3 rims 5-28 ms, the other 28 snares and claps 35-167)
SNARE_NOISE_MAX_DB = -1.0   # snare: the noise share of the body is at most this (a tone under the noise), and over...
SNARE_NOISE_MIN_DB = -12.0  # ...this (a body that is nearly all tone is a tom or a kick). Library, 108 samples, rim rule first: the 9
#                             snares read -9.1 to -1.0 (9 of 9 named), the 9 toms -40.6 to -14.8 (0 of 9; under -1 alone: 9 of 9) and the
#                             33 kicks -39.7 to -23.0 (0 of 33; 31 of 33 under -1 alone). Still named: 1 clave (Clave Mian, -8.9) and 7
#                             hats (3 closed, 4 open, -3.3 to -1.7: nothing here reads a hat's noise as noise), and the 14 claps stay neutral
KICK_MAX_HZ = 70.0       # kick: pitched, body pitch at most this (library: 33 kicks at 28-56 Hz)
AGREE_MEMBERS = 8        # the rules that read a clean hit's measurements read this many members nearest the medoid instead...
AGREE_MIN = 3            # ...from at least this many complete members read (fewer: neutral, "too few complete members")
AGREE_SHARE = 2 / 3      # ...and name only when at least this share of them read the same (Flare snare group 2: 6 snare, 5 rim, 3 neither)
NEAR_SEPARATION = 1.15   # two groups whose separation is at most this times the threshold may be one sound...
SAME_PATTERN = 0.8       # ...when the cosine of the sets of 16ths they play at (one hit is enough) is at least this

# Bleed (mark_bleed). Reasons in docs/DRUMROLES.md.
COINCIDE_S = 0.025       # two hits of different stems are one event when their onsets are this close (the reviews' reading)
BLEED_RATE = 0.7         # a group is bleed only when at least this share of its hits coincide with the other group's...
BLEED_MARGIN = 0.25      # ...by this much more than the two patterns give by chance...
BLEED_P = 0.01           # ...and the chance of that many coincidences from the patterns alone is at most this
CHANCE_FLOOR = 0.02      # a 16th the other group never plays in the track still gets this chance of a coincidence
BLEED_LOUDER_DB = 6.0    # the other group is louder by at least this at the coincident hits...
BLEED_LOUDER_SHARE = 0.75  # ...in at least this share of them
BLEED_DISTANCE = 2.0     # the two clean hits sound alike: spectral distance at most this
OWN_CLOSE = 0.10         # `p` is not evidence of leakage when the chance over the group's own bars is within this of the observed rate
# When the patterns give the coincidence away (the other group plays at the group's 16ths in almost every bar) the
# coincidence test cannot tell a leak from two parts that share a pattern: p is 1 whatever the group is, the margin cannot be
# met. The group is then judged on level and likeness alone, each stricter than above, and the evidence says "co-located".
CO_EXPECTED = 0.9        # the expected coincidence rate from which the test is "co-located: chance cannot tell"
CO_RATE = 0.9            # ...then at least this share of the group's hits coincide (replaces the rate, margin and p)...
CO_LOUDER_DB = 12.0      # ...the other group is louder by at least this at the coincident hits (a separator leak is 15-30 dB down)...
CO_LOUDER_SHARE = 0.9    # ...in at least this share of them...
CO_DISTANCE = THRESHOLD  # ...and the clean hits are within the grouping's own threshold (the same sound, not a similar one)


def _fade(x: np.ndarray, sr: int) -> np.ndarray:
    """x [n, channels] with its last FADE_S (at most a quarter of it) faded out by a half Hann window."""
    n = min(int(round(FADE_S * sr)), len(x) // 4)
    if n < 2:
        return x
    x = x.copy()
    x[-n:] *= (0.5 + 0.5 * np.cos(np.pi * (np.arange(n) + 1) / n))[:, None]
    return x


def _db_rms(x: np.ndarray, sr: int) -> float:
    """dBFS of the loudest 1 ms of x [n, channels] (power mean over the channels)."""
    return 20.0 * math.log10(max(oneshot._loudest_ms(x, sr), 1e-6))


def _onset_after_tail(x: np.ndarray, sr: int, start: int) -> int:
    """Sample of the hit whose segment begins at `start` (up to 10 ms before it, after an earlier hit): hit_cut's
    onset, the first rise out of the tail with the tail read as its loudest ms over 20 ms, so the phase a low tone's
    tail is at when the segment begins does not move the onset."""
    onset = oneshot.hit_cut(x, sr, start / sr)[1]
    return start if onset is None else int(round(onset * sr))


def _agglomerate(dm: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage clusters of the points whose pairwise distances are dm [m, m] (symmetric): the two
    clusters with the least mean distance between their members merge until the least is over `threshold`.
    Ties go to the lowest indices, so the same distances in the same order always give the same clusters."""
    clusters = [[i] for i in range(len(dm))]
    d = np.array(dm, dtype=np.float64)
    np.fill_diagonal(d, np.inf)
    while len(clusters) > 1:
        i, j = divmod(int(np.argmin(d)), len(clusters))
        if d[i, j] > threshold:
            break
        ni, nj = len(clusters[i]), len(clusters[j])
        d[i, :] = d[:, i] = (ni * d[i, :] + nj * d[j, :]) / (ni + nj)       # Lance-Williams, average linkage
        d[i, i] = np.inf
        d = np.delete(np.delete(d, j, axis=0), j, axis=1)
        clusters[i] += clusters.pop(j)
    return clusters


def _pair_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """spectral_distance of two cuts over the shorter of the two, both ended by a fade."""
    n = min(len(a), len(b))
    return oneshot.spectral_distance(_fade(a[:n], sr), _fade(b[:n], sr), sr)


def group_cuts(cuts: list, keys: list, sr: int, threshold: float = THRESHOLD, sample: int = SAMPLE_HITS,
               max_groups: int = MAX_GROUPS, shift: float = 0.0, cache: dict | None = None) -> tuple[list[dict], list[int]]:
    """Group hit cuts ([n, channels] arrays, each from its hit's onset) by how alike they sound: (groups,
    ungrouped). `keys` are the hits' onset times (any numbers, distinct): the hits are put in that order first,
    so the order of `cuts` never changes the answer. Two cuts are compared over the shorter of the two (spectral_distance,
    mel, 60 dB floor). Each round: a deterministic sample of `sample` hits, evenly spaced in time (`shift`, in
    spacings, moves them along: 0 starts at the first hit), is grouped (average linkage, cut at `threshold`); every
    other remaining hit joins the group with the nearest medoid when it is within `threshold` of it; the hits left
    over go round again, until none is left or `max_groups` groups exist (the rest are `ungrouped`, not forced).
    A group is {"members": [indices into cuts], "medoid": index (the sample member with the least mean distance to
    the others), "distance": {index: distance to the medoid}}. `cache` ({(i, j): distance}, i < j) is filled and can
    be passed to a later call on the same cuts."""
    order = sorted(range(len(cuts)), key=lambda i: keys[i])
    seen = {} if cache is None else cache

    def dist(i: int, j: int) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            seen[(a, b)] = _pair_distance(cuts[a], cuts[b], sr)
        return seen[(a, b)]

    groups, remaining = [], order
    while remaining and len(groups) < max_groups:
        m = len(remaining)
        pick = remaining if m <= sample else [
            remaining[int(round((k + shift) * (m - 1) / (sample - 1 + shift)))] for k in range(sample)]
        pick = list(dict.fromkeys(pick))
        dm = np.zeros((len(pick), len(pick)))
        for a in range(len(pick)):
            for b in range(a + 1, len(pick)):
                dm[a, b] = dm[b, a] = dist(pick[a], pick[b])
        new = []
        for cl in _agglomerate(dm, threshold)[:max_groups - len(groups)]:
            mean = [np.mean([dm[a, b] for b in cl]) if len(cl) > 1 else 0.0 for a in cl]
            medoid = pick[cl[int(np.argmin(mean))]]
            new.append({"members": [pick[a] for a in cl], "medoid": medoid,
                        "distance": {pick[a]: dist(pick[a], medoid) for a in cl}})
        taken = {i for g in new for i in g["members"]}
        left = []
        for i in remaining:
            if i in taken:
                continue
            near = min(new, key=lambda g: (dist(i, g["medoid"]), g["medoid"]))
            d = dist(i, near["medoid"])
            if d <= threshold:
                near["members"].append(i)
                near["distance"][i] = d
            else:
                left.append(i)
        groups += new
        remaining = left
        if len(new) == 0:
            break
    return groups, remaining


def _hat_like(x: np.ndarray, sr: int) -> float:
    """Share of the stem's energy above HAT_HZ, from 8192-sample blocks (never the whole stem in one FFT)."""
    size = 8192
    freqs = np.fft.rfftfreq(size, 1.0 / sr)
    window = np.hanning(size)
    total = np.zeros(len(freqs))
    for a in range(0, max(1, len(x) - size + 1), size * 128):
        blocks = x[a:a + size * 128]
        m = len(blocks) // size
        if m == 0:
            blocks = np.concatenate([blocks, np.zeros((size - len(blocks), x.shape[1]))])
            m = 1
        mono = blocks[:m * size].mean(axis=1).reshape(m, size)
        total += (np.abs(np.fft.rfft(mono * window, axis=1)) ** 2).sum(axis=0)
    return float(total[freqs >= HAT_HZ].sum() / total.sum()) if total.sum() > 0 else 0.0


def _grid_onsets(x: np.ndarray, sr: int, onsets: list, t0: float, step_s: float) -> list[int]:
    """Onsets (samples) of hits that oneshot._segments cannot see because they start out of a tail: at each 16th
    line from the first onset on, with no onset within GRID_QUIET_S, the rise of the first-difference power (the
    mean over the channels of the squared sample-to-sample difference, which favours hats) from the 5 ms before a
    point to the 3 ms after it (1 ms apart), looked for every 0.5 ms within GRID_LEAD_S of the line; the best point
    counts as a hit when the rise is at least GRID_RISE_DB, the 3 ms after it are no more than GRID_FLOOR_DB under
    the stem's peak, and no onset lies within GRID_QUIET_S of it. Only a window of the stem around each line is
    read."""
    if not len(onsets):
        return []
    on = np.array(sorted(onsets))
    post, pre, gap, q, g, quiet = (max(1, int(round(v * sr))) for v in (0.003, 0.005, 0.001, 0.0005, GRID_LEAD_S, GRID_QUIET_S))
    peak2 = float(np.abs(x).max()) ** 2 * 10 ** (-GRID_FLOOR_DB / 10)
    found = []
    n0 = int(math.ceil((on[0] / sr - t0) / step_s))
    n1 = int(math.floor((len(x) / sr - t0) / step_s))
    for n in range(n0, n1 + 1):
        line = int(round((t0 + n * step_s) * sr))
        if float(np.abs(on - line).min()) < quiet:
            continue
        a, b = line - g - pre - gap - 1, line + g + post + 1
        if a < 0 or b > len(x):
            continue
        w = x[a:b]
        d2 = (np.diff(w, axis=0) ** 2).mean(axis=1)          # d2[k] is the step into sample a + k + 1
        p2 = (w[1:] ** 2).mean(axis=1)
        cd, cp = (np.concatenate([[0.0], np.cumsum(v)]) for v in (d2, p2))
        idx = np.arange(pre + gap, 2 * g + pre + gap + 1, q)   # candidate points, as offsets into w[1:]
        after = (cd[idx + post] - cd[idx]) / post
        before = (cd[idx - gap] - cd[idx - gap - pre]) / pre
        level = (cp[idx + post] - cp[idx]) / post
        ratio = np.where(level > peak2, after / np.maximum(before, 1e-30), 0.0)
        k = int(np.argmax(ratio))
        point = a + 1 + int(idx[k])
        if ratio[k] >= 10 ** (GRID_RISE_DB / 10) and float(np.abs(on - point).min()) >= quiet:
            found.append(point)
    return found


def _body_pitch(cut: np.ndarray, sr: int) -> float | None:
    """oneshot's body pitch of one hit's cut, None when it has none (unpitched, or too short to measure)."""
    try:
        return oneshot.profile_array(cut, sr, hit=0, single=True)["pitch_body_hz"]
    except ValueError:
        return None


def _split_by_pitch(x: np.ndarray, sr: int, hits: list, groups: list, pitch: dict) -> list:
    """Groups with those split whose members' body pitches (oneshot's, on each member's cut) fall in clusters more than
    PITCH_SPLIT_ST apart: the sorted pitches are cut at their largest gap over that, each side keeping at least two
    hits, again within each side. A group is read only when at least half of PITCH_SCREEN members spread over it have a
    pitch. Members without a pitch stay with the largest cluster. Each new group's medoid is the member whose pitch is
    nearest its median (the old medoid, where it stays) and its distances are measured to that."""
    out = []
    for g in groups:
        ids = g["members"]
        n = len(ids)
        screen = ids if n <= PITCH_SCREEN else [ids[int(round(k * (n - 1) / (PITCH_SCREEN - 1)))] for k in range(PITCH_SCREEN)]
        for i in screen:
            if i not in pitch:
                pitch[i] = _body_pitch(x[hits[i]["start"]:hits[i]["end"]], sr)
        if n < 4 or sum(pitch[i] is not None for i in screen) * 2 < len(screen):
            out.append(g)
            continue
        for i in ids:
            if i not in pitch:
                pitch[i] = _body_pitch(x[hits[i]["start"]:hits[i]["end"]], sr)
        clusters = [sorted((12 * math.log2(pitch[i]), i) for i in ids if pitch[i])]
        done = []
        while clusters:
            c = clusters.pop()
            best = max(((c[k + 1][0] - c[k][0], k) for k in range(1, len(c) - 2)), default=None)
            if best is not None and best[0] > PITCH_SPLIT_ST:           # each side keeps at least two hits
                clusters += [c[:best[1] + 1], c[best[1] + 1:]]
            else:
                done.append(c)
        if len(done) < 2:
            out.append(g)
            continue
        done.sort(key=lambda c: (-len(c), c[0][0]))
        unpitched = [i for i in ids if not pitch[i]]
        for j, c in enumerate(done):
            mem = sorted([i for _, i in c] + (unpitched if j == 0 else []))
            if g["medoid"] in mem:
                medoid = g["medoid"]
            else:
                mid = float(np.median([st for st, _ in c]))
                medoid = min((i for _, i in c), key=lambda i: (abs(12 * math.log2(pitch[i]) - mid), i))
            mc = x[hits[medoid]["start"]:hits[medoid]["end"]]
            out.append({"members": mem, "medoid": medoid, "distance": {
                i: (g["distance"][i] if medoid == g["medoid"] else _pair_distance(x[hits[i]["start"]:hits[i]["end"]], mc, sr))
                for i in mem}})
    return out


def _clean_hit(x: np.ndarray, sr: int, hits: list, ids: list, dist: dict) -> tuple[dict | None, str | None, list]:
    """(the clean hit, or None and why, the members read) among hits `ids` of one group: oneshot._clean_hit's rule on the group's
    members. Candidates, nearest their group's medoid first (at most MAX_CANDIDATES), must be complete (their cut
    falls 40 dB before it ends, oneshot._measure's `truncated`); of those the ones within CLEAN_BAND_DB of the loudest,
    then the ones with a quiet lead-in (the 20 ms before the onset LEAD_DB under the peak), if any, then the ones with
    at least ROOM_FRAC of the median room, then the ones within TYPICAL_DB of the median peak, if any. Of the
    last the one nearest the medoid wins (oneshot's rule takes the earliest; here the group is already known, so
    the most typical sound is taken). The members read are the first AGREE_MEMBERS complete candidates, nearest the medoid
    first: [{"hit", "audio"}] (empty when none is complete)."""
    order = sorted(ids, key=lambda i: (dist[i], hits[i]["start"]))[:MAX_CANDIDATES]
    cands = []
    for i in order:
        h = hits[i]
        try:
            got = oneshot._measure(x[h["start"]:h["end"]], sr, "the cut ends", quick=True)
        except ValueError:
            continue
        if got["truncated"]:
            continue
        lead = np.abs(x[max(0, h["start"] - int(oneshot.LEAD_S * sr)):h["start"]]).max(axis=1)
        quiet = not len(lead) or 20 * math.log10(max(float(lead.max()), 1e-12)) <= got["peak_db"] - oneshot.LEAD_DB
        cands.append({"i": i, "peak": got["peak_db"], "room": h["room"], "quiet": quiet})
    if not cands:
        return None, ("none of the %d member%s tried is complete: each is cut off by the next hit or by the %g s "
                      "limit before it has fallen 40 dB" % (len(order), "" if len(order) == 1 else "s", MAX_CUT_S)), []
    near = [{"hit": hits[c["i"]]["id"], "audio": x[hits[c["i"]]["start"]:hits[c["i"]]["end"]].copy()} for c in cands[:AGREE_MEMBERS]]
    loud = max(c["peak"] for c in cands)
    cands = [c for c in cands if c["peak"] >= loud - oneshot.CLEAN_BAND_DB]
    cands = [c for c in cands if c["quiet"]] or cands
    med_room = float(np.median([c["room"] for c in cands]))
    cands = [c for c in cands if c["room"] >= oneshot.ROOM_FRAC * med_room]
    med_peak = float(np.median([c["peak"] for c in cands]))
    cands = [c for c in cands if abs(c["peak"] - med_peak) <= oneshot.TYPICAL_DB] or cands
    h = hits[cands[0]["i"]]                     # candidates are in order of distance to the medoid
    return {"hit": h["id"], "start": h["start"], "end": h["end"], "onset_s": h["onset_s"], "level_db": h["level_db"],
            "audio": x[h["start"]:h["end"]].copy()}, None, near


def _evenly(ids: list, n: int) -> list:
    """At most `n` of `ids` (in time order), evenly spaced, the first and the last included."""
    return ids if len(ids) <= n else [ids[int(round(k * (len(ids) - 1) / (n - 1)))] for k in range(n)]


def _separation(x: np.ndarray, sr: int, hits: list, found: list) -> None:
    """Sets g["separation"], g["nearest_group"], g["separation_any"] and g["nearest_any"] on each group of `found`. The
    nearest other group is the one whose medoid is nearest this group's medoid (spectral distance, ties to the lower id);
    separation is the mean distance from SEP_MEMBERS of this group's measured members to SEP_OTHERS of that group's, both
    taken evenly spaced in time (the first and the last included). `separation_any` / `nearest_any` are measured against any
    other group; `separation` / `nearest_group` against the nearest group that is not tentative (at least TENTATIVE_N
    measured hits), because a one-to-three hit satellite beside a big group says nothing about whether two sounds are apart.
    None when there is no such group (a lone group, or only tentative ones). One distance per pair of groups plus at most
    192 per group and nearest group."""
    cut = lambda i: x[hits[i]["start"]:hits[i]["end"]]  # noqa: E731

    def mean_distance(g: dict, og: dict) -> float:
        ds = [_pair_distance(cut(i), cut(j), sr) for i in _evenly(g["members"], SEP_MEMBERS)
              for j in _evenly(og["members"], SEP_OTHERS)]
        return round(float(np.mean(ds)), 3)

    for gid, g in found:
        g["separation"] = g["nearest_group"] = g["separation_any"] = g["nearest_any"] = None
        mine = cut(g["medoid"])
        ranked = sorted(((_pair_distance(mine, cut(og["medoid"]), sr), o, og) for o, og in found if o != gid),
                        key=lambda t: (t[0], t[1]))
        if not ranked:
            continue
        _, o, og = ranked[0]
        g["nearest_any"], g["separation_any"] = o, mean_distance(g, og)
        big = [t for t in ranked if len(t[2]["members"]) >= TENTATIVE_N]
        if big:
            if big[0][1] == o:
                g["nearest_group"], g["separation"] = o, g["separation_any"]
            else:
                g["nearest_group"], g["separation"] = big[0][1], mean_distance(g, big[0][2])


def _scope(x: np.ndarray, sr: int, hits: list, found: list, hidden: set, ids: set, first: int, bars: int, steps: int) -> dict:
    """The groups of `found` (hit ids; see hit_groups) restricted to the hits in `ids`, over the `bars` bars from bar
    `first`. Ordered by count, then by the first member's time; "id" is the group's number in the whole stem.
    Assigned hits (overlapped hits placed in a group) count in `pattern`, `bars_with`, `off_grid`; not in `count`,
    `level_db`, `spread` or the clean hit."""
    groups = []
    for gid, g in found:
        mine = [i for i in g["members"] if i in ids]      # in time order
        extra = [i for i in g["assigned"] if i in ids]
        if not mine and not extra:
            continue
        on = [hits[i] for i in mine + extra if hits[i]["on_grid"]]
        on_assigned = [hits[i] for i in extra if hits[i]["on_grid"]]
        clean, why, near = _clean_hit(x, sr, hits, mine, g["distance"]) if mine else (None, "no measured member in this scope", [])
        dists = [g["distance"][i] for i in mine]
        groups.append({
            "id": gid, "count": len(mine), "assigned": len(extra), "tentative": len(mine) < TENTATIVE_N,
            "stability": g["stability"], "stability_status": g["stability_status"], "separation": g["separation"],
            "nearest_group": g["nearest_group"], "separation_any": g["separation_any"], "nearest_any": g["nearest_any"], "bars_with": len({hits[i]["bar"] for i in mine + extra}),
            "pattern": [len({h["bar"] for h in on if h["step"] == s}) / bars if bars else 0.0 for s in range(steps)],
            "assigned_pattern": [len({h["bar"] for h in on_assigned if h["step"] == s}) / bars if bars else 0.0
                                 for s in range(steps)],
            "level_db": float(np.median([hits[i]["level_db"] for i in mine])) if mine else None,
            "spread": {"median": float(np.median(dists)), "max": float(max(dists))} if mine else None,
            "off_grid": len(mine) + len(extra) - len(on), "members": mine, "assigned_members": extra,
            "clean_hit": clean, "clean_note": why, "read_members": near})
    groups.sort(key=lambda g: (-g["count"], (g["members"] + g["assigned_members"])[0]))
    lost = [h for h in (hits[i] for i in ids) if h["on_grid"] and first <= h["bar"] < first + bars and h["status"] in (
        "overlapped", "too_short", "no_lead", "ungrouped", "faint")]
    pat = lambda hs: [len({h["bar"] for h in hs if h["step"] == s}) / bars if bars else 0.0 for s in range(steps)]  # noqa: E731
    cells_h = sorted(c for c in hidden if first <= c[0] < first + bars)
    tail_u = {}                  # the level of the tail an unmeasured hit started out of (its level and its lead), None when unknown
    for h in lost:
        db = None if h["level_db"] is None or h["lead_db"] is None else round(h["level_db"] + h["lead_db"], 1)
        c = (h["bar"], h["step"])
        tail_u[c] = db if c not in tail_u or tail_u[c] is None else tail_u[c] if db is None else max(db, tail_u[c])
    cells_u = sorted(tail_u)
    return {"hits": len(ids), "bars": bars, "first_bar": first, "groups": groups, "unmeasured_pattern": pat(lost),
            "hidden_pattern": [len({b for b in range(first, first + bars) if (b, s) in hidden}) / bars if bars else 0.0
                               for s in range(steps)],
            "hidden_cells": [[b, s] for b, s in cells_h], "hidden_tail_db": [hidden[c] for c in cells_h],
            "unmeasured_cells": [[b, st] for b, st in cells_u], "unmeasured_tail_db": [tail_u[c] for c in cells_u],
            "unmeasured": [{"hit": i, "status": hits[i]["status"], "reason": hits[i]["reason"]}
                           for i in sorted(ids) if hits[i]["status"] not in ("grouped", "assigned")]}


def _assign_overlapped(x: np.ndarray, sr: int, hits: list, found: list, threshold: float) -> None:
    """Place overlapped hits in groups that already exist, never forming or moving one. A hit whose tail is
    ASSIGN_FULL_DB or more under it is compared with each group's medoid over its whole cut, one whose tail is between
    that and ASSIGN_SHORT_DB over its first ASSIGN_SHORT_S (the tail is lost under a decaying hit's body in the first
    tens of ms and keeps growing against it after), a louder tail: not at all. The hit goes to the nearest medoid when
    that is within `threshold` and ASSIGN_MARGIN times nearer than the second nearest; its status becomes "assigned".
    A group of fewer than TENTATIVE_N measured hits is a candidate only when its medoid rose out of near silence
    (`lead_db` of CLEAN_START_DB or less): a lone hit that sits on a tail is that tail's sound as much as its own, and
    attracted the hits of the real group that were heard over the same kind of tail."""
    for h in hits:
        if h["status"] != "overlapped" or h["lead_db"] is None:
            continue
        if h["lead_db"] > ASSIGN_SHORT_DB:
            h["reason"] += "; the tail is too loud to compare it with a group"
            continue
        window = None if h["lead_db"] <= ASSIGN_FULL_DB else int(round(ASSIGN_SHORT_S * sr))
        cut = x[h["start"]:h["end"]]
        near = []
        for gid, g in found:
            m = hits[g["medoid"]]
            if len(g["members"]) < TENTATIVE_N and not (m["lead_db"] is not None and m["lead_db"] <= CLEAN_START_DB):
                continue
            mc = x[m["start"]:m["end"]]
            n = min(len(cut), len(mc)) if window is None else min(len(cut), len(mc), window)
            near.append((oneshot.spectral_distance(_fade(cut[:n], sr), _fade(mc[:n], sr), sr), gid, n))
        near.sort()
        if not near:
            continue
        if near[0][0] > threshold:
            h["reason"] += "; not assigned: the nearest group is %.2f away (over %.2f)" % (near[0][0], threshold)
        elif len(near) > 1 and near[1][0] < ASSIGN_MARGIN * near[0][0]:
            h["reason"] += "; not assigned: groups %d and %d are %.2f and %.2f away (not %.0f %% apart)" % (
                near[0][1], near[1][1], near[0][0], near[1][0], 100 * (ASSIGN_MARGIN - 1))
        else:
            d, gid, n = near[0]
            h["status"], h["group"], h["distance"] = "assigned", gid, d
            h["reason"] = "overlapped (%s); assigned to group %d over its first %.0f ms (distance %.2f%s)" % (
                h["reason"].split(";")[0].removeprefix("it starts out of "), gid, 1000 * n / sr, d,
                "" if len(near) == 1 else ", the next group %.2f" % near[1][0])
            dict(found)[gid]["assigned"].append(h["id"])


def hit_groups(stem, tempo: float, t0: float, sections: list | None = None, sr: int | None = None,
               beats_per_bar: int = 4, threshold: float = THRESHOLD, max_cut_s: float = MAX_CUT_S,
               sample: int = SAMPLE_HITS, max_groups: int = MAX_GROUPS, grid_onsets: bool = False,
               stability_runs: int = STABILITY_RUNS, ref_power: float | None = None) -> dict:
    """Every hit in one drum stem, with its place on the grid, grouped by how alike the hits sound (see the module
    docstring for the choices). `stem` is a path or an array [n] / [n, channels] with its rate in `sr`; bar 0 starts
    at `t0` seconds, 16ths are a quarter of a beat of `tempo` (4 * beats_per_bar to a bar); `sections` are
    {name, start_bar, bars} (as reference_profile takes them). `grid_onsets` (off by default) also looks for hits at
    16th lines that `oneshot._segments` cannot see, and is meant for hat-like stems (see `hat_like`): on a kick, snare or
    toms stem it reports bleed and tails as hits (118-355 of them on the Flare stems). `stability_runs` is the number of
    regroupings with a shifted sample that `stability` is measured against (0: not measured). `ref_power` is for a slice of a longer stem: the peak power of the whole stem's envelope (`oneshot.envelope_peak`), so that hits are found against the stem's loudest 5 ms and not the slice's own (a slice with nothing within 40 dB of it has no hits).

    Returns {"sample_rate", "tempo", "t0", "threshold", "linkage": "average", "max_cut_s", "grid_onsets", "hat_like"
    (the share of the stem's energy above 4 kHz), "grid_hits" (hits found by `grid_onsets`), "hits", "groups",
    "unmeasured_pattern", "hidden_pattern", "unmeasured", "sections", "notes", "bars", "first_bar", "hidden_cells",
    "unmeasured_cells", "hidden_tail_db", "unmeasured_tail_db"}.
    A hit is {"id", "source" ("segment" or "grid"), "onset_s", "bar", "step" (16th, 0-15, of the nearest grid line),
    "offset_ms" (onset minus that line), "on_grid" (within 20 ms of it), "level_db" (loudest ms, dBFS), "lead_db" (the
    10 ms before the hit against its loudest ms; None at the start of the stem), "through" (16th lines inside its cut
    that its own tail still sounds through), "start", "end" (cut, in samples of the stem), "room" (samples to the
    next hit), "status" ("grouped", "assigned" (overlapped, placed in an existing group), "too_short", "no_lead",
    "faint", "overlapped", "silent" or "ungrouped"), "reason" (why not grouped), "group", "distance" (to its group's
    medoid, over the window that was compared)}.
    A group is {"id", "count" (measured members), "assigned" (overlapped hits placed in it), "tentative" (under 4 measured
    members), "stability", "stability_status", "separation", "nearest_group", "separation_any", "nearest_any" (see below), "bars_with" (bars it plays in), "pattern" (per 16th, the share of the scope's
    bars it plays at, on-grid hits, assigned ones included), "assigned_pattern" (the same for the assigned hits alone),
    "level_db" (median), "spread" ({"median", "max"} of the members' distances to the medoid), "off_grid", "members"
    (hit ids), "assigned_members", "clean_hit" ({"hit", "start", "end", "onset_s", "level_db", "audio"} or None; never
    an assigned hit), "clean_note" (why None), "read_members" ([{"hit", "audio"}]: up to AGREE_MEMBERS complete members nearest the
    medoid, which the role rules read alongside the clean hit)}.
    `stability` is how the group holds when the sample changes: the stem is grouped again `stability_runs` times with
    the sample moved along; the k-th of the `sample` hits of a round is the one at position
    round((k + shift) * (m - 1) / (sample - 1 + shift)) of the m hits still ungrouped, with shift 0 in the grouping
    itself and (r + 1) / (stability_runs + 1) in regrouping r (1/3 and 2/3 by default). For each run, the Jaccard index
    (members in both / members in either) of the group with the group of that run it shares most members with; the
    figure is the mean over the runs. 1.0: the same members every time; a group that a run splits or merges with another
    reads lower. Material whose distances lie around the threshold (a continuum of decays, a velocity gradient) is cut
    where the sample puts the cut, and reads well under 1; the figure is for the stem as a whole, section groups carry it
    unchanged. It is None, and `stability_status` says why, when the sample cannot move ("stable by construction": no more
    measured hits than `sample`, so the first round's sample took every hit, the grouping is exact average linkage and no
    regrouping can differ; none is run, and a note says so), with `stability_runs=0` ("not asked for"), and for a tentative
    group ("not measured"); "measured" otherwise.
    `separation` is how far the group lies from its neighbour: the nearest other group that is not tentative (at least 4
    measured hits) is the one whose medoid is nearest this group's medoid (ties to the lower id), its id is `nearest_group`,
    and `separation` is the mean spectral distance from 16 of this group's measured members to 12 of that group's, each set
    evenly spaced in time with the first and the last included (all of them when fewer). Above `threshold`: the two sounds
    are apart; near or under it: they are a continuum the threshold cut. A one-to-three hit group is not a neighbour: it is
    a stray that the sample left out, and a big group sitting 1.2 from a 3-hit satellite had read as "not apart". The figure
    against ANY other group, tentative ones included, is kept as `separation_any` with `nearest_any`. `separation` is None
    when there is no non-tentative neighbour (a lone group, or only tentative others): such a group has no rival, and
    `name_roles` treats it as it treats a lone group. It is measured on the final groups, so a pitch split is included.
    `groups` is the whole stem (its bars run from the first real hit's bar to the last's); each of `sections` has the same
    groups restricted to its bars, with the section's own count, pattern over its `bars`, level, spread and clean hit.
    `unmeasured_pattern` (per 16th, the share of the scope's bars with an on-grid hit there that is overlapped and not
    assigned, too short, without a lead, faint or ungrouped) leaves out the hits that were never found: a hit inside another's
    tail is not a hit to the detector. `hidden_pattern` (per 16th, the share of the scope's bars in which that line falls
    inside the tail of a hit, `through`, with no hit found at it) is where such hits could be; with `grid_onsets` on it
    is what is left after the search. The same two, per bar rather than per 16th, for the roles' judgement of which steps a
    group could be seen playing at: `first_bar` and `bars` (the scope: the bars from the first real hit's, or the
    section's) and `hidden_cells` / `unmeasured_cells`, the [bar, step] pairs that were hidden under a tail, or held an
    on-grid hit that was not measured (the statuses of `unmeasured_pattern`); `hidden_tail_db` and `unmeasured_tail_db`
    are, cell by cell, the level of the tail there (the 1 ms at the line, dBFS, the scale of a hit's `level_db`; for an unmeasured
    hit the level of the tail it started out of, None when unknown). Whether a hidden cell hides a given group is `name_roles`'s
    judgement: only when the tail is within the rise `_segments` needs of that group's level (see `_use`). The sections carry their own."""
    if not (tempo > 0 and math.isfinite(tempo)) or not math.isfinite(t0):
        raise ValueError("tempo must be above 0 and t0 a number of seconds, got %r and %r" % (tempo, t0))
    if beats_per_bar < 1 or sample < 2 or max_groups < 1 or threshold <= 0 or stability_runs < 0:
        raise ValueError("beats_per_bar and max_groups need at least 1, sample at least 2, threshold above 0, "
                         "stability_runs 0 or more")
    if sections is not None:
        check_sections(sections)
    if isinstance(stem, (str, pathlib.Path)):
        x, sr = load(stem)
    else:
        if sr is None:
            raise ValueError("an array needs its sample rate: pass sr=")
        x = np.asarray(stem)
        x = x[:, None] if x.ndim == 1 else x
    check_channels(x)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("the stem holds NaN or infinite samples")
    sr = int(sr)
    x = oneshot._undc(x, sr)
    out = {"sample_rate": sr, "tempo": tempo, "t0": t0, "threshold": threshold, "linkage": "average",
           "max_cut_s": max_cut_s, "grid_onsets": grid_onsets, "hat_like": None, "grid_hits": 0, "hits": [],
           "groups": [], "unmeasured_pattern": [], "hidden_pattern": [], "unmeasured": [], "sections": [], "notes": [],
           "bars": 0, "first_bar": 0, "hidden_cells": [], "unmeasured_cells": [], "hidden_tail_db": [], "unmeasured_tail_db": []}
    if not len(x) or float(np.abs(x).max()) <= 0:
        out["notes"].append("the stem is silent: no hits")
        return out
    out["hat_like"] = round(_hat_like(x, sr), 4)
    segs = oneshot._segments(x, sr, ref_power)
    onsets = [(oneshot._hit_onset(x[s:e], sr) if s == 0 else _onset_after_tail(x, sr, s), "segment") for s, e in segs]
    step_s, steps = 60.0 / tempo / 4.0, 4 * beats_per_bar
    if grid_onsets:
        onsets = sorted(onsets + [(o, "grid") for o in _grid_onsets(x, sr, [o for o, _ in onsets], t0, step_s)])
        out["grid_hits"] = sum(src == "grid" for _, src in onsets)
        if out["hat_like"] < 0.5:
            out["notes"].append("grid_onsets is meant for hat-like stems and this one reads hat_like %.2f: the %d hits it added "
                                "are bleed, tails or the phase of a low tone as often as hits" % (out["hat_like"], out["grid_hits"]))
    guard, lead_end, lead_len = (int(round(v * sr)) for v in (END_GUARD_S, 0.001, LEAD_WINDOW_S))
    hits, cuts, keys, tails = [], [], [], {}
    frame, glead = max(1, int(round(0.001 * sr))), int(round(GRID_LEAD_S * sr))
    for k, (o, source) in enumerate(onsets):
        last = k + 1 == len(onsets)
        nxt = len(x) if last else onsets[k + 1][0]
        end = min(nxt if last else nxt - guard, o + int(round(max_cut_s * sr)))
        rel = o / sr - t0
        line = int(round(rel / step_s))
        bar, step = divmod(line, steps)
        off = rel - line * step_s
        h = {"id": k, "source": source, "onset_s": o / sr, "bar": bar, "step": step, "offset_ms": round(1000 * off, 2),
             "on_grid": abs(off) <= GRID_LEAD_S, "level_db": None, "lead_db": None, "through": 0, "start": o,
             "end": end, "room": nxt - o, "status": "grouped", "reason": None, "group": None, "distance": None}
        hits.append(h)
        cut = x[o:end]
        if float(np.abs(cut).max()) <= 0:
            h["status"], h["reason"] = "silent", "the cut holds only zeros"
            continue
        h["level_db"] = _db_rms(cut, sr)
        a = max(0, o - lead_end - lead_len)
        if o - lead_end - a >= lead_len // 2:           # something of the stem lies before the hit
            before = float((x[a:o - lead_end] ** 2).mean())
            h["lead_db"] = round(10.0 * math.log10(max(before, 1e-12)) - h["level_db"], 1)
        loud = oneshot._loudest_ms(cut, sr) ** 2
        for n in range(line + 1, line + steps + 1):
            i = int(round((t0 + n * step_s) * sr)) - o
            if i + frame > len(cut) - glead:
                break
            at_line = float((cut[i:i + frame] ** 2).mean())
            if i > glead and at_line >= loud * 10 ** (-THROUGH_DB / 10):
                h["through"] += 1
                tails.setdefault(k, []).append((n, round(10.0 * math.log10(max(at_line, 1e-12)), 1)))
        if nxt - o < int(round(MIN_CUT_S * sr)):
            h["status"], h["reason"] = "too_short", "the next hit starts %.0f ms after it (under %.0f ms)" % (
                1000 * (nxt - o) / sr, 1000 * MIN_CUT_S)
        elif h["lead_db"] is None:
            h["status"], h["reason"] = "no_lead", (
                "it starts within %.0f ms of the start of the stem: a tail or bed under it cannot be ruled out"
                % (1000 * (lead_end + lead_len // 2) / sr))
        elif h["lead_db"] > OVERLAP_DB:
            h["status"], h["reason"] = "overlapped", (
                "it starts out of a tail or bed only %.0f dB under its own loudest ms (it needs %.0f)"
                % (-h["lead_db"], -OVERLAP_DB))
    for h in hits:
        if h["status"] != "grouped":
            continue
        around = hits[max(0, h["id"] - NEIGHBOURS):h["id"]] + hits[h["id"] + 1:h["id"] + NEIGHBOURS + 1]
        near = [a["level_db"] for a in around if a["level_db"] is not None]
        if len(near) >= 3 and h["level_db"] < float(np.median(near)) - FAINT_DB:
            h["status"], h["reason"] = "faint", (
                "%.0f dB under the hits around it (over %.0f dB): bed, bleed or a ghost note; it is counted in "
                "`unmeasured_pattern`" % (float(np.median(near)) - h["level_db"], FAINT_DB))
        else:
            h["_cut"] = len(cuts)
            cuts.append(x[h["start"]:h["end"]])
            keys.append(h["start"])
    index = {h["_cut"]: h["id"] for h in hits if "_cut" in h}
    seen, pitch = {}, {}

    def cluster(shift: float):
        groups, rest = group_cuts(cuts, keys, sr, threshold, sample, max_groups, shift, seen)
        for g in groups:                                  # from cut numbers to hit ids
            g["members"] = sorted(index[i] for i in g["members"])
            g["medoid"] = index[g["medoid"]]
            g["distance"] = {index[i]: d for i, d in g["distance"].items()}
        return _split_by_pitch(x, sr, hits, groups, pitch), [index[c] for c in rest]

    groups, rest = cluster(0.0) if cuts else ([], [])
    movable = len(cuts) > sample                   # with no more measured hits than the sample every regrouping is the same
    runs = [cluster((r + 1) / (stability_runs + 1))[0] for r in range(stability_runs)] if cuts and movable else []
    if stability_runs and cuts and not movable:
        out["notes"].append("stability is not measured: %d measured hits are no more than the sample (%d), so the sample "
                            "cannot move: every group is stable by construction (the grouping is exact average linkage and "
                            "no regrouping can differ)" % (len(cuts), sample))
    groups.sort(key=lambda g: (-len(g["members"]), g["members"][0]))
    found = list(enumerate(groups))
    for gid, g in found:
        g["assigned"] = []
        shares = []
        for other in runs:
            best = max(other, key=lambda o: len(set(o["members"]) & set(g["members"])))
            both = len(set(best["members"]) & set(g["members"]))
            shares.append(both / (len(g["members"]) + len(best["members"]) - both))
        g["stability"] = round(float(np.mean(shares)), 3) if shares and len(g["members"]) >= TENTATIVE_N else None
        g["stability_status"] = ("measured" if g["stability"] is not None else "not asked for" if not stability_runs
                                 else "stable by construction" if not movable else "not measured")
        for i, d in g["distance"].items():
            hits[i]["group"], hits[i]["distance"] = gid, d
    for i in rest:
        hits[i]["status"], hits[i]["reason"] = "ungrouped", "more than %d distinct sounds: not forced into a group" % max_groups
    _separation(x, sr, hits, found)
    _assign_overlapped(x, sr, hits, found, threshold)
    for h in hits:
        h.pop("_cut", None)
    out["hits"] = hits
    if not hits:
        return out
    hidden = {}                                  # (bar, step) -> the tail's level there (dBFS, loudest-ms scale of `level_db`)
    for ns in tails.values():                    # a tail's cut ends at the next hit: no line with a hit
        for n, db in ns:
            hidden[divmod(n, steps)] = max(db, hidden.get(divmod(n, steps), -math.inf))
    real = [h["bar"] for h in hits if h["status"] not in ("faint", "silent", "no_lead")]
    first, last = (min(real), max(real)) if real else (0, 0)
    whole = _scope(x, sr, hits, found, hidden, {h["id"] for h in hits}, first, last - first + 1, steps)
    out.update({k: whole[k] for k in ("groups", "unmeasured_pattern", "hidden_pattern", "unmeasured", "bars", "first_bar",
                                      "hidden_cells", "unmeasured_cells", "hidden_tail_db", "unmeasured_tail_db")})
    for sec in sections or []:
        a, b = sec["start_bar"], sec["start_bar"] + sec["bars"]
        got = _scope(x, sr, hits, found, hidden, {h["id"] for h in hits if a <= h["bar"] < b}, a, sec["bars"], steps)
        out["sections"].append({"name": sec.get("name"), "start_bar": a, "bars": sec["bars"], **got})
    if any(h["through"] for h in hits):
        out["notes"].append("a hit played inside another hit's tail, without rising 15 dB out of it, is not found "
                            "(grid_onsets=True looks for it at the 16th lines): `through` lists the 16th lines each "
                            "tail still sounds through, `hidden_pattern` where they fall")
    return out


# ---------------------------------------------------------------- role names

def _use(result: dict, g: dict) -> dict:
    """How a group plays on the grid, from its measured and assigned hits that sit on it: "counts" (hits per 16th), "n"
    (their number) and "steps", one row per 16th: the bars the group plays there (`played`), the bars in which that
    16th could be seen (`observable`), and the shares of the group's own bars (every bar it has a hit in) and of the
    observable ones. A bar is not observable at a 16th when the line fell inside another hit's tail with no hit found
    at it AND the tail there is within the rise a hit needs to be found (oneshot.RISE_DB, 15 dB; GRID_RISE_DB with
    `grid_onsets`) of the group's own level (`hidden_cells` with `hidden_tail_db`: a hat hidden under an open hat's tail
    cannot be seen playing there, but one that rises 15 dB or more out of a quiet tail would have been found: the cell is
    seen and empty), or held an on-grid hit that was not measured (`unmeasured_cells`: overlapped, too short, without a
    lead, faint or ungrouped; it may be this group's, whatever the tail), and the group does not play there. Only the group's own bars count (the bars in which it plays at
    all), as in `bars_with`, so a group that lives in one section is judged over that section. A 16th observable in
    fewer than OBSERVE_MIN bars is "unknown"; otherwise "in use" when its share of observable bars is at least STEP_MIN."""
    steps = max(len(g["pattern"]), 1)
    hits = [result["hits"][i] for i in g["members"] + g["assigned_members"]]
    on = [h for h in hits if h["on_grid"]]
    own = {h["bar"] for h in hits}
    blind = {(b, st) for b, st in result.get("unmeasured_cells", [])}      # a hit was found there: it may be this group's
    tails = result.get("hidden_tail_db")
    rise = GRID_RISE_DB if result.get("grid_onsets") else oneshot.RISE_DB
    # Without tail levels (a hand-built result, or one stored before they existed) or a group level, a hidden cell cannot
    # be judged and stays hidden, which can make a sparse part read dense; `hidden_unjudged` says so in the output
    tails_known = tails is not None and g.get("level_db") is not None
    for k, (b, st) in enumerate(result.get("hidden_cells", [])):
        # hidden FOR THIS GROUP only when a hit of its level would not have risen out of the tail: the tail within `rise` dB of the
        # group's level
        if not tails_known or tails[k] > g["level_db"] - rise:
            blind.add((b, st))
    counts, rows = np.zeros(steps), []
    for st in range(steps):
        at = [h for h in on if h["step"] == st]
        played = {h["bar"] for h in at}
        seen = len(own) - len([b for b in own if (b, st) in blind and b not in played])
        share_obs = len(played) / seen if seen else None
        counts[st] = len(at)
        rows.append({"step": st, "played_bars": len(played), "own_bars": len(own), "observable_bars": seen,
                     "share_all": round(len(played) / len(own), 3) if own else 0.0,
                     "share_observable": None if share_obs is None else round(share_obs, 3),
                     "state": "unknown" if seen < OBSERVE_MIN else "in use" if share_obs >= STEP_MIN else "not in use"})
    return {"counts": counts, "n": int(counts.sum()), "steps": rows,
            "hidden_unjudged": bool(result.get("hidden_cells")) and not tails_known}


def _read(audio: np.ndarray, sr: int) -> tuple[dict | None, str | None]:
    """(oneshot's readings of one hit that the rules use, or None and why)."""
    try:
        p = oneshot.profile_array(audio, sr, hit=0, single=True)
    except ValueError as e:
        return None, str(e)
    return {"fall_20_ms": p["fall_20_ms"], "fall_40_ms": p["fall_40_ms"], "pitched": p["pitched"],
            "pitch_hz": p["pitch_body_hz"], "noise_body_db": (p["noise_db"] or {}).get("body")}, None


def _clean_measures(g: dict, sr: int, cache: dict) -> tuple[dict | None, str | None]:
    """(the readings of the group's clean hit, or None and why). `cache` holds the readings by hit id, so that a hit that
    is both the clean hit and one of the members read is measured once."""
    c = g["clean_hit"]
    if c is None:
        return None, g["clean_note"] or "no clean hit"
    if c["hit"] not in cache:
        cache[c["hit"]] = _read(c["audio"], sr)
    m, why = cache[c["hit"]]
    return (m, None) if m else (None, "the clean hit cannot be measured: %s" % why)


def _member_readings(g: dict, sr: int, cache: dict) -> list[dict]:
    """The readings of the members the measurement rules read: the group's `read_members` (up to AGREE_MEMBERS complete
    members nearest the medoid), or the clean hit alone in a result that has none. A member that cannot be measured is left out."""
    members = g.get("read_members")
    if members is None:
        members = [] if g["clean_hit"] is None else [{"hit": g["clean_hit"]["hit"], "audio": g["clean_hit"]["audio"]}]
    out = []
    for mem in members:
        if mem["hit"] not in cache:
            cache[mem["hit"]] = _read(mem["audio"], sr)
        if cache[mem["hit"]][0] is not None:
            out.append(cache[mem["hit"]][0])
    return out


def _pattern_numbers(use: dict) -> dict:
    """The pattern numbers the hh rules read: "share_off" (share of the on-grid hits on the offbeats), and for the offbeats
    ("off") and the odd 16ths ("odd") (in use, unknown, known, the number of 16ths of that kind), and "off_per_bar" and
    "odd_per_bar": hits a bar on those 16ths over all the group's own bars, and "off_density" and "odd_density": the density
    the rules read, the sum over the known 16ths of that kind of each one's share of the bars it could be seen in (an unknown
    16th adds nothing; the known-16ths floors cover those)."""
    rows = use["steps"]
    kind = lambda pick: [r for r in rows if pick(r["step"])]  # noqa: E731
    off, odd = kind(lambda st: st % 4 == 2), kind(lambda st: st % 2 == 1)
    count = lambda rs: (sum(r["state"] == "in use" for r in rs), sum(r["state"] == "unknown" for r in rs),  # noqa: E731
                        sum(r["state"] != "unknown" for r in rs), len(rs))
    bars = rows[0]["own_bars"] if rows else 0
    return {"share_off": float(sum(use["counts"][r["step"]] for r in off) / use["n"]) if use["n"] else 0.0,
            "off": count(off), "odd": count(odd),
            "off_per_bar": float(sum(use["counts"][r["step"]] for r in off) / bars) if bars else 0.0,
            "odd_per_bar": float(sum(use["counts"][r["step"]] for r in odd) / bars) if bars else 0.0,
            "off_density": float(sum(r["played_bars"] / r["observable_bars"] for r in off if r["state"] != "unknown")),
            "odd_density": float(sum(r["played_bars"] / r["observable_bars"] for r in odd if r["state"] != "unknown"))}


def _enough(in_use: int, known: int, total: int, needed: int, known_min: int) -> bool:
    """`needed` of `total` 16ths in use, judged over the known ones: at least `known_min` known and no more than
    `total - needed` of the known ones empty (not in use), so an unknown 16th counts neither for nor against. Monotone: a
    16th that is in use and becomes unknown (hidden) leaves the number of empty ones as it was, so it never takes a name away;
    the share of the known ones this replaces did (8 known, 6 in use: named; 7 known, 5 in use: not)."""
    return known >= known_min and known - in_use <= total - needed


def _unknown_text(unknown: int, what: str) -> str:
    return "" if not unknown else " (%d of the %s unknown: observable in under %d bars)" % (unknown, what, OBSERVE_MIN)


def _label(kind: str, m: dict) -> str | None:
    """What one hit's readings say it is under the measurement rules of stem `kind` (None: nothing)."""
    fall = m["fall_20_ms"]
    if kind == "hh":
        return "hh open" if fall is not None and fall >= OPEN_FALL_20_MS else None
    if kind == "snare":
        if fall is not None and fall < RIM_FALL_20_MS:
            return "rim"
        return "snare" if m["noise_body_db"] is not None and SNARE_NOISE_MIN_DB < m["noise_body_db"] <= SNARE_NOISE_MAX_DB else None
    if kind == "toms":
        return "tom" if m["pitched"] else None
    if kind == "kick":
        return "kick" if m["pitched"] and m["pitch_hz"] is not None and m["pitch_hz"] <= KICK_MAX_HZ else None
    return None


def _votes(kind: str, reads: list) -> dict:
    """{name: members that read it} over `reads` for the measurement rules of stem `kind` (None: nothing)."""
    votes = {}
    for m in reads:
        lab = _label(kind, m)
        votes[lab] = votes.get(lab, 0) + 1
    return votes


def _vote_text(votes: dict, n: int) -> str:
    """"6 of 14 snare, 5 rim, 3 neither", the readings that agreed first."""
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0] is None, kv[0] or ""))
    return ", ".join(("%d of %d %s" % (v, n, lab or "neither")) if k == 0 else "%d %s" % (v, lab or "neither")
                     for k, (lab, v) in enumerate(ranked))


def _apply_rules(kind: str, use: dict, reads: list, need: int = 1) -> tuple[str | None, str | None]:
    """(name, the rule that gave it) from the first rule of stem `kind` that fits, else (None, None). The pattern rules read
    the group's pattern. The measurement rules (`hh open`, `rim`, `snare`, `tom`, `kick`) read `reads`, the readings of up to
    AGREE_MEMBERS members (see `_member_readings`), and name only when at least AGREE_SHARE of them read the same and at least
    `need` were read (name_roles passes AGREE_MIN for a result that has `read_members`); with none they do not fire. "tom" is ranked low / mid / high by name_roles."""
    pn = _pattern_numbers(use)
    if kind == "hh":
        (off_used, off_unk, off_known, _), (odd_used, odd_unk, odd_known, _) = pn["off"], pn["odd"]
        if pn["share_off"] >= OFFBEAT_SHARE and _enough(off_used, off_known, 4, OFFBEAT_STEPS, OFFBEAT_KNOWN) \
                and pn["off_density"] >= OFFBEAT_PER_BAR:
            return "hh offbeat", ("%.0f %% of its hits on the offbeats, %d of %d known offbeats in use%s, %.1f hits a bar on them over the "
                                  "bars they could be seen in (%.1f over all its bars)" % (
                100 * pn["share_off"], off_used, off_known, _unknown_text(off_unk, "offbeats"), pn["off_density"], pn["off_per_bar"]))
        if _enough(odd_used, odd_known, 8, ODD_STEPS, ODD_KNOWN) and pn["odd_density"] >= ODD_PER_BAR:
            return "hh 16th", "%d of %d known odd 16ths in use%s, %.1f hits a bar on them over the bars they could be seen in (%.1f over all its bars)" % (
                odd_used, odd_known, _unknown_text(odd_unk, "odd 16ths"), pn["odd_density"], pn["odd_per_bar"])
    if kind not in ("hh", "snare", "toms", "kick") or not reads or len(reads) < need:
        return None, None
    votes = _votes(kind, reads)
    agreed = [lab for lab, v in votes.items() if lab is not None and v >= AGREE_SHARE * len(reads) - 1e-9]
    if not agreed:
        return None, None
    name, of = agreed[0], [m for m in reads if _label(kind, m) == agreed[0]]
    pre = "%d of %d members: " % (len(of), len(reads))
    med = lambda key: float(np.median([m[key] for m in of]))  # noqa: E731
    if name == "hh open":
        return name, pre + "falls 20 dB in %.0f ms (at least %.0f)" % (med("fall_20_ms"), OPEN_FALL_20_MS)
    if name == "rim":
        return name, pre + "falls 20 dB in %.1f ms (under %.0f)" % (med("fall_20_ms"), RIM_FALL_20_MS)
    if name == "snare":
        return name, pre + "a tone under the noise: the body's noise share is %.1f dB (over %.1f, at most %.1f)" % (
            med("noise_body_db"), SNARE_NOISE_MIN_DB, SNARE_NOISE_MAX_DB)
    if name == "tom":
        return name, pre + "pitched, body at %.0f Hz" % med("pitch_hz")
    return name, pre + "pitched, body at %.0f Hz (at most %.0f)" % (med("pitch_hz"), KICK_MAX_HZ)


def _no_rule(kind: str, use: dict, m: dict | None, why_no_clean: str | None, reads: list, need: int = 1) -> str:
    """Why no rule of stem `kind` fits, with the numbers the rules read."""
    if kind not in ("hh", "snare", "toms", "kick"):
        return "the stem kind %r has no rules" % (kind,)
    parts = []
    if kind == "hh":
        pn = _pattern_numbers(use)
        (off_used, off_unk, off_known, _), (odd_used, odd_unk, odd_known, _) = pn["off"], pn["odd"]
        if odd_known < ODD_KNOWN:
            parts.append("only %d of 8 odd 16ths could be observed (%d needed to name a 16th hat)" % (odd_known, ODD_KNOWN))
        if off_known < OFFBEAT_KNOWN:
            parts.append("only %d of 4 offbeats could be observed (%d needed to name an offbeat hat)" % (off_known, OFFBEAT_KNOWN))
        parts.append("%.0f %% of its hits on the offbeats, %d of %d known offbeats%s and %d of %d known odd 16ths%s in use over the "
                     "bars they could be seen in (a name needs %.0f %% and at most %d of the 4 offbeats empty, or at most %d of the 8 odd 16ths "
                     "empty; at least %d and %d known)" % (
                         100 * pn["share_off"], off_used, off_known, _unknown_text(off_unk, "offbeats"), odd_used, odd_known,
                         _unknown_text(odd_unk, "odd 16ths"), 100 * OFFBEAT_SHARE, 4 - OFFBEAT_STEPS, 8 - ODD_STEPS, OFFBEAT_KNOWN, ODD_KNOWN))
        parts.append("%.1f hits a bar on the offbeats and %.1f on the odd 16ths over the bars they could be seen in (%.1f and %.1f over all "
                     "its own bars; a name needs %.0f and %.0f)" % (
                         pn["off_density"], pn["odd_density"], pn["off_per_bar"], pn["odd_per_bar"], OFFBEAT_PER_BAR, ODD_PER_BAR))
    if m is None:
        parts.append("no clean hit to measure: %s" % why_no_clean)
    else:
        parts.append("clean hit: falls 20 dB in %s ms, %s, noise share of the body %s dB" % (
            m["fall_20_ms"], "pitched at %.0f Hz" % m["pitch_hz"] if m["pitched"] else "unpitched", m["noise_body_db"]))
    if reads and len(reads) < need:
        parts.append("too few complete members: %d read, nearest the medoid (a name needs at least %d)" % (len(reads), need))
    elif reads:
        parts.append("the %d member%s read, nearest the medoid, do not agree (a name needs %.0f %% to read the same): %s" % (
            len(reads), "" if len(reads) == 1 else "s", 100 * AGREE_SHARE, _vote_text(_votes(kind, reads), len(reads))))
    return "no rule of the %s stem fits: %s" % (kind, "; ".join(parts))


def _eligibility(g: dict, threshold: float) -> list[str]:
    """Why the group may not be named (an empty list when it may): fewer than NAME_MIN_HITS measured hits, stability not
    measured (except "stable by construction": the sample took every hit, so no regrouping can differ) or under NAME_STABILITY, separation not over the grouping threshold. `separation` is measured against the
    nearest group of at least TENTATIVE_N measured hits; a group with none (separation None: a lone group, or only
    tentative others) has no rival to be taken for and passes that test."""
    why = []
    if g["count"] < NAME_MIN_HITS:
        why.append("only %d measured hit%s (a name needs %d)" % (g["count"], "" if g["count"] == 1 else "s", NAME_MIN_HITS))
    status = g.get("stability_status")
    if g["stability"] is None:
        if g["count"] >= NAME_MIN_HITS and status != "stable by construction":
            why.append("stability is not measured: " + (
                "it was not asked for (stability_runs=0)" if status == "not asked for" else "the result does not say why"))
    elif g["stability"] < NAME_STABILITY:
        why.append("stability %.2f is under %.2f" % (g["stability"], NAME_STABILITY))
    if g["separation"] is not None and g["separation"] <= threshold:
        why.append("separation %.2f is not over the grouping threshold %.2f (nearest group %s)" % (
            g["separation"], threshold, g["nearest_group"]))
    return why


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _close_alike(a: dict, b: dict, ua: dict, ub: dict, threshold: float) -> bool:
    """Two groups that may be one sound: one is the other's nearest group within NEAR_SEPARATION times the threshold,
    and they play at the same 16ths (cosine of the two sets of 16ths with at least one hit, at least SAME_PATTERN)."""
    near = any(x["nearest_group"] == y["id"] and x["separation"] is not None and x["separation"] <= NEAR_SEPARATION * threshold
               for x, y in ((a, b), (b, a)))
    return near and _cosine((ua["counts"] > 0).astype(float), (ub["counts"] > 0).astype(float)) >= SAME_PATTERN


def name_roles(result: dict, stem_kind: str | None, bleed: dict | None = None) -> dict:
    """A role name for each group of one stem's `hit_groups` result, by rule (never recognition; the rules and the source
    of each number are in docs/DRUMROLES.md). `stem_kind` is "hh", "snare", "toms", "kick" or anything
    else (every group then stays neutral). `bleed` is that stem's entry of `mark_bleed`'s answer: a group marked bleed has
    no role. `result` is not changed; sections are not named (they carry the stem's group ids).

    A group is NAMED only when it is eligible: at least NAME_MIN_HITS measured hits, `stability` measured and at least
    NAME_STABILITY, `separation` (against the nearest group of at least 4 measured hits; None passes) over the result's grouping threshold. Then the first rule of its stem that fits gives
    the name: hh: "hh offbeat", "hh 16th", "hh open"; snare: "rim", "snare"; toms: "tom" ("tom low" / "tom mid" /
    "tom high" by body pitch, when several roles are toms); kick: "kick". The rules that read measurements (everything but the two pattern rules) read up to AGREE_MEMBERS members nearest
    the medoid (`read_members`) and name only when at least AGREE_SHARE of them read the same; the reason of a group where
    they disagree has the counts. The pattern counts the measured hits and the
    ones assigned to the group under another hit's tail (`assigned` says how many). A group that is not eligible, or that
    no rule fits, keeps "<stem> <id + 1>" with the reason. Two roles with one name get " 2", " 3".
    Groups of one name that are within NEAR_SEPARATION times the threshold of each other and play the same 16ths are one
    role with `variants` ("possibly one sound": the cut between them depends on the sample); a group that is not
    eligible but is that near and alike a named group is listed in that role's `possibly_also`.

    Returns {"stem", "groups": [{"id", "name", "named", "rule", "reason" (why neutral), "bleed", "role" (index into
    "roles", None for bleed), "measures", "note"}], "roles": [{"name", "named", "rule", "groups", "count", "assigned",
    "level_db", "pattern", "steps", "density" ({"off_per_bar", "off_density", "odd_per_bar", "odd_density"}: hits a bar on the offbeats and the odd 16ths over all the group's bars, and the density over the bars each 16th could be seen in, which the hh rules read), "note", "variants" (None, or one entry per group), "possibly_also"}], "notes"}.
    "measures" holds, per 16th, `steps` (see `_use`): the bars the group plays there, the bars it could be seen in, and the
    shares of its own bars and of the observable ones, with "in use", "not in use" or "unknown" (observable in under
    OBSERVE_MIN bars: it counts neither for nor against); the hh pattern rules judge in-use over the known 16ths only
    (at most 8 - ODD_STEPS of the known odd 16ths empty, with at least ODD_KNOWN known; the offbeats 4 - OFFBEAT_STEPS and OFFBEAT_KNOWN),
    and a name that was given with unknown 16ths says how many in `note`. A result without `hidden_cells` and
    `unmeasured_cells` has nothing hidden: every bar the group plays in is observable."""
    thr, sr = result["threshold"], result["sample_rate"]
    kind = stem_kind if stem_kind in ("hh", "snare", "toms", "kick") else None
    label = stem_kind if isinstance(stem_kind, str) and stem_kind else "group"
    bleed = bleed or {}
    by_id = {g["id"]: g for g in result["groups"]}
    uses = {g["id"]: _use(result, g) for g in result["groups"]}
    info, notes, cache = {}, [], {}
    for g in result["groups"]:
        gid = g["id"]
        u = uses[gid]
        m, why_clean = _clean_measures(g, sr, cache)
        reads = _member_readings(g, sr, cache) if kind else []
        need = AGREE_MIN if "read_members" in g else 1        # a result without the members has only the clean hit to read
        e = {"id": gid, "name": "%s %d" % (label, gid + 1), "named": False, "rule": None, "reason": None,
             "bleed": bool(bleed.get(gid, {}).get("bleed")), "role": None, "note": None,
             "measures": {"hits": g["count"] + g["assigned"], "measured": g["count"], "assigned": g["assigned"],
                          "offbeat_share": round(_pattern_numbers(u)["share_off"], 3), "steps": u["steps"],
                          "hidden_unjudged": u["hidden_unjudged"],
                          "unknown_steps": [r["step"] for r in u["steps"] if r["state"] == "unknown"],
                          "density": {k: round(_pattern_numbers(u)[k], 3) for k in ("off_per_bar", "off_density", "odd_per_bar", "odd_density")},
                          "members_read": len(reads), "stability_status": g.get("stability_status"), **(m or {"clean_hit": why_clean})}}
        info[gid] = e
        if e["bleed"]:
            e["reason"] = "marked as bleed: %s" % bleed[gid]["why"]
            continue
        why = _eligibility(g, thr)
        if why:
            e["reason"] = "not named: " + "; ".join(why)
            continue
        name, rule = _apply_rules(kind, u, reads, need) if kind else (None, None)
        if name is None:
            e["reason"] = _no_rule(stem_kind, u, m, why_clean, reads, need)
        else:
            e["name"], e["named"], e["rule"] = name, True, rule
            if g.get("stability_status") == "stable by construction":
                e["note"] = ("stability not measured: no more measured hits than the sample, so every hit was in the first "
                             "sample, the grouping is exact average linkage and no regrouping can differ (stable by construction)")
            if name == "tom":                      # ranked by the median of the pitched members, not by the clean hit alone
                e["measures"]["role_pitch_hz"] = float(np.median([r["pitch_hz"] for r in reads if r["pitched"]]))
            pn = _pattern_numbers(u)
            unk = {"hh offbeat": ("off", "offbeats"), "hh 16th": ("odd", "odd 16ths")}.get(name)
            if unk and pn[unk[0]][1]:
                e["note"] = (e["note"] + "; " if e["note"] else "") + "named without %d of the %d %s: %s, each observable in under %d bars (hidden under a tail or held an " \
                            "unmeasured hit), so they counted neither for nor against" % (
                                pn[unk[0]][1], pn[unk[0]][3], unk[1], [r["step"] for r in u["steps"] if r["state"] == "unknown"
                                                                       and (r["step"] % 4 == 2 if unk[0] == "off" else r["step"] % 2 == 1)], OBSERVE_MIN)
    live = [g for g in result["groups"] if not info[g["id"]]["bleed"]]
    parent = {g["id"]: g["id"] for g in live}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    named = [g for g in live if info[g["id"]]["named"]]
    for k, a in enumerate(named):
        for b in named[k + 1:]:
            if info[a["id"]]["name"] == info[b["id"]]["name"] and _close_alike(a, b, uses[a["id"]], uses[b["id"]], thr):
                parent[find(b["id"])] = find(a["id"])
    clusters = {}
    for g in live:
        clusters.setdefault(find(g["id"]), []).append(g["id"])
    roles = []
    for ids in clusters.values():
        gs = [by_id[i] for i in ids]
        big = max(gs, key=lambda g: g["count"])
        e = info[big["id"]]
        also = []
        if e["named"]:
            for g in live:
                if g["id"] not in ids and not info[g["id"]]["named"] and any(
                        _close_alike(g, h, uses[g["id"]], uses[h["id"]], thr) for h in gs):
                    also.append(g["id"])
                    info[g["id"]]["note"] = ("possibly the same sound as group %d (%s): they are within %.2f of each other "
                                             "and play the same 16ths" % (big["id"], e["name"], NEAR_SEPARATION * thr))
        total = np.zeros(len(big["pattern"]))
        for g in gs:
            total = total + np.array(g["pattern"])
        roles.append({"name": e["name"], "named": e["named"], "rule": e["rule"], "groups": sorted(ids),
                      "count": sum(g["count"] for g in gs), "assigned": sum(g["assigned"] for g in gs),
                      "level_db": big["level_db"], "pattern": [float(min(1.0, v)) for v in total],
                      "steps": info[big["id"]]["measures"]["steps"], "density": info[big["id"]]["measures"]["density"], "note": e["note"] if e["named"] else None,
                      "variants": None if len(ids) == 1 else [
                          {"group": g["id"], "count": g["count"], "level_db": g["level_db"],
                           "stability": g["stability"], "separation": g["separation"],
                           "fall_20_ms": info[g["id"]]["measures"].get("fall_20_ms")} for g in gs],
                      "possibly_also": also})
        if len(ids) > 1:
            notes.append("groups %s are one role (%s) with variants: each is within %.2f of the other and they play the same "
                         "16ths, so they may be one sound" % (sorted(ids), e["name"], NEAR_SEPARATION * thr))
    roles.sort(key=lambda r: (-r["count"], r["groups"][0]))
    toms = [r for r in roles if r["name"] == "tom" and r["named"]]
    pitch = lambda r: info[max(r["groups"], key=lambda i: by_id[i]["count"])]["measures"]["role_pitch_hz"]  # noqa: E731
    levels = sorted({round(pitch(r)) for r in toms})                  # toms at one pitch (to the Hz printed) share a rank
    if len(levels) > 1:
        for r in toms:
            k = levels.index(round(pitch(r)))
            r["name"] = "tom low" if k == 0 else "tom high" if k == len(levels) - 1 else "tom mid"
        notes.append("toms are ranked by body pitch among this stem's tom roles (%s): low, mid and high are relative to each "
                     "other, not to a scale; roles at one pitch share a rank" % ", ".join(
                         "%s %.0f Hz" % (r["name"], pitch(r)) for r in sorted(toms, key=pitch)))
    elif len(toms) > 1:
        notes.append("the tom roles of this stem all read %d Hz: they are not ranked low, mid or high" % levels[0])
    seen = {}
    for r in roles:
        seen[r["name"]] = seen.get(r["name"], 0) + 1
        if seen[r["name"]] > 1 and r["named"]:
            r["name"] += " %d" % seen[r["name"]]
    for k, r in enumerate(roles):
        for i in r["groups"]:
            info[i]["name"] = r["name"] if r["named"] else info[i]["name"]
            info[i]["role"] = k
    return {"stem": stem_kind, "groups": [info[g["id"]] for g in result["groups"]], "roles": roles, "notes": notes}


# ---------------------------------------------------------------- bleed

def _tail(probs: list, k: int) -> float:
    """P(at least k successes) of independent trials with the success probabilities `probs`."""
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for p in probs:
        nxt = pmf * (1.0 - p)
        nxt[1:] += pmf[:-1] * p
        pmf = nxt
    return float(pmf[k:].sum())


def _bleed_pair(res_a: dict, ga: dict, res_b: dict, gbs: list, span: int, sr: int) -> dict:
    """The evidence that group `ga` (stem a) is bleed of the group `gbs[0]` (stem b), or of the union of the groups `gbs`
    (one or more groups of stem b: every hit of theirs is a candidate partner), and the first of the three conditions it
    fails ("coincidence", "level", "likeness", or None when it passes all). See mark_bleed."""
    hits_a = [res_a["hits"][i] for i in ga["members"] + ga["assigned_members"] if res_a["hits"][i]["level_db"] is not None]
    owner = {}
    for gb in gbs:
        for i in gb["members"] + gb["assigned_members"]:
            if res_b["hits"][i]["level_db"] is not None:
                owner[i] = gb
    hits_b = [res_b["hits"][i] for i in owner]
    n = len(hits_a)
    ev = {"source_stem": None, "source_group": gbs[0]["id"], "source_groups": [g["id"] for g in gbs], "union": len(gbs) > 1,
          "hits": n, "coincide": 0, "rate": 0.0, "expected": None, "own_expected": None, "own_bars": 0, "p_evidence": None, "p": None, "louder_db": None, "louder_share": None,
          "distance": None, "distances": None, "co_located": False, "strict": False, "louder_min_db": BLEED_LOUDER_DB,
          "louder_min_share": BLEED_LOUDER_SHARE, "distance_max": BLEED_DISTANCE, "failed": "coincidence"}
    if not n or not hits_b:
        return ev
    tb = np.array([h["onset_s"] for h in hits_b])
    order = np.argsort(tb)
    tb = tb[order]
    pairs = []
    for h in hits_a:
        k = int(np.searchsorted(tb, h["onset_s"]))
        near = [j for j in (k - 1, k) if 0 <= j < len(tb) and abs(tb[j] - h["onset_s"]) <= COINCIDE_S]
        if near:
            j = min(near, key=lambda j: abs(tb[j] - h["onset_s"]))
            pairs.append((h, hits_b[int(order[j])]))
    cells = {}
    for h in hits_b:
        if h["on_grid"]:
            cells.setdefault(h["step"], set()).add(h["bar"])
    chance = [max(CHANCE_FLOOR, len(cells.get(h["step"], ())) / span) for h in hits_a]
    own = {h["bar"] for h in hits_a}                    # the bars this group plays in: the same question over them
    chance_own = [max(CHANCE_FLOOR, len(cells.get(h["step"], set()) & own) / len(own)) for h in hits_a]
    ev.update(coincide=len(pairs), rate=len(pairs) / n, expected=float(np.mean(chance)),
              p=_tail(chance, len(pairs)) if pairs else 1.0, own_expected=float(np.mean(chance_own)), own_bars=len(own))
    ev["p_evidence"] = abs(ev["rate"] - ev["own_expected"]) > OWN_CLOSE
    co = ev["co_located"] = ev["expected"] >= CO_EXPECTED
    strict = ev["strict"] = co or ev["own_expected"] >= CO_EXPECTED     # over its own bars the other group is there almost always
    louder_db, louder_share, limit = (CO_LOUDER_DB, CO_LOUDER_SHARE, CO_DISTANCE) if strict else (
        BLEED_LOUDER_DB, BLEED_LOUDER_SHARE, BLEED_DISTANCE)
    ev.update(louder_min_db=louder_db, louder_min_share=louder_share, distance_max=limit)
    if pairs:                                   # measured even when the coincidence fails, so the report can say it
        diff = [b["level_db"] - a["level_db"] for a, b in pairs]
        ev.update(louder_db=float(np.median(diff)), louder_share=float(np.mean([d >= louder_db for d in diff])))
    if co:
        passed = ev["rate"] >= CO_RATE
    else:
        passed = ev["rate"] >= BLEED_RATE and ev["rate"] - ev["expected"] >= BLEED_MARGIN and ev["p"] <= BLEED_P
    if not passed:
        return ev
    ev["failed"] = "level"
    if not (ev["louder_db"] >= louder_db and ev["louder_share"] >= louder_share):
        return ev
    ev["failed"] = "likeness"
    paired = {}                                  # the groups the coincident pairs belong to, by id
    for _, b in pairs:
        paired[owner[b["id"]]["id"]] = owner[b["id"]]
    ev["source_groups"] = sorted(paired)
    missing = [g for g in paired.values() if g["clean_hit"] is None] + ([ga] if ga["clean_hit"] is None else [])
    if missing:
        ev["distance_note"] = "no clean hit to compare (%s)" % "; ".join(
            "%s: %s" % ("this group" if g is ga else "group %d" % g["id"], g["clean_note"]) for g in missing)
        return ev
    ev["distances"] = {gid: float(_pair_distance(ga["clean_hit"]["audio"], g["clean_hit"]["audio"], sr)) for gid, g in paired.items()}
    ev["distance"] = max(ev["distances"].values())
    ev["failed"] = None if ev["distance"] <= limit else "likeness"
    return ev


def _bleed_why(ev: dict) -> str:
    """One line for the evidence of one pair (the reason it was not marked, or the proof that it was)."""
    src = ("the union of groups %s of `%s`" % (ev["source_groups"], ev["source_stem"]) if ev["union"]
           else "group %d of `%s`" % (ev["source_group"], ev["source_stem"]))
    co = "%d of %d hits coincide with %s (%.0f %%; the two patterns give %s by chance over the track's bars%s, p %s%s)" % (
        ev["coincide"], ev["hits"], src, 100 * ev["rate"],
        "-" if ev["expected"] is None else "%.0f %%" % (100 * ev["expected"]),
        "" if ev["own_expected"] is None else " and %.0f %% over the %d bars this group plays in" % (100 * ev["own_expected"], ev["own_bars"]),
        "-" if ev["p"] is None else "%.3g" % ev["p"],
        "" if ev["p_evidence"] is not False or ev["p"] is None or ev["p"] > BLEED_P else "; p is over the track's bars and is not evidence of leakage here: over this "
        "group's own bars the other group plays at these 16ths about as often as they coincide")
    if ev["co_located"]:
        co = "co-located: chance cannot tell (%s); judged on level and likeness alone" % co
    elif ev["strict"]:
        co = "%s; over this group's own bars the other group is there almost always, so the level and likeness limits are the strict ones" % co
    lv = ("the level of that group against this group's hits is %+.1f dB (median; %.0f %% of the pairs at %+.0f dB or more)"
          % (ev["louder_db"], 100 * ev["louder_share"], ev["louder_min_db"])) if ev["louder_db"] is not None else ""
    if ev["failed"] == "coincidence":
        if not ev["coincide"]:
            return "no hit coincides with %s within %.0f ms" % (src, 1000 * COINCIDE_S)
        if ev["co_located"]:
            return "%s: too few of the hits coincide (it needs %.0f %%); %s" % (co, 100 * CO_RATE, lv)
        return "%s: not well above chance (it needs %.0f %% of the hits, %.0f points over chance, p at most %g); %s" % (
            co, 100 * BLEED_RATE, 100 * BLEED_MARGIN, BLEED_P, lv)
    if ev["failed"] == "level":
        return "%s; %s: it is not under that group (it needs %+.0f dB in %.0f %% of the pairs)" % (
            co, lv, ev["louder_min_db"], 100 * ev["louder_min_share"])
    if ev["distance"] is None:
        return "%s; %s; likeness not tested: %s" % (co, lv, ev.get("distance_note"))
    sounds = "the clean hits are %.2f apart%s (at most %.2f)" % (
        ev["distance"], "" if not ev["union"] else " (the furthest of the groups the pairs belong to)", ev["distance_max"])
    return "%s; %s; %s%s" % (co, lv, sounds, "" if ev["failed"] is None else ": they do not sound alike")


def mark_bleed(results_by_stem: dict) -> dict:
    """Mark the groups that are one stem's bleed of another stem's sound. `results_by_stem` is {stem name: hit_groups result}
    for stems of one track (same tempo, t0 and sample rate). A group is bleed of a group in ANOTHER stem when all three hold:
    (a) its hits coincide (onsets within COINCIDE_S) with that group's hits at least BLEED_RATE of the time, at least
    BLEED_MARGIN more than the two patterns give by chance, and with a chance of at most BLEED_P of so many from the patterns
    alone; (b) that group is louder at the coincident hits by at least BLEED_LOUDER_DB in at least BLEED_LOUDER_SHARE of the
    pairs (`level_db`: loudest ms of each hit's cut); (c) the two clean hits are at most BLEED_DISTANCE apart (spectral
    distance; no clean hit on either side: not tested, not marked). Coincidence alone is never enough.
    The chance rate: for each hit of the group at 16th s, the share of the track's bars in which the other group plays at
    s (at least CHANCE_FLOOR), the hits taken as independent: `expected` is the mean of those shares and `p` the chance of
    at least as many coincidences (Poisson-binomial). If the other group plays every beat, a group that sits on the beats
    coincides with it every time and the chance rate is 100 %: that is not evidence. The track's bars run from the first to
    the last hit of any stem. Hits that are not on the grid count in the observed rate and get the chance of their nearest 16th.
    When `expected` is at least CO_EXPECTED the coincidence cannot tell a leak from two parts that share a pattern
    ("co_located" in the evidence): the group is then marked on level (CO_LOUDER_DB in CO_LOUDER_SHARE of the pairs) and
    likeness (CO_DISTANCE) alone, with at least CO_RATE of its hits coincident.
    When that chance over the group's own bars (below) is at least CO_EXPECTED too ("strict" in the evidence), the level and
    likeness limits are the co-located ones (CO_LOUDER_DB in CO_LOUDER_SHARE, CO_DISTANCE) though the coincidence test is the
    ordinary one: a sectional fill copied 20 dB down stays provable, the same sound only 8 dB under does not.
    The same chance is also reported over the bars the group itself plays in (`own_expected`, `own_bars`), and `p_evidence`
    is False when that is within OWN_CLOSE of the observed rate: `p` is then over the track's bars, not over the group's, and
    says only that both parts live in the same bars. The test itself keeps the track-wide chance: a leak is only there in the bars
    where its source plays, so a chance taken over the group's own bars has the answer built in (a leak of a tom fill would
    always read 100 %) and no leak of a sectional source could be proved; the track-wide chance asks the question the
    test is for, whether the group's hits would fall on the other group's 16ths if they did not come from it. What the own
    bars add is the warning that, for a part that lives in one section, the track-wide `p` overstates.
    Second test, reported separately: a separator may split one source over several groups of its stem, so that no single
    group is coincident enough. A group that fails (a) against EVERY group of another stem is also tested against the union
    of that stem's groups that the first test did not mark (at least two), the chance rate computed the same way from the
    16ths where any of them plays. Coincidence alone never marks here either: (b) is read over all coincident pairs, each
    hit at its own level, and (c) against the clean hit of every group that a coincident pair belongs to, the furthest
    counting; a group without a clean hit is not tested.
    Returns {stem: {group id: {"bleed": bool, "why": str, "evidence": dict or None, "union": {other stem: {"evidence", "why"}}}}}
    for every group of every stem. A marked group's `evidence` is the pair or union that proves it ({"source_stem",
    "source_group", "source_groups", "union", "hits", "coincide", "rate", "expected", "p", "louder_db", "louder_share",
    "distance", "distances", "failed"}); an unmarked group's is the single group that came closest and `why` says which
    condition it failed; `union` holds what the second test read, when it was run. Pass each stem's entry to
    `name_roles(..., bleed=)`."""
    stems = list(results_by_stem)
    if not stems:
        raise ValueError("no stems")
    first = results_by_stem[stems[0]]
    for s in stems[1:]:
        r = results_by_stem[s]
        if (r["tempo"], r["t0"], r["sample_rate"]) != (first["tempo"], first["t0"], first["sample_rate"]):
            raise ValueError("stems %r and %r do not share a tempo, t0 and sample rate: %r against %r" % (
                stems[0], s, (first["tempo"], first["t0"], first["sample_rate"]), (r["tempo"], r["t0"], r["sample_rate"])))
    sr = first["sample_rate"]
    bars = [h["bar"] for r in results_by_stem.values() for h in r["hits"]]
    span = max(bars) - min(bars) + 1 if bars else 1
    stage = ["coincidence", "level", "likeness", None]
    out = {s: {} for s in stems}
    singles = {}                                               # (stem, group id) -> {other stem: [evidence per group]}
    for sa in stems:
        for ga in results_by_stem[sa]["groups"]:
            ok, best, per_stem = [], None, {}
            for sb in stems:
                if sb == sa:
                    continue
                per_stem[sb] = []
                for gb in results_by_stem[sb]["groups"]:
                    ev = _bleed_pair(results_by_stem[sa], ga, results_by_stem[sb], [gb], span, sr)
                    ev["source_stem"] = sb
                    per_stem[sb].append(ev)
                    if ev["failed"] is None:
                        ok.append(ev)
                    rank = (stage.index(ev["failed"]), ev["rate"])
                    if best is None or rank > best[0]:
                        best = (rank, ev)
            singles[(sa, ga["id"])] = per_stem
            if ok:
                ev = min(ok, key=lambda e: e["distance"])
                out[sa][ga["id"]] = {"bleed": True, "why": _bleed_why(ev), "evidence": ev, "union": {}}
            elif best is None:
                out[sa][ga["id"]] = {"bleed": False, "why": "there is no other stem to compare with", "evidence": None, "union": {}}
            else:
                out[sa][ga["id"]] = {"bleed": False, "why": _bleed_why(best[1]), "evidence": best[1], "union": {}}
    left = {s: [g for g in results_by_stem[s]["groups"] if not out[s][g["id"]]["bleed"]] for s in stems}
    for sa in stems:
        for ga in results_by_stem[sa]["groups"]:
            entry = out[sa][ga["id"]]
            if entry["bleed"]:
                continue
            ok = []
            for sb, evs in singles[(sa, ga["id"])].items():
                if len(left[sb]) < 2 or any(e["failed"] != "coincidence" for e in evs):
                    continue
                ev = _bleed_pair(results_by_stem[sa], ga, results_by_stem[sb], left[sb], span, sr)
                ev["source_stem"] = sb
                entry["union"][sb] = {"evidence": ev, "why": _bleed_why(ev)}
                if ev["failed"] is None:
                    ok.append(ev)
            if ok:
                ev = min(ok, key=lambda e: e["distance"])
                entry.update(bleed=True, why=_bleed_why(ev), evidence=ev)
            elif entry["union"]:
                entry["why"] += "; against the union of a stem's groups: " + "; ".join(
                    "`%s`: %s" % (sb, u["why"]) for sb, u in entry["union"].items())
    return out
