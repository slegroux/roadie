#!/usr/bin/env python3
"""Track card: a Markdown study note for one analysed track.

    card.py <stems_dir> [--start-bar 9] [-o FILE] [--no-write] [--no-transitions]

What a producer writes down when studying a reference by hand: the sections, which
elements play where, when each one comes in and goes out, what happens in the bar
before each change, and a few measured observations. It is built from the same
analysis as `roadie sections` (per-bar stem activity, the cached beat grid) and
adds no new guessing: the section NAMES are rules, everything else is measured.

The card prints to stdout and is saved as `card.md` in the stems folder (`-o`
writes it elsewhere, for example into a notes vault; `--no-write` skips the file).
Bars are Live bars, with the clips starting at `--start-bar` as `roadie load` and
`roadie als` place them, whenever the downbeat is known.
"""

import os
import sys

# Re-exec with local .venv python if present to ensure librosa/numpy environment
_real_script = os.path.realpath(__file__)
_venv_py = os.path.join(os.path.dirname(_real_script), ".venv", "bin", "python")
if os.path.exists(_venv_py) and sys.executable != _venv_py and "VIRTUAL_ENV" not in os.environ:
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import math
import re

from beat_aligner import cached_grid
from sections import (BEATS_PER_BAR, COMMENT_ORDER, STEM_MIN_RUN, analyze_sections,
                      cli_grid, detect_transitions, live_bars)

# Element-map rows, low to high; stems not listed here follow in name order.
ROW_ORDER = ("drums_kick", "drums", "bass", "drums_toms", "drums_snare",
             "drums_hh", "drums_ride", "drums_crash", "synths", "vocals")
BARS_PER_CHAR = 4          # one element-map column per 4 bars: an 8-bar phrase is 2
RULER_EVERY = 32           # bar numbers on the element map every 32 bars
MINI_BREAK_BARS = 4        # a section this short or shorter is reported as a mini-break


def track_name(stems_dir):
    """"Off Gerlach" from ".../Off_Gerlach_125bpm_stems"."""
    base = os.path.basename(os.path.normpath(stems_dir))
    base = re.sub(r"_stems$", "", base)
    base = re.sub(r"_\d+(\.\d+)?bpm$", "", base)
    return base.replace("_", " ")


def short(stem):
    """Row name for a stem file: "kick" for drums_kick, the file name otherwise."""
    return stem[len("drums_"):] if stem.startswith("drums_") else stem


def runs(bits):
    """[(first, last), ...] bar indices, inclusive, of the "1" runs in a presence string."""
    out, start = [], None
    for i, c in enumerate(bits + "0"):
        if c == "1" and start is None:
            start = i
        elif c != "1" and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _row_key(stem):
    return (ROW_ORDER.index(stem), "") if stem in ROW_ORDER else (len(ROW_ORDER), stem)


def _fmt_bar(b):
    return "%g" % round(b, 2)


def _db(rel):
    return 20.0 * math.log10(rel) if rel > 0 else float("-inf")


def _changes(prev, sec):
    """Every role that came in or went out against the section before."""
    if prev is None or "stems" not in sec or "stems" not in prev:
        return ""
    now, before = set(sec["stems"]), set(prev["stems"])
    out = ["%s in" % r for r in COMMENT_ORDER if r in now and r not in before]
    out += ["%s out" % r for r in COMMENT_ORDER if r in before and r not in now]
    return ", ".join(out)


def _bar_index(res, beat):
    """Bar index of a grid beat, counted from the first section's start."""
    return int(round((float(beat) - res["sections"][0]["beat"]) / BEATS_PER_BAR))


def _section_bars(sec):
    return max(1, int(round(sec["duration_beats"] / float(BEATS_PER_BAR))))


def _mmss(t):
    s = int(round(t))
    return "%d:%02d" % (s // 60, s % 60)


def element_map(presence, starts, labels, first_bar, bars_per_char=BARS_PER_CHAR):
    """Text grid: one row per stem, one column per `bars_per_char` bars.

    A cell is "█" when the stem plays in every bar of it, "▒" in some, "·" in none.
    A ruler of bar numbers and a row of section names sit on top.
    """
    if RULER_EVERY % bars_per_char:
        raise ValueError("bars_per_char must divide %d, the ruler step" % RULER_EVERY)
    n_bars = len(next(iter(presence.values())))
    cols = (n_bars + bars_per_char - 1) // bars_per_char
    names = sorted(presence, key=_row_key)
    width = max(len(short(n)) for n in names + ["bar"]) + 2

    ruler = [" "] * (cols + 8)
    for idx in range(0, n_bars, RULER_EVERY):
        text = _fmt_bar(first_bar + idx)
        c = idx // bars_per_char
        ruler[c:c + len(text)] = list(text)
    sect = [" "] * cols
    cstarts = [s // bars_per_char for s in starts] + [cols]
    for i, label in enumerate(labels):
        room = cstarts[i + 1] - cstarts[i]
        text = ("|" + label)[:max(room - 1, 1)] if room > 1 else "|"
        sect[cstarts[i]:cstarts[i] + len(text)] = list(text)

    lines = ["bar".ljust(width) + "".join(ruler).rstrip(),
             "".ljust(width) + "".join(sect[:cols]).rstrip()]
    for n in names:
        bits = presence[n]
        row = []
        for c in range(cols):
            cell = bits[c * bars_per_char:(c + 1) * bars_per_char]
            on = cell.count("1")
            row.append("█" if on == len(cell) else ("▒" if on else "·"))
        lines.append(short(n).ljust(width) + "".join(row))
    return lines


def longest_static_stretch(presence):
    """(first, last) bar indices of the longest run where no stem comes in or goes out."""
    n_bars = len(next(iter(presence.values())))
    cuts = [0] + [i for i in range(1, n_bars)
                  if any(b[i] != b[i - 1] for b in presence.values())] + [n_bars]
    a, b = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: ab[1] - ab[0])
    return a, b - 1


def render_card(name, res, first_bar, bar_kind="Live", start_bar=9, transitions=None,
                bars_per_char=BARS_PER_CHAR, drift_pct=None):
    """The card as Markdown. Pure: everything comes from the arguments.

    `res` is an `analyze_sections` result; `first_bar` the bar number of its first
    section (a Live bar when `bar_kind` is "Live", the file's own bar otherwise);
    `transitions` a `detect_transitions` list, or None when they were not looked for;
    `drift_pct` the tempo drift the beat aligner measured, when it found one.
    """
    secs = res["sections"]
    presence = res.get("presence") or {}
    starts = [_bar_index(res, s["beat"]) for s in secs]
    bar_of = lambda idx: first_bar + idx          # noqa: E731
    n_bars = (len(next(iter(presence.values()))) if presence
              else res["beat_count"] // BEATS_PER_BAR)

    out = ["# %s" % name, ""]
    out.append("**%g BPM · %s · %d bars · %d sections**" % (
        res["bpm"], _mmss(res["duration_sec"]), n_bars, len(secs)))
    out.append("")
    if bar_kind == "Live":
        out.append("Bar numbers are Live bars, with the clips starting at bar %d "
                   "(`roadie load` and `roadie als` place them there)." % start_bar)
    else:
        out.append("Bar numbers count from the start of the audio file: no downbeat "
                   "is known, so they are not Live bars.")
    if drift_pct:
        out += ["", "**The tempo drifts %.1f%% across this track.** The analysis counts "
                "bars at one steady tempo, so bar numbers far from the start may be off "
                "by a bar or more: check them against Live's ruler." % drift_pct]

    out += ["", "## Arrangement", "",
            "| Bar | Section | Bars | Playing | What changed | Level |",
            "|---:|---|---:|---|---|---:|"]
    for i, s in enumerate(secs):
        level = _db(s["rel_energy"])
        out.append("| %s | %s | %d | %s | %s | %s |" % (
            _fmt_bar(bar_of(starts[i])), s["label"], _section_bars(s),
            ", ".join(s["stems"]) if s.get("stems") else "—",
            _changes(secs[i - 1] if i else None, s) or "—",
            "%+.1f dB" % level if math.isfinite(level) else "silent"))
    out += ["", "Level is the section's loudness against the track's average."]

    if presence:
        out += ["", "## Element map", "",
                "One column per %d bars: █ plays throughout, ▒ in part, · not at all."
                % bars_per_char, "", "```"]
        out += element_map(presence, starts, [s["label"] for s in secs], first_bar,
                           bars_per_char)
        out += ["```", "", "## Entries and exits", ""]
        firsts = []
        for n in sorted(presence, key=_row_key):
            rs = runs(presence[n])
            if not rs:
                continue
            spans = ", ".join("%s–%s" % (_fmt_bar(bar_of(a)), _fmt_bar(bar_of(b)))
                              for a, b in rs)
            total = sum(b - a + 1 for a, b in rs)
            out.append("- **%s**: %s (%d bars)" % (short(n), spans, total))
            firsts.append((rs[0][0], short(n)))     # already in row order
        order = []
        for idx in sorted({f for f, _ in firsts}):
            who = [n for f, n in firsts if f == idx]
            order.append("%s at %s" % (", ".join(who), _fmt_bar(bar_of(idx))))
        out += ["", "First entries, in order: %s." % "; ".join(order), "",
                "Entries and exits are bar by bar. Section starts are not: a change "
                "within %d bars of another is merged into one section start, which is "
                "then snapped to the 8-bar grid when close to it. So an entry and the "
                "section it opens can differ by a few bars." % STEM_MIN_RUN]
    else:
        out += ["", "No element map: it needs the stem-activity method, which reads "
                "a kick (`drums_kick`, or `drums` without drum parts) and a `bass` stem."]

    if transitions is not None:
        out += ["", "## Into each section", ""]
        if transitions:
            for t in transitions:
                idx = _bar_index(res, t["beat"])
                what = ("a fill: %.1f× the drum hits of the section's usual bar"
                        if t["kind"] == "fill" else
                        "a dropout: %.1f× fewer drum hits than the section's usual bar")
                out.append("- Bar %s, the bar before **%s**: %s." % (
                    _fmt_bar(bar_of(idx)), t["into"], what % t["ratio"]))
        else:
            out.append("No fill or dropout in the bar before any section change.")

    notes = []
    if presence:
        a, b = longest_static_stretch(presence)
        notes.append("Longest stretch with no element coming in or going out: "
                     "bars %s–%s (%d bars)." % (_fmt_bar(bar_of(a)), _fmt_bar(bar_of(b)),
                                                b - a + 1))
    loud = [_db(s["rel_energy"]) for s in secs if s["label"].startswith(("Drop", "Main"))]
    # "Break" and "Quiet" are the spectral method's names for a breakdown.
    quiet = [_db(s["rel_energy"]) for s in secs
             if s["label"].startswith(("Breakdown", "Break", "Quiet"))]
    if loud and quiet and math.isfinite(max(loud)) and math.isfinite(min(quiet)):
        notes.append("The deepest breakdown sits %.1f dB under the loudest drop."
                     % (max(loud) - min(quiet)))
    for i, s in enumerate(secs[:-1]):
        if i and _section_bars(s) <= MINI_BREAK_BARS:     # a short Intro is not a break
            notes.append("Mini-break at bar %s: %d bars of %s before %s." % (
                _fmt_bar(bar_of(starts[i])), _section_bars(s),
                ", ".join(s.get("stems") or []) or "near silence", secs[i + 1]["label"]))
    odd = [s for s in secs[:-1] if _section_bars(s) % 8]
    if odd:
        notes.append("Sections whose length is not a multiple of 8 bars: %s." % ", ".join(
            "%s (%d)" % (s["label"], _section_bars(s)) for s in odd))
    if notes:
        out += ["", "## Observations", ""] + ["- " + n for n in notes]

    out += ["", "---", "",
            "*Measured: which stems play in each bar, the section boundaries, levels and "
            "drum-hit counts. Rules: the section names (no model recognises a drop). The "
            "stems come from source separation, so `synths` holds every pad, lead, arp "
            "and pluck together, and some bleed between stems is normal.*", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Write a Markdown study note (track card) for an analysed track")
    ap.add_argument("stems_dir", help="Path to a roadie stems folder")
    ap.add_argument("--start-bar", type=int, default=9,
                    help="Live bar the clips start on, as `roadie load` / `roadie als` "
                         "place them (default 9)")
    ap.add_argument("-o", "--out", default=None,
                    help="write the card here instead of <stems_dir>/card.md; a "
                         "folder gets <track name>.md")
    ap.add_argument("--no-write", action="store_true",
                    help="print the card only; do not save it")
    ap.add_argument("--no-transitions", action="store_true",
                    help="skip the fill/dropout scan of the bar before each section")
    args = ap.parse_args()

    if not os.path.isdir(args.stems_dir):
        ap.error("not a stems folder: %s" % args.stems_dir)
    period, grid_t0, downbeat = cli_grid(args.stems_dir)
    res = analyze_sections(args.stems_dir, period=period, grid_t0=grid_t0,
                           downbeat_sec=downbeat)
    live = live_bars(res, downbeat, args.start_bar)
    first_bar, kind = (live[0], "Live") if live else (res["sections"][0]["bar"], "file")

    transitions = None
    if not args.no_transitions:
        try:
            transitions = detect_transitions(args.stems_dir, res["sections"], res["period"])
        except Exception as e:      # the card stands without this part
            print("card: transition scan failed (%s: %s); left out"
                  % (type(e).__name__, e), file=sys.stderr)

    cached = cached_grid(args.stems_dir) or {}
    drift = cached.get("drift_pct") if cached.get("has_drift") else None
    text = render_card(track_name(args.stems_dir), res, first_bar, kind,
                       args.start_bar, transitions, drift_pct=drift)
    print(text, end="")

    path = None if args.no_write else (args.out or os.path.join(args.stems_dir, "card.md"))
    if path and os.path.isdir(path):         # -o a notes folder: one file per track
        path = os.path.join(path, track_name(args.stems_dir) + ".md")
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print("card: saved %s" % path, file=sys.stderr)
        except OSError as e:
            print("card: could not write %s: %s" % (path, e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
