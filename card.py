"""The track card's parts, for `roadie study report` (from PR #6, `roadie card`, folded into the report).

What a producer writes down when studying a reference by hand: which elements play where (the element map), what
happens in the bar before each change (`sections.detect_transitions`, measured by `roadie micro`), and a few measured
observations. Pure functions: the report hands them the study's records (sections.json's `bar_activity` for who plays
in each bar, its sections, micro.json's transitions) and the bar numbering to print.

From the PR as it was, except: the per-bar presence is sections.json's `bar_activity` (bar by bar, not despeckled to a
4-bar minimum as the PR's `presence` was), so a one-bar gap shows as ▒; the PR's `render_card` and `main` are gone (the
report renders the parts), and its arrangement table and entries/exits are the report's own sections.
"""

from __future__ import annotations

import math

# Element-map rows, low to high; stems not listed here follow in name order.
ROW_ORDER = ("drums_kick", "drums", "bass", "drums_toms", "drums_snare",
             "drums_hh", "drums_ride", "drums_crash", "synths", "vocals")
BARS_PER_CHAR = 4          # one element-map column per 4 bars: an 8-bar phrase is 2
RULER_EVERY = 32           # bar numbers on the element map every 32 bars
MINI_BREAK_BARS = 4        # a section this short or shorter is reported as a mini-break
LEGEND = "One column per %d bars: █ plays throughout, ▒ in part, · not at all." % BARS_PER_CHAR


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
    return 20.0 * math.log10(rel) if rel and rel > 0 else float("-inf")


def element_map(presence, starts, labels, first_bar, bars_per_char=BARS_PER_CHAR, flag=""):
    """Text grid: one row per stem, one column per `bars_per_char` bars.

    A cell is "█" when the stem plays in every bar of it, "▒" in some, "·" in none.
    A ruler of bar numbers (`first_bar` for index 0, each followed by `flag`) and a row of section names sit on top.
    `starts` are the sections' first bars as indices into the presence strings.
    """
    if RULER_EVERY % bars_per_char:
        raise ValueError("bars_per_char must divide %d, the ruler step" % RULER_EVERY)
    n_bars = len(next(iter(presence.values())))
    cols = (n_bars + bars_per_char - 1) // bars_per_char
    names = sorted(presence, key=_row_key)
    width = max(len(short(n)) for n in names + ["bar"]) + 2

    ruler = [" "] * (cols + 8)
    for idx in range(0, n_bars, RULER_EVERY):
        text = _fmt_bar(first_bar + idx) + flag
        c = idx // bars_per_char
        ruler[c:c + len(text)] = list(text)
    sect = [" "] * cols
    cstarts = [max(0, s) // bars_per_char for s in starts] + [cols]
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


def transition_lines(transitions, bar_of, flag=""):
    """"Into each section" as Markdown bullets, from `detect_transitions` records ({"bar": the file bar before the
    change, "kind", "ratio", "into"}); `bar_of` turns a file bar into the bar to print."""
    if not transitions:
        return ["No fill or dropout in the bar before any section change."]
    out = []
    for t in transitions:
        what = ("a fill: %.1f× the drum hits of the section's usual bar" if t["kind"] == "fill" else
                "a dropout: %.1f× fewer drum hits than the section's usual bar")
        out.append("- Bar %s%s, the bar before **%s**: %s." % (_fmt_bar(bar_of(t["bar"])), flag, t["into"],
                                                              what % t["ratio"]))
    return out


def observations(sections, presence, bar_of, flag=""):
    """The card's measured observations, as sentences. `sections` are [{label, start (index into the presence
    strings), bars, stems, rel_energy}], `presence` {stem: "0101..."} or None, `bar_of` an index -> the bar to print."""
    notes = []
    if presence:
        a, b = longest_static_stretch(presence)
        notes.append("Longest stretch with no element coming in or going out: "
                     "bars %s–%s%s (%d bars)." % (_fmt_bar(bar_of(a)), _fmt_bar(bar_of(b)), flag, b - a + 1))
    loud = [_db(s.get("rel_energy")) for s in sections if s["label"].startswith(("Drop", "Main"))]
    # "Break" and "Quiet" are the spectral method's names for a breakdown.
    quiet = [_db(s.get("rel_energy")) for s in sections
             if s["label"].startswith(("Breakdown", "Break", "Quiet"))]
    if loud and quiet and math.isfinite(max(loud)) and math.isfinite(min(quiet)):
        notes.append("The deepest breakdown sits %.1f dB under the loudest drop." % (max(loud) - min(quiet)))
    for i, s in enumerate(sections[:-1]):
        if i and s["bars"] <= MINI_BREAK_BARS:     # a short Intro is not a break
            notes.append("Mini-break at bar %s%s: %d bars of %s before %s." % (
                _fmt_bar(bar_of(s["start"])), flag, s["bars"],
                ", ".join(s.get("stems") or []) or "near silence", sections[i + 1]["label"]))
    odd = [s for s in sections[:-1] if s["bars"] % 8]
    if odd:
        notes.append("Sections whose length is not a multiple of 8 bars: %s." % ", ".join(
            "%s (%d)" % (s["label"], s["bars"]) for s in odd))
    return notes
