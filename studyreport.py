#!/usr/bin/env python3
"""One study's report as Markdown: `roadie study report <title> [--write] [--json]`.

Built from the records only, nothing measured here: the study record (`study.load_study`: bpm, key, grid, sections,
mix profile), the structure a reference gives (`arrangement.reference_metrics`: groove blocks, entries, kick-outs,
drops) and the micro layer (`micro.json`, `roadie micro`). A part that is not on disk is listed as `missing: <layer>`
with the command that makes it.

Macro is the structure (sections, entries and exits, kick-outs, groove blocks, the mix per section); micro is what each
instrument plays inside a section. Every bar position and every 16-step pattern depends on where bar 1 is: when the
grid's downbeat is chance-level, there is no alignment.json, or the grid has a warp map, those numbers carry a FLAG mark
and the report says why at the top.

The report is generated: a rebuild overwrites it, so notes belong elsewhere (the vault). `--write` puts `report.md` in
the study folder; a study on the pCloud mount is written with rclone (a temp file, `copyto`, a read-back), never through
the mount, and its micro.json is read the same way.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arrangement  # noqa: E402
import card  # noqa: E402
import study as studymod  # noqa: E402
from roadie_analysis import key as keymod  # noqa: E402

KIND, VERSION = "study_report", 1
FLAG = "†"
ROLE_ORDER = ("kick", "bass", "hats", "snare", "perc", "synths", "vocals")
BANDS = ("sub", "bass", "low_mid", "mid", "high_mid", "high")
MIX_ROLES = ("kick", "bass", "hats", "snare", "toms", "ride", "crash", "synths", "vocals")


# ---- reading

def read_text(path: Path) -> str | None:
    """A small file of the study: through rclone for a study on the pCloud mount (the mount's view can lag behind a
    write that was just made through rclone), directly otherwise; None when it is not there."""
    remote = keymod.pcloud_remote(path)
    if remote:
        return studymod._remote_read(remote)
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return None


def read_micro(st: studymod.Study) -> tuple[dict | None, str | None]:
    """(micro.json record, None) or (None, why it is missing or unreadable)."""
    sd = studymod._stems_dir(st.root)
    text = read_text(sd / "micro.json") if sd else None
    if text is None:
        return None, "roadie micro \"%s/%s\" (writes micro.json beside drums.json)" % (st.artist, st.title)
    try:
        rec = json.loads(text)
    except ValueError:
        return None, "micro.json exists but could not be read"
    if not isinstance(rec, dict) or rec.get("kind") != "micro":
        return None, "micro.json is not a micro record"
    return rec, None


def _sizes(folder: Path) -> dict[str, int] | None:
    """{file name: size} of a stems folder: through rclone for a pCloud study, by listing otherwise; None when it cannot be
    listed."""
    remote = keymod.pcloud_remote(folder)
    try:
        if remote:
            out = {}
            for line in studymod._rclone("lsf", "--files-only", "--format", "sp", "--separator", "\t", remote).splitlines():
                size, _, name = line.partition("\t")
                if name:
                    out[name] = int(size)
            return out
        return {p.name: p.stat().st_size for p in Path(folder).iterdir() if p.is_file()}
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def micro_stale(st: studymod.Study, mc: dict) -> list[str]:
    """Why micro.json no longer describes the study (empty when it still does): another key, grid or section list than
    the records now give, or an input file (a stem, alignment.json, sections.json, drums.json) of another size, gone, or
    new. drums.json and the sizes are read through rclone for a pCloud study."""
    why = []
    try:
        key = keymod.key_name(*keymod.parse_key(st.key)) if st.key else None
    except ValueError:
        key = st.key
    if ((mc.get("key") or {}).get("name")) != key:
        why.append("key %s, now %s" % ((mc.get("key") or {}).get("name"), key))
    res, g = st.layers["grid"].data or {}, mc.get("grid") or {}
    bpm, down = res.get("bpm"), res.get("downbeat_sec")
    if isinstance(bpm, (int, float)) and isinstance(down, (int, float)) and bpm > 0:
        bar_s = 240.0 / bpm
        t0 = down - math.floor(down / bar_s) * bar_s
        if abs(float(g.get("tempo") or 0) - bpm) > 0.001 or abs(float(g.get("t0") or 0) - t0) > 0.005:
            why.append("grid %s bpm at %s s, now %g bpm at %.4f s" % (g.get("tempo"), g.get("t0"), bpm, t0))
    now = [(s.label, s.bar, s.bars) for s in st.sections()]
    was = [(s.get("name"), s.get("file_bar"), s.get("bars")) for s in mc.get("sections") or []]
    if now != was:
        why.append("the sections changed")
    sd = studymod._stems_dir(st.root)
    sizes = _sizes(sd) if sd else None
    if sizes is not None:
        recorded = {i["name"]: i.get("size") for i in mc.get("inputs") or []}
        changed = sorted(n for n, size in recorded.items() if sizes.get(n) != size)
        new = sorted(n for n in ("alignment.json", "sections.json", "drums.json") if n in sizes and n not in recorded)
        if changed:
            why.append("changed since: %s" % ", ".join(changed))
        if new:
            why.append("made since: %s" % ", ".join(new))
    text = read_text(sd / "drums.json") if sd else None
    try:
        made = json.loads(text).get("made") if text else None
    except (ValueError, AttributeError):
        made = None
    if made and mc.get("drums_made") and made != mc["drums_made"]:
        why.append("its drum roles are from an older drums.json (%s, now %s)" % (mc["drums_made"], made))
    return why


def grid_state(st: studymod.Study) -> dict:
    """{reliable, why, verdict, confidence, detector, warp_map, has_drift} from the grid layer."""
    g = st.layers["grid"]
    res = g.data or {}
    why = []
    if g.data is None:
        why.append("no alignment.json: no measured downbeat" if g.source is None else "alignment.json could not be read")
    if res.get("downbeat_verdict") == "chance-level":
        why.append("the downbeat is chance-level (confidence %s): bar 1 may start on the wrong beat"
                   % res.get("downbeat_confidence"))
    warp = res.get("warp_map")
    if warp:
        why.append("the grid has a warp map (the tempo drifts): bar positions late in the track may be off")
    return {"reliable": not why, "why": why, "verdict": res.get("downbeat_verdict"),
            "confidence": res.get("downbeat_confidence"), "detector": res.get("detector"),
            "warp_map": None if g.data is None else bool(warp), "has_drift": res.get("has_drift")}


def entries_exits(st: studymod.Study) -> dict | None:
    """Per role (study.STEM_ROLES): the bar it enters (its first stretch of 4+ bars, as `arrangement` reads an entry),
    the stretches it is out between its first and last such stretch, and its last bar; None without bar_activity."""
    act = st.activity()
    if act is None:
        return None
    first = st.activity_first_bar()
    secs = st.sections()

    def section_at(bar):
        return next((s.label for s in secs if s.bar <= bar < s.bar + s.bars), None)

    n = len(next(iter(act.values())))
    steps = [{"bar": first + i, "bars": 1, "section": section_at(first + i), "cells": {}} for i in range(n)]
    out = {}
    for role in [r for r in ROLE_ORDER if r in act]:
        play = act[role]
        span = arrangement._stretch(steps, play)
        if span is None:
            out[role] = {"enters": None, "out": [], "last_bar": None, "note": "never plays 4 bars in a row"}
            continue
        gaps, trailing = arrangement._internal_gaps(steps, play)
        out[role] = {"enters": steps[span[0]]["bar"], "enters_in": steps[span[0]]["section"],
                     "out": [{"bar": g["bar"], "bars": g["bars"], "section": g["section"]} for g in gaps],
                     "last_bar": steps[span[1]]["bar"], "trailing_out_bars": trailing}
    return out


def card_data(st: studymod.Study, micro: dict | None) -> dict:
    """The track card's parts (card.py, from PR #6): who plays in each bar (sections.json's bar_activity, by stem file),
    the sections in card form, micro.json's transitions, and the bar numbering (Live bars when the offset is known)."""
    act = st.layers["activity"].data
    secs = st.sections()
    first = act["first_bar"] if act else (secs[0].bar if secs else 1)
    raw = (st.layers["sections"].data or {}).get("sections") or []          # in the order st.sections() reads them
    return {"presence": act["stems"] if act else None, "first_bar": first, "bar_offset": st.bar_offset,
            "bars": "Live" if st.bar_offset is not None else "file",
            "sections": [{"label": s.label, "start": s.bar - first, "bars": s.bars, "stems": s.roles,
                          "rel_energy": raw[i].get("rel_energy") if i < len(raw) else None} for i, s in enumerate(secs)],
            "transitions": (micro or {}).get("transitions"), "has_micro": micro is not None,
            "drift_pct": (st.layers["grid"].data or {}).get("drift_pct") if (st.layers["grid"].data or {}).get("has_drift") else None}


def card_markdown(c: dict, f: str) -> list[str]:
    """Element map, into each section and observations, in the report's bar numbering (`f` marks bars on an
    unreliable grid)."""
    off = c["bar_offset"] or 0
    idx_bar = (lambda i: c["first_bar"] + i + off)          # noqa: E731  presence index -> printed bar
    file_bar = (lambda b: b + off)                          # noqa: E731
    L = ["", "## Element map", ""]
    if c["presence"]:
        L += ["%s %s bars%s." % (card.LEGEND, c["bars"], ", from bar_activity: bar by bar, so a one-bar gap shows as ▒"),
              "", "```"]
        L += card.element_map(c["presence"], [s["start"] for s in c["sections"]], [s["label"] for s in c["sections"]],
                              idx_bar(0), flag=f)
        L += ["```"]
    else:
        L.append("missing: activity (bar_activity in sections.json: re-run `roadie sections`)")
    L += ["", "## Into each section", ""]
    if c["transitions"] is not None:
        L += ["The bar before each change, from the drum parts' hits (micro.json; tuned on one track, so none found "
              "means none found, not none there). %s bars.%s" % (c["bars"], (
                  " On this unreliable grid a fill one bar before a section line may be the line itself, a bar late."
                  if f else "")), ""]
        L += card.transition_lines(c["transitions"], file_bar, f)
    elif c["has_micro"]:
        L.append("not measured: no drum parts (drums_<kind> stems), or a micro.json made before the scan (re-run "
                 "`roadie micro`)")
    else:
        L.append("missing: micro (the fill or dropout scan is part of `roadie micro`)")
    # The map is bar by bar; the observations read the presence despeckled to 4-bar runs (#6's reading), so a one-bar
    # flicker does not cut "the longest stretch with nothing in or out".
    notes = card.observations(c["sections"], card.despeckled(c["presence"]) if c["presence"] else None, idx_bar, f) \
        if c["sections"] else []
    if c["drift_pct"]:
        notes.insert(0, "The tempo drifts %.1f%% across this track: bars are counted at one steady tempo, so bar numbers "
                        "far from the start may be off by a bar or more." % c["drift_pct"])
    L += ["", "## Observations", ""] + (["- " + n for n in notes] or ["none"])
    return L


def report_data(st: studymod.Study, micro: dict | None = None, micro_missing: str | None = None) -> dict:
    """The report as one record (what `--json` prints and the Markdown renders)."""
    grid = grid_state(st)
    data = {"kind": KIND, "version": VERSION, "artist": st.artist, "title": st.title, "root": st.source,
            "path": str(st.root.absolute()),
            "bpm": {"measured": st.bpm, "listed": st.listed_bpm, "listed_url": st.bpm_url},
            "key": None, "grid": grid, "bar_offset": st.bar_offset,
            "stem_note": studymod.stem_model_note(st.stem_model),
            "sections": [{"label": s.label, "file_bar": s.bar, "live_bar": s.live_bar, "bars": s.bars, "stems": s.roles}
                         for s in st.sections()],
            "entries": entries_exits(st), "structure": None, "mix": None, "micro": micro, "missing": [], "caveats": []}
    if st.key:
        short, cam = studymod._key_cells(vars(st))
        data["key"] = {"key": st.key, "short": short, "camelot": cam, "source": st.key_source,
                       "confidence": st.key_confidence, "url": st.key_url, "note": st.key_note}
    else:
        data["missing"].append({"layer": "key", "command": "roadie key \"%s\", then roadie study tag \"%s\" --key <key>"
                                % (st.title, st.title)})
    for name, layer in st.layers.items():
        if layer.source is None:
            data["missing"].append({"layer": name, "command": layer.missing})
        elif layer.data is None and name != "activity":
            data["caveats"].append("%s: %s" % (name, "; ".join(layer.caveats) or "could not be read"))
        data["caveats"] += ["%s: %s" % (name, c) for c in layer.caveats if layer.data is not None]
    if st.layers["activity"].data is not None:
        ref = arrangement.reference_metrics(st)
        m = ref["metrics"]
        data["structure"] = {"blocks": m["blocks"], "kick": m["kick"], "bass": m["bass"], "hats": m["hats"],
                             "drops": arrangement.labelled_drops(m), "drop_ratio": m["drop_ratio"]}
    data["mix"] = st.mix() or None
    data["card"] = card_data(st, micro)
    if micro is None:
        data["missing"].append({"layer": "micro", "command": micro_missing})
    else:
        stale = micro_stale(st, micro)
        data["micro_stale"] = stale
        if stale:
            data["caveats"].append("micro: micro.json is stale (%s): re-run roadie micro \"%s/%s\""
                                   % ("; ".join(stale), st.artist, st.title))
    return data


# ---- Markdown

def _num(x, fmt="%g") -> str:
    return "-" if x is None else fmt % x


def _pct(x) -> str:
    return "-" if x is None else "%d %%" % round(100 * x)


def _code(s) -> str:
    return "`%s`" % s if s else "-"


def to_markdown(d: dict) -> str:
    rel = d["grid"]["reliable"]
    f = "" if rel else FLAG

    def bar(x):
        return "-" if x is None else ("%g" % x) + f

    L = ["# %s - %s" % (d["artist"], d["title"]), "",
         "Generated by `roadie study report` from the study's records; a rebuild overwrites it (notes go to the vault).", ""]
    bpm = d["bpm"]
    L.append("- **bpm**: %s measured%s" % (_num(bpm["measured"]), "" if bpm["listed"] is None else
                                           ", %g listed%s" % (bpm["listed"], " (%s)" % bpm["listed_url"] if bpm["listed_url"] else "")))
    k = d["key"]
    if k:
        why = "source %s" % (k["source"] or "not recorded")
        if k["confidence"] is not None:
            why += ", confidence %g" % k["confidence"]
        if k["url"]:
            why += ", %s" % k["url"]
        if k["note"]:
            why += ", note: %s" % k["note"]
        L.append("- **key**: %s (%s, Camelot %s), %s" % (k["key"], k["short"], k["camelot"], why))
    else:
        L.append("- **key**: missing")
    g = d["grid"]
    L.append("- **grid**: downbeat %s%s (%s), warp map %s; %s" % (
        g["verdict"] or "not measured", "" if g["confidence"] is None else " %g" % g["confidence"], g["detector"] or "no detector",
        {None: "unknown", True: "present", False: "none"}[g["warp_map"]],
        "Live bar = file bar + %d" % d["bar_offset"] if d["bar_offset"] is not None else "no Live bar known"))
    if not rel:
        L.append("- **%s grid unreliable**: %s. Every number marked %s depends on bar 1 (bar positions, 16-step patterns, "
                 "chords per bar, phrase starts): read it as a guess." % (FLAG, "; ".join(g["why"]), FLAG))
    if d["stem_note"]:
        L.append("- %s" % d["stem_note"])
    L += ["", "## Section map", ""]
    if d["sections"]:
        L += ["| Section | File bar | Live bar | Bars | Stems playing |", "|---|---|---|---|---|"]
        for s in d["sections"]:
            L.append("| %s | %s | %s | %d | %s |" % (s["label"], bar(s["file_bar"]), bar(s["live_bar"]), s["bars"],
                                                    " ".join(s["stems"]) or "-"))
    else:
        L.append("missing: sections (`roadie sections <stems>`)")
    L += ["", "## Entries and exits", ""]
    if d["entries"] is None:
        L.append("missing: activity (bar_activity in sections.json: re-run `roadie sections`)")
    else:
        L.append("File bars. An entry is the first stretch of 4 bars or more; out = a stretch without it between its first "
                 "and last such stretch.")
        L += ["", "| Role | Enters | Out | Last bar |", "|---|---|---|---|"]
        for role, e in d["entries"].items():
            if e["enters"] is None:
                L.append("| %s | - | - | - (%s) |" % (role, e["note"]))
                continue
            outs = ", ".join("%s-%s (%d, %s)" % (bar(o["bar"]), bar(o["bar"] + o["bars"] - 1), o["bars"], o["section"] or "no label")
                             for o in e["out"]) or "never"
            L.append("| %s | %s (%s) | %s | %s |" % (role, bar(e["enters"]), e["enters_in"] or "no label", outs,
                                                    bar(e["last_bar"])))
    L += card_markdown(d["card"], f)
    S = d["structure"]
    L += ["", "## Kick-outs", ""]
    if S is None or S["kick"] is None:
        L.append("missing: activity" if S is None else "no kick stem")
    else:
        kick = S["kick"]
        if not kick["out"]:
            L.append("none: the kick plays from bar %s to %s without a gap" % (bar(kick["first_bar"]), bar(kick["last_bar"])))
        else:
            L += ["| From | Bars | In |", "|---|---|---|"]
            for o in kick["out"]:
                L.append("| %s | %d | %s |" % (bar(o["bar"]), o["bars"], o["section"] or "no label"))
            L.append("")
            L.append("Longest: %d bars." % kick["longest_out_bars"])
    L += ["", "## Groove blocks", ""]
    if S is None or S["blocks"] is None:
        L.append("missing: activity" if S is None else "no kick or bass stem: no groove block")
    elif not S["blocks"]:
        L.append("none: kick and bass never play 8 bars together")
    else:
        L.append("A block is a stretch in which kick and bass both play (a fill under 4 bars ignored, from 8 bars).")
        L += ["", "| # | From | Bars | Gap before | Kick out before | Bass lead |", "|---|---|---|---|---|---|"]
        for i, b in enumerate(S["blocks"], 1):
            L.append("| %d | %s | %d | %s | %s | %d |" % (i, bar(b["bar"]), b["bars"], _num(b.get("gap_before")),
                                                         _num(b.get("kick_out_before")), b["bass_lead"]))
    if S is not None and S["drops"]:
        L.append("")
        L.append("Labelled drops: " + "; ".join("%s at %s, %d bars (+%d of Main after)" % (x["label"], bar(x["bar"]), x["bars"], x["mains_after"])
                                               for x in S["drops"])
                 + ("; Drop 2 / Drop 1 = %g" % S["drop_ratio"]["ratio"] if S["drop_ratio"] else ""))
    L += ["", "## Mix profile per section", ""]
    if not d["mix"]:
        L.append("missing: mix (reference_profile, the roadie MCP tool, on the mix and stems)")
    else:
        L.append("Mix level and bands in dB (the bounced track), then each stem's level in dB (`absent`: not playing, "
                 "parenthesised: a trace).")
        L += ["", "| Section | Bars | Level | LUFS | " + " | ".join(BANDS) + " |", "|---|---|---|---|" + "---|" * len(BANDS)]
        for s in d["mix"]:
            m = s.get("mix") or {}
            bands = m.get("bands_db") or {}
            L.append("| %s | %s | %s | %s | %s |" % (s.get("name"), s.get("bars"), _num(m.get("level_db"), "%.1f"),
                                                    _num(m.get("integrated_lufs"), "%.1f"),
                                                    " | ".join(_num(bands.get(b), "%.1f") for b in BANDS)))
        roles = [r for r in MIX_ROLES if any(r in (s.get("roles") or {}) for s in d["mix"])]
        L += ["", "| Section | " + " | ".join(roles) + " |", "|---|" + "---|" * len(roles)]
        for s in d["mix"]:
            cells = []
            for r in roles:
                x = (s.get("roles") or {}).get(r)
                if not x:
                    cells.append("-")
                elif x.get("absent"):
                    cells.append("absent")
                else:
                    v = _num(x.get("level_db"), "%.1f")
                    cells.append("(%s)" % v if x.get("trace") else v)
            L.append("| %s | %s |" % (s.get("name"), " | ".join(cells)))
    L += ["", "## Micro", ""]
    L += micro_markdown(d["micro"])
    L += ["", "## Missing layers", ""]
    L += ["- missing: %s: `%s`" % (m["layer"], m["command"]) for m in d["missing"]] or ["none"]
    if d["caveats"]:
        L += ["", "## Notes", ""] + ["- %s" % c for c in d["caveats"]]
    return "\n".join(L) + "\n"


def micro_markdown(mc: dict | None) -> list[str]:
    if mc is None:
        return ["missing: micro (see Missing layers)"]
    rel = mc["grid"]["reliable"]
    f = "" if rel else FLAG
    L = ["Per section, from `micro.json` (`roadie micro`, made %s). Patterns are 16 steps from the bar line "
         "(`o` = an onset); bars inside a section count from 1. Degrees are of the key %s." % (
             mc.get("made", "?"), mc["key"]["name"] if mc.get("key") else "(none: absolute note names)")]
    if not rel:
        L.append("")
        L.append("%s The grid is unreliable (%s): every pattern, chord per bar and phrase start below is marked." % (
            FLAG, "; ".join(mc["grid"]["unreliable_because"])))
    secs = mc["sections"]

    def why(x, default="silent"):
        return "- (%s)" % ((x or {}).get("notes") or [default])[-1] if x is not None else "- (no stem)"

    # bass
    L += ["", "### Bass", "", "Rhythm: the most common one-bar pattern, with its share of the bars that have an onset on the grid. "
          "Level dips: the 16ths at which the stem's mean level over its playing bars is %g dB or more under its loudest 16th. "
          "A dip says the part is quieter there on average: a sidechain duck, the gap between notes, or the tail of notes "
          "that start elsewhere, not which; read it against the kick and the rhythm, and listen before calling it a "
          "sidechain." % DIP_DB, "",
          "| Section | Bars playing | Rhythm (share of bars with onsets) | 16th steps | Onsets/bar | On 1 / on 8th offbeats "
          "| Level dips | Notes (degrees) | Register | Pedal or moving | Duty | Repeat |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in secs:
        b = s["bass"]
        if not b or not b["bars_playing"]:
            L.append("| %s | 0 | %s | | | | | | | | | |" % (s["name"], why(b)))
            continue
        degs = b["degrees"] or b.get("pitch_names")
        notes = ", ".join("%s %s" % (x.get("degree") or x["note"], _pct(x["share"])) for x in degs) if degs else "-"
        L.append("| %s | %d of %d | %s | %s | %s | %s | %s | %s | %s (%s) | %s | %s %s | %s |" % (
            s["name"], b["bars_playing"], s["bars"], _rhythm(b, f),
            (", ".join(map(str, b["rhythm_steps"])) + f) if b["rhythm_steps"] else "-", _num(b["onsets_per_bar"]),
            ("%s / %s%s" % (_pct(b["on_step1_share"]), _pct(b["on_offbeat_8ths_share"]), f))
            if b["on_step1_share"] is not None else "-", _dips(b.get("level_by_16th_db"), f), notes, b["register"] or "-",
            b.get("register_range") or "-",
            "%s (%s %s, %s class%s)" % (b["movement"], b["top_pitch_class"], _pct(b["top_pitch_class_share"]), b["pitch_classes"],
                                        "" if b["pitch_classes"] == 1 else "es")
            if b["movement"] else "-", _num(b["duty"], "%.2f"), b["style"] or "", _repeat(b, f)))
    L += _notes(secs, "bass")
    # synths
    L += ["", "### Synths", "", "Leads and pads are one stem: the rhythm and the chords are of both together.", "",
          "Chords change when the root does (VII, VII5 and VIIsus2 in a row are one VII). The progression is a loop of 2 to 4 "
          "chords, with the share of the changes that follow it when that is not all of them; with no loop, every change in "
          "order. A loop is written on the phase most sections of the track start it on, so it need not begin on a "
          "section's first chord (micro.json keeps the section's own rotation as section_progression).", "",
          "| Section | Bars playing | Rhythm (share of bars with onsets) | Onsets/bar | Level dips | Progression | Bars per chord | Chords "
          "| Held (90 % of bars) | Repeat |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in secs:
        y = s["synths"]
        if not y or not y["bars_playing"]:
            L.append("| %s | 0 | %s | | | | | | | |" % (s["name"], why(y)))
            continue
        drone = y.get("drone") or {}
        held = ", ".join("%s%s %s" % (pc, " (%s)" % dg if dg else "", _pct(sh)) for pc, dg, sh in zip(
            drone.get("pitch_classes", []), drone.get("degrees", [None] * 12), drone.get("bars_share", []))) or "none"
        prog = (y["progression"] or "-") + f + (" (%s)" % _pct(y["progression_share"])
                                                if y.get("progression_share") not in (None, 1.0) else "")
        L.append("| %s | %d of %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s["name"], y["bars_playing"], s["bars"], _rhythm(y, f),
            _num(y["onsets_per_bar"]), _dips(y.get("level_by_16th_db"), f), prog, ("%g" % y["bars_per_chord"] + f) if y["bars_per_chord"] else
            ("whole section" if y.get("runs") and len(y["runs"]) == 1 else "-"),
            (", ".join(y["distinct_chords"]) + f) if y["distinct_chords"] else "-", held, _repeat(y, f)
            + (", loop %g bars" % y["loop_bars"] if y.get("loop_bars") else "")))
    L += _notes(secs, "synths")
    # vocals
    L += ["", "### Vocals", "", "| Section | Bars with vocal | Share | Phrase starts (bar.beat) |", "|---|---|---|---|"]
    for s in secs:
        v = s["vocals"]
        if not v or not v["bars_playing"]:
            L.append("| %s | 0 | 0 %% | %s |" % (s["name"], why(v)))
            continue
        starts = ", ".join("%d.%d" % (p["bar"], p["beat"]) for p in v["phrase_starts"] or [])
        L.append("| %s | %s%s | %s | %s |" % (s["name"], v["bars_with_vocal"], f, _pct(v["share"]),
                                              (starts + f) if starts else "- (%s)" % (v["notes"] or ["none"])[-1]))
    L += _notes(secs, "vocals")
    # drums
    L += ["", "### Drums", "", "Roles and patterns from drums.json (`x` most bars, `o` some, `.` fewer), the roles with 4 or more hits "
          "in the section; not measured again here.", "",
          "| Section | Stem | Role | Pattern | Hits | Bars with | Hat accent |", "|---|---|---|---|---|---|---|"]
    for s in secs:
        dr = s["drums"]
        if dr is None:
            L.append("| %s | - | - | - (no drums.json: `roadie drums`) | | | |" % s["name"])
            continue
        if not dr["roles"]:
            L.append("| %s | - | - | - (%s) | | | |" % (s["name"], dr["notes"][-1]))
            continue
        for r in dr["roles"]:
            acc = "-" if r["stem"] not in ("hh", "ride") else (
                "16th %s" % r["accent_16th"] if r.get("accent_16th") is not None else "missing (no per-hit levels in drums.json)")
            L.append("| %s | %s | %s%s | %s%s | %d | %d | %s |" % (
                s["name"] + (" (as %s)" % dr["drums_section"] if dr["drums_section"] != s["name"] else ""), r["stem"],
                r["role"], "" if r.get("named") else " (neutral)", _code(r["pattern"]), f, r["hits"], r["bars_with"], acc))
    L += _notes(secs, "drums")
    notes = list(mc.get("notes") or [])
    if notes:
        L += ["", "Micro notes:"] + ["- %s" % n for n in notes]
    return L


DIP_DB = 6.0


def _rhythm(x, f) -> str:
    """The rhythm cell: the pattern, its share and of how many bars, or `-` with the reason."""
    if not x.get("rhythm"):
        why = [n for n in x.get("notes") or [] if n.startswith(("no onset", "silent"))]
        return "- (%s)" % (why or ["no rhythm"])[0]
    return "%s%s (%s of %s)" % (_code(x["rhythm"]), f, _pct(x["rhythm_share"]), x.get("bars_with_onsets", "?"))


def _notes(secs, part) -> list[str]:
    """The per-section notes of one instrument (what a `-` cell stands for), the stem-wide synths note left out."""
    rows = []
    for s in secs:
        x = s.get(part)
        ns = [n for n in (x or {}).get("notes") or [] if not n.startswith("the synths stem holds")]
        if ns:
            rows.append("- %s: %s" % (s["name"], "; ".join(ns)))
    return ["", "Notes:"] + rows if rows else []


def _dips(levels, f) -> str:
    """The 16ths (1-based) DIP_DB or more under the loudest, from level_by_16th_db."""
    if not levels:
        return "-"
    low = [i + 1 for i, v in enumerate(levels) if v is None or v <= -DIP_DB]
    return ("16ths %s%s" % (", ".join(map(str, low)), f)) if low else "none"


def _repeat(x, f) -> str:
    if x.get("repeat_bars") is None:
        return "none of 1, 2, 4, 8 bars" if x.get("changes_at_bars") is not None else "- (too few playing bars)"
    ch = x.get("changes_at_bars") or []
    more = ", ..." if len(ch) > 8 else ""
    return "%d bar%s%s" % (x["repeat_bars"], "" if x["repeat_bars"] == 1 else "s", (
        "; changes at %s%s%s" % (", ".join(map(str, ch[:8])), more, f) if ch else "; no change"))


# ---- writing

def write_report(st: studymod.Study, text: str) -> str:
    """`report.md` in the study folder; a pCloud study through rclone with a read-back. Returns where it went."""
    return write_file(st.root / "report.md", text)


def write_file(path: Path, text: str) -> str:
    """One generated file: on the pCloud mount through rclone (a temp file, `copyto`, a read-back compared), never
    through the mount; locally by a temp file and a rename. Returns where it went."""
    remote = keymod.pcloud_remote(path)
    if remote:
        with tempfile.TemporaryDirectory(prefix="roadie-report-") as tmp:
            Path(tmp, path.name).write_text(text)
            studymod._rclone("copyto", str(Path(tmp, path.name)), remote)
        if studymod._remote_read(remote) != text:
            raise OSError("%s: read back after writing and it is not what was written" % remote)
        return remote
    tmp_path = path.with_name("." + path.name + ".tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return str(path)


# ---- the artist report

ARTIST_KIND = "artist_report"
# Which aggregates need bar 1 where the grid put it. A chance-level downbeat (or a warp map, or no alignment.json) can put
# bar 1 on the wrong beat: that moves every 16-step pattern, on-1 share and phrase-start beat, and every boundary's place
# in its 8-bar phrase, so those leave out such tracks. Lengths, ratios, rates and chord numerals do not move when the
# whole grid shifts by a beat or two, and positions given as a share of the track move by under a bar: those use every
# track.
BAR1_RULE = ("Two kinds of aggregate. Bar-1-dependent ones (where section boundaries fall in their 8-bar phrase, 16-step "
             "patterns, the bass's share on step 1, vocal phrases starting on beat 1) leave out tracks whose grid is "
             "unreliable (chance-level downbeat, a warp map, no alignment.json): a downbeat on the wrong beat moves all of "
             "them. Shift-invariant ones (lengths of sections and kick-outs, Drop 2 / Drop 1, bars per chord, onsets per "
             "bar, duty, pedal or moving, progressions as numerals, and positions as a share of the track, which move by "
             "under a bar) use every track. Each row says how many tracks it used.")
KICKOUT_CLASSES = ((0, "no kick-out"), (1, "1-7 bars (a fill)"), (8, "8-15 bars"), (16, "16-31 bars"), (32, "32+ bars"))


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _majority(pairs):
    """The value with the most weight in [(value, weight)], None for none."""
    w = {}
    for v, n in pairs:
        if v is not None:
            w[v] = w.get(v, 0) + (n or 0)
    return max(w, key=lambda v: (w[v], str(v))) if w else None


def track_row(st: studymod.Study, d: dict) -> dict:
    """One study of the artist as the numbers the artist report compares (from its report record)."""
    secs = d["sections"]
    length = sum(s["bars"] for s in secs) or None
    first = secs[0]["file_bar"] if secs else None
    S = d["structure"] or {}
    kick = S.get("kick") or {}
    longest = max(kick.get("out") or [], key=lambda o: o["bars"], default=None)
    bass_entry = (S.get("bass") or {}).get("entry_bar")
    row = {"title": st.title, "path": str(st.root), "bpm": st.bpm, "key": (d["key"] or {}).get("short"),
           "camelot": (d["key"] or {}).get("camelot"), "key_source": (d["key"] or {}).get("source"),
           "verdict": d["grid"]["verdict"], "reliable": d["grid"]["reliable"], "old_stems": st.stem_model != studymod.STEM_MODEL_CURRENT,
           "length_bars": length, "sections": [(s["label"], s["bars"]) for s in secs],
           "section_bars": [s["bars"] for s in secs],
           "drop_ratio": (S.get("drop_ratio") or {}).get("ratio"),
           "kickout_start": longest["bar"] if longest else None, "kickout_bars": longest["bars"] if longest else (0 if kick else None),
           "kickout_in": longest["section"] if longest else None,
           "kickout_at": round((longest["bar"] - first) / length, 2) if longest and length and first is not None else None,
           "bass_entry": bass_entry,
           "bass_entry_at": round((bass_entry - first) / length, 2) if bass_entry is not None and length and first is not None else None,
           "bass_before_drop": (S.get("bass") or {}).get("before_first_drop"),
           "why_bass entry": None if (S.get("bass") or {}).get("before_first_drop") is not None else
           "no activity" if not S or (S.get("bass") or {}).get("entry_bar") is None else "no Drop label",
           "phrase_lines": None, "micro": d["micro"] is not None}
    bounds = [s["file_bar"] - first for s in secs[1:]] if first is not None else []
    if bounds:
        row["phrase_lines"] = round(sum(b % 8 == 0 for b in bounds) / len(bounds), 2)
    mc = d["micro"] or {}
    msecs = mc.get("sections") or []
    bass = [(s["bass"], s["bars"]) for s in msecs if s.get("bass") and s["bass"].get("bars_playing")]
    row.update(bass_movement=_majority((b["movement"], b["bars_playing"]) for b, _ in bass),
               bass_style=_majority((b["style"], b["bars_playing"]) for b, _ in bass),
               bass_duty=(round(sum(b["duty"] * b["bars_playing"] for b, _ in bass if b["duty"] is not None)
                               / sum(b["bars_playing"] for b, _ in bass if b["duty"] is not None), 2)
                          if any(b["duty"] is not None for b, _ in bass) else None),     # weighted by the bars it plays
               bass_onsets=_median([b["onsets_per_bar"] for b, _ in bass]),
               bass_on1=_median([b["on_step1_share"] for b, _ in bass]),
               bass_rhythm=max(bass, key=lambda x: x[0].get("bars_with_onsets") or 0)[0]["rhythm"] if bass else None)
    syn = [s["synths"] for s in msecs if s.get("synths") and s["synths"].get("bars_playing")]
    loops = [(y["progression"], y["bars_playing"]) for y in syn if y.get("progression_share") is not None]
    row.update(progression=_majority(loops), bars_per_chord=_median([y.get("bars_per_chord") for y in syn]),
               synth_onsets=_median([y["onsets_per_bar"] for y in syn]),
               loops=sorted({y["progression"] for y in syn if y.get("progression_share") is not None}))
    voc = [s["vocals"] for s in msecs if s.get("vocals") is not None]
    starts = [p for v in voc for p in v.get("phrase_starts") or []]
    row.update(vocal_share=round(sum(v["bars_playing"] for v in voc) / sum(s["bars"] for s in msecs), 2) if voc else None,
               vocal_beat1=round(sum(p["beat"] == 1 for p in starts) / len(starts), 2) if starts else None)
    roles = {}
    for s in msecs:
        for r in (s.get("drums") or {}).get("roles") or []:
            if r.get("named"):
                roles.setdefault(r["role"], set()).add(r["pattern"])
    row["drum_roles"] = {k: sorted(v) for k, v in roles.items()}
    row["has_drums"] = any(s.get("drums") is not None for s in msecs)
    row["key_reliable"] = bool(d["key"]) and (d["key"] or {}).get("source") != "detected"
    row["missing"] = d["missing"]
    row["caveats"] = [c for c in d["caveats"] if c.startswith("micro:")]
    return row


# (field, label, bar-1-dependent, format)
AGGREGATES = (("bpm", "bpm", False, "%g"), ("length_bars", "length, bars", False, "%g"),
              ("n_sections", "sections", False, "%g"), ("median_section", "section length (median), bars", False, "%g"),
              ("drop_ratio", "Drop 2 / Drop 1", False, "%.2f"), ("kickout_bars", "longest kick-out, bars", False, "%g"),
              ("kickout_at", "longest kick-out starts at (share of the track)", False, "%.2f"),
              ("bass_entry_at", "bass enters at (share of the track)", False, "%.2f"),
              ("bass_duty", "bass duty", False, "%.2f"), ("bass_onsets", "bass onsets per bar", False, "%.2f"),
              ("bars_per_chord", "bars per chord", False, "%g"), ("synth_onsets", "synth onsets per bar", False, "%.2f"),
              ("vocal_share", "share of bars with vocal", False, "%.2f"),
              ("phrase_lines", "boundaries on an 8-bar phrase line", True, "%.2f"),
              ("bass_on1", "bass onsets on step 1", True, "%.2f"),
              ("vocal_beat1", "vocal phrases starting on beat 1", True, "%.2f"))


def artist_data(artist: str, roots=None) -> dict:
    """The artist report as one record: a row per analysed study, the aggregates (each saying the tracks it used and
    which it left out, see BAR1_RULE), the patterns two or more tracks share, and the layers each lacks."""
    roots = studymod._roots(roots)
    label, _, rest = artist.partition(":")
    if rest.strip() and studymod._norm(label) in {studymod._norm(r.label) for r in roots}:
        roots, artist = [r for r in roots if studymod._norm(r.label) == studymod._norm(label)], rest.strip()
    refs = [r for r in studymod.find_studies(roots) if r.analysed and studymod._norm(r.artist) == studymod._norm(artist)]
    if not refs:
        raise ValueError("no analysed study by %r under %s" % (artist, ", ".join("%s=%s" % (r.label, r.path) for r in roots)))
    folders = {r.root.parent for r in refs}
    if len(folders) > 1:
        raise ValueError("%r is in several roots (%s): give one, as %s:%s" % (
            artist, ", ".join(sorted({r.source for r in refs})), refs[0].source, artist))
    rows, unreadable = [], []
    for r in refs:
        try:
            st = studymod.load_study(str(r.root), roots)
            mc, why = read_micro(st)
            rows.append(track_row(st, report_data(st, mc, why)))
        except (ValueError, KeyError, TypeError, AttributeError, OSError, subprocess.CalledProcessError) as e:
            unreadable.append({"title": r.title, "why": "%s: %s" % (type(e).__name__, e)})
    for row in rows:
        row["n_sections"] = len(row["sections"]) or None
        row["median_section"] = _median(row["section_bars"])
    aggs = []
    for field, name, bar1, fmt in AGGREGATES:
        use = [x for x in rows if x.get(field) is not None and (x["reliable"] or not bar1)]
        out = [x["title"] for x in rows if x.get(field) is not None and bar1 and not x["reliable"]]
        none = [x["title"] for x in rows if x.get(field) is None]
        vals = [x[field] for x in use]
        med = _median(vals)
        aggs.append({"field": field, "name": name, "bar1": bar1, "fmt": fmt, "median": None if med is None else round(med, 3),
                     "min": min(vals) if vals else None, "max": max(vals) if vals else None, "used": len(use),
                     "of": len(rows), "left_out_grid": out, "no_value": none})
    keys = {}
    for x in rows:
        if x["camelot"]:
            keys.setdefault(x["camelot"], []).append(x["title"])
    return {"kind": ARTIST_KIND, "version": 1, "artist": refs[0].artist, "folder": str(refs[0].root.parent),
            "rule": BAR1_RULE, "rows": rows, "aggregates": aggs, "keys": keys, "patterns": pattern_kinds(rows),
            "unreadable": unreadable}


def _kick_class(bars):
    return next(n for lo, n in reversed(KICKOUT_CLASSES) if bars >= lo)


def _drop_class(r):
    return "Drop 2 shorter (under 0.9)" if r < 0.9 else "Drop 2 longer (over 1.1)" if r > 1.1 else "about equal (0.9-1.1)"


# (scope, kind, the layer a track needs, needs a reliable grid, needs a reliable key, several values a track, value(s))
PATTERN_KINDS = (
    ("macro", "bass entry", "activity", False, False, False,
     lambda x: None if x["bass_before_drop"] is None else
     "bass in before Drop 1" if x["bass_before_drop"] else "bass held until Drop 1"),
    ("macro", "longest kick-out", "activity", False, False, False,
     lambda x: None if x["kickout_bars"] is None else _kick_class(x["kickout_bars"])),
    ("macro", "Drop 2 / Drop 1", "Drop 1 and Drop 2 labels", False, False, False,
     lambda x: None if x["drop_ratio"] is None else _drop_class(x["drop_ratio"])),
    ("micro", "chord loop (numerals, any rotation)", "micro", False, True, True,
     lambda x: None if not x["micro"] else (sorted({micro_rotation(lp.split()) for lp in x["loops"]}) or ["no loop"])),
    ("micro", "bars per chord (median)", "micro", False, False, False,
     lambda x: None if x["bars_per_chord"] is None else "%g" % x["bars_per_chord"]),
    ("micro", "bass type", "micro", False, False, False,
     lambda x: "%s, %s" % (x["bass_movement"], x["bass_style"]) if x["bass_movement"] else None),
    ("micro", "drum role", "drums", False, False, True,
     lambda x: sorted(x["drum_roles"]) or ["no named role"] if x["has_drums"] else None),
    ("micro", "drum role pattern (16 steps)", "drums", True, False, True,
     lambda x: sorted("%s `%s`" % (r, ln) for r, lines in x["drum_roles"].items() for ln in lines) or ["no named role"]
     if x["has_drums"] else None),
    ("micro", "bass rhythm (16 steps)", "micro", True, False, False,
     lambda x: "`%s`" % x["bass_rhythm"] if x["bass_rhythm"] else None),
)


def pattern_kinds(rows: list[dict]) -> list[dict]:
    """Each kind of pattern across the tracks, as a full distribution: {scope, kind, of (the tracks that qualify: with the
    layer, and a reliable grid or key where the kind needs one), values [{pattern, tracks}] by count, left_out {reason:
    [titles]}, qualifiers}. A 16-step line needs a reliable grid; a chord loop in numerals a key that is not only
    detected (a detected key is to harmony what a chance-level downbeat is to bar 1)."""
    out = []
    for scope, kind, layer, grid, key, multi, value in PATTERN_KINDS:
        left = {"no %s" % layer: [], "unreliable grid": [], "detected or no key": []}
        vals, of = {}, []
        shown = {}
        for x in rows:
            v = value(x)
            if v is None:
                left.setdefault(x.get("why_" + kind) or "no %s" % layer, []).append(x["title"])
                continue
            if grid and not x["reliable"]:
                left["unreliable grid"].append(x["title"])
                continue
            if key and not x["key_reliable"]:
                left["detected or no key"].append(x["title"])
                continue
            of.append(x["title"])
            for p in (v if multi else [v]):
                vals.setdefault(p, []).append(x["title"])
            if kind.startswith("chord loop"):
                for lp in x["loops"]:
                    shown.setdefault(micro_rotation(lp.split()), []).append(lp)
        name = (lambda p: max(set(shown[p]), key=lambda w: (shown[p].count(w), w)) if p in shown else p)
        quals = ["tracks with %s" % layer] + (["a reliable grid"] if grid else []) + (["a key that is not only detected"]
                                                                                       if key else [])
        out.append({"scope": scope, "kind": kind, "of": len(of), "qualifiers": ", ".join(quals), "multi": multi,
                    "values": sorted(({"pattern": name(p), "tracks": sorted(ts)} for p, ts in vals.items()),
                                     key=lambda g: (-len(g["tracks"]), g["pattern"])),
                    "left_out": {k: v for k, v in left.items() if v}})
    return out


def micro_rotation(loop: list[str]) -> str:
    """A loop written from its rotation that comes first in a fixed order, so two tracks' loops compare."""
    w = tuple(loop)
    return " ".join(min(w[i:] + w[:i] for i in range(len(w))))


def artist_markdown(a: dict) -> str:
    rows = a["rows"]
    unread = ", %d unreadable" % len(a["unreadable"]) if a["unreadable"] else ""
    L = ["# %s: artist report" % a["artist"], "",
         "Generated by `roadie study report --artist` from the %d analysed studies' records; a rebuild overwrites it "
         "(notes go to the vault). Each track's own report is `report.md` in its folder." % len(rows), "", a["rule"], "",
         "## Tracks", "",
         "| Track | bpm | Key | Grid | Bars | Sections | Drop 2 / Drop 1 | Longest kick-out | Bass enters | Bass | "
         "Synth loop, bars per chord | Vocal |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        f = "" if x["reliable"] else FLAG
        L.append("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            x["title"], " (old stems)" if x["old_stems"] else "", _num(x["bpm"]),
            "%s %s (%s)" % (x["key"], x["camelot"], x["key_source"]) if x["key"] else "missing",
            (x["verdict"] or "none") + ("" if x["reliable"] else " " + FLAG), _num(x["length_bars"]),
            " ".join("%s %d" % (lab, b) for lab, b in x["sections"]) or "-", _num(x["drop_ratio"], "%.2f"),
            ("bar %d%s, %d bars (%s)" % (x["kickout_start"], f, x["kickout_bars"], x["kickout_in"] or "no label")
             if x["kickout_start"] is not None else ("none" if x["kickout_bars"] == 0 else "-")),
            ("bar %d%s%s" % (x["bass_entry"], f, "" if x["bass_before_drop"] is None else
                             ", before Drop 1" if x["bass_before_drop"] else ", with or after Drop 1"))
            if x["bass_entry"] is not None else "-",
            ("%s, duty %.2f" % (x["bass_movement"], x["bass_duty"]) if x["bass_movement"] else "-") if x["micro"] else "missing: micro",
            ("%s, %s" % (x["progression"] or "no loop", "%g" % x["bars_per_chord"] if x["bars_per_chord"] else "-"))
            if x["micro"] else "missing: micro",
            _pct(x["vocal_share"]) if x["micro"] else "missing: micro"))
    if any(not x["reliable"] for x in rows):
        L += ["", "%s grid unreliable: its bar positions are a guess (the rule above)." % FLAG]
    L += ["", "## Medians and spreads", "", "| Measure | Kind | Median | Range | Tracks used | Left out |", "|---|---|---|---|---|---|"]
    for g in a["aggregates"]:
        fmt = g["fmt"]
        left = []
        if g["left_out_grid"]:
            left.append("unreliable grid: " + ", ".join(g["left_out_grid"]))
        if g["no_value"]:
            left.append("no value: " + ", ".join(g["no_value"]))
        L.append("| %s | %s | %s | %s | %d of %d%s | %s |" % (
            g["name"], "bar-1-dependent" if g["bar1"] else "shift-invariant",
            _num(g["median"], fmt) if g["used"] else "- (no track)",
            ("%s-%s" % (_num(g["min"], fmt), _num(g["max"], fmt))) if g["used"] else "-",
            g["used"], g["of"], unread, "; ".join(left) or "-"))
    L += ["", "Keys: " + ", ".join("%s (%s)" % (cam, ", ".join(ts)) for cam, ts in sorted(a["keys"].items())) if a["keys"] else "Keys: none recorded"]
    L += ["", "## Patterns across tracks", "",
          "Each kind in full, out of the tracks that qualify for it (the layer it needs, and a reliable grid for 16-step "
          "lines, a key that is not only detected for chord numerals). With this few tracks a coarse kind (two to five "
          "classes) puts two tracks in one class by construction: a class is a finding only against the library's rate "
          "for it.", "", "| Scope | Kind | Out of | Distribution | Left out |", "|---|---|---|---|---|"]
    for g in a["patterns"]:
        left = "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in g["left_out"].items()) or "-"
        if g["of"] < 2:
            dist = "fewer than 2 tracks qualify (%d): not compared" % g["of"]
        elif not g["multi"] and len(g["values"]) == 1:
            dist = "all %d: %s" % (g["of"], g["values"][0]["pattern"])
        else:
            dist = "; ".join("%s: all %d" % (v["pattern"], g["of"]) if len(v["tracks"]) == g["of"] else
                             "%s: %d of %d (%s)" % (v["pattern"], len(v["tracks"]), g["of"], ", ".join(v["tracks"]))
                             for v in g["values"])
        L.append("| %s | %s | %d of %d%s (%s) | %s | %s |" % (
            g["scope"], g["kind"], g["of"], len(rows), unread, g["qualifiers"], dist, left))
    L += ["", "## Missing layers", ""]
    miss = [(x["title"], m) for x in rows for m in x["missing"]]
    if miss:
        L += ["| Track | Missing | Command |", "|---|---|---|"] + ["| %s | %s | `%s` |" % (t, m["layer"], m["command"]) for t, m in miss]
    else:
        L.append("none")
    notes = [(x["title"], c) for x in rows for c in x["caveats"]] + [(u["title"], "could not be read: " + u["why"])
                                                                       for u in a["unreadable"]]
    old = [x["title"] for x in rows if x["old_stems"]]
    if old:
        notes.append((", ".join(old), "stems from an older separation model (re-split planned in studies/08)"))
    if notes:
        L += ["", "## Notes", ""] + ["- %s: %s" % n for n in notes]
    return "\n".join(L) + "\n"


def main(args) -> int:
    """`roadie study report` (args from study.py's parser: name or --artist, write, json)."""
    if bool(getattr(args, "artist", None)) == bool(args.name):
        print("study: report: give a study name or --artist <name>, one of the two", file=sys.stderr)
        return 2
    try:
        if args.artist:
            data = artist_data(args.artist)
            text = artist_markdown(data)
            where = write_file(Path(data["folder"]) / "_artist.md", text) if args.write else None
        else:
            st = studymod.load_study(args.name)
            mc, why = read_micro(st)
            data = report_data(st, mc, why)
            text = to_markdown(data)
            where = write_report(st, text) if args.write else None
    except (ValueError, KeyError, TypeError, AttributeError, OSError, subprocess.CalledProcessError) as e:
        print("study: report: %s" % (e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e),
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif where:
        print("wrote %s" % where)
    else:
        print(text, end="")
    return 0
