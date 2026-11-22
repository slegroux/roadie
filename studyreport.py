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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arrangement  # noqa: E402
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
    if st.mix():
        data["mix"] = st.mix()
    if micro is None:
        data["missing"].append({"layer": "micro", "command": micro_missing})
    else:
        made = (st.layers["drums"].data or {}).get("made")
        if made and micro.get("drums_made") and made != micro["drums_made"]:
            data["caveats"].append("micro: its drum roles come from an older drums.json (%s, now %s): re-run roadie micro"
                                   % (micro["drums_made"], made))
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
    L += ["", "### Bass", "", "| Section | Bars playing | Rhythm (share of bars) | 16th steps | Onsets/bar | On 1 / on 8th offbeats "
          "| Notes (degrees) | Register | Pedal or moving | Duty | Repeat |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in secs:
        b = s["bass"]
        if not b or not b["bars_playing"]:
            L.append("| %s | 0 | %s | | | | | | | | |" % (s["name"], why(b)))
            continue
        degs = b["degrees"] or b.get("pitch_names")
        notes = ", ".join("%s %s" % (x.get("degree") or x["note"], _pct(x["share"])) for x in degs) if degs else "-"
        L.append("| %s | %d of %d | %s%s (%s) | %s | %s | %s / %s | %s | %s (%s) | %s | %s %s | %s |" % (
            s["name"], b["bars_playing"], s["bars"], _code(b["rhythm"]), f, _pct(b["rhythm_share"]),
            ", ".join(map(str, b["rhythm_steps"] or [])) or "-", _num(b["onsets_per_bar"]),
            _pct(b["on_step1_share"]), _pct(b["on_offbeat_8ths_share"]), notes, b["register"] or "-",
            b.get("register_range") or "-",
            "%s (%s %s, %s classes)" % (b["movement"], b["top_pitch_class"], _pct(b["top_pitch_class_share"]), b["pitch_classes"])
            if b["movement"] else "-", _num(b["duty"], "%.2f"), b["style"] or "", _repeat(b, f)))
    # synths
    L += ["", "### Synths", "", "Leads and pads are one stem: the rhythm and the chords are of both together.", "",
          "| Section | Bars playing | Rhythm (share of bars) | Onsets/bar | Progression | Bars per chord | Chords | Drone | Repeat |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in secs:
        y = s["synths"]
        if not y or not y["bars_playing"]:
            L.append("| %s | 0 | %s | | | | | | |" % (s["name"], why(y)))
            continue
        drone = y.get("drone") or {}
        held = ", ".join("%s%s %s" % (pc, " (%s)" % dg if dg else "", _pct(sh)) for pc, dg, sh in zip(
            drone.get("pitch_classes", []), drone.get("degrees", [None] * 12), drone.get("bars_share", []))) or "none"
        L.append("| %s | %d of %d | %s%s (%s) | %s | %s%s | %s | %s | %s | %s |" % (
            s["name"], y["bars_playing"], s["bars"], _code(y["rhythm"]), f, _pct(y["rhythm_share"]), _num(y["onsets_per_bar"]),
            y["progression"] or "-", f, ("%g" % y["bars_per_chord"] + f) if y["bars_per_chord"] else
            ("whole section" if y.get("runs") and len(y["runs"]) == 1 else "-"),
            ", ".join(y["distinct_chords"] or []) or "-", held, _repeat(y, f)
            + (", loop %d bars" % y["loop_bars"] if y.get("loop_bars") else "")))
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
    # drums
    L += ["", "### Drums", "", "Roles and patterns from drums.json (`x` most bars, `o` some, `.` fewer); not measured again here.", "",
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
                "16th %s" % r["accent_16th"] if r.get("accent_16th") else "missing (no per-hit levels in drums.json)")
            L.append("| %s | %s | %s%s | %s%s | %d | %d | %s |" % (
                s["name"] + (" (as %s)" % dr["drums_section"] if dr["drums_section"] != s["name"] else ""), r["stem"],
                r["role"], "" if r.get("named") else " (neutral)", _code(r["pattern"]), f, r["hits"], r["bars_with"], acc))
    notes = list(mc.get("notes") or [])
    if notes:
        L += ["", "Micro notes:"] + ["- %s" % n for n in notes]
    return L


def _repeat(x, f) -> str:
    if x.get("repeat_bars") is None:
        return "none of 1, 2, 4, 8 bars"
    ch = x.get("changes_at_bars") or []
    more = ", ..." if len(ch) > 8 else ""
    return "%d bar%s%s" % (x["repeat_bars"], "" if x["repeat_bars"] == 1 else "s", (
        "; changes at %s%s%s" % (", ".join(map(str, ch[:8])), more, f) if ch else "; no change"))


# ---- writing

def write_report(st: studymod.Study, text: str) -> str:
    """`report.md` in the study folder; a pCloud study through rclone with a read-back. Returns where it went."""
    path = st.root / "report.md"
    remote = keymod.pcloud_remote(path)
    if remote:
        with tempfile.TemporaryDirectory(prefix="roadie-report-") as tmp:
            Path(tmp, "report.md").write_text(text)
            studymod._rclone("copyto", str(Path(tmp, "report.md")), remote)
        if studymod._remote_read(remote) != text:
            raise OSError("%s: read back after writing and it is not what was written" % remote)
        return remote
    tmp_path = path.with_name(".report.md.tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)
    return str(path)


def main(args) -> int:
    """`roadie study report` (args from study.py's parser: name, write, json)."""
    try:
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
