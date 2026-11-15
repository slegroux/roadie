"""The reference-profile half of the analysis: `reference_profile`, and `session_compare`, the element and
drum layers added to Sideman's mix report of a capture, with their markdown.

Copied from `sideman/listen.py` (Sideman main bc9ba08) with behaviour unchanged: the same inputs give
byte-identical profile JSON, report JSON and markdown to Sideman main's `mix_report` with a profile
(`test_analysis_equivalence.py` pins that against files written by Sideman main). The measurement core it
builds on is Sideman's, imported by its public names from `sideman.listen`; this module holds only what
needs a reference track, stems or samples.

`session_compare` does not build the plain report itself: it takes it from Sideman (`load_capture`,
`mix_layer`), adds its layers, and has Sideman write it (`write_report`), so the mix layer, the audio-file
reference and the plain markdown exist once, in Sideman.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import soundfile as sf

from sideman.listen import (BANDS, CHANGE_BLOCK_BARS, FLOOR_DB, amp_to_db, analyze_array, band_power,
                            check_sections, compare, load, load_capture, mix_layer, peak_rms, slice_span, spectra,
                            to_db, to_stereo, write_report)

# ------------------------------------------------------- reference profile
#
# A reference track that comes with stems, measured once per section so a
# capture of the Set can be compared with it later (mix_report's `reference`).
# Levels are mean power in dBFS over stereo (mono is centred); a role's
# "vs_kick_db" is its level minus the kick's in the same section, which cancels
# a gain the stems and the mix share.

PROFILE_KIND = "reference_profile"
PROFILE_VERSION = 6        # 6: `core_bar_db` (presence on each bar's core); 5: windows start GRID_LEAD_S early (no edge guard)
ABSENT_DB = 60.0           # a role with under a quarter of its bars within this of the stems' sum is absent, not -120 dB
TRACE_DB = 40.0            # ...and within ABSENT_DB but not this of it, a trace: listed, not a gap against an absent one
GRID_LEAD_S = 0.020        # a hit that starts up to this long before a grid line (real drums do, and a capture is never
                           # sample-aligned) belongs to the bar after it, on both sides: a bar's level window starts at
                           # such a hit's onset, otherwise on the line, so a note or a tail that ends on the line stays
                           # in the bar before. Presence is judged on each bar's core, this far inside both of its lines:
                           # what sounds within it of a line never makes a role present in a bar or a section
ELEMENT_GAP_DB = 1.0       # an element's ratio to the kick differing by more than this is a gap
WIDTH_GAP = 0.1            # an element's width (0 mono .. 1 all side) differing by more than this is a width gap
CAPTURE_IS_REF_DB = 0.3    # a master capture is the reference when, with the level offset removed,
CAPTURE_IS_REF_WIDTH = 0.01  # its per-bar curve and bands agree within this and its width within this
STALE_MTIME_S = 2.0        # an input whose mtime moved by less than this (a cloud re-sync) is not changed
KICK_OUT_DB = 30.0         # the kick is out where it sits this far under the loudest other role
DRUM_ROLES = ("kick", "toms", "snare", "hh", "clap", "rim")   # roles the drum layer reads


def _level_db(x: np.ndarray) -> float:
    """Mean power of a stereo segment in dBFS; FLOOR_DB when it is silent or empty."""
    return amp_to_db(peak_rms(x)[1])


def _line_start(x: np.ndarray, sr: int, t: float) -> int:
    """Sample of x [n, channels] where the level window of the bar beginning at line t (s) starts: the onset
    of a hit that rises within GRID_LEAD_S before the line (oneshot's onset rule: out of the loudest 1 ms of
    the 20 ms before), else the line itself. A line under 2 * GRID_LEAD_S into x has no lead."""
    from .oneshot import _onset
    g, line = int(round(GRID_LEAD_S * sr)), int(round(t * sr))
    if line < 2 * g or line >= len(x):
        return max(0, min(line, len(x)))
    win = x[line - 2 * g:line + g]
    i = _onset(win, np.abs(win).max(axis=1), sr, True, GRID_LEAD_S)
    return line - 2 * g + i if g <= i < 2 * g else line


def _bar_levels(x: np.ndarray, sr: int, t0: float, bar_s: float, first_bar: int, n: int, core: bool = False) -> list:
    """Level in dBFS of each of `n` bars from `first_bar` (bar 0 at t0 s); None for a silent bar. A bar's
    window runs from its _line_start to the next one's; with `core`, from GRID_LEAD_S after its line to
    GRID_LEAD_S before the next (presence)."""
    out = []
    lines = [t0 + (first_bar + i) * bar_s for i in range(n + 1)]
    edges = None if core else [_line_start(x, sr, t) for t in lines]
    for i in range(n):
        if core:
            seg = slice_span(x, sr, lines[i] + GRID_LEAD_S, lines[i + 1] - GRID_LEAD_S)
        else:
            seg = x[edges[i]:edges[i + 1]]
        lv = _level_db(seg)
        out.append(None if lv <= FLOOR_DB else round(lv, 2))
    return out


def _presence(per_bar: list, mix_level: float | None) -> str:
    """"present", "trace" or "absent" from the per-bar levels of the bars' cores against a reference
    level (the power mean of the per-bar cores of the stems' sum, or of our element sources' sum; never
    the mix or the master capture). Present when at least a quarter of the bars (and one) are within
    TRACE_DB of it; otherwise a trace when at least a quarter are within ABSENT_DB of it; otherwise
    absent. A phrase in a few bars counts; what sounds within GRID_LEAD_S of a line does not."""
    if mix_level is None or not per_bar:
        return "absent"
    need = max(1.0, len(per_bar) / 4.0)
    within = lambda db: sum(v is not None and v >= mix_level - db for v in per_bar)   # noqa: E731
    if within(TRACE_DB) >= need:
        return "present"
    return "trace" if within(ABSENT_DB) >= need else "absent"


def _bands_width(x: np.ndarray, sr: int) -> tuple[dict, float | None]:
    """Six-band mean power (dBFS) and width of a stereo segment, as analyze_array reads them."""
    sp = spectra(x, sr)
    lr = (sp["l"] + sp["r"]) / 2.0
    bands = {name: round(to_db(band_power(lr, sp["freqs"], lo, hi)), 2) for name, lo, hi in BANDS}
    pm, ps = float(sp["m"].sum()), float(sp["s"].sum())
    return bands, (round(math.sqrt(ps) / (math.sqrt(pm) + math.sqrt(ps)), 3) if pm + ps > 0 else None)


def _mean_db(vals: list) -> float | None:
    """Power mean of per-bar dB values (None counts as silence); None when all are silent."""
    p = float(np.mean([0.0 if v is None else 10.0 ** (v / 10.0) for v in vals])) if vals else 0.0
    return 10.0 * math.log10(p) if p > 0 else None


def _arc(role_bars: list, kick_bars: list) -> list:
    """The role's level against the kick for each CHANGE_BLOCK_BARS block of bars (None where
    either is silent)."""
    out = []
    for i in range(0, len(role_bars), CHANGE_BLOCK_BARS):
        r, k = _mean_db(role_bars[i:i + CHANGE_BLOCK_BARS]), _mean_db(kick_bars[i:i + CHANGE_BLOCK_BARS])
        out.append(None if r is None or k is None else round(r - k, 2))
    return out


def _input(p: pathlib.Path) -> dict:
    st = p.stat()
    return {"path": str(p), "mtime": st.st_mtime, "size": st.st_size}


def reference_profile(mix: str | pathlib.Path, stems: dict, tempo: float, t0: float, sections: list,
                      beats_per_bar: int = 4, out: str | pathlib.Path | None = None) -> dict:
    """Profile of a reference track and its stems, per section; written as JSON.

    `stems` maps a role to a file path or a list of paths summed into that role
    (`drums` = hh + snare + toms). Roles are free-form; `kick` is the yardstick
    for every level against the kick, and "mix" is reserved. `sections` is a list
    of {"name", "start_bar", "bars"} as in a capture manifest, names unique;
    `t0` is where bar 0 sits in the files, in seconds.

    Per section: the mix's level (mean power, dBFS), integrated LUFS, six bands,
    width and per-bar levels; and for each role its level, its level against the
    kick (`vs_kick_db`), per-bar levels, six bands and width. Presence is judged against
    the stems together (`stems_sum`: their power sum, per section and per bar), not the
    mix, so a master's make-up gain does not change it. A role with under a quarter of its
    bars (and one) within ABSENT_DB of the stems' sum is {"absent": true, "per_bar_db": [...]}
    there, with no other numbers; with a quarter within ABSENT_DB but not within TRACE_DB the
    entry carries "trace": true. Presence is judged on each bar's core (`core_bar_db`, GRID_LEAD_S
    inside both lines, for every role and the stems' sum). For the roles' levels, bands and widths and
    all per-bar levels a section or bar window starts at the onset of a hit that rises within GRID_LEAD_S
    before its line, else on the line (the mix's bands, width and LUFS use the plain window). Files must share one sample rate
    (refused otherwise); of different lengths they are trimmed to the shortest
    and `notes` says so.

    Writes <mix stem>.profile.json next to the mix (`out`: a file path ending
    in .json, or a folder) and returns the profile with its `json_path`. The
    profile records each input's path, mtime and size under `inputs`;
    profile_stale() says which have changed since."""
    import json
    if not isinstance(tempo, (int, float)) or not math.isfinite(tempo) or tempo <= 0:
        raise ValueError("tempo must be a number above 0, got %r" % (tempo,))
    if not isinstance(t0, (int, float)) or not math.isfinite(t0) or t0 < 0:
        raise ValueError("t0 must be a number of seconds >= 0, got %r" % (t0,))
    if not isinstance(beats_per_bar, (int, float)) or not math.isfinite(beats_per_bar) or beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be a number above 0")
    if not sections:
        raise ValueError("no sections")
    check_sections(sections)
    names = [s.get("name", "bar %d" % s["start_bar"]) for s in sections]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError("section names must be unique (mix_report pairs sections by name): %s" % ", ".join(dup))
    if not stems:
        raise ValueError("no stems")

    def existing(label: str, path) -> pathlib.Path:
        p = pathlib.Path(path).expanduser().resolve()
        if not p.is_file():
            raise ValueError("%s: no such audio file: %s" % (label, p))
        return p

    mixp = existing("mix", mix)
    files: dict[str, list[pathlib.Path]] = {}
    for role, v in stems.items():
        if not isinstance(role, str) or not role or role == "mix":
            raise ValueError("stem role must be a non-empty name other than 'mix', got %r" % (role,))
        paths = [v] if isinstance(v, (str, pathlib.Path)) else list(v)
        if not paths:
            raise ValueError("stem %r has no files" % role)
        files[role] = [existing("stem %r" % role, p) for p in paths]
    infos = {p: sf.info(str(p)) for p in [mixp] + [p for ps in files.values() for p in ps]}
    rates = {p: i.samplerate for p, i in infos.items()}
    if len(set(rates.values())) > 1:
        raise ValueError("files must share one sample rate: " + ", ".join(
            "%s %d Hz" % (p.name, r) for p, r in rates.items()))
    sr = next(iter(rates.values()))
    frames = {p: i.frames for p, i in infos.items()}
    n = min(frames.values())
    notes = []
    if len(set(frames.values())) > 1:
        shortest = min(frames, key=frames.get)
        notes.append("files differ in length (%.2f to %.2f s); all trimmed to the shortest, %s (%.2f s)"
                     % (min(frames.values()) / sr, max(frames.values()) / sr, shortest.name, n / sr))
    bar_s = float(beats_per_bar) * 60.0 / float(tempo)
    end_s = t0 + max(s["start_bar"] + s["bars"] for s in sections) * bar_s
    if n / sr + 0.01 < end_s:
        raise ValueError("the files are %.2f s long but the sections run to %.2f s" % (n / sr, end_s))
    if "kick" not in files:
        notes.append("no kick stem: levels against the kick are left out")

    def span(sec):
        a = t0 + sec["start_bar"] * bar_s
        return a, a + sec["bars"] * bar_s

    x, _ = load(mixp)
    x = x[:n]
    mix_rows = []
    for sec in sections:
        a = analyze_array(slice_span(x, sr, *span(sec)), sr)
        mix_rows.append({"level_db": a["rms_db"], "integrated_lufs": a["integrated_lufs"],
                         "width": a["width"], "bands_db": a["bands_db"],
                         "per_bar_db": _bar_levels(to_stereo(x), sr, t0, bar_s, sec["start_bar"], sec["bars"])})
    nsec = len(sections)
    measured: dict[str, list] = {}
    for role, paths in files.items():
        y = None
        for p in paths:
            z = to_stereo(load(p)[0])[:n]
            y = z.copy() if y is None else y + z
        measured[role] = []
        for sec in sections:
            seg = y[_line_start(y, sr, span(sec)[0]):_line_start(y, sr, span(sec)[1])]
            lv = _level_db(seg)
            bands, width = _bands_width(seg, sr)
            measured[role].append({"level": lv, "bands": bands, "width": width,
                                   "bars": _bar_levels(y, sr, t0, bar_s, sec["start_bar"], sec["bars"]),
                                   "core": _bar_levels(y, sr, t0, bar_s, sec["start_bar"], sec["bars"], core=True)})
    # presence is judged against the stems together, not the mix: the mix can carry a master's make-up gain
    def bar_sum(i, key, n):
        power = [sum(0.0 if m[i][key][j] is None else 10 ** (m[i][key][j] / 10) for m in measured.values())
                 for j in range(n)]
        return [round(to_db(v), 2) if v > 0 else None for v in power]

    sums = []
    for i, sec in enumerate(sections):
        total = sum(10 ** (m[i]["level"] / 10) for m in measured.values() if m[i]["level"] > FLOOR_DB)
        sums.append({"level_db": round(to_db(total), 2), "per_bar_db": bar_sum(i, "bars", sec["bars"]),
                     "core_bar_db": bar_sum(i, "core", sec["bars"])})
    role_rows: dict[str, list] = {}
    raw: dict[str, list] = {}
    for role, rows in measured.items():
        role_rows[role], raw[role] = [], []
        for i in range(nsec):
            m, per_bar = rows[i], rows[i]["bars"]
            kind = _presence(m["core"], _mean_db(sums[i]["core_bar_db"]))
            if kind == "absent":
                role_rows[role].append({"absent": True, "per_bar_db": per_bar, "core_bar_db": m["core"]})
                raw[role].append(None)
                continue
            role_rows[role].append({"absent": False, "trace": kind == "trace", "level_db": round(m["level"], 2),
                                    "vs_kick_db": None, "per_bar_db": per_bar, "core_bar_db": m["core"],
                                    "bands_db": m["bands"],
                                    "width": m["width"]})
            raw[role].append(m["level"])
    if "kick" in files:
        for i in range(nsec):
            for role in files:
                if raw[role][i] is not None and raw["kick"][i] is not None:
                    role_rows[role][i]["vs_kick_db"] = round(raw[role][i] - raw["kick"][i], 2)
    profile = {
        "kind": PROFILE_KIND, "version": PROFILE_VERSION, "mix": str(mixp), "tempo": tempo,
        "beats_per_bar": beats_per_bar, "t0": t0, "sample_rate": sr, "notes": notes,
        "inputs": {"mix": [_input(mixp)], "stems": {r: [_input(p) for p in ps] for r, ps in files.items()}},
        "sections": [{"name": nm, "start_bar": sec["start_bar"], "bars": sec["bars"], "mix": mix_rows[i],
                      "stems_sum": sums[i], "roles": {role: role_rows[role][i] for role in files}}
                     for i, (nm, sec) in enumerate(zip(names, sections))]}
    dest = pathlib.Path(out).expanduser() if out else mixp.parent
    if dest.is_dir() or not str(dest).lower().endswith(".json"):
        dest = dest / (mixp.stem + ".profile.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    profile["json_path"] = str(dest.resolve())
    dest.write_text(json.dumps(profile, indent=1))
    return profile


def profile_stale(profile: dict) -> list:
    """The inputs of a reference profile that have changed since it was built, as
    {"role", "path", "reason"} ("changed": size differs or mtime moved by more than
    STALE_MTIME_S, so a cloud re-sync that touches mtimes is not a change; "missing").
    Empty when the profile still matches its files."""
    out = []
    groups = [("mix", profile["inputs"]["mix"])] + list(profile["inputs"]["stems"].items())
    for role, items in groups:
        for it in items:
            p = pathlib.Path(it["path"])
            if not p.is_file():
                out.append({"role": role, "path": it["path"], "reason": "missing"})
                continue
            st = p.stat()
            if st.st_size != it["size"] or abs(st.st_mtime - it["mtime"]) > STALE_MTIME_S:
                out.append({"role": role, "path": it["path"], "reason": "changed"})
    return out


def _check_profile(prof: dict) -> None:
    """Raise ValueError (saying to rebuild) unless `prof` has the shape reference_profile writes."""
    if not isinstance(prof, dict) or prof.get("kind") != PROFILE_KIND:
        raise ValueError("not a reference profile (build one with reference_profile)")
    if prof.get("version") != PROFILE_VERSION:
        raise ValueError("reference profile version %r, this Sideman reads version %d: rebuild it with "
                         "reference_profile" % (prof.get("version"), PROFILE_VERSION))

    def need(ok: bool) -> None:
        if not ok:
            raise KeyError

    def num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    try:
        for key in ("tempo", "beats_per_bar", "t0", "sample_rate"):
            need(num(prof[key]))
        for items in [prof["inputs"]["mix"], *prof["inputs"]["stems"].values()]:
            need(isinstance(items, list))
            for it in items:
                need(isinstance(it["path"], str) and num(it["mtime"]) and num(it["size"]))
        need(isinstance(prof["sections"], list) and len(prof["sections"]) > 0)
        for sec in prof["sections"]:
            n = sec["bars"]
            need(isinstance(sec["name"], str) and isinstance(sec["start_bar"], int) and isinstance(n, int) and n >= 1)
            m = sec["mix"]
            need(num(m["level_db"]) and len(m["per_bar_db"]) == n and "width" in m and "integrated_lufs" in m)
            need(all(num(m["bands_db"][name]) for name, _, _ in BANDS))
            need(num(sec["stems_sum"]["level_db"]) and len(sec["stems_sum"]["per_bar_db"]) == n)
            need(len(sec["stems_sum"]["core_bar_db"]) == n)
            need(isinstance(sec["roles"], dict))
            for e in sec["roles"].values():
                need(isinstance(e["absent"], bool) and len(e["per_bar_db"]) == n and len(e["core_bar_db"]) == n)
                if not e["absent"]:
                    need(isinstance(e["trace"], bool) and num(e["level_db"]) and "width" in e)
                    need(all(name in e["bands_db"] for name, _, _ in BANDS))
    except (KeyError, TypeError, AttributeError, IndexError):
        raise ValueError("the reference profile is incomplete or malformed: rebuild it with reference_profile")


def _load_profile(reference) -> dict | None:
    """The reference as a validated profile dict when it is one (a dict, JSON text, or the path of
    a .json file); None for an audio file path or no reference. Raises ValueError for a profile
    that is not the current shape."""
    import json
    if isinstance(reference, dict):
        prof = reference
    elif isinstance(reference, str) and reference.lstrip().startswith("{"):
        prof = json.loads(reference)
    elif isinstance(reference, (str, pathlib.Path)) and str(reference).lower().endswith(".json"):
        p = pathlib.Path(reference).expanduser()
        if not p.is_file():
            raise ValueError("no such profile: %s" % p)
        prof = json.loads(p.read_text())
        if isinstance(prof, dict):
            prof["json_path"] = str(p.resolve())            # where it was read, not where it was built
    else:
        return None
    _check_profile(prof)
    return prof


def _pair(prof: dict, sec: dict, name: str) -> tuple[dict | None, int, list]:
    """(profile section, first bar inside it, notes) for a manifest section. By default the
    profile section of the same name from its first bar; the manifest section's
    "reference": {"section": <name>, "from_bar": N} (both optional) picks another section or
    starts N bars in. The range must lie inside the profile section: a shorter manifest
    section pairs with its first bars (noted), a longer one is not paired."""
    ref = sec.get("reference")
    if ref is not None:
        ok = (isinstance(ref, dict) and set(ref) <= {"section", "from_bar"}
              and isinstance(ref.get("section", ""), str)
              and isinstance(ref.get("from_bar", 0), int) and ref.get("from_bar", 0) >= 0)
        if not ok:
            raise ValueError('section %r: "reference" must be {"section": name, "from_bar": whole number >= 0}, '
                             "got %r" % (name, ref))
    ref = ref or {}
    pname, fb = ref.get("section", name), ref.get("from_bar", 0)
    psec = next((s for s in prof["sections"] if s["name"] == pname), None)
    if psec is None:
        return None, 0, ["no section named %r in the profile (it has: %s): mix layer only"
                         % (pname, ", ".join(x["name"] for x in prof["sections"]))]
    if fb + sec["bars"] > psec["bars"]:
        return None, 0, ["section %r is %d bars here; from bar %d of the profile's %r (%d bars) only %d fit: "
                         "mix layer only" % (name, sec["bars"], fb + 1, pname, psec["bars"],
                                             max(0, psec["bars"] - fb))]
    notes = []
    if fb or sec["bars"] != psec["bars"]:
        notes.append("paired with bars %d-%d of the profile's %r (%d bars); the reference's bands, widths "
                     "and loudness are for the whole section, so width gaps are not ranked and the "
                     "capture-is-the-reference check is not made" % (fb + 1, fb + sec["bars"], pname, psec["bars"]))
    return psec, fb, notes


def _same_shape(a: dict, pm: dict, ours_bars: list, ref_bars: list) -> bool:
    """Whether the capture's master is the reference mix, level aside: width within
    CAPTURE_IS_REF_WIDTH, and after subtracting the median difference, the per-bar levels and the
    bands (those within ABSENT_DB of the loudest band on both sides) within CAPTURE_IS_REF_DB. A fader a few tenths off
    still matches; one element 15 dB louder moves its band and does not."""
    if a["width"] is None or pm["width"] is None or abs(a["width"] - pm["width"]) > CAPTURE_IS_REF_WIDTH:
        return False
    bars = [o - r for o, r in zip(ours_bars, ref_bars) if o is not None and r is not None]
    top = (max(a["bands_db"].values()) - ABSENT_DB, max(pm["bands_db"].values()) - ABSENT_DB)
    bands = [a["bands_db"][n] - pm["bands_db"][n] for n, _, _ in BANDS      # bands 60 dB under the loudest are
             if a["bands_db"][n] > top[0] and pm["bands_db"][n] > top[1]]  # window leakage, not material
    if not bars or not bands or len(ours_bars) != len(ref_bars):
        return False
    return all(max(abs(np.array(d) - float(np.median(d)))) <= CAPTURE_IS_REF_DB for d in (bars, bands))


def _element_layer(prof: dict, psec: dict, fb: int, notes: list, sec: dict, a: dict, audio: dict, sr: int,
                   t0: float, bar_s: float) -> tuple[dict, dict]:
    """(vs_reference, elements) for one manifest section paired with `psec` of the profile
    (starting `fb` bars in). Levels are ratios to the kick, or, where the kick is out on either
    side (absent, or KICK_OUT_DB under the loudest other role), to the loudest role present on
    both sides, the same one for both: `yardstick` says which. Presence is judged on the bars' cores
    against our element sources together and the profile's stems together, never the master capture;
    levels on windows that start at a hit rising within GRID_LEAD_S before a line (_line_start). A role
    present on one side and a trace on the other is compared (`trace_side` says which); a role that
    is a trace or absent on both sides is listed in `traces`. A kick out on one side only is
    listed in the one-side lists as well."""
    n = sec["bars"]
    partial = bool(fb) or n != psec["bars"]
    a0 = max(0, int(round((t0 - 2 * GRID_LEAD_S) * sr)))     # our windows are read as the profile's: with
    a1 = int(round((t0 + n * bar_s + GRID_LEAD_S) * sr))     # the 40 ms before each line and 20 ms after
    rel = t0 - a0 / sr
    pm = psec["mix"]
    c = compare(a, {"bands_db": pm["bands_db"], "side_minus_mid_db": {},
                    "integrated_lufs": pm["integrated_lufs"], "width": pm["width"]})
    vs = {"band_balance_db": c["band_balance_db"], "flagged_bands": [f["band"] for f in c["flagged_bands"]],
          "lufs_diff": c["diff"].get("integrated_lufs"), "width_diff": c["diff"].get("width")}
    notes = list(notes)
    master_bars = _bar_levels(to_stereo(audio["master"][0][a0:a1]), sr, rel, bar_s, 0, n)
    is_ref = not partial and _same_shape(a, pm, master_bars, pm["per_bar_db"])
    if is_ref:
        notes.append("the capture is the reference: check solos")
    rng = slice(fb, fb + n)

    # the reference side: presence from the paired bars' cores against the stems' sum
    sum_lv = _mean_db(psec["stems_sum"]["core_bar_db"][rng])
    ref_lv, ref_kind, ref_width = {}, {}, {}
    for role, e in psec["roles"].items():
        if role == "master":
            continue
        bars = e["per_bar_db"][rng]
        ref_kind[role] = _presence(e["core_bar_db"][rng], sum_lv)
        ref_lv[role] = None if ref_kind[role] == "absent" else (_mean_db(bars) if partial else e["level_db"])
        ref_width[role] = e.get("width")
    # our side: the same, against our element sources together
    ours, core, raw_lv, ours_kind = {}, {}, {}, {}
    for role in audio:
        if role == "master":                     # sits after the master chain: never ratioed against elements
            continue
        y = to_stereo(audio[role][0][a0:a1])
        ours[role] = _bar_levels(y, sr, rel, bar_s, 0, n)
        core[role] = _bar_levels(y, sr, rel, bar_s, 0, n, core=True)
        raw_lv[role] = y[_line_start(y, sr, rel):_line_start(y, sr, rel + n * bar_s)]
    ours_power = [sum(0.0 if v[j] is None else 10 ** (v[j] / 10) for v in core.values()) for j in range(n)]
    ours_sum = _mean_db([round(to_db(p), 2) if p > 0 else None for p in ours_power])
    ours_lv, ours_width = {}, {}
    for role, seg in raw_lv.items():
        ours_kind[role] = _presence(core[role], ours_sum)
        if ours_kind[role] == "absent":
            ours_lv[role], ours_width[role] = None, None
        else:
            ours_lv[role], ours_width[role] = _level_db(seg), _bands_width(seg, sr)[1]

    def kick_under(lv):
        """dB the kick sits under the loudest other role in `lv`; None when it is silent, -inf when alone."""
        if lv["kick"] is None:
            return None
        others = [v for r, v in lv.items() if r != "kick" and v is not None]
        return max(others) - lv["kick"] if others else -math.inf

    def kick_out(lv):
        under = kick_under(lv)
        return under is None or under > KICK_OUT_DB

    yard, kick_is_out, kick_notes = None, False, []
    if "kick" not in ours_lv:
        notes.append("no kick source in the manifest: elements against the kick left out")
    elif "kick" not in ref_lv:
        notes.append("no kick in the profile: elements against the kick left out")
    elif not (kick_out(ref_lv) or kick_out(ours_lv)):
        yard = "kick"
    else:
        kick_is_out = True                        # decided for both sides together: one yardstick for both
        r_out, o_out = kick_out(ref_lv), kick_out(ours_lv)
        where = " and ".join(w for w, o in (("the reference", r_out), ("ours", o_out)) if o)
        both = sorted(r for r in ref_lv if r != "kick" and ref_kind[r] == "present" and ours_kind.get(r) == "present")
        if both:
            yard = max(both, key=lambda r: ref_lv[r])
            notes.append("the kick is out in %s (absent, or more than %g dB under the loudest other role): "
                         "elements against the %s, the loudest role in both"
                         % (where, KICK_OUT_DB, yard))
            if r_out != o_out:                    # out on one side only: that is the biggest difference there is
                out_lv, in_lv = (ref_lv, ours_lv) if r_out else (ours_lv, ref_lv)
                under = kick_under(out_lv)
                kick_notes.append({
                    "side": "ours" if r_out else "reference",
                    "vs_yardstick_db": None if in_lv.get("kick") is None else round(in_lv["kick"] - in_lv[yard], 2),
                    "reason": ("kick out in %s: silent" % ("the reference" if r_out else "ours")) if under is None else
                              "kick out in %s: %.0f dB under the loudest other role" % (
                                  "the reference" if r_out else "ours", under)})
        else:
            notes.append("the kick is out in %s and no other element sounds in both: elements left out" % where)
    table, ref_only, ours_only, traces, arc, silent_both = [], [], [], [], {}, []
    skip = {yard} | ({"kick"} if kick_is_out else set())
    for role in sorted((set(ref_lv) | set(ours_lv)) - skip):
        r_lv, o_lv = ref_lv.get(role), ours_lv.get(role)
        r_ratio = round(r_lv - ref_lv[yard], 2) if yard and r_lv is not None else None
        o_ratio = round(o_lv - ours_lv[yard], 2) if yard and o_lv is not None else None
        rk, ok = ref_kind.get(role), ours_kind.get(role)
        if "present" in (rk, ok) and rk in ("present", "trace") and ok in ("present", "trace"):
            if yard:
                table.append({"role": role, "reference_db": r_ratio, "ours_db": o_ratio,
                              "difference_db": round(o_ratio - r_ratio, 2),
                              "reference_width": ref_width.get(role), "ours_width": ours_width.get(role),
                              "trace_side": None if rk == ok else ("reference" if rk == "trace" else "ours")})
                if n > CHANGE_BLOCK_BARS:
                    arc[role] = {"reference": _arc(psec["roles"][role]["per_bar_db"][rng],
                                                   psec["roles"][yard]["per_bar_db"][rng]),
                                 "ours": _arc(ours[role], ours[yard])}
        elif rk == "present":
            ref_only.append({"role": role, "vs_yardstick_db": r_ratio,
                             "reason": "not in the manifest" if ok is None else "silent in ours"})
        elif ok == "present":
            ours_only.append({"role": role, "vs_yardstick_db": o_ratio,
                              "reason": "not in the profile" if rk is None else "silent in the reference"})
        elif "trace" in (rk, ok):
            traces.append({"role": role, "reference": rk or "not in the profile", "ours": ok or "not in the manifest",
                           "reference_db": r_ratio, "ours_db": o_ratio})
        else:
            silent_both.append(role)
    for k in kick_notes:
        (ours_only if k["side"] == "ours" else ref_only).append(
            {"role": "kick", "vs_yardstick_db": k["vs_yardstick_db"], "reason": k["reason"]})
    gaps = sorted(({"role": t["role"], "difference_db": t["difference_db"]} for t in table
                   if abs(t["difference_db"]) > ELEMENT_GAP_DB), key=lambda g: -abs(g["difference_db"]))
    width_gaps = sorted(({"role": t["role"], "reference": t["reference_width"], "ours": t["ours_width"],
                          "difference": round(t["ours_width"] - t["reference_width"], 3)} for t in table
                         if t["reference_width"] is not None and t["ours_width"] is not None
                         and abs(t["ours_width"] - t["reference_width"]) > WIDTH_GAP),
                        key=lambda g: -abs(g["difference"]))
    el = {"paired": True, "paired_with": {"section": psec["name"], "from_bar": fb, "bars": n},
          "notes": notes, "capture_is_reference": bool(is_ref), "yardstick": yard,
          "kick_out": kick_is_out,
          "table": table if yard else None, "gaps": gaps if yard else [], "width_gaps": width_gaps if yard and not partial else [],
          "reference_only": ref_only, "ours_only": ours_only, "traces": traces, "silent_in_both": silent_both,
          "arc": arc if yard else {},
          "presence": {r: {"reference": ref_kind.get(r), "ours": ours_kind.get(r)} for r in set(ref_kind) | set(ours_kind)}}
    return vs, el


def _read_span(items: list, rate: int, start_s: float, end_s: float) -> tuple[np.ndarray, float] | None:
    """(samples [n, 2], start in seconds) of the files `items` (profile inputs), summed, between
    start_s and end_s, read from disk without loading the rest; None when the span is outside them."""
    a, b = max(0, int(round(start_s * rate))), int(round(end_s * rate))
    ys = []
    for it in items:
        total = sf.info(it["path"]).frames
        if min(b, total) <= a:
            return None
        y, _ = sf.read(it["path"], start=a, frames=min(b, total) - a, dtype="float32", always_2d=True)
        ys.append(to_stereo(y))
    m = min(len(y) for y in ys)
    return sum(y[:m] for y in ys), a / rate


def _drum_layer(prof: dict, psec: dict, fb: int, sec: dict, audio: dict, sr: int, t0: float,
                bar_s: float, beats: float, stale: set, presence: dict) -> dict:
    """For each drum role in both the manifest's sources and the profile's stems, and present on
    both sides in the paired bars: the readings of oneshot.compare(ours, reference) on a typical
    clean hit of each side. Each side is cut with one beat of pre-roll and post-roll (clamped to the
    file), each ending GRID_LEAD_S + NEAR_S short of its beat line, our post-roll NEAR_S longer than the
    reference's (a hit of the same audio read up to NEAR_S apart never has less room on our side); only hits whose onset lies inside the section are candidates, its window starting
    GRID_LEAD_S early like the elements layer's, so the tail of a hit from before the line is never
    one and a hit a few ms early belongs to its own section. The reference's hit is picked first; on
    our side the hit within 30 ms of the same place in the section is taken when there is one. When our
    segmentation offers none there, the hit is looked for at that place anyway (oneshot.hit_cut) and taken
    only if it rises out of what comes before it within 30 ms of the place, with a note; otherwise our
    own typical hit, with a note that the hits are not at the same place and the readings compare
    different hits. The two hits are then measured again on cuts made from the hits themselves
    (oneshot.hit_cut: onset found again, the cut as long as the reference hit's room to its next onset,
    or both cut to the shorter when the reference's hit is the last in its slice), so the slice a hit
    was picked in does not change its reading; `ours_at_ms` and `reference_at_ms` are those onsets. The
    reference stems are read again from the profile's paths, only the span needed; a role whose
    stems changed or are missing since the profile is skipped. Any failure on one role is that
    role's note."""
    from . import oneshot
    n = sec["bars"]
    pbar = float(prof["beats_per_bar"]) * 60.0 / float(prof["tempo"])
    r0 = prof["t0"] + (psec["start_bar"] + fb) * pbar
    rrate = prof["sample_rate"]
    out = []
    for role in DRUM_ROLES:
        if role not in audio or role not in prof["inputs"]["stems"]:
            continue
        entry = {"role": role, "ours_hit": None, "reference_hit": None, "readings": [], "notes": []}
        out.append(entry)
        try:
            if role in stale:
                entry["notes"].append("a stem of this role changed or is missing since the profile was built: skipped")
                continue
            pr = presence.get(role, {})
            if pr.get("reference") != "present" or pr.get("ours") != "present":
                entry["notes"].append("not present on both sides in these bars (reference %s, ours %s): skipped"
                                      % (pr.get("reference") or "-", pr.get("ours") or "-"))
                continue
            # a beat of pre-roll and post-roll, each ending EDGE_S short of the beat line, where hits that start a
            # few ms early sit; ours runs NEAR_S longer than the reference's, so a hit of the same audio read up to
            # NEAR_S apart never has less room to fall on our side than on the reference's
            edge = GRID_LEAD_S + oneshot.NEAR_S
            pbeat = pbar / prof["beats_per_bar"]
            got = _read_span(prof["inputs"]["stems"][role], rrate, r0 - pbeat - edge, r0 + n * pbar + pbeat - edge)
            if got is None:
                entry["notes"].append("the reference stem does not reach these bars")
                continue
            ref_x, ref_start = got
            ref_pre = r0 - ref_start
            profs, errs = {}, []
            try:
                profs["reference"] = oneshot.profile_array(
                    ref_x, rrate, hit="clean", within=(ref_pre - GRID_LEAD_S, ref_pre + n * pbar - GRID_LEAD_S))
            except ValueError as e:
                errs.append("the reference: %s" % e)
            a0 = max(0, int(round((t0 - bar_s / beats - edge) * sr)))
            a1 = min(len(audio[role][0]), int(round((t0 + n * bar_s + bar_s / beats - GRID_LEAD_S) * sr)))
            our_x, our_pre = to_stereo(audio[role][0][a0:a1]), t0 - a0 / sr
            near = None
            if "reference" in profs:
                pos = profs["reference"]["onset_ms"] / 1000.0 - ref_pre          # s into the section, reference time
                near = our_pre + pos * bar_s / pbar
            try:
                profs["ours"] = oneshot.profile_array(
                    our_x, sr, hit="clean", within=(our_pre - GRID_LEAD_S, our_pre + n * bar_s - GRID_LEAD_S), near=near)
            except ValueError as e:
                errs.append("ours: %s" % e)

            if "reference" in profs:
                rp_ = profs["reference"]
                ons = rp_["onsets_ms"]
                # the reference slice does not move with the capture's offset, so its hit's length (to the next
                # onset) is the one both cuts use: the same audio gives the same two cuts at any offset
                length = ons[rp_["hit"] + 1] / 1000.0 - rp_["onset_ms"] / 1000.0 - 0.001 if rp_["hit"] + 1 < len(ons) else None
                ref_cut, ref_on = oneshot.hit_cut(ref_x, rrate, rp_["onset_ms"] / 1000.0, length)
                if ref_on is not None:           # positions are reported where the cuts found the onsets
                    pos = ref_on - ref_pre
                same_place = "ours" in profs and abs(profs["ours"]["onset_ms"] / 1000.0 - near) <= oneshot.NEAR_S + 0.0005
                our_cut, our_at = None, None
                if same_place:
                    our_cut, on = oneshot.hit_cut(our_x, sr, profs["ours"]["onset_ms"] / 1000.0, length)
                    our_at = profs["ours"]["onset_ms"] / 1000.0 if on is None else on
                else:
                    # our own segmentation did not offer a complete hit there (its thresholds follow the slice's
                    # loudest hit, and a fast pattern splits differently): look for the hit at that place anyway,
                    # and take it only if it rises out of what comes before it there, not 30 ms or more away
                    cut, on = oneshot.hit_cut(our_x, sr, near, length)
                    win = our_x[max(0, int((near - oneshot.NEAR_S) * sr)):int((near + oneshot.NEAR_S) * sr + 1)]
                    if (on is not None and abs(on - near) <= oneshot.NEAR_S and len(win)
                            and np.abs(win).max() >= np.abs(our_x).max() * 10 ** (-30 / 20)):
                        our_cut, our_at = cut, on
                        errs = [e for e in errs if not e.startswith("ours:")]
                        entry["notes"].append("ours: measured at the reference's place; our own segmentation did "
                                              "not offer a complete hit there")
                    elif "ours" in profs:
                        our_cut, on = oneshot.hit_cut(our_x, sr, profs["ours"]["onset_ms"] / 1000.0, length)
                        our_at = profs["ours"]["onset_ms"] / 1000.0 if on is None else on
                        entry["notes"].append("the hits are not at the same place: the reference's is %.0f ms into "
                                              "the section, ours %.0f ms; the readings compare different hits"
                                              % (pos * 1000, (our_at - our_pre) * 1000))
                if our_cut is not None:          # a cut that reached the end of its slice: both as long as the shorter
                    dur = min(len(ref_cut) / rrate, len(our_cut) / sr)
                    ref_cut, our_cut = ref_cut[:int(round(dur * rrate))], our_cut[:int(round(dur * sr))]
                entry["notes"] += errs
                if our_cut is not None:
                    entry["readings"] = oneshot.compare(oneshot.profile_array(our_cut, sr, hit=0, single=True),
                                                        oneshot.profile_array(ref_cut, rrate, hit=0, single=True))["readings"]
                    entry["ours_hit"] = profs["ours"]["hit"] if "ours" in profs and same_place else None
                    entry["reference_hit"] = profs["reference"]["hit"]
                    entry["ours_at_ms"], entry["reference_at_ms"] = round((our_at - our_pre) * 1000, 1), round(pos * 1000, 1)
            else:
                entry["notes"] += errs
        except Exception as e:                    # one role failing must not take the report down
            entry["readings"] = []
            entry["notes"].append("failed: %s: %s" % (type(e).__name__, e))
    return {"roles": out}


def session_compare(manifest: dict | str | pathlib.Path, reference: dict | str | pathlib.Path,
                    out_dir: str | pathlib.Path | None = None) -> dict:
    """A capture of the Set against a reference profile, section by section: Sideman's mix report
    (sideman.listen: load_capture, mix_layer, write_report) with the element and drum layers added.
    Writes mix_report.json and mix_report.md into out_dir (default: the master file's folder),
    replacing earlier ones there, and returns the report with their paths; the same bytes as
    Sideman's mix_report with that profile.

    `reference` is a reference_profile: the profile dict, its JSON text or the path of its
    .profile.json. An audio file or no reference is refused: that report needs only the Set and is
    Sideman's mix_report. Sections pair BY NAME (the manifest's
    section "A" with the profile's section "A"); a manifest section may carry
    "reference": {"section": <profile section name>, "from_bar": N} (both
    optional) to pair with another section, or N bars into it. The range must
    fit inside the profile's section: a shorter manifest section pairs with its
    first bars (noted; the reference's bands, widths and loudness are then those of
    the whole section: no width gaps are ranked and the capture-is-the-reference check is not made); a
    section with no match, or a longer one, reports the mix layer only, with a note.
    A paired section gets the mix compared with the profile's section (bands,
    width, loudness), an `elements` layer and a `drums` layer.
    `elements`: each source in both the manifest and the profile as a ratio to the
    kick (reference, ours, difference), roles in one side only, the ratio per 4 bars,
    and `gaps` (ratios differing by more than ELEMENT_GAP_DB, largest first); each row
    also carries the element's width on both sides (0 mono .. 1 all side; the reference's
    is the whole section's) and `width_gaps` lists those more than WIDTH_GAP apart, apart
    from the level gaps. Where
    the kick is out on either side (absent, or KICK_OUT_DB under the loudest other
    role) the yardstick is the loudest role present on both sides, named in
    `yardstick` (none: the table is left out with a note). The `master` source is never
    ratioed against the elements: it sits after the master chain. A master capture
    that is the reference mix, level offset aside (per-bar curve and bands within 0.3 dB,
    width within 0.01), is flagged ("the capture is the reference: check solos"). Without a `kick` source (or kick stem) the table is left
    out with a note; the mix layer still reports.
    `drums`: for each of DRUM_ROLES in both the manifest's sources and the profile's
    stems and present on both sides in the paired bars, the `oneshot.compare` readings
    between a typical clean hit of ours and of the reference stem. Each side is cut
    with a beat of pre-roll and post-roll; only hits that start inside the section
    count; the reference's hit is picked first and ours is the one at the same place
    (within 30 ms) when there is one. The reference stems are read again from the
    profile's paths, only the span needed; a role whose stem changed since the profile
    is skipped, with a note; a side with no clean hit gives a note and no readings.
    `reference_stale` lists profile inputs
    that changed since it was built."""
    cap = load_capture(manifest)
    prof = _load_profile(reference)
    if prof is None:
        raise ValueError("session_compare needs a reference profile (from reference_profile: the path of its "
                         ".profile.json, its JSON text or the profile itself); for an audio-file reference or none, "
                         "use Sideman's mix_report")
    stale = profile_stale(prof)
    report, spans = mix_layer(cap)
    stale_roles = {x["role"] for x in stale}
    m, audio, sr, bar_s = cap["manifest"], cap["audio"], cap["sr"], cap["bar_s"]
    for row, span in zip(report["sections"], spans):
        sec, t0 = span["section"], span["t0"]
        psec, fb, pnotes = _pair(prof, sec, row["name"])
        if abs(float(prof["tempo"]) - float(m["tempo"])) > 0.01:
            pnotes = pnotes + ["tempo differs: the manifest says %g BPM, the profile %g: bars are compared "
                               "bar for bar, not second for second" % (m["tempo"], prof["tempo"])]
        if psec is None:
            row["elements"] = {"paired": False, "notes": pnotes}
        else:
            row["vs_reference"], row["elements"] = _element_layer(prof, psec, fb, pnotes, sec, span["analysis"],
                                                                  audio, sr, t0, bar_s)
            row["drums"] = _drum_layer(prof, psec, fb, sec, audio, sr, t0, bar_s,
                                       float(m.get("beats_per_bar", 4)), stale_roles, row["elements"]["presence"])
    report["reference"] = prof.get("json_path") or prof["mix"]
    report["reference_kind"] = "profile"
    report["reference_stale"] = stale
    return write_report(report, cap, out_dir, intro=_intro_markdown(report),
                        layers=_elements_markdown(report) + _drums_markdown(report))


def _fmt(v, nd=1):
    return "—" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def _elements_markdown(r: dict) -> list:
    lines = ["", "## Elements against the kick", "",
             "Each element's mean power minus the kick's over the same bars (dB; negative = quieter than the "
             "kick). A gap is a difference of more than %.1f dB. Where the kick is out (absent, or more than "
             "%g dB under the loudest other role, on either side) the section says which element is the "
             "yardstick instead. The master capture is never ratioed against the elements: it sits after "
             "the master chain." % (ELEMENT_GAP_DB, KICK_OUT_DB), ""]
    for s in r["sections"]:
        e = s["elements"]
        lines += ["### %s" % s["name"], ""]
        if e["paired"] and e["kick_out"] and e["yardstick"]:
            lines += ["**Elements against the %s (kick out)**" % e["yardstick"], ""]
        lines += ["- ⚠ %s" % n if "check solos" in n else "- %s" % n for n in e["notes"]]
        if not e["paired"]:
            lines.append("")
            continue
        if e["table"] is not None:
            lines += ["| Element | Reference | Ours | Diff | Width (ref / ours) |", "|---|---|---|---|---|"]
            lines += ["| %s | %s | %s | %s | %s / %s |" % (
                t["role"] + ("" if not t["trace_side"] else " (trace in %s)" % t["trace_side"]),
                _fmt(t["reference_db"]), _fmt(t["ours_db"]), "%+.1f" % t["difference_db"],
                _fmt(t["reference_width"], 2), _fmt(t["ours_width"], 2)) for t in e["table"]]
            lines.append("")
            lines.append("Gaps, largest first: " + ("; ".join("%s %+.1f dB" % (g["role"], g["difference_db"])
                                                             for g in e["gaps"]) or "none") + ".")
            lines.append("Width gaps (more than %g apart): " % WIDTH_GAP + ("; ".join(
                "%s %s -> %s" % (g["role"], _fmt(g["reference"], 2), _fmt(g["ours"], 2)) for g in e["width_gaps"])
                or "none") + ".")
        if e["traces"]:
            lines.append("Traces (%g to %g dB under the elements, and not present on the other side: never a gap): "
                         % (TRACE_DB, ABSENT_DB) + "; ".join(
                "%s (reference %s%s, ours %s%s)" % (
                    t["role"], t["reference"], "" if t["reference_db"] is None else " %.1f dB" % t["reference_db"],
                    t["ours"], "" if t["ours_db"] is None else " %.1f dB" % t["ours_db"]) for t in e["traces"]) + ".")
        for key, label in (("reference_only", "In the reference only"), ("ours_only", "In ours only")):
            if e[key]:
                lines.append("%s: " % label + "; ".join(
                    "%s (%s%s)" % (x["role"], x["reason"], "" if x["vs_yardstick_db"] is None
                                   else ", %+.1f dB vs %s" % (x["vs_yardstick_db"], e["yardstick"]))
                    for x in e[key]) + ".")
        if e["arc"]:
            lines += ["", "Per 4 bars (reference / ours):"]
            for role, a in e["arc"].items():
                lines.append("- %s: " % role + " | ".join("%s / %s" % (_fmt(x), _fmt(y))
                                                         for x, y in zip(a["reference"], a["ours"])))
        lines.append("")
    return lines


def _drums_markdown(r: dict) -> list:
    lines = ["", "## Drum sounds", "",
             "Our clean hit of each drum role against the reference's, over the section's bars "
             "(oneshot readings: a is ours, b the reference; none = nothing past the thresholds).", ""]
    for s in r["sections"]:
        if "drums" not in s:
            continue
        lines += ["### %s" % s["name"], ""]
        if not s["drums"]["roles"]:
            lines += ["- no drum role is in both the manifest and the profile", ""]
        for d in s["drums"]["roles"]:
            body = "; ".join(d["readings"]) or ("none" if not d["notes"] else "")
            lines.append("- %s: %s" % (d["role"], "; ".join(x for x in [body] + d["notes"] if x)))
        lines.append("")
    return lines


def _intro_markdown(r: dict) -> list:
    lines = ["Reference profile: `%s`. Sections are paired with the profile's by name; band balance "
             "is relative to the matching section after removing the overall level offset, and bands "
             "more than 3 dB away are listed." % r["reference"], ""]
    lines += ["- ⚠ %s `%s` %s since the profile was built" % (s["role"], s["path"], s["reason"])
              for s in r["reference_stale"]]
    lines += [""] if r["reference_stale"] else []
    return lines
