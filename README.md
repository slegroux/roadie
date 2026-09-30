# Roadie

*Formerly yt2live.*

**From finished track to Live set.** Roadie is stem separation for Ableton
Live 12. Give it a recording of your own track or set, as a local file (or, as one
more input, the URL of your own YouTube upload); it separates the audio into stems
and builds the arrangement in your running Live session: one track per stem, on
the bar line, ordered by frequency, colour-coded, levelled, with a locator per
section. The stems are plain audio files any DAW can open; everything past the
split is Ableton-only.

Every default here was measured on real material, not taken from a benchmark
leaderboard; where the two disagree, the measurement is documented alongside the
setting. The calibration material was deep/melodic house, so that is where the
defaults are known to hold. Each measurement is written down below so it can be
re-derived for other material rather than trusted.

```
roadie split my-set.wav -o <dir> -d     # audio -> stems -> drum parts
roadie load <dir>/<name>_stems          # -> Ableton, tempo measured from the audio
```

> **Use it on recordings you have the rights to.** A DJ set contains other
> artists' music: stems you extract are for your own practice and analysis, not
> for release. Downloading from YouTube is subject to YouTube's Terms of Service.
> For your own uploads, YouTube Studio's download gives you the original file;
> pass it to `roadie split` as a local file. `roadie split <url>` also accepts a
> URL and runs `yt-dlp` on it, and what you point it at is your responsibility.

---

## Commands

| command | what it does | script |
|---|---|---|
| `roadie split <file\|url>` | audio → 44.1 kHz WAV → stems (→ drum parts with `-d`) | `yt2stems.sh` |
| `roadie load <stems_dir>` | stems → the Live set that is open now | `stems2live.py` |
| `roadie open <stems_dir>` | stems → a **new** Live project (copies a template `.als`, opens it, loads) | `stems2als.sh` |
| `roadie sections <stems_dir>` | detect song sections, print locator positions | `sections.py` |
| `roadie scenes <stems_dir>` | one Session View scene per section, one looping clip per stem | `session_view.py` |

`roadie <command> --help` shows that command's own options. Every argument after
the command passes through untouched, except that `split` turns a bare local path
into `--file <path>`. The underlying scripts still work directly, under the same
names without extension (`yt2stems`, `stems2live`, `stems2als`) once installed.

---

## What it produces

From one input:

| File | What it is |
|---|---|
| `<title>_<bpm>bpm.wav` | 44.1 kHz / 24-bit working copy of the source, tempo-tagged |
| `<title>_<bpm>bpm_stems/bass.flac` | 16-bit FLAC — see *Output format* below |
| `…/drums.flac` | composite kit |
| `…/synths.wav` | float32 WAV — see *the synths merge* below |
| `…/vocals.flac` | |
| `…/drums_{kick,snare,toms,hh,ride,crash}.flac` | six-way drum split (`-d`) |
| `…/alignment.json` | cached beat grid, reused by `roadie load` |

All stems are sample-identical in length, so they drop into Live at bar 1 and
lock without warping — unless the set itself drifts in tempo, in which case a warp
map is written. See *Warping: only when the tempo actually drifts*.

`roadie load` then loads them into a running Live set, and can optionally mark
the arrangement:

```bash
roadie load <stems_dir> --locators        # + a locator per detected section
roadie load <stems_dir> --start-bar 1     # butt the audio against bar 1 (default 9)
roadie load <stems_dir> --energy-map      # + a silent MIDI track mapping who plays where
roadie load <stems_dir> --session         # + a Session View grid, same beat grid
```

`roadie open <stems_dir>` does the same into a fresh project: it copies
`~/Music/Ableton/User Library/Templates/empty.als` (`--template` overrides) next
to the stems, opens it in the newest `/Applications/Ableton Live*.app`
(`ABLETON_APP` overrides), waits for the bridge and runs `roadie load` into it.
It cannot save for you — Live's Object Model has no save command — but because
the `.als` already exists on disk, ⌘S saves in place with no dialog.

`roadie scenes <stems_dir>` builds the Session View grid on its own, over tracks
`roadie load` already created (it decorates, it does not create tracks).
`--dry-run` prints the scene/clip plan without writing; `--replace` overwrites an
existing grid. It reuses the beat grid and downbeat `roadie load` cached in the
stems folder's `alignment.json`, so its scenes start on the same bar lines as the
arrangement. With no cache it says so once and assumes bar 1 is the first grid
beat; `--tempo`/`--grid-t0`/`--downbeat` set the grid by hand.

See *Section locators*, *Downbeat detectors* and *Gain staging* below.

---

## Install

```bash
brew install ffmpeg deno aubio uv
uv tool install yt-dlp
uv tool install demucs --with numpy
uv tool install "audio-separator[cpu]"
./install.sh
```

`./install.sh` checks the tools above, builds `.venv` from `requirements.txt`
(Python 3.13, via `uv`) if it does not exist yet, and symlinks `roadie` plus the
per-script names into `~/.local/bin` (`--bin DIR` to change). `./install.sh
--check` reports without changing anything.

Three of those install flags are load-bearing and easy to get wrong:

- **`demucs --with numpy`** — demucs 4.1.0 omits numpy from its metadata. A plain
  install "succeeds", then dies on `import numpy` at runtime.
- **`audio-separator[cpu]`** — the base package doesn't pull `onnxruntime`.
  Worse, `audio-separator --list_models` **exits 0** while printing a
  `ModuleNotFoundError` traceback, so a naive exit-code check reports success.
- **`deno`** — yt-dlp needs a JS runtime or YouTube extraction runs in a
  deprecated mode where some formats silently go missing. Only needed for URLs.

### Ableton side: Sideman

`roadie load`, `roadie open` and `roadie scenes` talk to
**[Sideman](https://github.com/slegroux/sideman)**
([site](https://slegroux.github.io/sideman/)), an Ableton Live remote script
that exposes the Live Object Model over a local socket on port **9878**. Install
it by its own instructions (a `.pkg` from its Releases, or `./scripts/install.sh`
from a clone), then in Live: **Preferences → Link, Tempo & MIDI → Control
Surface = AbletonLOM** (the name Sideman's remote script registers under),
Input/Output = None. Restart Live once if it was running during the install.

Sideman ships `warp_markers_set` natively, so no patch is needed on the Live side.
Verify with:

```bash
./install.sh --check
```

It asks the bridge whether it knows `warp_markers_set`. **Port-open is not
evidence**: the socket answers normally whatever version of the handlers Live has
loaded.

**Why Sideman rather than AbletonMCP.** Sideman exposes the Live Object Model
generically — every call is a `get`/`set`/`call`/`count` against a path like
`live_set tracks 3 arrangement_clips 0` — so what the AbletonMCP patches below
add is reachable without patching:

| | |
|---|---|
| `create_audio_track` | native — patch 0001 unnecessary |
| locators (`set_or_delete_cue`, cue naming) | native — patch 0003's fixes done client-side |
| warp markers | native `warp_markers_set` — patch 0002 unnecessary |
| track and clip colour | native |
| clip deletion | unreachable, and unnecessary — creating a clip at the same position replaces it |

**The decisive difference is reload.** Sideman's `handlers.py` is re-imported on
`{"op": "reload"}` — **no Live restart**. The AbletonMCP path needs a full quit
and reopen for every remote-script change, because Python keeps the module in
`sys.modules` and toggling the Control Surface re-instantiates the class from
already-loaded code. That restart cycle is the single most expensive thing about
bridge work, and Sideman removes it for everything except the initial install.

Nothing warns you when a bridge is running stale code. Besides `--check`, the
bytecode cache tells you: delete `__pycache__` next to the remote script; a real
restart recreates `__init__.cpython-311.pyc` (3.11 is Live 12's interpreter).

Also check **which copy Live loads** — `~/Music/Ableton/User Library/Remote
Scripts/`, not the Preferences path (trap 3 below). Updating the wrong one looks
exactly like a failed reload.

### Don't use `activate` with a uv venv

`uv venv` does not install pip. After `source .venv/bin/activate`, `python`
points at the venv but **`pip` falls through to whatever pip is next on PATH** —
here, miniconda's, which silently installed 16 packages into the conda base
environment. Always:

```bash
uv pip install --python .venv/bin/python <pkgs>
```

---

## Legacy: AbletonMCP patches

Roadie no longer uses these. They are kept because they are the reference for
what Live's API will and will not accept, and because the MCP bridge remains the
only way to DELETE an arrangement clip — needed once here, to clear stale clips
after changing `--start-bar`.

All three are patches against
[uisato/ableton-mcp-extended](https://github.com/uisato/ableton-mcp-extended)
(MIT, © 2025 uisato) at commit `1116449`, and apply in order: `0001`, then
`0002`, then `0003` on top of both. They touch only that project's
`AbletonMCP_Remote_Script/__init__.py`.

<details>
<summary>The AbletonMCP setup these patches were written against</summary>

```bash
git clone https://github.com/uisato/ableton-mcp-extended
cd ableton-mcp-extended
git checkout 1116449
git apply /path/to/roadie/patches/0001-create_audio_track.patch
git apply /path/to/roadie/patches/0002-warp_markers.patch
git apply /path/to/roadie/patches/0003-cue_points.patch

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "mcp[cli]>=1.3.0,<2" \
    "elevenlabs>=0.2.26" "python-dotenv>=1.0.0"

mkdir -p ~/Music/Ableton/"User Library"/"Remote Scripts"/AbletonMCP
cp AbletonMCP_Remote_Script/__init__.py \
   ~/Music/Ableton/"User Library"/"Remote Scripts"/AbletonMCP/
```

Control Surface = AbletonMCP, port 9877. The two bridges coexist by design —
Sideman's source says *"deliberately NOT 9877 — coexists with an existing
AbletonMCP"* — so both can be enabled at once.
</details>

Four traps from the MCP setup, all of which cost real time here:

1. **`pip install -e .` fails.** The repo's `pyproject.toml` declares a package
   `AbletonMCP_UDP` that doesn't exist in the tree. Install the three
   dependencies directly instead and run via `PYTHONPATH`.
2. **Pin `mcp<2`.** mcp 2.0.0 removed `mcp.server.fastmcp`, which the server
   imports. Unpinned resolves to 2.x and the server won't start.
3. **The documented install path is wrong.** The project says
   `~/Library/Preferences/Ableton/Live X.X.X/User Remote Scripts/`. Live 11+
   scans `~/Music/Ableton/User Library/Remote Scripts/`. Installing to the
   documented path means AbletonMCP never appears in the Control Surface list.
4. **Clear `__pycache__` after patching.** Live's embedded Python will load a
   stale `.pyc` compiled from the pre-patch source. Symptom: your patch is
   provably in the file on disk and Live still reports `Unknown command`.

### `patches/0001-create_audio_track.patch`

The MCP server advertises a `create_audio_track` tool, but the remote script has
no handler for it, so the bridge can place audio clips yet cannot create anywhere
to put them. Every stem import required adding tracks by hand first.

The fix needs **two** edits, and this is the part worth knowing if you re-derive
it: adding the handler alone is not enough. There is a second gate — a whitelist
of commands routed to Live's main thread. A command missing from that list never
reaches the dispatch chain and falls through to `Unknown command`, which reads
exactly like a missing handler. Both sites are marked `LOCAL PATCH`.

### `patches/0002-warp_markers.patch`

Adds one command, `set_clip_warp_markers`, taking `track_index`, `clip_index`, a
list of `[beat_time, sample_time]` pairs, and an optional `warp_mode`.

**This one cannot be done from the client side, and that is the whole reason it
exists.** `Clip.add_warp_marker()` wants a C++ `TWarpMarker`, and boost.python
registers a from-Python converter for exactly one Python shape. **Which shape is
undocumented and not guessable.** Measured against Live 12.2.7, all four of these
are rejected with `No registered converter was able to produce a C++ rvalue of
type NApiHelpers::TWarpMarker from this Python object of type X`:

| passed | result |
|---|---|
| `dict` | rejected |
| `list` | rejected |
| `tuple` | rejected |
| `int` | rejected |

So the handler does not assume a shape — `_make_warp_marker` **probes** five
constructions in order (`Live.Clip.WarpMarker` by keyword, the same positionally,
a `namedtuple`, a duck-typed object with `.beat_time`/`.sample_time`, and a plain
tuple), caches the first that Live accepts, and re-probes if a cached one later
fails. If every one fails it raises with all five error strings joined, because a
silent no-op would look like a successful warp and leave the clip subtly out of
time. The accepted shape comes back over the socket as `marker_strategy`.

On Live 12.2.7 the winner is the first candidate,
`Live.Clip.WarpMarker(beat_time=…, sample_time=…)` — but the probe stays, because
that is a measurement of one build, not a documented contract.

Whatever that shape turns out to be, it has to be constructed **inside Live** —
the socket carries JSON, which cannot express a Live API object. That is the part
that makes this irreducibly a remote-script change, and why Sideman carries
`warp_markers_set` as a dedicated op rather than a generic `call`.

### `patches/0003-cue_points.patch`

Fixes `create_cue_point`, which was wrong in three ways — all found by inserting
nine locators and reading them back, none visible from reading the code:

1. **`set_or_delete_cue()` is a toggle.** Asking for a locator where one already
   sits *deletes* it.
2. **Live snaps `current_song_time`.** The old handler then looked the new cue up
   by the *requested* time, never matched, and silently skipped naming — all nine
   locators came back as Live's defaults `"1"`…`"9"`. The fix reads the actual time
   back and identifies the new cue by diffing the cue list.
3. **Cue points cannot be created past `song_length`**, which is why locators are
   written after the clips.

Bulk delete is deliberately a `NotImplementedError`: it cannot work inside one
main-thread task (see *Section locators*), and an endpoint that looks like it works
is worse than one that refuses. `stems2live.py` applies the same three rules
client-side over Sideman.

> An earlier version of this README claimed `add_warp_marker` "accepts a real
> Python tuple and nothing else". That was inferred from seeing dict and list
> rejected, never verified, and is **wrong** — tuple is rejected too. The handler
> built on it compiled, installed and reviewed clean, and would have written zero
> markers. Only an end-to-end write against Live caught it.

Same two-site rule as 0001: the command also has to join the main-thread
whitelist, or clip mutation happens off the main thread and Live no-ops or
crashes.

---

## Why these settings

### Model: `htdemucs_6s` — a safe default, for narrower reasons than first claimed

**Corrected finding.** An earlier version of this README claimed stock `htdemucs`
emits a silent bass stem (−72.4 dB) and that the SDR leaderboard is therefore
inverted on house material. That came from a **single 60-second window**, and it
does not survive proper sampling.

Bass deficit (source 20–120 Hz level minus bass-stem level; lower is better),
5 windows across one 9:23 house set:

| model | 56s | 154s | 251s | 349s | 446s | **median** |
|---|---|---|---|---|---|---|
| htdemucs | 1.0 | 1.8 | 19.4 | 26.9 | 4.2 | **4.2** |
| htdemucs_ft | 1.1 | 2.0 | 14.9 | 13.5 | 3.9 | **3.9** |
| htdemucs_6s | 0.9 | 1.4 | 11.8 | 9.3 | 4.0 | **4.0** |
| mdx_extra | 1.0 | 1.8 | 12.9 | 8.9 | 3.7 | **3.7** |

All four medians sit within **0.5 dB** of each other. Window choice moves the
result by up to **26 dB on the same track** — far more than model choice does.

What *does* replicate, across both overlap settings: in **bass-sparse** passages
(251s, 349s — where the source sub-band drops from −14 dB to −22 dB) the models
diverge and the ordering is stable. `htdemucs` lands 19–29 dB down; `htdemucs_6s`
and `mdx_extra` 9–13 dB. Where the low end is strong, all four are identical.

So `htdemucs_6s` is kept as the default because it is **never worse and clearly
better when the bass thins out** — not because published SDR is wrong. `-m`
picks another model.

**The real lesson: never benchmark an audio model on one excerpt.** Use
`benchmark.py --samples 5`.

### The synths merge

`htdemucs_6s` also emits `piano` and `guitar`. On synth-based music these measure
as the *same spectral contour* as `other` at lower level — fragmentation, not
separation:

```
other   −28.2  −23.0  −29.2  −42.5  −51.5
piano   −33.2  −31.4  −41.8  −61.2  −78.5   <- same shape, just quieter
```

So all three are summed back into one `synths.wav`. The merged result lands at
−19.6 dB versus the plain 4-stem model's `other` at −19.4 dB — i.e. recombining
recovers exactly what the simpler model gave you. `htdemucs_6s` is kept purely for
its bass.

Written **float32 WAV**, deliberately: the three inputs sum past full scale
(measured true peak 1.0018, 0.016 dB over) and FLAC has no float format, so any
integer format would clip. float32 keeps the sum exact, so
`synths + bass + drums + vocals` still reconstructs the mix. It is the one WAV
among the FLAC stems, on purpose.

### `--overlap`: demucs's 0.25 default — 0.5 was tried and did not earn its place

**Also corrected.** 0.5 was once recommended on a reconstruction residual of
−33.7 dB versus −32.5 dB at the 0.25 default. That was one window. Re-run across
5 windows measuring bass deficit, `0.25` and `0.50` are indistinguishable:

| model | median @ 0.25 | median @ 0.50 | Δ |
|---|---|---|---|
| htdemucs | 4.2 | 4.1 | −0.1 |
| htdemucs_ft | 3.9 | 4.0 | +0.1 |
| htdemucs_6s | 4.0 | 4.1 | +0.1 |
| mdx_extra | 3.7 | 3.8 | +0.1 |

Per-window values agree to within 2.2 dB, mostly under 0.5. So `roadie split`
leaves demucs at its own 0.25 default. (`benchmark.py` still defaults to 0.5,
the setting that table was measured against.)

`--shifts` is skipped on the same reasoning: that comparison (shifts=5 scoring
*worse* than shifts=2) was also single-window, so it demonstrates measurement noise
rather than a property of the setting.

### Output format: 16-bit FLAC stems

The source is converted to a 44.1 kHz / **24-bit** WAV working copy. demucs is
then run with `--flac` and without `--int24`, so the stems are **16-bit** FLAC
(`soundfile` reports `PCM_16`); `synths.wav` is the float32 exception above.

That is a choice, not an accident. On the path this was built on, the source is a
lossy stream — a ~133 kbps Opus transcode off YouTube — whose noise floor sits far
above 16-bit's ~96 dB of range, so 24-bit stems would store more bits of the same
information. FLAC then cuts the output by ~57% against WAV (a 9-minute bass stem:
149 MB → 64 MB; a whole set ~1.6 GB → ~700 MB), and Live imports `.flac` from an
absolute path exactly like `.wav`.

For a **lossless local source** that argument is weaker. demucs's own code passes
24 bits to its FLAC writer when `--int24` is given alongside `--flac`; that
combination is not exposed or measured here.

### BPM: from the fitted beat grid, kick first

Stage 4 of `roadie split` runs `beat_aligner.py` on the stems **directory**. It
fits a beat grid to the cleanest percussive stem it finds — `drums_kick`, then
`drums`, then `bass`, then the mix — resolves bar phase off the summed mix, and
caches the result in `alignment.json`, which `roadie load` reads instead of
repeating the analysis. The filename gets that tempo rounded to an integer.

The kick is preferred because no pads or vocals confuse onset detection:
measured on one set, interval IQR was 3.9% (kick) / 6.2% (drums) / 4.6% (mix),
same ~126 BPM from all three.

When the aligner or `.venv` is unavailable, or returns no tempo, it falls back
to `aubiotrack` over `drums_kick` → `drums` → the WAV, gated at interval
IQR ≤ 10%; above that
the filename is left untagged rather than guessing. An earlier gate on the
p10–p90 spread rejected two of three *correct* answers — IQR is the stable
statistic here. **Calibrated on one track**; widen it if a steady set comes back
untagged.

### Warping: only when the tempo actually drifts

A warp map is written **only if drift is detected**. A set that holds one tempo is
left unwarped, which is why the stems still drop in at bar 1 and lock on their own.

The reason is that the grid fit is not exact. Residual jitter of the detected beats
against the fitted grid runs **~35 ms RMS**, which is aubio's placement noise, not
the track moving. Writing a marker at every detected beat would pin the audio to
that noise — you would be warping a steady track onto a jittery grid and making it
worse. So when drift *is* found, markers go at **smoothed per-segment boundaries**,
not at raw beat detections.

**The units are the easy thing to get wrong:**

| field | unit |
|---|---|
| `beat_time` | beats from the **sample start** |
| `sample_time` | **seconds** from the sample start |

`sample_time` is *seconds*, not sample frames — do not multiply by the sample rate.
Nothing in the API name suggests this and nothing errors if you get it wrong; the
map simply lands somewhere absurd.

Requires Sideman's `warp_markers_set` op. A bridge whose handlers predate it
answers `unknown op`; `roadie load` places the clips anyway, unwarped, and ends
with a warning that the set will drift against the grid. Updating the handlers needs **no Live restart** — send
`{"op": "reload"}` to port 9878.

**A nonzero `remove_failed` in the response is normal.** Setting a map means adding
the new markers and then removing the old ones, but a freshly created audio clip
carries a **shadow marker** encoding its detected tempo, and Live refuses to move or
remove it (`The shadow marker can't be moved.`). That refusal is expected, so each
removal is attempted independently and merely counted. Markers whose `beat_time`
collides with one of the new markers are skipped rather than removed — removal
addresses a marker *by beat time*, so removing a colliding old marker would delete
the new one just written at that position.

### Section locators (`--locators`)

`roadie load --locators` drops a Live locator at each detected section start:

```
Intro   bar   9      Build    bar  17      Drop 1   bar  25
Main 1  bar  41      Break    bar  73      Quiet    bar  81
Drop 2  bar  89      Main 2   bar 137      Outro    bar 153
```

That is with the default `--start-bar 9`: the clips start at bar 9, and every
section starts a whole number of 8-bar phrases after it.

Existing locators are cleared first; `--keep-locators` adds to them instead.
`roadie sections <stems_dir>` runs the detection standalone (`--json` available,
`--phrase-bars` to change the snap length). Like `roadie scenes`, it snaps to the
grid and downbeat cached in `alignment.json` when there is one, and otherwise
says that bar 1 is assumed at the first grid beat.

**Boundaries are measured; names are rules.** Boundaries come from beat-synchronous
MFCC + chroma + RMS through `librosa.segment.agglomerative` — classical clustering,
no model. Then every boundary is **snapped to an 8-bar (32-beat) phrase multiple
counted from the detected downbeat**, which is what makes the output musically
usable: unsnapped boundaries land a beat or two off and read as a mistake in the
arrangement.

The names are hand-written rules over energy *and order*: first section is Intro,
last is Outro, loud-after-quiet is a Drop, quiet-after-loud is a Break, and a
mid-energy section is a Build only because a full one follows it — which is why
labelling needs a second pass over the whole list. Nothing here recognises a drop;
`Drop 2` means *energy > 1.12 and the previous section was quieter*. Expect odd
names on tracks that open loud or drop without a preceding break.

**Locators are written after the clips, and that ordering is load-bearing.** A cue
point cannot be created past `song_length`, and `song_length` is whatever the
arrangement currently reaches — so on an empty Set every locator past the end is
rejected with `Cannot set the Songtime behind the Songlength`. The clips are what
extend the song.

**They are also cleared and written one round trip at a time**: set
`current_song_time`, then call `set_or_delete_cue`. Locators can only be toggled
at the play head (`set_or_delete_cue()` takes no position), and Live does not
finish moving the play head within a single main-thread task — so a bulk loop
fires the toggle at the *previous* position, where it deletes an existing cue or
creates a stray one. Measured: names attached one section behind, a stray locator
at the play head, and four of nine silently missing, because the toggle deletes
when it lands on an existing cue. The single-cue rules are the ones
`patches/0003-cue_points.patch` found (see *Legacy*).

### Downbeat detectors — `--detector`

Tempo and *phase* are different problems. Tempo comes from the kick; **phase —
which beat is bar 1 — cannot**, because four-on-the-floor is periodic at beat level
by construction, so every bar position looks identical in a kick stem. Phase is
detected on the summed mix, where bass movement and chord changes live.

| detector | needs | reference track |
|---|---|---|
| `heuristic` (**default**) | nothing | phase 0, conf 0.302, **8.3 s** |
| `madmom` | git-main install, see below | phase 2, conf 0.541, **31.2 s** |

**The higher confidence is on the WRONG answer.** madmom's 0.541 is a vote
share, and the phase it picked is the one the arrangement-boundary check has to
override; the heuristic's 0.302 is attached to the phase that check agrees with.
Both end at the same place because the tiebreaker corrects madmom — but the
heuristic gets there in a quarter of the time and needs no install. madmom stays
worth trying on material the heuristic is not tuned for: it weighs kick and
snare energy, which is a four-on-the-floor assumption.

**`beat_this` was dropped, on purpose.** It used to be listed as a third backend
and was **never once run here**: it needs torch, torch was only ever installed
while evaluating `allin1` (below, which does not work), and removing that 614 MB
took `beat_this`'s only dependency with it. Advertising a backend nobody has
exercised is worse than not offering one — the dispatch degrades to the
heuristic silently, so a broken or absent model looks identical to a working
one. madmom already covers the trained-model case.

To bring it back: install torch plus
`beat-this tqdm einops soxr rotary-embedding-torch`, restore a
`_beat_this_downbeats(y, sr)` returning downbeat times in seconds, and add
`"beat_this"` to `DETECTORS`. Worth doing only if the heuristic and madmom
disagree on material you care about.

**torch is deliberately not in the venv.** Removing it took the venv from 986 MB
to 372 MB. madmom needs none of it, and the demucs CLI `roadie split` uses is a
separate `uv tool` install this does not touch.

The heuristic reads barely above chance on this material and says so in its output.
That is not a bug in the heuristic: kick energy is symmetric between beats 1 and 3,
and a backbeat is symmetric between 2 and 4, so the percussive cues cancel and only
harmonic change is left to break the tie — which is the arrangement-boundary
check's job.

**madmom does install on Python 3.13**, contrary to its reputation. The PyPI release
(0.16.1, 2018) does not — it does `from collections import MutableSequence`, removed
in 3.10 — but git main does:

```bash
uv pip install --python .venv/bin/python "setuptools<81" cython     # 81+ dropped pkg_resources
uv pip install --python .venv/bin/python --no-build-isolation \
    "madmom @ git+https://github.com/CPJKU/madmom"
```

Asking for madmom without it installed **degrades to the heuristic rather than
aborting**. `./install.sh --check` reports whether it is present.

**The detectors can disagree about bar 1 by two beats** — the half-bar ambiguity —
and on the reference track they do. Only one is right, and no confidence number
settles it. Check bar 1 by ear before building an edit on it.

### Why not `allin1`

[All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one) is the
obvious upgrade for section labels — a transformer trained on Harmonix that emits
functional labels (intro/verse/chorus/bridge) rather than energy rules. It does not
work here, and the blocker is not the one you would guess.

`allin1 1.1.0` imports `natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb` from
NATTEN ≤ 0.14, but declares `natten` **unpinned**, so you resolve 0.21.7 where those
symbols no longer exist. Pinning `natten==0.14.6` then fails to compile against
torch 2.13 (the ATen API moved; 20 clang errors). It needs an era-matched
Python + torch + NATTEN together.

Patching `dinat.py` onto the modern NATTEN API was deliberately **not** done — that
is a model's attention implementation, and getting it subtly wrong yields
plausible-looking but wrong segmentation with nothing to catch it. The sound route
is a separate Python 3.9/3.10 venv driving allin1 through its CLI as a subprocess.

### Gain staging — `--headroom` (default 6 dB)

Every stem track is pulled to **−6 dB** on import. Measured on the reference
track, with the `drums` composite muted as the Set has it:

| | |
|---|---|
| the nine stems summed | **+0.15 dBFS — clips** |
| samples over full scale | 10 (0.0001%) |
| the source YouTube master | **exactly 0.00 dBFS** |

The clipping itself is trivial — ten samples, inaudible, and it is separation
residue rather than a mistake, since demucs stems do not sum bit-exactly back to
their source. **The real problem is that the source has no headroom at all.**
A mastered track is brickwalled at 0 dBFS, the stems sum back to roughly that, so
at unity the first EQ boost or compressor you add clips and you spend the session
fighting the master instead of mixing.

Applied per stem track rather than to the master on purpose: the point is that
YOUR faders read near unity while the SUM has room. Pulling the master down
instead leaves every stem at unity and hides the headroom. `--headroom 0` leaves
the faders alone.

**Live's fader is a 0..1 taper, not dB**, and the mapping is undocumented.
Measured by setting values and reading `display_value` back:

| value | dB | value | dB | value | dB |
|---|---|---|---|---|---|
| 1.00 | +6.0 | 0.80 | −2.0 | 0.60 | −10.0 |
| 0.85 | 0.0 | 0.75 | −4.0 | 0.50 | −14.0 |
| 0.92 | +2.8 | 0.70 | **−6.0** | 0.40 | −18.0 |

Dead linear at 40 dB per unit across that span, so `value = 0.85 + dB/40` and
unity is 0.85 (also Live's default). It stops being linear below ~0.4 — 0.30
reads −24.2 where the formula predicts −22 — so the conversion refuses anything
past −18 dB rather than quietly returning a wrong fader.

### Arrangement layout

- **Kick is pinned first**, ahead of the sort. Measured, the kick came in at 66 Hz
  against bass at 68 Hz — a 2 Hz margin that flips on a sub-heavy track.
- Everything else is ordered by **log-frequency spectral centroid**, low at the
  top.
- Clips **and their tracks** are coloured on a **red→blue hue ramp** following
  the same order.
- The composite `drums` clip is **muted automatically** when `drums_*` parts are
  present — the parts sum back to it, so both playing is the kit twice (~+6 dB).
- Writing starts at track 2 (`--first-track`), leaving Live's two default MIDI
  tracks alone.

---

## Known limitations

**No pad / arp / stab separation, and no model in this family will do it.** These
are *arrangement roles*, not sources; no training corpus labels them, so a
separator can only emit the classes it learned. Re-running `other` through
demucs does not help — it asks the network to find vocals/drums/bass in a signal
that is by construction none of those. HPSS (harmonic-percussive separation) on
the merged `synths` stem is the closest available technique. **Untested here.**

**Live's API cannot reorder tracks.** "Ordering" means *which stem is assigned to
which track index*, which is why re-running rewrites every track rather than
shuffling them.

**`ride`/`crash` come out near-silent** on electronic material (−61 / −59 dB).
That is correct — there are no acoustic cymbals — not a failure.

**The drum pass is slow**: ~1.3× realtime versus ~14.5× for demucs, so a 9-minute
set costs ~7 minutes for `-d` alone. Hence opt-in.

**Live must already be running** with Sideman active (`roadie open` launches it
for you). The socket (127.0.0.1:9878) lives inside Live; there is no headless
mode, and the port is 9878 (only `roadie scenes` takes `--port`). The socket answers identically whatever handlers
are loaded, so "the port is open" proves nothing — `./install.sh --check` probes
for `warp_markers_set`, which is the reliable test.

**Which beat is bar 1 is the weakest link.** Tempo is solid — the grid fit is
cross-checked two independent ways — but phase detection is a genuinely hard problem
on four-on-the-floor, and the two detectors disagree by two beats on the reference
track. Confidence is reported on a 0.25-is-chance scale and a weak reading is called
out explicitly rather than hidden. Check bar 1 by ear.

**Section labels are rules, not recognition.** The boundaries are measured; the
names (`Drop`, `Break`, `Build`) are thresholds over energy and order. See *Section
locators*.

**Tempo in the filename is not trustworthy for alignment.** It is rounded to an
integer, and on the reference track a simple median-interval estimate was off by
~1.4% anyway — the true tempo was 125.000 where it said 126.713. `roadie load`
measures the tempo from the audio; do not feed the filename BPM back in as
`--tempo`.

---

## Tests

```bash
.venv/bin/python -m pytest -q
```

The real-stem tests need a split of a reference track, which is not
redistributable and never committed. Point `ROADIE_REF_STEMS` at a stems folder
of your own (or put one at `./reference/`); without it those tests skip.

---

## Packaging notes

Not yet a distributable package. What is and isn't there:

- `./install.sh` builds `.venv` from pinned `requirements.txt` and links the
  commands; the separation tools (`demucs`, `audio-separator`, `yt-dlp`) stay
  separate `uv tool` installs.
- The Live side is Sideman, installed on its own; nothing here patches Live.
  Only its first install needs a Live restart.
- `stems2live.py` needs only the stdlib to talk to Live, but imports
  `beat_aligner.py` for the grid, which needs the venv.
- The BPM gate and the model choice are calibrated on a single genre. Both are
  documented above with their measurements so they can be re-derived rather than
  trusted.
