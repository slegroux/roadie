# Design notes and measurements

Why each default is what it is, with the measurement behind it. The calibration material was deep/melodic house, so that is where the defaults are known to hold; each measurement is written down so it can be re-derived for other material rather than trusted.

## MCP server details

The first three have the names, arguments and output of Sideman's tools of the
same name; `session_compare` is what Sideman's `mix_report` returns when its
`reference` is a profile, and refuses an audio file or no reference (that is
Sideman's `mix_report`). The tools' docstrings are the full documentation; the
model reads them. A `source` is a file path or an audio clip's LOM path (`live_set
tracks 3 clip_slots 0 clip`): its `file_path` is read through
`sideman.client.Live`, the only thing sent to Live. Everything else runs locally
on the files.

Roadie builds on Sideman here too: `roadie_analysis/` imports the measurement
core from `sideman.listen` (and `session_compare` takes the plain report from it
before adding its layers); nothing in Sideman imports Roadie. The server needs
the `mcp` package, which `install.sh` installs from `requirements-mcp.txt`
(every package pinned); without it `roadie mcp` exits with that hint.

## The `.als` format

The `.als` format is gzipped XML. The track and clip layout match a project Live
12.2.7 saved itself. Live also stores each sample's size and a 16-bit checksum
(`OriginalCrc`, CRC-16 of the first 16 KiB) and reports the file missing if they
disagree; `roadie als` computes both, a rule that reproduces every reference in a
project Live saved. Clip colours are palette indices copied from that project, so
treat them as approximate. Samples are referenced by a path relative to the
`.als` (the way Live 12 does for files outside a project), so keep the `.als`
next to its `_stems` folder.

## Sideman: why, and how it is called

**Why Sideman rather than AbletonMCP.** Sideman exposes the Live Object Model
generically — every call is a `get`/`set`/`call`/`count` against a path like
`live_set tracks 3 arrangement_clips 0` — so what the [AbletonMCP patches](ABLETONMCP-PATCHES.md)
add is reachable without patching:

| | |
|---|---|
| `create_audio_track` | native — patch 0001 unnecessary |
| locators (`set_or_delete_cue`, cue naming) | native — patch 0003's fixes done client-side |
| warp markers | native `warp_markers_set` — patch 0002 unnecessary |
| track and clip colour | native |
| clip deletion | unreachable, and unnecessary — creating a clip at the same position replaces it |

**Few requests, few undo steps.** A request to Live costs about 400 ms whatever it
carries (it waits on Live's main thread), so `roadie load` and `roadie scenes` send
reads that do not depend on each other in one `get_batch` and writes in one
`transaction`. A ten-stem load with `--session` into a fresh Set is 38 requests
instead of 439 (measured on 10 stems of 92 s: 185 s before, 38.6 s after), and
each transaction is one Cmd-Z on Live 12.4+ (about 23 to undo
a load instead of about 280). Anything that depends on an earlier step (a clip
addressed after the request that made it, a read-back after its write, the play-head
moves for locators) still waits for a later request: `live_batch.py` lists them.
**Don't touch the Set while a load or `roadie scenes` runs**: each batch is one undo step,
and on Live 12.4+ a knob you move while one lands can be swept into it. A batch Live does
not answer in time ends the run, since Live may still be applying it.

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
Scripts/`, not the Preferences path (trap 3 in [ABLETONMCP-PATCHES.md](ABLETONMCP-PATCHES.md)). Updating the wrong one looks
exactly like a failed reload.

### Don't use `activate` with a uv venv

`uv venv` does not install pip. After `source .venv/bin/activate`, `python`
points at the venv but **`pip` falls through to whatever pip is next on PATH** —
here, miniconda's, which silently installed 16 packages into the conda base
environment. Always:

```bash
uv pip install --python .venv/bin/python <pkgs>
```

## Model: BS-Roformer-SW, scored against true stems

`roadie split -s` separates with **BS-Roformer-SW** (a six-stem BS-RoFormer, through
`audio-separator`) and merges its `other`, `piano` and `guitar` into `synths.wav`
exactly as `htdemucs_6s` was merged. It replaced `htdemucs_6s` on a measurement with
ground truth: Rodriguez Jr.'s "Synthwave #1" remix pack (30 stems, summed to a mix,
124 BPM) separated by both, each stem scored against its true role. SI-SDR in dB,
higher is better:

| stem | BS-Roformer-SW | htdemucs_6s | change |
|---|---|---|---|
| kick | 19.3 | 17.0 | +2.3 |
| snare | 7.1 | 6.4 | +0.7 |
| hats | 1.6 | 1.8 | −0.2 (even) |
| bass | 5.7 | 3.7 | +2.0 |
| synths | −0.4 | −1.7 | +1.3 |
| drums | 18.8 | 16.4 | +2.4 |

BS-Roformer-SW wins 86–99 % of the bars where a stem is active on kick, snare, bass
and synths, and the bass energy left in the synths stem falls from 29.9 % to 15.0 %.
**That is one track**, a remix pack with no master bus, so easier than a rip: read the
margins as a direction. The scores come from `benchmark.py --truth` with
`bench/synthwave1.json`; re-run it, with a truth file of your own, on your own
material before trusting the size of them. Its other candidates (Mel-Band RoFormer, a
soft-mask re-mask, a residual track, cross-stem gating) did not help. (`benchmark.py`'s
corpus mode, without `--truth`, still separates with Demucs only, so it needs
`demucs` installed.)

**Cost, on an Apple M3 Pro (both models' timings back to back, Live open in the
background)**: a 5:43 mix separates in 2:25 against 0:34 for `htdemucs_6s`; the drum
pass (3:00) is unchanged, so the whole `roadie split -s` takes 5:26 against 3:35
(+51 %). It runs on the GPU (the log says `setting Torch device to MPS`), peaks at
about 2.3 GB resident (demucs: 2.7 GB), and scales linearly: an 11:25 mix took 10:32
(separation 4:31, drum pass 5:47, 3.3 GB resident, no CPU fallback, no swap growth).

The stems are real **24-bit** FLAC, where demucs's were 16-bit. That needs
`--use_soundfile`: audio-separator's default writer converts every stem to 16 bits and
only then labels the file 24-bit (checked: the low 8 bits of every sample were zero).
Measured on a 60 s excerpt of the benchmark mix, the whole `-s` run is 52 MB against
43 MB with the default writer: bass, drums and vocals go from 6.5 to 15.2 MB, the
float32 `synths.wav` is 21.2 MB either way, and the drum parts (still written by the
default writer) are 15.8 MB. That is 1.2× the disk overall and 2.3× for the three FLAC
stems, about 470 MB for a 9-minute set. `-m htdemucs_6s` (or `htdemucs_ft`,
`mdx_extra`, `htdemucs`) selects Demucs again.

`--normalization 1.0` keeps the stems adding up to the mix. audio-separator's default
(0.9) scales the input down to a 0.9 peak when it is louder, and then each stem on its
own by the same rule, so the stems no longer sum to the file you gave it. On the same
excerpt, made 1.9 dB hotter (peak 0.963), the stem sum was 0.61 dB under the input at
the default and 0.02 dB under at 1.0; SI-SDR of the sum against the input is 32.2 dB
either way (it ignores scale), so only the level moves. At 1.0 nothing is amplified
and a stem is scaled only if its own peak passes full scale, so a stem that clips is
still brought down by itself. The input is a WAV we wrote, so its peak is at most 1.0
and it is never scaled. The drum pass keeps audio-separator's default writer and
normalization, as it was measured.

**Re-running with another model.** The stems folder holds a `.stem-model` file naming
the model that wrote it. `-m` with a different model re-separates, redoes the drum
parts, and prints `stems in <dir> are from <X>; re-separating with <Y>`; a folder from
before the marker counts as Demucs. The same model skips.

**A silent stem.** audio-separator writes no file for a stem whose peak is under 1e-6
(BS-Roformer-SW's vocals on this mix sit at −116 dB). `bass`, `drums` and `other` are
required; a missing `vocals`, `piano` or `guitar` is replaced by silence of the input's
length, with a warning, so the merge still runs.

**Not tested.** Sets of an hour or more: the longest run was 11:25 at 3.3 GB resident,
and audio-separator suggests `--chunk_duration 600` past an hour but joins the chunks
without a crossfade. The detected tempo of the Synthwave pack is tagged 62 bpm (half
tempo) in both pipelines; that is a `beat_aligner` matter, left alone here.

## Model: `htdemucs_6s` — the previous default, for narrower reasons than first claimed

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

So `htdemucs_6s` was kept as the default because it is **never worse and clearly
better when the bass thins out** — not because published SDR is wrong. `-m`
still picks it.

**The real lesson: never benchmark an audio model on one excerpt.** Use
`benchmark.py --samples 5`.

## The synths merge

`htdemucs_6s` and BS-Roformer-SW also emit `piano` and `guitar`. On synth-based music these measure
as the *same spectral contour* as `other` at lower level — fragmentation, not
separation:

```
other   −28.2  −23.0  −29.2  −42.5  −51.5
piano   −33.2  −31.4  −41.8  −61.2  −78.5   <- same shape, just quieter
```

So all three are summed back into one `synths.wav`. For `htdemucs_6s` the merged
result lands at −19.6 dB versus the plain 4-stem model's `other` at −19.4 dB — i.e.
recombining recovers exactly what the simpler model gave you (the 6-stem model was
kept purely for its bass).

Written **float32 WAV**, deliberately: the three inputs sum past full scale
(`htdemucs_6s`: measured true peak 1.0018, 0.016 dB over) and FLAC has no float format, so any
integer format would clip. float32 keeps the sum exact, so
`synths + bass + drums + vocals` still reconstructs the mix. It is the one WAV
among the FLAC stems, on purpose.

## `--overlap`: demucs's 0.25 default — 0.5 was tried and did not earn its place

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

## Output format: FLAC stems

The source is converted to a 44.1 kHz / **24-bit** WAV working copy. With the
default model, `audio-separator --use_soundfile` writes 24-bit FLAC (it follows the
input's depth; without the flag it rounds to 16 bits first).
With demucs, run with `--flac` and without `--int24`, the stems are **16-bit** FLAC
(`soundfile` reports `PCM_16`), which is what the rest of this section is about.
`synths.wav` is the float32 exception above.

That is a choice, not an accident. On the path this was built on, the source is a
lossy stream — a ~133 kbps Opus transcode off YouTube — whose noise floor sits far
above 16-bit's ~96 dB of range, so 24-bit stems would store more bits of the same
information. FLAC then cuts the output by ~57% against WAV (a 9-minute bass stem:
149 MB → 64 MB; a whole set ~1.6 GB → ~700 MB), and Live imports `.flac` from an
absolute path exactly like `.wav`.

For a **lossless local source** that argument is weaker. demucs's own code passes
24 bits to its FLAC writer when `--int24` is given alongside `--flac`; that
combination is not exposed or measured here.

## BPM: from the fitted beat grid, kick first

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

**The tempo is checked against the kick stem.** The beat detector can settle a
little off: on one 9-minute track it said 121.12 where the kick, hats, snare and
bass all lock to 120.00, so the set opened in Live at the wrong tempo and the
grid ran two bars out by the end. After fitting the grid, `beat_aligner` asks the
kick stem (`drums_kick`, else `drums`) which tempo within 3 BPM of the detected
one it really locks to, scoring each half of the track on its own. When both
halves agree, the measured tempo is more than 0.3 BPM from the detected one and
it scores at least 1.5x better, the grid is refitted at the measured tempo and
the change is said on stderr; the result's `tempo_check` holds both numbers.
`--tempo` is never second-guessed, and `--no-tempo-check` (on `beat_aligner`,
`roadie load` and `roadie als`) keeps the detector's answer. A tempo in a file
name is a label, not evidence: on five reference tracks four were 2 BPM above
what the audio locks to.

## Warping: only when the tempo actually drifts

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

## Section locators (`--locators`)

`roadie load --locators` drops a Live locator at each detected section start:

```
Intro   bar   9      Build    bar  17      Drop 1   bar  25
Main 1  bar  41      Break    bar  73      Quiet    bar  81
Drop 2  bar  89      Main 2   bar 137      Outro    bar 153
```

That is with the default `--start-bar 9`: the clips start at bar 9, and every
section starts a whole number of 8-bar phrases after it.

**Names carry the length and what changed.** With the stem-activity method each
locator reads `Drop 1 (bass in) · 64`: the label, up to two stems that came or went
against the section before (`kick out`, `bass in, hats out`), and the section's
length in bars. The comment and the length are left off when the section does not
carry what they come from, so the spectral method gives `Drop 1 · 64`.

Existing locators are cleared first; `--keep-locators` adds to them instead.
`roadie sections <stems_dir>` runs the detection standalone (`--json` available,
`--phrase-bars` to change the snap length). Like `roadie scenes`, it snaps to the
grid and downbeat cached in `alignment.json` when there is one, and otherwise
says that bar 1 is assumed at the first grid beat.

**Two detectors, chosen by `--method` (default `auto`).** With the separated stems
on disk (`drums_kick`, or the `drums` composite when there are no drum parts, plus `bass`) it reads
**per-bar stem activity**: each stem's level per bar, relative to that stem's own loud
level, with a hysteresis threshold for "playing". Boundaries are where the set of
playing stems changes (kick, bass, hats, synths; vocals are reported, not cut on). A
**muffled breakdown kick** (about 22 dB under its loud level) counts as no kick. The
8-bar phrase cycle is **estimated from where the changes land** instead of assumed
to start at the detected downbeat, and a boundary is snapped to it only when it is
within a bar. A stem that is effectively silent (under -50 dB, or audible in fewer than 4 bars, like an empty ride) is ignored. Names are rules over which stems play: no kick is a Breakdown, kick and
bass returning is a Drop, a kickless section with the hats back just before that is a
Build, and the last section is an Outro once the kick or bass has gone, or when it is a
stripped-down copy of the one before (a track that ends on the full groove has none). Without those stems, or with `--method spectral`, it uses the spectral method
below. Checked by ear on one track (Gerlach, "Off"): quiet at Live bar 69, drop at 101,
where the spectral method placed them four bars late.

`roadie sections` prints a **Live** bar column next to the file's own bar count. Use
the Live one: it is the bar in Live's ruler with the clips starting at `--start-bar`
(default 9, as `roadie load` and `roadie als` place them), the lead-in trim included.

**Spectral method: boundaries are measured; names are rules.** Boundaries come from beat-synchronous
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
`patches/0003-cue_points.patch` found (see [ABLETONMCP-PATCHES.md](ABLETONMCP-PATCHES.md)).

## Downbeat detectors — `--detector`

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
to 372 MB. madmom needs none of it, and the separation CLIs `roadie split` uses are
separate `uv tool` installs this does not touch.

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

## Why not `allin1`

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

## Gain staging — `--headroom` (default 6 dB)

Every stem track is pulled to **−6 dB** on import. Measured on the reference
track, with the `drums` composite muted as the Set has it:

| | |
|---|---|
| the nine stems summed | **+0.15 dBFS — clips** |
| samples over full scale | 10 (0.0001%) |
| the source YouTube master | **exactly 0.00 dBFS** |

The clipping itself is trivial — ten samples, inaudible, and it is separation
residue rather than a mistake, since separated stems (demucs's, here) do not sum bit-exactly back to
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

## Arrangement layout

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

**The drum pass is slow**: ~1.9× realtime on an Apple GPU (MPS; the stems
themselves run at ~2× on BS-Roformer-SW, ~10× on demucs), so a 9-minute set costs
~4.7 minutes for the drum pass alone. It is on by default anyway: a study
wants the parts, and the beat grid is fitted to `drums_kick` first, so a split
without it gets a looser tempo. `-D` (`--no-drums`) stops at the four stems; a
later `-s` run on the same source adds only the drum pass.

**Live must already be running** with Sideman active (`roadie open` launches it
for you). Sideman's socket (127.0.0.1:9878, reached through `sideman.client`)
lives inside Live; there is no headless mode, and the port is 9878 (only
`roadie scenes` takes `--port`). The socket answers identically whatever handlers
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

## Packaging notes

Not yet a distributable package. What is and isn't there:

- `./install.sh` builds `.venv` from pinned `requirements.txt` and links the
  commands; the separation tools (`audio-separator`, `yt-dlp`, and `demucs` if you
  want the old models) stay separate `uv tool` installs.
- The Live side is Sideman, installed on its own; nothing here patches Live.
  Only its first install needs a Live restart.
- `stems2live.py` needs only the stdlib to talk to Live, but imports
  `beat_aligner.py` for the grid, which needs the venv.
- The BPM gate and the model choice are calibrated on a single genre. Both are
  documented here with their measurements so they can be re-derived rather than
  trusted.
