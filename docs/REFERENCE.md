# Usage reference

The full detail behind the README: every command's behaviour, the output files, configuration and installation. Why the defaults are what they are is in [DESIGN.md](DESIGN.md).

## Commands and the scripts behind them

| command | what it does | script |
|---|---|---|
| `roadie split <file\|url>` | audio → 44.1 kHz WAV → stems with `-s` (BS-Roformer-SW; `-m htdemucs_6s` for Demucs) → drum parts (skip with `-D`) | `yt2stems.sh` |
| `roadie load <stems_dir>` | stems → the Live set that is open now | `stems2live.py` |
| `roadie open <stems_dir>` | stems → a **new** Live project (copies a template `.als`, opens it, loads) | `stems2als.sh` |
| `roadie sections <stems_dir>` | detect song sections, print locator positions | `sections.py` |
| `roadie scenes <stems_dir>` | one Session View scene per section, one looping clip per stem | `session_view.py` |
| `roadie als <stems_dir>` | stems → a Live project **file** (`.als`), written directly; Live need not be running | `als_writer.py` |
| `roadie batch plan\|run\|verify-sources` | unattended overnight batch over many local files, stored on pCloud | `batch.py` |
| `roadie stats` | runs and distinct tracks from the local usage log | — |
| `roadie mcp` | MCP server (stdio) for the analysis tools below; an MCP client starts it | `roadie_mcp.py` |

`roadie <command> --help` shows that command's own options. Every argument after
the command passes through untouched, except that `split` turns a bare local path
into `--file <path>`. The underlying scripts still work directly, under the same
names without extension (`yt2stems`, `stems2live`, `stems2als`) once installed.

## What `roadie split` produces

From one input:

| File | What it is |
|---|---|
| `<title>_<bpm>bpm.wav` | 44.1 kHz / 24-bit working copy of the source, tempo-tagged |
| `<title>_<bpm>bpm_stems/bass.flac` | 16-bit FLAC — see [Output format](DESIGN.md#output-format-flac-stems) |
| `…/drums.flac` | composite kit |
| `…/synths.wav` | float32 WAV — see [the synths merge](DESIGN.md#the-synths-merge) |
| `…/vocals.flac` | |
| `…/drums_{kick,snare,toms,hh,ride,crash}.flac` | six-way drum split (on with `-s`; `-D` skips it) |
| `…/alignment.json` | cached beat grid, reused by `roadie load` |

All stems are sample-identical in length, so they drop into Live at bar 1 and
lock without warping — unless the set itself drifts in tempo, in which case a warp
map is written. See [Warping](DESIGN.md#warping-only-when-the-tempo-actually-drifts).

## `roadie load`, `open` and `scenes`

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

See [Section locators](DESIGN.md#section-locators---locators), [Downbeat detectors](DESIGN.md#downbeat-detectors----detector) and [Gain staging](DESIGN.md#gain-staging----headroom-default-6-db).

## MCP server: analysis against a reference

`roadie mcp` is an MCP server with four tools for the analysis that needs **a
reference track, its stems or samples**. Measuring the Set on its own needs only
Live and the Set's signal, and stays in Sideman's server: `audio_analyze`,
`audio_compare`, and `mix_report` with no reference or an audio file as the
reference. Register both:

```bash
claude mcp add --scope user sideman -- /path/to/sideman/.venv/bin/python -m mcp_server.server
claude mcp add --scope user roadie -- "$PWD/roadie" mcp          # from this clone
```

| tool | what it does |
|---|---|
| `sample_analyze(source, hit=None)` | a drum one-shot in synth terms: envelope, pitch and its drop, harmonics, noise, click, bands, width |
| `sample_compare(a, b, hit_a=None, hit_b=None)` | two hits compared, with one plain `readings` line per difference past its threshold |
| `reference_profile(mix, stems, tempo, t0, sections, beats_per_bar=4, out=None)` | a reference track and its stems measured once per section; writes `<mix>.profile.json`, returns a summary |
| `session_compare(manifest, reference, out_dir=None)` | a capture of the Set against a profile, per section: Sideman's mix report plus each element against the kick and each drum sound against the reference's; writes `mix_report.json` and `mix_report.md` |

More detail in [DESIGN.md](DESIGN.md#mcp-server-details).

## `roadie als`: a project file without Live

`roadie als <stems_dir>` writes `<name>.als` next to the stems folder (`-o` to
choose, `-f` to overwrite), and makes that folder a Live project (an empty
`Ableton Project Info` folder), so ⌘S in Live saves the set in place. It copies the audio tracks of your empty template
(`~/Music/Ableton/User Library/Templates/empty.als`, `--template` overrides), one
per stem, each with a warped clip on the same bar line `roadie load` uses, the
measured tempo, `--headroom` dB of fader, and a locator per detected section
(`--no-locators` skips the detection). The `drums` composite is muted when the
`drums_*` parts are there. `roadie sections` also saves `sections.json` into the
stems folder (`--no-write` skips it).

**Energy lanes.** `roadie als` also draws each stem's energy as automation under
its clip. Every stem track gets an empty Audio Effect Rack named **Energy** (no
chains, so the audio passes through untouched) whose Macro 1 carries the stem's
mean power per bar: the top of the lane is the loudest bar of any stem, the bottom
is 40 dB below it, so a quiet hat line reads as quiet next to the bass. The Main
track gets the same lane for the whole mix, 12 dB deep, because a mastered track
moves much less than a stem (on Off Gerlach 95% of bars sit within 6 dB of the
loudest). The set opens in automation mode with each track showing its Energy
lane, aligned with the clips and the locators (`A` toggles the view). The macro is mapped to nothing, so the lanes are a
read-only map; `--no-energy` leaves them out. Reading the stems adds about 8 s
on a 5-minute track.

## Configuration

Roadie reads one optional file, `~/.config/roadie/config` (`$XDG_CONFIG_HOME` is
honoured). It is plain `key = value` text: one setting per line, `#` starts a
comment, a leading `~` expands to your home directory. The file is parsed, never
executed, and a missing file is fine.

```
# ~/.config/roadie/config
out = ~/Projects/Music/slgMusProd/Studies
```

| setting | meaning | default |
|---|---|---|
| `out` | folder `roadie split` writes the WAV and `_stems` folder into | the current directory |

A configured `out` is treated as a studies library: `roadie split` gives each
track its own folder, `<out>/<Artist>/<Title>/`, holding the WAV and the `_stems`
folder; `roadie als` and `roadie open` then write the `.als` beside the stems, so
one folder holds the whole analysis project. Artist and title split at the first
` - ` (`Matador - Eternity` → `Matador/Eternity/`, `PML - Under Water` →
`PML/Under Water/`). An existing folder named for the artist or for their last
name is reused, so `Marc Holstege - Mangata` lands in `Holstege/Mangata/` if you
already have a `Holstege/`. A title with no ` - ` (a live set) gets
`<out>/<Title>/`; pass `-o` to place it. An explicit `-o` always writes exactly
there (`yt2stems.sh -a` applies the layout by hand).

Precedence, highest first: the command-line flag (`roadie split -o DIR`), then the
config file, then the built-in default. Config applies only through the `roadie`
command; the underlying scripts run directly (`yt2stems`, `stems2live`) ignore it.

Two environment variables sit beside the file:

| variable | effect |
|---|---|
| `ROADIE_CONFIG=<file>` | read this config file instead; set it empty to ignore config |
| `ROADIE_LOG=<file>` | write the usage log here; set it empty to turn the log off |

**Usage log.** Each `roadie` run on a source or a stems dir appends one
tab-separated line (time, command, exit code, seconds, absolute path or URL) to
`~/.local/state/roadie/usage.log` (`$XDG_STATE_HOME` is honoured). It stays on your
machine; nothing is sent. `roadie stats` summarises it. Runs that never reach a
script (`--help`, a mistyped directory) are not logged.

## Install details

```bash
brew install ffmpeg deno aubio uv
uv tool install yt-dlp
uv tool install "audio-separator[cpu]==0.44.5"
./install.sh
```

`demucs` is optional now: only `-m htdemucs_6s` (and the other Demucs models)
needs it, with `uv tool install demucs --with numpy`.

`./install.sh` checks the tools above, builds `.venv` from `requirements.txt`
(Python 3.13, via `uv`) if it does not exist yet, and symlinks `roadie` plus the
per-script names into `~/.local/bin` (`--bin DIR` to change). `./install.sh
--check` reports without changing anything.

Four install details are easy to get wrong:

- **`audio-separator[cpu]==0.44.5`** — the pin is the version everything here was
  tested at (its flags, its output file names, how it writes stems); a newer one may
  work, untested. The base package doesn't pull `onnxruntime`.
  Worse, `audio-separator --list_models` **exits 0** while printing a
  `ModuleNotFoundError` traceback, so a naive exit-code check reports success.
  Despite the name, the extra only picks the ONNX runtime: on an Apple-silicon Mac
  the PyTorch models (BS-Roformer-SW, DrumSep) still run on the GPU, and its log
  says `setting Torch device to MPS`.
- **`demucs --with numpy`** (optional) — demucs 4.1.0 omits numpy from its
  metadata. A plain install "succeeds", then dies on `import numpy` at runtime.
- **Models** — `audio-separator` keeps them in `/tmp/audio-separator-models` by
  default, which a reboot empties. `roadie split` points it at
  `ROADIE_MODEL_DIR`, else `AUDIO_SEPARATOR_MODEL_DIR`, else
  `~/.cache/audio-separator-models`. The first run downloads BS-Roformer-SW
  (~700 MB) and the DrumSep model (~440 MB), about 1.1 GB. A DrumSep model left
  in `/tmp` by an earlier version is not found there: it downloads once more, and a
  line says so before it does.
- **`deno`** — yt-dlp needs a JS runtime or YouTube extraction runs in a
  deprecated mode where some formats silently go missing. Only needed for URLs.

## Ableton side: Sideman

`roadie load`, `roadie open` and `roadie scenes` talk to
**[Sideman](https://github.com/slegroux/sideman)**
([site](https://slegroux.github.io/sideman/)), an Ableton Live remote script
that exposes the Live Object Model over a local socket on port **9878**. Install
it by its own instructions (a `.pkg` from its Releases, or `./scripts/install.sh`
from a clone), then in Live: **Preferences → Link, Tempo & MIDI → Control
Surface = AbletonLOM** (the name Sideman's remote script registers under),
Input/Output = None. Restart Live once if it was running during the install.

Roadie does not speak that socket itself: `roadie load` and `roadie scenes` go
through Sideman's Python client, `sideman.client.Live`. `install.sh` installs
the `sideman` package into Roadie's `.venv` from `requirements-sideman.txt`
(pinned to a commit, installed with `--no-deps`, so Sideman's own dependency
`mcp` is not pulled in unpinned; `roadie mcp`'s copy comes from
`requirements-mcp.txt`, every package pinned) and re-syncs it on every run, so a bumped pin reaches an
existing `.venv` with one `./install.sh`. A `.venv` built before Sideman was a
dependency says so and asks for that run.

Sideman ships `warp_markers_set` natively, so no patch is needed on the Live side.
Verify with:

```bash
./install.sh --check
```

It asks the bridge whether it knows `warp_markers_set`. **Port-open is not
evidence**: the socket answers normally whatever version of the handlers Live has
loaded.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The real-stem tests need a split of a reference track, which is not
redistributable and never committed. Point `ROADIE_REF_STEMS` at a stems folder
of your own (or put one at `./reference/`); without it those tests skip.
