<p align="center">
  <img src="docs/logo.svg" alt="roadie" width="320">
</p>

<p align="center">
  <strong>From finished track to Live set.</strong><br>
  Split a recording into stems and get a ready-to-study Ableton Live 12 arrangement.
</p>

<p align="center">
  <a href="https://github.com/slegroux/roadie/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/slegroux/roadie/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://slegroux.github.io/roadie/"><img alt="site" src="https://img.shields.io/badge/site-slegroux.github.io%2Froadie-C93C25"></a>
  <img alt="Python 3.13" src="https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="Ableton Live 12" src="https://img.shields.io/badge/Ableton%20Live-12-000000?logo=abletonlive&logoColor=white">
  <img alt="macOS" src="https://img.shields.io/badge/platform-macOS-lightgrey?logo=apple">
  <a href="LICENSE"><img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-blue"></a>
</p>

---

Roadie takes a recording of your own track or set and does two things:

1. **Splits it** into stems (bass, drums, synths, vocals) and the drum kit into
   kick, snare, toms, hats, ride and crash.
2. **Builds the arrangement** in Live: one track per stem, on the bar line, at the
   measured tempo, sorted low to high, colour-coded, 6 dB down, with a locator
   per section.

The stems are plain audio files any DAW can open. Everything after the split is
Ableton-only. *Roadie was formerly called yt2live.*

## Quick start

```bash
roadie split my-track.wav -s -o ~/Studies          # audio -> stems -> drum parts
roadie load ~/Studies/my-track_124bpm_stems         # stems -> the Live set that is open
```

Don't want Live running? `roadie als <stems_dir>` writes a `.als` project file
you can open later.

> [!IMPORTANT]
> **Use it on recordings you have the rights to.** Stems extracted from a DJ set
> are other artists' music: fine for practice and analysis, not for release.
> `roadie split <url>` runs `yt-dlp`, which is subject to YouTube's Terms of
> Service. For your own uploads, download the original from YouTube Studio and
> pass the file instead.

## Features

- **Measured defaults.** BS-Roformer-SW separation, chosen by scoring stems
  against true multitracks rather than a leaderboard. Tuned on deep and melodic
  house.
- **Tempo from the audio.** The beat grid is fitted to the kick stem, then
  cross-checked against it, so a set doesn't drift two bars out by the end.
- **Warps only when needed.** Steady tracks drop in unwarped. Drifting ones get a
  smoothed warp map.
- **Section locators** named by which stems come and go: `Drop 1 (bass in) · 64`.
- **Energy lanes.** `roadie als` draws each stem's loudness per bar as automation,
  a read-only map of who plays where.
- **Session View scenes**, one per section, with a looping clip per stem.
- **Overnight batch** over a whole library, uploaded to pCloud with `rclone`,
  using about 1 GB of local disk.
- **MCP server** for comparing your Set against a reference track and its stems.

## Install

macOS, with Homebrew:

```bash
brew install ffmpeg deno aubio uv
uv tool install yt-dlp
uv tool install "audio-separator[cpu]==0.44.5"
./install.sh            # builds .venv, links `roadie` into ~/.local/bin
./install.sh --check    # reports what's missing, changes nothing
```

The first split downloads about 1.1 GB of models to
`~/.cache/audio-separator-models`.

To work with a running Live set, install
**[Sideman](https://github.com/slegroux/sideman)**, the Live remote script Roadie
talks to. Then in Live, set **Preferences → Link, Tempo & MIDI → Control
Surface** to `AbletonLOM`. `./install.sh --check` confirms the bridge answers.

Demucs is optional: `uv tool install demucs --with numpy`, then `-m htdemucs_6s`.
Pin details and pitfalls are in [Install details](docs/REFERENCE.md#install-details).

## Commands

| Command | What it does |
|---|---|
| `roadie split <file\|url>` | Audio → 44.1 kHz WAV; `-s` adds stems and drum parts. `-D` skips the drum parts; `-m htdemucs_6s` uses Demucs. |
| `roadie load <stems_dir>` | Stems → the Live set that is open now. Add `--locators`, `--session`, `--energy-map`. |
| `roadie open <stems_dir>` | Stems → a **new** Live project, from your template `.als`. |
| `roadie als <stems_dir>` | Stems → a `.als` file directly. Live need not be running. |
| `roadie sections <stems_dir>` | Detect song sections and print their bars. |
| `roadie scenes <stems_dir>` | One Session View scene per section. |
| `roadie batch plan\|run\|verify-sources` | Unattended batch over many files. See [BATCH.md](docs/BATCH.md). |
| `roadie stats` | Runs and tracks from the local usage log. |
| `roadie mcp` | MCP server for reference-based analysis. |

`roadie <command> --help` lists each command's options.

## What you get

```
my-track_124bpm.wav                 44.1 kHz / 24-bit working copy
my-track_124bpm_stems/
├── bass.flac  drums.flac  vocals.flac
├── synths.wav                      float32, so the summed parts can't clip
├── drums_{kick,snare,toms,hh,ride,crash}.flac
├── alignment.json                  beat grid, reused by load / als / scenes
└── sections.json                   written by sections / als
```

All stems are the same length, so they line up at bar 1.

## Configuration

One optional file, `~/.config/roadie/config`:

```
out = ~/Projects/Music/Studies
```

With `out` set, each track gets its own folder: `Matador - Eternity` lands in
`<out>/Matador/Eternity/`. A `-o` on the command line always wins. Details,
environment variables and the usage log are in
[Configuration](docs/REFERENCE.md#configuration).

## MCP server

```bash
claude mcp add --scope user roadie -- "$PWD/roadie" mcp
```

| Tool | What it does |
|---|---|
| `sample_analyze` | One drum hit in synth terms: envelope, pitch drop, harmonics, noise, click, width. |
| `sample_compare` | Two hits compared, one line per real difference. |
| `reference_profile` | A reference track and its stems, measured per section. |
| `session_compare` | A capture of your Set against that profile, per section. |

Measuring the Set on its own, with no reference, stays in Sideman's own MCP
server. The tools' docstrings are the full documentation.

## Documentation

| Doc | What's in it |
|---|---|
| [REFERENCE.md](docs/REFERENCE.md) | Every command in full, outputs, configuration, install |
| [DESIGN.md](docs/DESIGN.md) | Why each default is what it is, the measurements, known limitations |
| [BATCH.md](docs/BATCH.md) | The overnight batch: manifest, resume, safety rules |
| [DRUMROLES.md](docs/DRUMROLES.md) | How drum sounds are grouped and named (`roadie_analysis.drumroles`) |
| [ABLETONMCP-PATCHES.md](docs/ABLETONMCP-PATCHES.md) | Legacy AbletonMCP patches and what they taught about Live's API |

## Known limitations

- **Which beat is bar 1** is the weakest link on four-on-the-floor. Check it by ear.
- **Section names are rules**, not recognition: `Drop` means the kick and bass
  came back.
- **No pad, arp or stab separation.** Those are arrangement roles, not sources.
- **The drum pass is slow**, about 2× realtime on an Apple GPU.
- **Live must be running** for `load`, `open` and `scenes`. `als` doesn't need it.

More in [DESIGN.md](docs/DESIGN.md#known-limitations).

## Development

```bash
.venv/bin/python -m pytest -q
```

Tests that need real stems skip unless `ROADIE_REF_STEMS` points at a stems
folder of your own.

## License

[GPL-3.0](LICENSE).
