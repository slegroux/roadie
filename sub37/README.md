# sub37: Moog Sub 37 patches → UADx Moog Minimoog Synth

Rebuilds a Sub 37 patch (a single-preset `.syx` dump) on the UADx Minimoog in Live, then tunes it against
recordings of the real patch. Written for Stephan Bodzin's *Boavista Deconstructed* pack (13 Sub 37 dumps,
13 MIDI files, per-song one-shots), 2026-11-13 to 2026-11-15. The pack is not in the repo: point `SUB37_WORK` at a
folder holding `decoded.json`, `tones/<song>/`, `mid/` (defaults to `~/.omc/reports/sub37-resynth/all`).

## Pipeline

| Step | Script | Out |
|---|---|---|
| 1. Fetch the knob map | `sh fetch_msp.sh` | `moog_sub37.msp` (bsp2/tks, pinned commit; the repo has no licence, so it is fetched, not vendored) |
| 2. Decode the patches | `python3 sub37_decode.py moog_sub37.msp syx/*.syx > decoded.json` | every NRPN field: value, range, 0–1 fraction |
| 3. Measure the references | `tone_info.py tones/*/` | `tones.json`: f0 → MIDI note, audible length per one-shot |
| 4. Build the Live tracks | `build_session.py` | per patch: `UAD <song>` (Sub37 Strip rack, mapped values, slot 1 = Bodzin's MIDI, slot 2 = a test clip of the reference pitches) + `CAP <song>` capture track; `session.json` |
| 5. Auto-tune | `passes.py [minutes]` | records every song at once, scores each tone, keeps improving moves; `tune_state.json`, `tune_log.txt` |

`sub37_map.py` holds the Sub 37 → UADx starting map (pure, tested in `test_sub37.py`). `uadlib.py` wraps the
Sideman client (`SIDEMAN_SRC`; `passes.py` needs `sideman.oneshot.distance_terms`, on `feat/sideman-patch-match`).

## Formats worth knowing

- **Sub 37 SysEx:** offsets count from the byte after `F0`. A field's high bits sit in byte `encOff` at
  `encBitOff` (`encBitNum` bits); `encBitNumExt` extension bits follow, 6 per byte, left-aligned from bit 5.
  Switches share lead bytes with values, so a plain 4-6-6 word decode mixes flags into values.
  `000_Earth.syx` in the pack is a byte-identical copy of `000_CooperStation.syx`.
- **UADx AU parameters:** Live exposes 41 of the Minimoog's 53 on its own. A `Default.appc` exposes all 53, but the
  AU parameter ID is the Java `String.hashCode` of the parameter's position in decimal (`"10"` → 1567), not the
  position. Found in the plug-in's NKS files (`extras/nks/*.nksf`, `NICA` chunk, msgpack); `nica.py` dumps them,
  `uad_state.py` reads the JSON state (real units) in the `PCHK` chunk.
- **The Minimoog has no drive control.** Multidrive maps to the mixer overload (oscillator levels) plus the
  feedback loop (`EXT. Mix`, `Input Source` = FEEDBACK); Decapitator in the strip does the post-filter part.

## Calibration (UADx Live 0–1 values, measured 2026-11-15)

- `OSC-2 Tune`: 0.497 = unison, ~20.1 semitones per unit (0.258 = −4.8 st).
- `LFO RATE`: 0.1 → 0.51 Hz, 0.49 → 2.74 Hz, 0.5 → 3.16 Hz, 0.6 → 7.25 Hz. `TEMPO SYNC` rewrites it.
- `MOD AMOUNT`: steep taper; 0.4 ≈ nothing, 0.6 ≈ a ±4–5 dB filter wobble, 1.0 ≈ 35 dB.

## Analysis tools (loops)

`bandreport.py` (per-zone balance, crest, width, LFO, pedal partials), `envelopes.py` (per-note amp/brightness
envelopes), `score.py` (alignment helpers), `mrstft_score.py` (whole-loop MR-STFT distance), `f0track.py`,
`lfo_cal.py`, `build_strip.py` (Instrument Rack `.adg` from a saved `.als` track, plug-in states included).

## Tuning strategy (learned 2026-11-15, 12 tracks, 38 passes)

The decoded patch does most of the work: 149 of 156 knobs ended within ±0.15 of the mapped values. Blind ± coordinate
descent (`passes.py` today) rejected about 84 % of its tries. Next version, in order:

1. **Verify the target** before tuning: reference pitches inside the song MIDI's range, plus a REF track to listen to.
   A wrong target wastes every pass.
2. **Corrected map:** compress the top of the cutoff map (high cutoffs always came down: 0.64→0.46, 0.61→0.43,
   0.46→0.28), sub at about 0.7× the Sub 37 level, amp decay ≥ 0.15 when the Sub 37 release is 0.
3. **Measure-driven steps:**
   - Mean brightness → cutoff; brightness decay rate → filter decay; how far it closes → contour.
   - Level at note end → release; sub band → sub level.
4. **Short ± polish** on cutoff, contour, release and sub only. Stop when a full cycle gains < 1 %.
   Judge LFO and mod depth only on long notes or loops.
5. **Speed:** all songs per pass, optionally 2–3 candidate copies per track, every LOM call batched.

Gain by knob in the 2026-11-15 run:
- Cutoff 65 %, inflated by leftovers from stopped runs.
- Contour 10 %, amp release 6 %, Decapitator drive 5 %, sub 5 %, osc2 3 %.
- Everything else about 1 % each.
