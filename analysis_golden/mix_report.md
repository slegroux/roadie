# Mix report

Tempo 120.0 BPM. Whole capture: -13.2 LUFS integrated, short-term max -13.2, true peak 2.6 dBTP, crest 15.7 dB, width 0.04.

Reference profile: `DIR/ref.profile.json`. Sections are paired with the profile's by name; band balance is relative to the matching section after removing the overall level offset, and bands more than 3 dB away are listed.

| Section | Bars | LUFS | ST max | TP | Crest | Width | Low corr | vs ref | Kick/bass overlap |
|---|---|---|---|---|---|---|---|---|---|
| A | 1–4 | -13.1 | -13.2 | 2.5 | 15.4 | 0.04 | 1.00 | sub +3.7 | 0.39 (quieter -8 dB) |
| B | 5–8 | -13.3 | -13.3 | 2.6 | 15.7 | 0.04 | 1.00 | sub +3.5 | 0.39 (quieter -8 dB) |
| C | 9–12 | -13.1 | -13.2 | 2.5 | 15.4 | 0.04 | 1.00 | sub +3.7 | 0.39 (quieter -8 dB) |

## Elements against the kick

Each element's mean power minus the kick's over the same bars (dB; negative = quieter than the kick). A gap is a difference of more than 1.0 dB. Where the kick is out (absent, or more than 30 dB under the loudest other role, on either side) the section says which element is the yardstick instead. The master capture is never ratioed against the elements: it sits after the master chain.

### A

| Element | Reference | Ours | Diff | Width (ref / ours) |
|---|---|---|---|---|
| bass | 4.1 | 8.1 | +4.0 | 0.00 / 0.00 |
| hh | -14.6 | -13.2 | +1.3 | 0.00 / 0.38 |
| snare | -5.7 | -5.7 | +0.0 | 0.00 / 0.00 |
| synth | -8.0 | -8.0 | +0.0 | 0.00 / 0.00 |

Gaps, largest first: bass +4.0 dB; hh +1.3 dB.
Width gaps (more than 0.1 apart): hh 0.00 -> 0.38.
In the reference only: drums (not in the manifest, -5.1 dB vs kick).

### B

| Element | Reference | Ours | Diff | Width (ref / ours) |
|---|---|---|---|---|
| bass | 4.1 | 8.1 | +4.0 | 0.00 / 0.00 |
| hh | -14.6 | -13.2 | +1.3 | 0.00 / 0.38 |
| snare | -5.7 | -5.7 | +0.0 | 0.00 / 0.00 |

Gaps, largest first: bass +4.0 dB; hh +1.3 dB.
Width gaps (more than 0.1 apart): hh 0.00 -> 0.38.
In the reference only: drums (not in the manifest, -5.1 dB vs kick).

### C

| Element | Reference | Ours | Diff | Width (ref / ours) |
|---|---|---|---|---|
| bass | 4.1 | 8.1 | +4.0 | 0.00 / 0.00 |
| hh | -14.6 | -13.2 | +1.3 | 0.00 / 0.38 |
| snare | -5.7 | -5.7 | +0.0 | 0.00 / 0.00 |
| synth | -8.0 | -8.0 | +0.0 | 0.00 / 0.00 |

Gaps, largest first: bass +4.0 dB; hh +1.3 dB.
Width gaps (more than 0.1 apart): hh 0.00 -> 0.38.
In the reference only: drums (not in the manifest, -5.1 dB vs kick).


## Drum sounds

Our clean hit of each drum role against the reference's, over the section's bars (oneshot readings: a is ours, b the reference; none = nothing past the thresholds).

### A

- kick: body pitch 2.0 st below the reference (71 Hz, reference 80 Hz)
- snare: none
- hh: width 0.38, reference 0.00 (wider)

### B

- kick: body pitch 2.0 st below the reference (71 Hz, reference 80 Hz)
- snare: none
- hh: width 0.38, reference 0.00 (wider)

### C

- kick: body pitch 2.0 st below the reference (71 Hz, reference 80 Hz)
- snare: none
- hh: width 0.38, reference 0.00 (wider)


## Change rate

Spectral change between consecutive 4-bar blocks (mean third-octave difference, dB). Under 1.0 dB counts as no change.

- ⚠ bars 1–12 (12 bars, A, B, C): no change for more than 8 bars

Steps: bar 5 0.7, bar 9 0.7
