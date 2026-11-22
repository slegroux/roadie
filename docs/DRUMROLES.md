# Drum roles: the rules and where each number comes from

Moved from Sideman's `docs/CONVENTIONS.md` (branch `feat/sideman-drum-roles`, `f877340`) with the code, unchanged
but for the module name. The README's "Drum roles" section is the short version.

`roadie_analysis.drumroles` (numpy and soundfile) lists the sounds
inside one drum stem. `hit_groups` finds the hits, puts them on the 16th grid
and groups the ones that sound alike. `name_roles` then gives each group a name
and `mark_bleed` marks the groups that are another stem's sound leaking in.

## Running it: `roadie drums`

    roadie drums <study name | stems_dir> [--jobs N] [--json] [--no-write]

`roadie_analysis.drums.drum_roles` analyses every `drums_<kind>` stem of a stems folder (`drums.flac`, the sum, is not
one) **section by section** and writes the roles. The MCP tool `drum_roles(stems, tempo, t0, sections, out, jobs,
timeout_s)` does the same and returns the record. Nothing is tuned: the thresholds are the ones below.

**Why by section.** `hit_groups` on a whole track groups every hit of a stem at once: a 5-minute hat stem is 1200
hits, the run takes minutes on one core, and the grouping moves with the sample it draws (Glacier 3000's hats came out
as eleven groups, none stable enough to name). A section holds a few dozen to a few hundred hits: it is grouped on its own
(a section of at most 32 hits of a stem is grouped exactly: "stable by construction"), and the (stem, section) jobs run in
a process pool (`--jobs N`, default the cores).

**How the sections are cut and merged** (`roadie_analysis/drumsections.py` has the choices):

1. The sections are `sections.json`'s. One shorter than 8 bars joins the next (the last joins the one before): too few
   hits to group. No `sections.json`: the whole track is one section. The first section starts at bar 0 and the last runs
   to the end of the file.
2. Each stem's loudest 5 ms is measured once over the whole file (`ref_power`), so a section is judged against that and not
   against its own loudest moment. A section's slice runs from about 40 ms before its first bar line to about 60 ms after its
   last, **rounded outwards to a multiple of lcm(1 ms, 10 ms) of samples** (19,404 at 44.1 kHz): the detector reads the audio on a
   1 ms grid and the DC offset on 10 ms blocks anchored at the slice start, so a slice that began a few samples later (a grid
   moved by 0.001 bpm moves a section line by about a millisecond) found hits a few samples apart and grouped them differently.
   With the anchor fixed the same audio gives the same hits and groups under any grid that is nearly the same. A hit belongs
   to the section whose bar lines, moved 20 ms earlier, enclose it, so a hit 8 ms early on the first beat is that section's
   and none is counted twice.
3. The groups of all the sections are matched by sound: each group's clean hit (else the cut of its member nearest the
   centre) is compared with every other group's with `spectral_distance`, and the groups merge by average linkage at the
   grouping threshold (1.5). Two groups of one section never merge (a pitch split inside a section stays split). The
   merged group is one role: its hits are all its sections' hits, its pattern is over the whole track, and it keeps one
   name.
4. `mark_bleed` and `name_roles` then run on the merged roles, with the rules below unchanged. A merged role's
   `stability` is the hit-weighted mean of its groups' measured stabilities, "stable by construction" when no group was
   measured and one of them is; its `separation` is the mean distance between the groups of the role and of the nearest other role of at least 4 hits
   (it can still be under the threshold: two groups that share a section are not merged however close they are). A sound
   with one hit in each of five sections is one role of five hits and can be named; a sound that only ever plays once has
   one hit and stays neutral.

**Library studies.** A study on the pCloud mount is never read or written through the FUSE mount (that fills a local cache
with audio and writes into the archive through a mount): `roadie drums` copies its drum stems, `alignment.json` and
`sections.json` to a temp folder with rclone, analyses there, uploads the new clips, then `drums.json`, then deletes the
clips the previous `drums.json` listed and the new one does not (nothing else is ever removed), and deletes the temp folder.
`--no-write` (MCP: `write=false`) uploads nothing. A local stems folder is read and written in place.

**The grid.** Tempo is `alignment.json`'s `result.bpm`, the bar line its `result.downbeat_sec`, the sections
`sections.json`'s (`label`, `time_sec`, `duration_beats`). Without `alignment.json` the first section's start is the bar
line. A default bar line is moved back by whole bars to the first one at or after the start of the file; the 16ths
are still counted from the downbeat (step 0). A hit before that line (a pickup, a hit in the first partial bar) has bar -1. A `t0` passed in is used as given. A downbeat the aligner
calls chance-level is repeated in the output's `notes`: the whole table is then counted from a bar line that may be the
wrong beat. The grid is one constant tempo: a track with drift reads hits a little early or late, and a hit more than
20 ms from a line is off the grid and counts in no pattern.

**The table**, one block per stem:

    role          hits  pattern               level   flags
    hh offbeat      96  ..x. ..x. ..x. ..x.  -6.5 dB  stability 1.00
        Intro           32  ..x. ..x. ..x. ..x.
        Drop            64  ..x. ..x. ..x. ..x.

- `pattern` is 16 characters, one per 16th of the bar from the downbeat, in groups of four (one beat each): `x` = the
  role plays there in at least half of the bars it plays in, `o` in 10 to 50 % of them, `.` in fewer. The first line of a
  role is over the whole track, the lines under it are one per section it plays in (hits, pattern over the bars it plays
  in within that section). The JSON has both readings of the whole-track pattern: `pattern` (the share of all the stem's
  bars, `name_roles`' own) and `own_pattern` (the share of the role's `bars_with`, which `line` is made from).
- A group of fewer than 4 measured hits is one summary line in the table (and in `study show`); it is a role entry in the
  JSON with its reason, like the others.
- `hits` counts measured hits and the ones assigned under another hit's tail. `level` is the median loudest-ms level of
  the measured hits (dBFS). `off_grid` counts hits more than 20 ms from a 16th line (in no pattern).
- `stability` and `separation` are the numbers `name_roles` judged (see above); none printed means stable by
  construction or no rival.
- **A neutral name (`hh 2`, `kick 1`) means no rule named the group**; the line under it says why (too few hits, not
  stable, not apart from its neighbour, no rule of that stem fits, members disagree). It is still a real group of hits that
  sound alike: listen to its clip. Stems with no rules (`ride`, `crash`) are always neutral.
- `BLEED` lines are roles marked as another stem's sound leaking in; they are not roles and have no clip.
- `variants`: one role made of several groups that may be one sound (see "Two groups that may be one sound").
- `no clip`: no member of the role is complete (cut off by the next hit or the 0.5 s limit before it falls 40 dB); the
  reason is in the JSON.
- The last line prints the elapsed time, the jobs and the time of each stage (sections, matching, bleed, naming).

**Files.** `drums.json` beside the stems (`kind: "drum_roles"`, `version: 1`, `by: "section"`): `grid` (tempo, `t0`, the
downbeat it came from, the sections as bar ranges after joining), `inputs` (each stem and grid file read, with name, size and
mtime: `roadie study show` says fresh or stale by name and size), `legend`, `notes`, `seconds`, `jobs`, `stage_seconds`, and
per stem `roles`, `bleed`, `per_section` (hits per role in each section) and `notes`. Each role has `name`, `named`, `rule`,
`reason`, `hits`, `bars_with`, `pattern`, `own_pattern`, `line`, `off_grid`, `level_db`, `stability`, `separation`, `groups`,
`variants`, `clip`, `clip_note` and `per_section` (a list: `section`, `start_bar`, `bars`, `hits`, `bars_with`, `pattern`,
`line`, only for the sections the role plays in). `drum_roles/<stem>-<role>.wav` is the role's clean hit (24-bit, the stem's
rate); no clean hit, no file. A run writes the new clips first, then `drums.json` (by temp file and rename, so a reader never sees half of it), and last removes the clips the earlier `drums.json` listed and the new one does not.
`roadie study show` prints each stem's roles with their pattern line when `drums.json` exists, and `study list` flags the
`drums` layer.

**Steadiness.** `python -m roadie_analysis.drumstability <stems_dir> [--stems hh,snare]` re-runs the analysis with the grid
moved by tempo +-0.001 and +-0.005 bpm and t0 +-1 ms (nothing written) and compares, hit by hit, which role each hit lands in
(`kept`: the share of a base role's hits in its best-matching role; `agreement`: the same over all roles of 4 or more hits).
The hit assignment is steady under those moves (Glacier's hat and snare stems: 1.0 in 11 of 12 runs, 0.995 in the other). What is
not steady is a **name that rests on a threshold the pattern sits on**: the patterns are made from the hits within 20 ms of a
16th line, and a hat whose offsets spread to the edge (Glacier's `hh 16th`: 60 of 1600 hat hits lie within 3 ms of it) loses or
gains a few hits when the grid moves, which moves its hits a bar on the odd 16ths across the 4.0 the `hh 16th` rule asks for
(4.3 at the base grid, 4.0 and 3.7 at +0.001 and +0.005 bpm) while its hits stay the same. A role whose name or whether it is
named changes when the on-grid tolerance is 3 ms tighter or 3 ms looser (about what 0.001 bpm moves the last section's bar line
by) has `name_steady: false` and a `name_steady_note` saying what changes, and the table says `name unsteady`: the name is a
reading of a pattern at a rule's edge. Nothing is tuned.

**Known limits.** Each section is judged against the whole stem's loudest 5 ms (the core's 40 dB figures, unchanged), so a section where the stem is near silent has no hits, as in a whole-track run; a stem's quiet hits within 40 dB of its loudest are still found. Matching is by one clean hit (or cut) per section group: two sounds that are close (under 1.5) in every
section are one role, and one sound that changes between sections (a filter sweep, a different sample) is two. A section
is grouped on what is in it: a hit more than 40 dB under the section's loudest moment is not found, and the hits at the
two ends of a slice have less context than in the whole stem. Hat stems are judged without `grid_onsets`: a hat under an
open hat's tail is placed only when it can be compared. One constant grid for the whole file. The kind of stem decides which
rules apply, and the separator's classes are all it has: congas and claps land in toms and snare and keep neutral names or
the nearest rule's. Without `sections.json` the whole track is one section, which is the slow and unstable case above.

**The names are rules over a group's stem, its pattern and the readings of its
clean hit. They are not recognition.** A group no rule fits keeps a neutral
name (`hh 2`: the stem and the group's number) with the reason. A name says
"this group passed this rule", never "this is a hi-hat". Listen to the clean
hit before trusting one.

A group is **named only when it is eligible**: at least 4 measured hits,
`stability` at least 0.75 (or **stable by construction**), and `separation`
over the grouping threshold (1.5). `stability` is None in three cases, and
`stability_status` says which: "stable by construction" (a stem with no more
measured hits than the sample of 32: the first round's sample took every hit, the
grouping is exact average linkage and no regrouping can differ; the group passes
the test, with a note on the role), "not asked for" (`stability_runs=0`: the group
stays neutral) and "not measured" (a tentative group). `separation` is measured against the nearest group
of at least 4 measured hits (`hit_groups` also keeps `separation_any` and
`nearest_any`, against any group): a one-to-three hit satellite beside a big group
says nothing about whether two sounds are apart (on Flare a 3-hit group at 1.21
had kept the 57-hit toms group from a name). A group with no such neighbour has
`separation` None, no rival, and passes that test, as a lone group does.
Otherwise it keeps the neutral name and the reason says which test it failed.
The rule is the second review's: on the Flare hats, groups at stability
0.79-0.91 were the ones the sample did not move, 0.13-0.69 the ones it did.

Pattern: the hits counted are the measured ones and the ones assigned to the
group under another hit's tail (`assigned` says how many). A 16th is *in use*
when the group plays there in at least 25 % of the bars it plays in **that it could
be seen in**. A bar does not count for a 16th when the line fell inside another
hit's tail with no hit found at it (`hidden_cells`: a closed hat under an open
hat's tail cannot be seen playing there, **if the tail is within 15 dB of the
group's own level**: a hit under a quieter tail would have risen out of it and been
found), or held an on-grid hit that was not measured (`unmeasured_cells`: it may have been this group's), and the group does
not play there. Without that, a hat that plays every "a" but is hidden in three
bars of four reads 25 %. A 16th observable in fewer than 4 bars is *unknown*: it
counts neither for nor against, and a name given with unknown 16ths says how many
in its `note`. The per-16th bar counts and shares (over all bars and over the
observable ones) are in each role's `steps`. "n of m in use" is read over the
known 16ths: **at most 2 of the known odd 16ths empty** (8 - 6) for `hh 16th`, at
most 1 of the known offbeats empty (4 - 3) for `hh offbeat`, and **at least as many
known as the rule asks in use**: 6 of the 8 odd 16ths, 3 of the 4 offbeats. (The rule
first took the same share of the known ones, rounded up. That is not monotone: 8
known with 6 in use was named and 7 known with 5 in use, the same group with one
in-use 16th hidden under a tail, was not. Counting the empty ones is monotone, and an
unknown 16th still counts neither for nor against.) A group seen at only half the
steps is not named from them: on Flare a hat group seen at its four "e" steps,
its four "a" steps hidden under a tail, passed 4 of 4 known and was named from half
the evidence. With fewer known the group stays neutral, "only N of 8 odd 16ths could
be observed". The offbeat floor was first 2 of 4; a group seen at two offbeats
with every hit on the offbeats would have been named from two steps, the same
weakness, so it is 3 (the two floors are the same 3/4). The denominator is the
group's own bars (every bar it has a hit in), so a group that lives in one section
is judged over that section, not over the stem.

| Stem | Name | Rule | Where the number comes from |
|------|------|------|-----------------------------|
| hh | `hh offbeat` | at least 90 % of the hits on steps 2, 6, 10, 14, at least 3 of those 4 in use (of the known ones), and **at least 2 hits a bar on them**, counted over the bars each could be seen in | 90 %: chosen, a tenth of the hits may be strays or assigned to the wrong 16th; 3 of 4: one offbeat may be taken by the other hat (Flare hh group 0 plays its four offbeats in 56 % to 89 % of the stem's bars: 2.2 to 3.6 a bar); 2 a bar: half of the 4, chosen (one hit a bar rotating over the offbeats put each in use in a quarter of the bars and was named) |
| hh | `hh 16th` | at least 6 of the 8 odd 16ths in use, and **at least 4 hits a bar on them**, counted over the bars each could be seen in | chosen: half of the 8 (2 hits a bar rotating over the odd 16ths put each in use in a quarter of the bars and was named); a 16th-note hat plays the "e" and "a" of each beat, and 6 of 8 leaves two gaps (4 of 8 let a hat that plays only the "e" of each beat, Flare hh group 2, take the name); an offbeat group has none of them |
| hh | `hh open` | neither pattern fits and the members read fall 20 dB in 140 ms or more | `fall_20_ms` of the 22 closed and open hats of `_PRODUCTION/Drums` that could be measured (below) |
| snare | `rim` | the members read fall 20 dB in under 30 ms | the library's 3 rims: 5, 19 and 28 ms; its other 30 claps and snares: 35 to 167 ms. Also under 30 ms, so named `rim` in a snare stem: 5 hats (1 closed, 4 open: 15 to 29 ms) and 1 clave (26.7 ms) |
| snare | `snare` | not a rim, and the noise share of the body is over -12.0 and at most -1.0 dB (a tone under the noise, not all tone) | the library (108 samples, rim rule first): its 9 snares (7 snares, 2 labelled clap + snare) read -9.1 to -1.0, 9 of 9 named; its 9 toms -40.6 to -14.8 and its 33 kicks -39.7 to -23.0, 0 of 42 (at most -1.0 alone would have named all 9 toms and 31 of the 33 kicks that could be read); the other 14 claps -0.3 to 0 stay neutral. **Still named: 1 clave (Clave Mian, -8.9 dB) and 7 hats (3 closed, 4 open, -3.3 to -1.7)**: nothing read here tells a hat's or a clave's noise from a snare's, which is why the name is only "passed this rule" |
| toms | `tom` | the members read are pitched | `profile_array`'s `pitched`; the library's 9 toms are all pitched (53 to 578 Hz). Also pitched: its 33 kicks, 4 snares or claps and 4 percussion samples, so a kick in the toms stem is a `tom` (the word means "pitched, in the toms stem") |
| kick | `kick` | pitched, body at most 70 Hz | the library's 33 kicks: 28 to 56 Hz. Also pitched at 70 Hz or under: the library's 53 Hz low tom (BZ LowTom Astronautin), which a kick stem would name `kick` |

**The measurement rules read members, not one hit.** `hh open`, `rim`,
`snare`, `tom` and `kick` read the readings of up to 8 complete members of the
group nearest its medoid (`read_members`, the clean hit usually among them) and
name only when **at least two thirds of the members read agree, from at least 3
complete members** (with fewer the group stays neutral: "too few complete
members"); otherwise the
group stays neutral and the reason has the counts ("3 of 8 rim, 3 snare, 2
neither"). The pattern rules do not read them. One hit is a poor witness: on
Flare the snare group 2's clean hit fell 20 dB in 30.0 ms, exactly the rim
limit, and over its members the readings were 6 snare, 5 rim, 3 neither, so the
name `snare` was a coin toss. A tom's rank is the median pitch of the members
that read pitched.

**The density is read like the step shares, over cells a hit of the group could
not have been found in.** "Hits a bar" on the odd 16ths or the offbeats is the sum,
over the *known* 16ths of that kind, of each one's share of the bars it could be
seen in (an unknown 16th adds nothing, and the known-16ths floors already cover
those). A cell is not seen **for a group** when it held a hit that was not measured
(it may have been the group's), or when its line fell inside another hit's tail
with no hit found AND that tail is within 15 dB (`RISE_DB`, the rise
`oneshot._segments` needs to find a hit; 6 dB with `grid_onsets`, which looks at the
line) of the group's own level: only then would a hit of the group have stayed
hidden. A tail more than 15 dB under the group's level is no cover: a hit of the
group would have risen out of it and been found, so the cell is seen and empty.
`hit_groups` gives each hidden cell the level of the tail there (`hidden_tail_db`,
`unmeasured_tail_db`, aligned with the cells). The first version counted every
cell under any tail within 40 dB of its hit as hidden; empty cells under quiet
tails were set aside while played cells stayed counted, which inflated the shares
and the density (a hat playing 2 hits a bar, with every empty odd line under a quiet
tail, read 8.0 a bar and was named `hh 16th`; on Flare 78 of the 83 "hidden" odd
cells of hh group 1 lay under a tail more than 15 dB under it). The role's `density`
gives both figures: `odd_per_bar` / `off_per_bar` over all the group's own bars,
and `odd_density` / `off_density`, which the rules read.

When several roles of one stem are toms they are ranked by body pitch:
`tom low`, `tom mid`, `tom high` (two toms: low and high; more mids get a
number). The rank is relative to the other toms in that stem, not a pitch class. Tom
roles that read the same pitch (to the Hz) share a rank: all at one pitch are
not ranked at all and just numbered, with a note.
Two roles that get one name are numbered (`hh offbeat`, `hh offbeat 2`).

**Rules that are not there, and why.**

- **No `hh closed`.** The library's closed hats fall 20 dB in 18 to 130 ms and
  its open hats in 15 to 269 ms (12 and 10 hats measured; four of the open ones
  are under 30 ms). Nothing under 140 ms separates them, so a short decay names
  nothing. Over 140 ms: 0 of 12 closed hats, 5 of 10 open ones.
- **No `clap`.** Nothing `profile_array` reads separates a noise clap from a
  noise snare or a stick. The best rule found (noise body and under -20 dB in
  the 60-250 Hz band) names 9 of the library's 14 claps and 5 samples that are
  not claps (CMPCT SNAD, FX SNARE, SNARE STICK, STICK, STICK 1): 64 % right. A
  noise-bodied group in a snare stem stays neutral, with its measurements.
- **No `conga`.** The library has one conga: it reads unpitched, falls 40 dB in
  264 ms, inside the toms' 116 to 476 ms, and its high toms (290 to 578 Hz) sit
  where a conga would. A pitched hit in the toms stem is a `tom`: the word means
  "pitched, in the toms stem".
- The snare rule misses 4 of 11 plain snares (noise snares: the body has no
  tone to read). They stay neutral.

**Two groups that may be one sound.** The grouping cuts a continuum where the
sample puts the cut (the threshold's own limit, above). Two groups of one name
that play at the same 16ths (cosine of the two sets of 16ths with at least one hit, at least 0.8; counts per 16th were tried first and gave 0.68 for the real Flare pair, because one of the two plays mostly on the second offbeat) and are
within 1.15 times the threshold (1.73) of each other, one the other's nearest
group, are reported as **one role with variants** (each variant has its
decay, level and stability): on Flare two offbeat hat groups 1.54 apart with
different decays are this. 1.15 is chosen: it takes in the 1.54 and 1.53 of the
real pair and stays under the 1.77-1.82 at which the three Flare snare groups
held (stability 0.95-0.98). A group of that kind that is not eligible stays
neutral, and the role it is near lists it under `possibly_also`.

**Bleed.** Separators leak: a tom fill can sit 20 dB down in the kick stem. A
group is marked bleed of a group in another stem only when **all three** hold:

1. *Coincidence well over chance.* At least 70 % of its hits fall within 25 ms
   of a hit of that group, at least 25 points more than the two patterns give by
   chance, and the chance of so many from the patterns alone is at most 1 %.
   Chance, for each hit of the group at 16th s: the share of the track's bars
   (first to last hit of any stem) in which the other group plays at s, at least
   2 %; the hits taken as independent, `expected` is their mean and the chance
   of at least as many coincidences is a Poisson-binomial tail (`p`, over the
   track's bars, the bars from the first to the last hit of any stem). **A 100 %
   chance rate means the patterns explain the coincidence completely**: a group
   that plays the beats against a group that plays every beat coincides every
   time whatever it is, so the coincidence is no evidence either way, and the
   first builder called such groups bleed from a 96 % coincidence. It is not a
   pass either; from an expected rate of 90 % the question goes to level and
   likeness (co-located, below). The three numbers each guard a case: the rate
   a partial overlap, the margin a long stem where a small excess is
   "significant", the p value a few hits.
   `p` is over the **track's** bars, which overstates the evidence for a part
   that lives in one section (on Flare, bars 32 to 64, the kick stem's second
   group coincides with the snare stem's groups in 24 of its 26 hits, 92 %: the
   snare plays at those 16ths in 38 % of the track's bars and in 93 % of the 7
   bars the group plays in; the level, 4.7 dB under, is what keeps it unmarked). The
   evidence therefore also gives the chance over the bars the group itself plays
   in (`own_expected`, `own_bars`), and says "not evidence of leakage here" when
   that is within 10 points of the observed rate and `p` is small enough to
   have passed. The test itself keeps the
   track-wide chance: a leak is only there in the bars where its source plays,
   so a chance over the group's own bars has the answer built in (a leaked tom
   fill would always read 100 %) and no leak of a sectional source could be
   proved. A group marked while `p_evidence` is False is marked on the level and
   likeness conditions with the coincidence only a consistency check.
2. *Louder there.* The other group's hits are louder at the coincident moments
   (`level_db`, loudest ms) by at least 6 dB in at least 75 % of the pairs. 6 dB
   is chosen (half the amplitude); on Flare the snare stem's groups are louder
   than the coincident toms by 3 to 5 dB and the kick stem's second group is
   4.7 dB under the snare's hits: neither is bleed by level.
3. *Sounds like it.* The two clean hits are at most 2.0 apart. Chosen between
   the grouping threshold (1.5: the same sound) and the nearest distinct
   sounds measured (2.5 over a bleed bed, 2.8 for closed against open hat).
   Without a clean hit on either side the test is not made and the group is
   not marked; the evidence says so.

**A leak of a constant pattern: co-located.** When the patterns give the
coincidence away, the test above cannot tell a leak from two parts that share a
pattern: a kick on every beat in one stem and a copy of it 20 dB down on every
beat in another has an expected rate of 100 %, `p` is 1 whatever the group is,
and the 25-point margin cannot be met. That is not evidence either way, and it
was also how a constant kick leak in the snare stem went unmarked and was named
`snare 2`. When the expected rate is **at least 90 %** the group is judged on
level and likeness alone, each stricter, and the evidence says "co-located:
chance cannot tell": at least 90 % of its hits coincide, the other group is
louder by **at least 12 dB** in at least 90 % of the coincident pairs (a
separator leak is 15 to 30 dB down; 8 dB under is a part doubling another, and
the 6 dB of the ordinary test would mark it), and the clean hits are within the
grouping threshold (**1.5**: the same sound, not a similar one). A different
sound that shares the pattern fails the likeness. **The 1.5 can miss a leak that
the separator has low-passed heavily:** the same snare leak low-passed at 1000 and
400 Hz reads 1.24 and 1.39 against the full-band copy and is marked, at 200 Hz it
reads 1.58 and is not. Such a group stays an unmarked, neutral group with the
numbers in its `why`.

The same strict limits (12 dB in 90 %, 1.5) apply when the **chance over the
group's own bars** is 90 % or more (the other group is there at the group's 16ths
in nearly every bar the group plays in), though the coincidence test stays the
ordinary one: a sectional tom fill copied 20 dB down is still marked, the same sound
8 dB under, or a 1.79-apart one, is not. The evidence says `strict`. The numbers are chosen, from
the cases built for them (20 dB and 8 dB down, a kick against a hat); on Flare
no group has an expected rate of 90 %, so nothing changes there.

**One source split over several groups.** A separator, or the grouping, can split one
source into several groups of its stem, so that no single group coincides enough.
A group that fails the first test against every group of another stem is also
tested against the union of that stem's groups that were not marked (at least two):
the chance rate is computed the same way from the 16ths where any of them plays.
It never marks on coincidence alone: the level is read over all coincident pairs,
each hit at its own level, and the clean hit must be within 2.0 of the clean hit
of every group a coincident pair belongs to (the furthest counts). It is reported
separately (`union` in the answer; the evidence says "the union of groups ...").

A marked group stays in the output with its `evidence` (hits that coincide,
`rate`, `expected`, `p`, `louder_db`, `distance`) and is left out of the stem's
roles. A group that is not marked says which test it failed and the numbers. A
group with one to three hits can hardly pass the first test: three hits where
the other plays in a quarter of the bars is a 1.6 % chance. A lone bled hit
(the Flare replica's 81 Hz hit in the kick stem) is not found this way; it stays
a neutral group. Why a group was not marked is in `mark_bleed`'s `why` and
`evidence` for that group, not in `name_roles`: there its `reason` is the one
for not being named (or "marked as bleed: ..." when it is).

