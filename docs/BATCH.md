# Overnight batch

`roadie batch` runs the whole pipeline (`split -d`, `sections`, `als`) over a list
of local files, one at a time, unattended. Each track is processed in a scratch
dir, uploaded with `rclone`, checked, and deleted, so local disk stays around
1 GB however long the list is.

```
roadie batch plan Studies References --out manifest.tsv     # inventory; writes manifest.excluded.tsv too
roadie batch run manifest.tsv --dry-run                     # what it would do; touches nothing
caffeinate -i nohup roadie batch run manifest.tsv > run.log 2>&1 &
roadie batch verify-sources manifest.tsv                    # exit 1 if any source changed
```

- `plan` takes one or more root folders (a root's name is its `tree`, e.g. `Studies`). It skips
  anything under `*_stems`, `* Project`, `Samples` and `Backup`, and lists every exclusion with its
  reason in `manifest.excluded.tsv`: pCloud `(conflicted)` copies, stem packs (2 or more audio files in a
  folder named for stems or a remix pack, or 5 or more that are mostly `.aif`), tracks already processed (a sibling
  `<name>_stems` folder), and duplicates. Duplicates are matched on a normalised title (BPM/key prefixes,
  YouTube ids and artist prefixes dropped; "Kolsch Full Circle Moment" equals "Full Circle Moment"), and
  versions such as remixes stay distinct. The best copy is kept: lossless over m4a over ogg over mp3,
  then root order. Existing `*_stems` folders become `topup` rows. `--dest` sets the pCloud base
  (default `pcloud:Music/Production/Roadie`); a track lands in `<dest>/<tree>/<subfolder>/<track>/`.
- `run` does `topup` rows first, then `new` rows in manifest order. A `topup` row only adds a missing
  `sections.json` and a missing `.als`; it never passes `-f`, never overwrites an `.als` and never
  deletes anything (`roadie als` does add its `Ableton Project Info/`, and `sections` may refresh a stale
  `alignment.json`).
  A `new` row: skip if the destination holds `.roadie-batch-ok`, the marker written only after a passing
  check; stop if free disk is under `--floor-gb` (4, exit 4); `split --file <src> -d` into
  `<work>/<id>` (the separator's temp files inside it too); `sections`; `als`; delete the 24-bit working wav;
  `rclone copy`, then `rclone check --one-way`, then the marker; only then delete the work dir. A step that
  fails or times out (3x duration + 10 min for the split, 15 min for the rest) marks the row `failed`
  and the run moves on. A failed upload or check keeps the work dir and stops the run (exit 3), so
  failed tracks cannot pile up on disk; the next run uploads that kept work dir without re-splitting. Options: `--dest` (override the manifest), `--work`, `--state`,
  `--logs`, `--report-dir`, `--limit N`, `--only-mode new|topup`, `--dry-run`.
- State is `~/.cache/roadie-batch/state.jsonl` (one line per event); a re-run skips rows whose last
  status is `done`, so a second night resumes. Per-row logs are in `~/.cache/roadie-batch/logs/`. A lock
  file in the work dir's parent refuses a second concurrent run. On exit, including SIGINT/SIGTERM/SIGHUP
  (the current child process group is killed) it writes `batch-report-YYYY-MM-DD.md` to `--report-dir`:
  start and end, counts per status, one line per row, minimum free disk, and the rows still remaining.
- Safety: sources are only read. Every delete goes through one helper that refuses any path not strictly
  inside the work dir. `ROADIE_BATCH_ROADIE` and `ROADIE_BATCH_RCLONE` point at other `roadie` / `rclone`
  commands (used by the tests).
