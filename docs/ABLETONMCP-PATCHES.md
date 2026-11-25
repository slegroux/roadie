# Legacy: AbletonMCP patches

Roadie no longer uses these; it talks to Live through [Sideman](https://github.com/slegroux/sideman). They are kept because they are the reference for
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
main-thread task (see [Section locators](DESIGN.md#section-locators---locators)), and an endpoint that looks like it works
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
