"""stems2live and session_view reach Live only through `sideman.client.Live`.

Two things are pinned here, and neither needs Live or a socket server:

1. WHAT GOES OUT. `test_live_wire_golden.json` is what the two modules sent
   before they moved to the shared client, captured by running the OLD code
   (commit df6c85a, its own raw sockets) against a loopback server on these
   exact inputs: every op, its params and the order, plus what the call
   returned and printed. The tests below replay the same inputs through the
   current code with `Live.request` replaced by a fake that records its
   arguments, and compare. Expectations come from the old code, not from the
   new code's own output.

   Not compared: the request `id` (the client sends "py", the old code counted
   up; the remote script only echoes it back) and the `id` key of the envelope
   `stems2live.lom` returns.

   Since the batching (live_batch.py), the functions that read or write many
   things send them in `get_batch` and `transaction` requests, so their wire
   changed by design and their golden entries are no longer compared request
   for request. They are kept in the golden file as the record of what was
   sent. What replaces them as the proof is test_batch_live.py: the old code
   and the new each run a whole load against a state model of Live, and the
   Set they leave is compared. Where a batched function still sends exactly
   the old operations in the old order, only grouped, the test below says so
   against the golden calls.

2. HOW FAILURES SURFACE. A refused connection, a closed one, a timeout and an
   error reply each ended the old code a particular way; they still do.

The last test fails if either module imports `socket` again.
"""

import ast
import json
import os
import socket

import pytest

import session_view as V
import stems2live as S
from sideman.client import Live, LiveError

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "test_live_wire_golden.json")) as _f:
    GOLDEN = json.load(_f)


def _generic(op, p):
    """The remote script's answers, to the extent the scenarios need them."""
    if op == "ping":
        return {"handlers": True}
    if op == "count":
        return {"count": {"tracks": 3, "scenes": 2, "arrangement_clips": 2,
                          "cue_points": 0}.get(p["child"], 1)}
    if op == "get":
        prop = p["property"]
        v = {"name": "x", "has_audio_input": True, "has_clip": True,
             "end_marker": 64.0, "start_marker": 8.0, "start_time": 3.5,
             "end_time": 31.7, "color": 255, "warping": True,
             "is_playing": False}.get(prop, 0)
        if prop == "name" and p["path"].endswith("tracks 1"):
            v = "bass"
        if prop == "name" and p["path"].endswith("tracks 2"):
            v = "ENERGY MAP"
        return {"path": p["path"], "property": prop, "value": v}
    if op == "set":
        return {"value": p["value"]}
    if op == "warp_markers_set":
        if not p:
            raise KeyError("path")
        return {"marker_count": 2, "added": 2, "removed": 0, "remove_failed": 1}
    return {"result": None}


def _unknown_warp(op, p):
    if op == "warp_markers_set":
        raise ValueError("unknown op 'warp_markers_set'; known: []")
    return _generic(op, p)


def _no_end_time(op, p):
    if op == "get" and p["property"] == "end_time":
        raise AttributeError("end_time")
    return _generic(op, p)


def _unknown_op(op, p):
    raise ValueError("unknown op 'warp_markers_set'")


def _no_handlers(op, p):
    return {"handlers": False, "handler_error": "boom"}


def _ping_fails(op, p):
    raise RuntimeError("nope")


class Timeout(Exception):
    """Named like the Remote Script's own timeout reply: it answers
    `{"ok": false, "error": {"type": "Timeout", ...}}` when Live's main thread
    stays blocked. The fake client turns it into LiveError("Timeout", ...)
    with no socket timeout behind it."""


def _remote_timeout(op, p):
    raise Timeout("Live's main thread was busy for 15s")


@pytest.fixture
def wire(monkeypatch):
    """Replace `Live.request`; returns (install, calls). `install(reply)` sets
    the remote script's behaviour. A raising reply becomes a LiveError, which
    is how the real client reports an error envelope."""
    calls = []
    state = {"reply": _generic}

    def request(self, op, params=None):
        # Round-trip through JSON: what the socket would have carried, so a
        # tuple arrives as a list and a non-serialisable param fails here.
        calls.append([op, json.loads(json.dumps(params or {}))])
        try:
            return state["reply"](op, params or {})
        except Exception as e:                           # noqa: BLE001
            raise LiveError(type(e).__name__, str(e))

    monkeypatch.setattr(Live, "request", request)

    def install(reply):
        state["reply"] = reply
    return install, calls


NOTES = [{"pitch": 84, "start_time": 3.0, "duration": 32.0, "velocity": 90}]
PERIOD = 60.0 / 125.003

# name -> (reply behaviour, callable on a Bridge-or-nothing)
S2L = {
    "s2l.lom.get": (_generic, lambda: S.lom("get", path="live_set tracks 0", property="name")),
    "s2l.lom.call_noargs": (_generic, lambda: S.lom("call", path="live_set", function="stop_playing")),
    "s2l.lom.call_args": (_generic, lambda: S.lom("call", path="live_set", function="create_audio_track", args=[-1])),
    "s2l.lom.set": (_generic, lambda: S.lom("set", path="live_set", property="tempo", value=126.0)),
    "s2l.lom.notes_add": (_generic, lambda: S.lom("notes_add", path="c", notes=[{"pitch": 84, "start_time": 0.0, "duration": 4.0, "velocity": 90}])),
    "s2l.probe_bridge.ok": (_generic, S.probe_bridge),
    "s2l.probe_bridge.nowarp": (_unknown_warp, S.probe_bridge),
    "s2l.lom.remote_timeout": (_remote_timeout, lambda: S.lom("call", path="live_set", function="create_audio_track", args=[-1])),
}

SV = {
    "sv.get": lambda b: b.get("live_set tracks 0", "name"),
    "sv.set": lambda b: b.set("live_set scenes 0", "name", "Intro"),
    "sv.call_noargs": lambda b: b.call("live_set", "x"),
    "sv.call_args": lambda b: b.call("live_set", "create_scene", [-1]),
    "sv.call_confirm": lambda b: b.call("live_set tracks 0 clip_slots 0", "delete_clip", confirm=True),
    "sv.count": lambda b: b.count("live_set", "tracks"),
    "sv.many_in_one_run": lambda b: [b.count("live_set", "tracks"), b.get("live_set", "tempo"), b.get("live_set", "tempo")],
}


def _as_json(x):
    return json.loads(json.dumps(x))


@pytest.mark.parametrize("name", sorted(S2L))
def test_stems2live_sends_what_the_raw_client_sent(name, wire, capsys):
    install, calls = wire
    reply, fn = S2L[name]
    install(reply)
    ret = fn()
    gold = GOLDEN[name]
    assert calls == gold["calls"]
    assert _as_json(ret) == gold["ret"] or (
        isinstance(ret, dict) and {k: v for k, v in ret.items() if k != "id"}
        == {k: v for k, v in gold["ret"].items() if k != "id"})
    assert capsys.readouterr().out == gold["out"].replace("{PORT}", str(S.PORT))


@pytest.mark.parametrize("name", sorted(SV))
def test_session_view_sends_what_the_raw_client_sent(name, wire):
    install, calls = wire
    install(_generic)
    ret = SV[name](V.Bridge())
    gold = GOLDEN[name]
    assert calls == gold["calls"]
    assert _as_json(ret) == gold["ret"]


# Batched, and still the old operations in the old order: (golden names, reply, call[, reads
# added at the end since]). The new call's return is the old one's, in the batched function's
# shape.
REGROUPED = {
    "s2l.trim_clip_start.warped": (
        ["s2l.trim_clip_start.warped"], _generic,
        lambda: S.trim_clip_starts([(2, 1)], 0.237, PERIOD, [])[0],
        # added since: `warping` read again with the read-back, to catch a unit change
        [["get", {"path": "live_set tracks 2 arrangement_clips 1", "property": "warping"}]]),
    "s2l.apply_warp_map": (
        ["s2l.apply_warp_map"], _generic,
        lambda: S.warp_outcome(S.lom_ops([S.warp_map_op(2, 1, [[0, 0.0], [4, 1.92]])])[0])),
    "s2l.apply_warp_map.unsupported": (
        ["s2l.apply_warp_map.unsupported"], _unknown_op,
        lambda: S.warp_outcome(S.lom_ops([S.warp_map_op(2, 1, [[0, 0.0]])])[0])),
    "sv.ensure_scenes": (
        ["sv.ensure_scenes"], _generic,
        lambda: V.ensure_scenes(V.Bridge(), [{"scene": i, "name": "S%d" % i} for i in range(4)])),
    "sv.write_clip": (
        ["sv.write_clip"], _generic,
        lambda: V.write_clips(V.Bridge(), [{"track": 1, "scene": 0, "path": "/tmp/a.flac",
                                            "start": 8.0, "end": 40.0, "name": "Intro",
                                            "color": 255}])[0]),
    "sv.clear_slot+write_clip": (
        ["sv.clear_slot", "sv.write_clip"], _generic,
        lambda: V.write_clips(V.Bridge(), [{"track": 1, "scene": 0, "path": "/tmp/a.flac",
                                            "start": 8.0, "end": 40.0, "name": "Intro",
                                            "color": 255}], replace=True)[0]),
}


@pytest.mark.parametrize("name", sorted(REGROUPED))
def test_batched_functions_still_send_the_old_operations_in_the_old_order(name, wire, capsys):
    """Flattened, the batches carry what the old code sent one request at a time: the same ops,
    the same params, the same order. The golden calls are the old code's, not regenerated."""
    import fake_live
    install, calls = wire
    golden, reply, fn, added = (REGROUPED[name] + ([],))[:4]
    install(fake_live.with_batches(reply))
    ret = fn()
    sent = [[op, {k: v for k, v in p.items() if k != "op"}]
            for op, params in calls for op, p in fake_live.leaves(op, params)]
    assert sent == [c for g in golden for c in GOLDEN[g]["calls"]] + added
    assert _as_json(ret) == GOLDEN[golden[-1]]["ret"]
    assert capsys.readouterr().out == "".join(GOLDEN[g]["out"] for g in golden)
    assert len(calls) < len(sent) or calls[0][0] == "transaction", "grouped in batches"


@pytest.mark.parametrize("name, reply", [
    ("s2l.probe_bridge.nohandlers", _no_handlers),
    ("s2l.probe_bridge.pingerror", _ping_fails),
])
def test_probe_bridge_still_exits_with_the_old_messages(name, reply, wire):
    """Expected text is what the old code exited with (see the golden run)."""
    install, calls = wire
    install(reply)
    with pytest.raises(SystemExit) as e:
        S.probe_bridge()
    if "nohandlers" in name:
        assert str(e.value) == (
            "AbletonLOM is listening but its handlers failed to load:\nboom\n"
            "Fix handlers.py and send {\"op\": \"reload\"} — no Live restart "
            "needed.")
    else:
        assert str(e.value) == "AbletonLOM did not answer ping: nope"
    assert calls == [["ping", {}]]


# --------------------------------------------------------------- failures

def _raises(exc):
    def reply(op, p):
        raise exc
    return reply


@pytest.fixture
def failing(monkeypatch):
    """`Live.request` that fails the way the real client does for a given
    connection-level problem (type strings are the client's own)."""
    def fail(type_, message):
        def request(self, op, params=None):
            if type_ == "Timeout":
                # The client raises it from inside `except socket.timeout`.
                try:
                    raise socket.timeout("timed out")
                except socket.timeout:
                    raise LiveError(type_, message)
            raise LiveError(type_, message)
        monkeypatch.setattr(Live, "request", request)
    return fail


def test_stems2live_nothing_listening_exits_with_the_setup_instructions(failing):
    failing("ConnectionRefused", "nothing listening")
    with pytest.raises(SystemExit) as e:
        S.lom("ping")
    assert str(e.value) == (
        "Nothing listening on 127.0.0.1:%d.\n"
        "Open Ableton Live and set Preferences -> Link, Tempo & MIDI ->\n"
        "Control Surface = AbletonLOM (Input/Output = None)." % S.PORT)


def test_stems2live_a_connection_closed_before_a_reply_is_a_runtime_error(failing):
    failing("ConnectionClosed", "closed")
    with pytest.raises(RuntimeError) as e:
        S.lom("ping")
    assert str(e.value) == "no reply from Live on port %d" % S.PORT


def test_stems2live_a_timeout_is_a_timeout_error(failing):
    failing("Timeout", "slow")
    with pytest.raises(TimeoutError):
        S.lom("ping")


def test_stems2live_lives_own_timeout_reply_fails_one_operation_not_the_run(wire):
    """REVIEW, blocking: the Remote Script answers `Timeout` when Live's main
    thread was blocked. The old code returned that as an error envelope and the
    run went on; only a timeout on the socket raised."""
    install, calls = wire
    install(_remote_timeout)
    r = S.lom("call", path="live_set", function="create_audio_track", args=[-1])
    assert r == {"ok": False, "error": {"type": "Timeout",
                                        "message": "Live's main thread was busy for 15s"}}
    # The track list reads (two tracks, so none to reuse past index 5); the
    # creates, one batch now, get Live's Timeout reply.
    install(lambda op, p: ({"results": [{"ok": True, "value": [{}, {}]}]
                            + [{"ok": False, "error": "IndexError: out of range"}]
                            * (len(p["specs"]) - 1)} if op == "get_batch"
                           else _remote_timeout(op, p)))
    # The old code exited with "create_audio_track failed: ...". The creates are one batch
    # now, and Live keeps running a task it gave up waiting on, so the tracks may still be
    # made: the run ends saying so (test_batch_live has the model where they land).
    with pytest.raises(SystemExit) as e:
        S.audio_track_targets(5, 2)
    assert str(e.value).startswith("Live did not answer the last batch of changes "
                                   "(Live's main thread was busy for 15s)")


def test_session_view_lives_own_timeout_reply_is_a_lom_error_the_run_survives(wire):
    """Old: LomError, which the per-clip `except LomError` counts and carries
    on from. Not the TimeoutError that ends the run."""
    install, _ = wire
    install(_remote_timeout)
    with pytest.raises(V.LomError) as e:
        V.Bridge().get("a", "b")
    assert str(e.value).startswith("get a: ")
    assert "Live's main thread was busy for 15s" in str(e.value)


def test_session_view_a_reset_mid_request_is_not_relabelled_nothing_listening(monkeypatch):
    for exc in (ConnectionResetError("reset"), BrokenPipeError("pipe")):
        def request(self, op, params=None, _exc=exc):
            raise _exc
        monkeypatch.setattr(Live, "request", request)
        with pytest.raises(type(exc)):
            V.Bridge().get("a", "b")


def test_stems2live_an_error_reply_comes_back_as_an_envelope_not_an_exception(wire):
    install, _ = wire
    install(_unknown_op)
    r = S.lom("get", path="a", property="b")
    assert r == {"ok": False, "error": {"type": "ValueError",
                                        "message": "unknown op 'warp_markers_set'"}}
    assert S.lom_err(r) == "unknown op 'warp_markers_set'"
    with pytest.raises(RuntimeError) as e:
        S.lom_get("a", "b")
    assert str(e.value) == "get a.b: unknown op 'warp_markers_set'"
    with pytest.raises(RuntimeError) as e:
        S.lom_count("a", "b")
    assert str(e.value) == "count a.b: unknown op 'warp_markers_set'"


def test_stems2live_on_a_closed_port_the_real_client_still_ends_in_the_old_exit():
    """No fake: the real client against a port with nothing on it. Pins the
    error type string `lom` keys on to what `sideman.client` really raises."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    old = S.PORT
    S.PORT = port
    try:
        with pytest.raises(SystemExit) as e:
            S.lom("ping")
    finally:
        S.PORT = old
    assert str(e.value).startswith("Nothing listening on 127.0.0.1:%d." % port)


def test_session_view_nothing_listening_is_a_lom_error_with_the_setup_hint(failing):
    failing("ConnectionRefused", "nothing listening")
    with pytest.raises(V.LomError) as e:
        V.Bridge(port=9999).get("a", "b")
    assert str(e.value) == (
        "nothing listening on 127.0.0.1:9999.\nOpen Ableton Live and set "
        "Preferences -> Link, Tempo & MIDI -> Control Surface = AbletonLOM.")


def test_session_view_a_closed_connection_is_a_lom_error(failing):
    failing("ConnectionClosed", "closed")
    with pytest.raises(V.LomError) as e:
        V.Bridge().get("a", "b")
    assert str(e.value) == "Live closed the connection during 'get'"


def test_session_view_a_timeout_is_a_timeout_error_not_a_lom_error(failing):
    """As before: the run does not mistake a slow Live for a missing one."""
    failing("Timeout", "slow")
    with pytest.raises(TimeoutError) as e:
        V.Bridge().get("a", "b")
    assert not isinstance(e.value, V.LomError)


def test_session_view_an_unreachable_host_is_a_lom_error(monkeypatch):
    """A connect that fails with something other than a refusal reaches the
    caller as a raw OSError from the client; the old Bridge turned any OSError
    at connect into the 'nothing listening' LomError, and still does."""
    def request(self, op, params=None):
        raise OSError("No route to host")
    monkeypatch.setattr(Live, "request", request)
    with pytest.raises(V.LomError) as e:
        V.Bridge().get("a", "b")
    assert "nothing listening on 127.0.0.1:9878 (No route to host)" in str(e.value)


def test_session_view_a_refusal_names_the_op_the_path_and_the_reason(wire):
    install, _ = wire
    install(_unknown_op)
    with pytest.raises(V.LomError) as e:
        V.Bridge().get("live_set tracks 99", "name")
    assert str(e.value).startswith("get live_set tracks 99: ")
    assert "unknown op 'warp_markers_set'" in str(e.value)


def test_session_view_on_a_closed_port_the_real_client_still_gives_a_lom_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(V.LomError) as e:
        V.Bridge(port=port).get("a", "b")
    assert str(e.value).startswith("nothing listening on 127.0.0.1:%d" % port)


# ------------------------------------------------- the real client, loopback

class _Loopback:
    """One-shot-per-connection server for the REAL `Live`: replies with an error
    envelope, stays silent, or hangs up, so the adapters meet the client's own
    exceptions rather than a fake's."""

    def __init__(self, mode, error=None):
        import threading
        self.mode, self.error = mode, error
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.held = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if self.mode == "error":
                conn.sendall(json.dumps({"id": "py", "ok": False,
                                         "error": self.error}).encode() + b"\n")
                conn.close()
            elif self.mode == "close":
                conn.close()
            else:                                   # silent: hold it open
                self.held.append(conn)

    def close(self):
        self.sock.close()
        for c in self.held:
            c.close()


@pytest.fixture
def loopback(monkeypatch):
    made = []

    def start(mode, error=None):
        srv = _Loopback(mode, error)
        made.append(srv)
        monkeypatch.setattr(S, "PORT", srv.port)
        monkeypatch.setattr(S, "LOM_TIMEOUT", 0.3)
        return srv
    yield start
    for srv in made:
        srv.close()


def test_loopback_a_silent_live_is_a_timeout_error_in_both_modules(loopback):
    srv = loopback("silent")
    with pytest.raises(TimeoutError):
        S.lom("ping")
    with pytest.raises(TimeoutError) as e:
        V.Bridge(port=srv.port, timeout=0.3).get("a", "b")
    assert not isinstance(e.value, V.LomError)


def test_loopback_a_connection_closed_before_a_reply(loopback):
    srv = loopback("close")
    with pytest.raises(RuntimeError) as e:
        S.lom("ping")
    assert str(e.value) == "no reply from Live on port %d" % srv.port
    with pytest.raises(V.LomError) as e:
        V.Bridge(port=srv.port, timeout=0.3).get("a", "b")
    assert str(e.value) == "Live closed the connection during 'get'"


def test_loopback_error_replies_including_lives_own_timeout(loopback):
    for err in ({"type": "ValueError", "message": "nope"},
                {"type": "Timeout", "message": "main thread blocked"}):
        srv = loopback("error", err)
        assert S.lom("get", path="a", property="b") == {"ok": False, "error": err}
        with pytest.raises(V.LomError) as e:
            V.Bridge(port=srv.port, timeout=0.3).get("a", "b")
        assert err["message"] in str(e.value)


def test_the_address_and_the_120_second_timeout_are_pinned(monkeypatch):
    seen = []

    def request(self, op, params=None):
        seen.append((self.host, self.port, self.timeout))
        return {}
    monkeypatch.setattr(Live, "request", request)
    S.lom("ping")
    V.Bridge().request("ping")
    assert seen == [("127.0.0.1", S.PORT, 120.0), ("127.0.0.1", 9878, 120.0)]


@pytest.mark.parametrize("script", ["stems2live.py", "session_view.py"])
def test_a_venv_without_sideman_says_what_to_run(script):
    """An old .venv built before sideman was a dependency: a one-line exit,
    not a ModuleNotFoundError traceback."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['sideman'] = None; "
            "sys.path.insert(0, %r); import %s" % (HERE, script[:-3]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120, check=False,
                       env=dict(os.environ, ROADIE_REEXEC="1"))
    assert r.returncode == 1
    assert r.stderr.strip() == "sideman is not installed in .venv: run ./install.sh"


# ------------------------------------------------------------ no raw socket

@pytest.mark.parametrize("module", [S, V], ids=["stems2live", "session_view"])
def test_the_modules_do_not_import_socket(module):
    """The protocol lives in sideman.client; a second copy here is what this
    change removed."""
    with open(module.__file__) as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "socket" not in imported
    assert not hasattr(module, "socket")
