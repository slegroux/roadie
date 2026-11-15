"""How stems2live and session_view pack many Live operations into few requests.

A request to the AbletonLOM Remote Script costs about 400 ms whatever it carries: it waits 3-4 of
Live's main-thread ticks (measured 2026-11-15 on a held connection: one `get` and a `get_batch` of 25
reads took the same 400 ms). A ten-stem load with a Session grid was 439 requests, so the time is
the request count. Reads that do not depend on each other go out together in one `get_batch`;
writes and calls in one `transaction`, which Live also records as ONE undo step.

WHAT IS NEVER PUT IN THE SAME REQUEST, because each was its own request before and something
measured depended on Live having finished the first before the second ran:
  - an op that addresses an object a create or delete in the same request made or moved (the new
    track, clip or scene is addressed in the next request);
  - a read of a property a write in the same request changed (read-backs stay a request later);
  - `start_marker` and the loop markers after `looping` or `warping` was written, and `looping`
    after a marker was written (the trim and the units it is in depend on that order);
  - anything after a move of the play head (see stems2live.settle).
Those splits are in the callers; this module only packs what they hand it.

Live runs a request on its main thread and gives up waiting after 15 s (MAIN_THREAD_TIMEOUT in the
Remote Script's __init__.py), and a request it gave up on still runs. So a request carries at most
HEAVY_PER_REQUEST ops that load audio or warp a clip and at most OPS_PER_REQUEST ops in all.
"""

# Loading a file into a clip, and turning warping on (Live may analyse the audio to place its own
# markers), are the only ops here whose main-thread time grows with the audio. Six at an assumed
# worst of 1 s each leaves the request at 6 s, under half of Live's 15 s.
HEAVY_PER_REQUEST = 6
# Property sets, reads and light calls are well under a millisecond each on Live's main thread.
OPS_PER_REQUEST = 256

_HEAVY_CALLS = ("create_audio_clip",)


def seq(ops, stop=True):
    """`ops` as ONE op of a batch: a nested transaction that runs them in order and, with `stop`,
    halts at the first that fails, the way the one-request-each code gave up on an item at its
    first failed request. A nested transaction adds no undo step of its own."""
    return {"op": "transaction", "ops": list(ops), "stop_on_error": stop,
            "rollback_on_error": False}


def weight(op):
    """(ops, heavy ops) one op puts in a request, counting into nested transactions."""
    if op.get("op") == "transaction":
        parts = [weight(o) for o in op["ops"]]
        return sum(n for n, _h in parts), sum(h for _n, h in parts)
    heavy = (op.get("op") == "warp_markers_set"
             or (op.get("op") == "call" and op.get("function") in _HEAVY_CALLS)
             or (op.get("op") == "set" and op.get("property") == "warping"))
    return 1, int(heavy)


def pack(ops):
    """Split `ops` into consecutive requests within the limits, keeping their order. An op over a
    limit on its own still goes, alone."""
    out, cur, n, h = [], [], 0, 0
    for op in ops:
        dn, dh = weight(op)
        if cur and (n + dn > OPS_PER_REQUEST or h + dh > HEAVY_PER_REQUEST):
            out.append(cur)
            cur, n, h = [], 0, 0
        cur.append(op)
        n, h = n + dn, h + dh
    if cur:
        out.append(cur)
    return out


def split_error(text):
    """(type, message) from a failed batch entry's "<Type>: <message>" (the Remote Script writes
    `"%s: %s" % (type(e).__name__, e)`, the same two fields an error envelope carries). An entry
    with no type in front, "unknown op 'x'", comes back with type ""."""
    text = str(text or "")
    head, sep, rest = text.partition(": ")
    if sep and head.isidentifier():
        return head, rest
    return "", text


def vector_len(value):
    """The length of a list property as `get` returns it: the items, or {"__vector__", "count"}
    past 64 of them. The same number op `count` gives."""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and value.get("__vector__"):
        return int(value["count"])
    raise TypeError("not a list: %r" % (value,))
