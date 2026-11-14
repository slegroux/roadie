"""Roadie's MCP server (roadie_mcp.py): four tools, the same output as Sideman main's tools.

sample_analyze, sample_compare and reference_profile came from Sideman's server with their names,
arguments, defaults and output; session_compare is Sideman's mix_report with a reference profile. The
fixtures are test_analysis_equivalence's (a reference track with six stems, a capture of it, a take of
three hits), and so are the goldens for what the tools return: Sideman main bc9ba08's tools return
exactly the text its library wrote there (json.dumps of the result, indent 2). What those goldens do not
hold, Sideman main's own server wrote to mcp_golden/sideman_main.json: its tools' input schemas and the
summary its reference_profile returns. Regenerate that file from a `git archive` of Sideman main:

    PYTHONPATH=<sideman main checkout> .venv/bin/python test_roadie_mcp.py --write mcp_golden

Tools are called through the server (`mcp.call_tool`, which validates arguments as a client's call
would), once over stdio through `roadie mcp`. A LOM-path source goes through a fake
`sideman.client.Live`: no Live, no network.
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

import test_analysis_equivalence as E

HERE = pathlib.Path(__file__).resolve().parent
MCP_GOLDEN = HERE / "mcp_golden" / "sideman_main.json"
TOOLS = {"sample_analyze", "sample_compare", "reference_profile", "session_compare"}
CLIP = "live_set tracks 2 clip_slots 0 clip"


def fixtures(d):
    """The equivalence fixtures in `d`: (stems for reference_profile, the capture manifest, the take)."""
    d = pathlib.Path(d)
    (d / "ref").mkdir()
    (d / "ours").mkdir()
    ref = E._stems(d / "ref", 80.0, -6.0, 0.0, 1.0)
    ours = E._stems(d / "ours", 80.0 * 2 ** (-2 / 12), -2.0, 0.6, 1.0)
    stems = {"kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"], "hh": ref["hh"],
             "snare": ref["snare"], "drums": [ref["hh"], ref["snare"]]}
    manifest = {"tempo": E.TEMPO, "offset_s": E.T0, "sections": E.SECTIONS,
                "sources": {"master": ours["mix"], "kick": ours["kick"], "bass": ours["bass"],
                            "synth": ours["synth"], "hh": ours["hh"], "snare": ours["snare"]}}
    take = np.concatenate([E._decay(100.0, 0.08, 0.7), E._decay(100.0, 0.08, 0.7) * 0.7, E._decay(120.0, 0.05, 0.7)])
    sf.write(str(d / "take.wav"), np.stack([take, take], 1).astype(np.float32), E.SR, subtype="FLOAT")
    return ref["mix"], stems, manifest, str(d / "take.wav")


@pytest.fixture(scope="module")
def S():
    import roadie_mcp
    return roadie_mcp


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("mcp")
    mix, stems, manifest, take = fixtures(d)
    return d, mix, stems, manifest, take


def call(S, name, **args):
    """The tool's text, called the way a client's call arrives (arguments validated by the server)."""
    res = asyncio.run(S.mcp.call_tool(name, args))
    assert not res.is_error and len(res.content) == 1, res
    return res.content[0].text


def failure(S, name, **args):
    """The error a client gets: a ToolError carrying the diagnosis, never the SDK's crash wrapper."""
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
    with pytest.raises(ToolError) as e:
        asyncio.run(S.mcp.call_tool(name, args))
    assert not isinstance(e.value, UnexpectedToolError), repr(e.value.__cause__)
    return str(e.value)


def same(text, golden_name, d):
    assert json.loads(E._norm(text, d)) == json.loads((E.GOLDEN / golden_name).read_text()), golden_name


def test_the_server_lists_exactly_four_tools_with_sidemans_arguments(S):
    tools = {t.name: t for t in asyncio.run(S.mcp.list_tools())}
    assert set(tools) == TOOLS
    assert all((t.description or "").strip() for t in tools.values())
    want = json.loads(MCP_GOLDEN.read_text())["input_schemas"]
    for name in ("sample_analyze", "sample_compare", "reference_profile"):
        assert tools[name].input_schema == want[name], name
    got, mix = tools["session_compare"].input_schema, want["mix_report"]
    assert got["properties"]["manifest"] == mix["properties"]["manifest"]
    assert got["properties"]["out_dir"] == mix["properties"]["out_dir"]
    assert got["required"] == ["manifest", "reference"] and "default" not in got["properties"]["reference"]


def test_each_tool_gives_what_sidemans_tool_gives_on_the_fixtures(S, made):
    d, mix, stems, manifest, take = made
    summary = call(S, "reference_profile", mix=mix, stems=stems, tempo=E.TEMPO, t0=E.T0, sections=E.SECTIONS,
                   out=str(d / "ref.profile.json"))
    same((d / "ref.profile.json").read_text(), "profile.json", d)
    assert json.loads(E._norm(summary, d)) == json.loads(MCP_GOLDEN.read_text())["reference_profile_summary"]
    report = call(S, "session_compare", manifest=manifest, reference=str(d / "ref.profile.json"), out_dir=str(d / "out"))
    same(report, "mix_report.json", d)
    assert E._norm((d / "out" / "mix_report.md").read_text(), d) == (E.GOLDEN / "mix_report.md").read_text()
    same(call(S, "sample_analyze", source=take, hit=0), "oneshot_profile.json", d)
    same(call(S, "sample_compare", a=take, b=take, hit_a=0, hit_b=2), "oneshot_compare.json", d)


def test_session_compare_takes_the_profile_as_a_path_its_json_or_the_dict(S, made):
    d, mix, stems, manifest, take = made
    prof = d / "again.profile.json"
    call(S, "reference_profile", mix=mix, stems=stems, tempo=E.TEMPO, t0=E.T0, sections=E.SECTIONS, out=str(prof))
    for reference in (prof.read_text(), json.loads(prof.read_text())):
        r = json.loads(call(S, "session_compare", manifest=manifest, reference=reference, out_dir=str(d / "o2")))
        assert r["reference_kind"] == "profile" and all(s["elements"]["paired"] for s in r["sections"])


def test_session_compare_refuses_an_audio_reference_or_none_and_points_to_sideman(S, made):
    d, mix, stems, manifest, take = made
    msg = failure(S, "session_compare", manifest=manifest, reference=mix, out_dir=str(d / "no"))
    assert "needs a reference profile" in msg and "Sideman's mix_report" in msg
    assert not (d / "no").exists()
    assert "reference" in failure(S, "session_compare", manifest=manifest)                 # required
    with pytest.raises(S.AnalysisError, match="Sideman's mix_report"):
        S.session_compare(manifest, None)


def test_errors_are_tool_errors_with_the_diagnosis(S, made, tmp_path):
    d, mix, stems, manifest, take = made
    # stems must be a mapping: the server refuses a list before the tool runs; called directly, as Sideman's was
    msg = failure(S, "reference_profile", mix=mix, stems=[stems["kick"]], tempo=E.TEMPO, t0=E.T0, sections=E.SECTIONS)
    assert "stems" in msg and "valid dictionary" in msg
    with pytest.raises(S.AnalysisError, match="cannot build the reference profile"):
        S.reference_profile(mix, [stems["kick"]], E.TEMPO, E.T0, E.SECTIONS)
    msg = failure(S, "reference_profile", mix=mix, stems={"kick": stems["kick"]}, tempo=E.TEMPO, t0=E.T0,
                  sections=E.SECTIONS + E.SECTIONS[:1], out=str(tmp_path / "dup.profile.json"))
    assert "cannot build the reference profile" in msg and "section names must be unique" in msg
    assert "no such profile file" in failure(S, "session_compare", manifest=manifest,
                                             reference=str(tmp_path / "nope.json"))
    bad = json.loads((E.GOLDEN / "profile.json").read_text())
    bad["sections"][0]["roles"] = []
    assert "rebuild it with reference_profile" in failure(S, "session_compare", manifest=manifest,
                                                          reference=json.dumps(bad))
    assert "out of range" in failure(S, "sample_analyze", source=take, hit=5)
    assert "hit must be" in failure(S, "sample_compare", a=take, b=take, hit_a="best")
    assert "no such audio file" in failure(S, "sample_analyze", source=str(tmp_path / "gone.wav"))
    assert "no audio source given" in failure(S, "sample_analyze", source="  ")


def test_the_reference_profile_summary_is_a_summary(S, made, tmp_path):
    d, mix, stems, manifest, take = made
    summary = json.loads(call(S, "reference_profile", mix=mix, stems=stems, tempo=E.TEMPO, t0=E.T0,
                              sections=E.SECTIONS, out=str(tmp_path / "s.profile.json")))
    assert summary["json_path"] == str((tmp_path / "s.profile.json").resolve())
    assert "per_bar_db" not in json.dumps(summary) and len(json.dumps(summary)) < 2000      # not the profile
    assert [s["name"] for s in summary["sections"]] == ["A", "B", "C"] and summary["roles"] == list(stems)
    assert summary["sections"][1]["absent"] == ["synth"] and summary["sections"][0]["absent"] == []


class FakeLive:
    """sideman.client.Live with no Live behind it: each get of a file_path answers from FILES."""
    FILES, CALLS = {}, []

    def __init__(self, *a, **k):
        pass

    def get(self, path, property, **params):
        FakeLive.CALLS.append((path, property))
        value = FakeLive.FILES[path]
        if isinstance(value, Exception):
            raise value
        return {"path": path, "property": property, "value": value, "type": "str"}


def test_a_lom_path_source_is_resolved_through_sideman_client(S, made, monkeypatch):
    import sideman.client
    d, mix, stems, manifest, take = made
    monkeypatch.setattr(sideman.client, "Live", FakeLive)
    FakeLive.FILES = {CLIP: take, "live_set tracks 0": None,
                      "live_set tracks 9 clip_slots 0 clip": sideman.client.LiveError("IndexError", "no track 9")}
    FakeLive.CALLS = []
    same(call(S, "sample_analyze", source=CLIP, hit=0), "oneshot_profile.json", d)
    same(call(S, "sample_compare", a=CLIP, b=take, hit_a=0, hit_b=2), "oneshot_compare.json", d)
    assert FakeLive.CALLS == [(CLIP, "file_path")] * 2
    assert "has no file_path; only audio clips do" in failure(S, "sample_analyze", source="live_set tracks 0")
    assert "IndexError: no track 9" in failure(S, "sample_analyze", source="live_set tracks 9 clip_slots 0 clip")


def test_over_stdio_through_roadie_mcp(made):
    """The way a client starts it: `roadie mcp`, the protocol on stdin and stdout."""
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters
    d, mix, stems, manifest, take = made

    async def run():
        params = StdioServerParameters(command=str(HERE / "roadie"), args=["mcp"], cwd=str(d),
                                       env={"ROADIE_LOG": str(d / "usage.log")})
        async with Client(params) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            ok = await c.call_tool("sample_analyze", {"source": take, "hit": 0})
            refused = await c.call_tool("session_compare", {"manifest": manifest, "reference": mix})
            return names, ok, refused

    names, ok, refused = asyncio.run(run())
    assert names == TOOLS
    assert not ok.is_error
    same(ok.content[0].text, "oneshot_profile.json", d)
    assert refused.is_error and "Sideman's mix_report" in refused.content[0].text


def test_without_mcp_the_server_exits_with_the_install_hint():
    code = ("import runpy, sys; sys.modules['mcp'] = None; sys.argv = ['roadie_mcp.py']; "
            "runpy.run_path(%r, run_name='__main__')" % str(HERE / "roadie_mcp.py"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False,
                       env={**os.environ, "ROADIE_REEXEC": "1"})
    assert r.returncode == 1 and r.stdout == ""
    assert "needs the `mcp` package" in r.stderr and "requirements-mcp.txt" in r.stderr


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        sys.exit(__doc__)
    import mcp_server.server as SM            # Sideman main's server, found through PYTHONPATH
    assert "site-packages" not in SM.__file__, SM.__file__
    dest = pathlib.Path(sys.argv[2]).resolve()
    dest.mkdir(exist_ok=True)
    schemas = {t.name: t.input_schema for t in asyncio.run(SM.mcp.list_tools())
               if t.name in ("sample_analyze", "sample_compare", "reference_profile", "mix_report")}
    with tempfile.TemporaryDirectory() as tmp:
        mix, stems, manifest, take = fixtures(tmp)
        summary = SM.reference_profile(mix, stems, E.TEMPO, E.T0, E.SECTIONS, out=str(pathlib.Path(tmp) / "ref.profile.json"))
        summary = json.loads(E._norm(summary, pathlib.Path(tmp)))
    (dest / "sideman_main.json").write_text(json.dumps({"input_schemas": schemas, "reference_profile_summary": summary},
                                                       indent=2) + "\n")
    print("wrote", dest / "sideman_main.json", "from", SM.__file__)
