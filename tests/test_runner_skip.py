"""generate() skips a manual SUT with no submission instead of failing the run."""
from __future__ import annotations

from hallmark.models import SystemSpec, Task
from hallmark.runner import _build_adapters, _select_adapter, generate

_TASK = Task.model_validate(
    {
        "id": "qa_x",
        "domain": "general_qa",
        "track": "reasoning",
        "turns": [{"prompt": "What is the structure of austenite?"}],
        "runs": 1,
    }
)


async def test_generate_skips_missing_submission(tmp_path, monkeypatch):
    from hallmark.config import settings

    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "transcripts_dir", tmp_path / "transcripts")

    system = SystemSpec(id="openai/chatgpt-web", adapter="manual", config={})
    msgs: list[str] = []
    written = await generate([_TASK], [system], use_cache=False, progress=msgs.append)

    assert written == []  # nothing generated
    assert any("skipped" in m for m in msgs)


# --------------------------------------------------------------------------- #
# Multiple adapters per system + capability-based routing                      #
# --------------------------------------------------------------------------- #

_MANUAL_TASK = Task.model_validate(
    {
        "id": "qa_manual",
        "domain": "general_qa",
        "track": "reasoning",
        "required_capabilities": ["manual"],
        "turns": [{"prompt": "Answer this by hand."}],
        "runs": 1,
    }
)


def _multi_system() -> SystemSpec:
    # `adapters:` map form — per-adapter config; the mock answers normal tasks,
    # the manual adapter serves any task that requires the `manual` capability.
    return SystemSpec.model_validate(
        {
            "id": "combo",
            "adapters": {
                "mock": {"answers": {"qa_x": "FCC."}},
                "manual": {"submissions_dir": "submissions"},
            },
        }
    )


def test_adapter_components_parsing():
    # "a + b" shorthand shares one config.
    s = SystemSpec(id="s", adapter="mock + manual", config={"k": 1})
    assert s.adapter_names() == ["mock", "manual"]
    assert s.adapter_components() == [("mock", {"k": 1}), ("manual", {"k": 1})]

    # list form
    assert SystemSpec(id="s", adapter=["mock", "manual"]).adapter_names() == ["mock", "manual"]

    # `adapters:` map layers per-adapter config over the shared config.
    s2 = _multi_system()
    assert s2.adapter_names() == ["mock", "manual"]
    assert dict(s2.adapter_components())["manual"]["submissions_dir"] == "submissions"


def test_select_adapter_routes_by_capability():
    adapters = _build_adapters(_multi_system())
    # normal task -> first non-manual adapter (mock)
    assert _select_adapter(adapters, _TASK).capabilities.manual is False
    # manual-required task -> manual adapter
    assert _select_adapter(adapters, _MANUAL_TASK).capabilities.manual is True


async def test_generate_routes_normal_task_to_mock(tmp_path, monkeypatch):
    from hallmark.config import settings

    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "transcripts_dir", tmp_path / "transcripts")

    qa_x = Task.model_validate(
        {
            "id": "qa_x",
            "domain": "general_qa",
            "track": "reasoning",
            "turns": [{"prompt": "What is austenite?"}],
            "runs": 1,
        }
    )
    written = await generate([qa_x], [_multi_system()], use_cache=False)
    assert len(written) == 1  # mock produced a transcript

    from hallmark.runner import _load_transcript

    tr = _load_transcript("combo", "qa_x", 0)
    assert tr is not None and tr.turns[0].answer_text == "FCC."
    assert tr.adapter_capabilities.manual is False  # served by mock, not manual


async def test_generate_routes_manual_task_to_manual(tmp_path, monkeypatch):
    from hallmark.config import settings

    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "transcripts_dir", tmp_path / "transcripts")

    sub = tmp_path / "submissions" / "combo" / "seed0" / "qa_manual.md"
    sub.parent.mkdir(parents=True)
    sub.write_text("## turn 1\nHand-written answer.\n", encoding="utf-8")

    written = await generate([_MANUAL_TASK], [_multi_system()], use_cache=False)
    assert len(written) == 1  # manual adapter ingested the submission

    from hallmark.runner import _load_transcript

    tr = _load_transcript("combo", "qa_manual", 0)
    assert tr is not None and tr.turns[0].answer_text == "Hand-written answer."
    assert tr.adapter_capabilities.manual is True
