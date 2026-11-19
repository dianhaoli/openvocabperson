from __future__ import annotations

import json

import pytest

from ovp.models.llm import extract_json
from ovp.pipeline.analyzer import PERSON_FIELDS, build_crop_prompt, parse_crop_reply


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("  [1, 2]  ", [1, 2]),
        ('```json\n{"a": {"b": [1]}}\n```', {"a": {"b": [1]}}),
        ('```\n[{"x": 1}]\n```', [{"x": 1}]),
        ('Here you go: {"a": "brace } in string", "b": "quote \\" }"} hope that helps', {"a": "brace } in string", "b": 'quote " }'}),
        ('{not json} but this is {"ok": true}', {"ok": True}),
        ('prefix [1, {"k": [2, 3]}] suffix', [1, {"k": [2, 3]}]),
    ],
)
def test_extract_json(text: str, expected: object) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", '{"truncated": ', None])
def test_extract_json_errors(text: str | None) -> None:
    with pytest.raises(ValueError):
        extract_json(text)  # type: ignore[arg-type]


def test_build_crop_prompt_questions_and_fields() -> None:
    prompt = build_crop_prompt("person", ["Is this person armed?", "Is this person running?"], describe=True)
    assert "detected 'person'" in prompt
    assert "q1: Is this person armed?" in prompt and "q2: Is this person running?" in prompt
    schema = json.loads(prompt.rsplit("shaped like: ", 1)[1])
    assert set(schema) == {"checks", *PERSON_FIELDS}


def test_build_crop_prompt_object_without_questions() -> None:
    prompt = build_crop_prompt("knife", [], describe=True)
    schema = json.loads(prompt.rsplit("shaped like: ", 1)[1])
    assert "checks" not in schema and "color" in schema and "clothing" not in schema
    verify_only = json.loads(build_crop_prompt("knife", ["Is it sharp?"], describe=False).rsplit("shaped like: ", 1)[1])
    assert list(verify_only) == ["checks"]


def test_parse_crop_reply_full() -> None:
    reply = json.dumps({
        "checks": [{"id": "q2", "answer": "no", "reason": "hands empty"}, {"id": "q1", "answer": True, "reason": "knife visible"}],
        "summary": " Man in a grey hoodie ",
        "clothing": "grey hoodie",
        "held_objects": "none",
        "appearance": "unclear",
        "actions": "",
    })
    checks, summary, attrs = parse_crop_reply(reply, 2, describe=True)
    assert checks == [(True, "knife visible"), (False, "hands empty")]
    assert summary == "Man in a grey hoodie"
    assert attrs == {"clothing": "grey hoodie"}


def test_parse_crop_reply_checks_without_ids_and_unknown_answers() -> None:
    reply = json.dumps({"checks": [{"answer": "YES"}, {"answer": "maybe", "reason": "blurry"}]})
    checks, summary, attrs = parse_crop_reply(reply, 3, describe=False)
    assert checks == [(True, ""), (None, "blurry"), (None, "")]
    assert summary is None and attrs is None


def test_parse_crop_reply_only_uninformative_attributes() -> None:
    _, summary, attrs = parse_crop_reply(json.dumps({"summary": "A dog", "color": "N/A", "details": "none"}), 0, describe=True)
    assert summary == "A dog" and attrs is None


@pytest.mark.parametrize(
    ("text", "answer"),
    [("Yes, clearly a knife.", True), ("no.", False), ("I cannot tell", None), ("", None)],
)
def test_parse_crop_reply_plain_text_fallback(text: str, answer: bool | None) -> None:
    checks, summary, _ = parse_crop_reply(text, 1, describe=True)
    assert checks == [(answer, text[:200])]
    assert summary == (text.strip() or None)


def test_parse_crop_reply_non_object_json() -> None:
    checks, summary, attrs = parse_crop_reply("[1, 2]", 1, describe=False)
    assert checks == [(None, "[1, 2]")] and summary is None and attrs is None
