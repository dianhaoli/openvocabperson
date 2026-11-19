from __future__ import annotations

import json

import pytest

from ovp.pipeline.planner import QueryPlanner, _singular, literal_concept, needs_llm, split_terms

from .fakes import FakeLLM


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("show me all dogs", ["dogs"]),
        ("find the buses", ["buses"]),
        ("dogs, cats and phones", ["dogs", "cats", "phones"]),
        ("knives; scissors\nforks", ["knives", "scissors", "forks"]),
        ("cars or trucks + bikes", ["cars", "trucks", "bikes"]),
        ("  Search for a Red Car!  ", ["red car"]),
        (", , ;", []),
    ],
)
def test_split_terms(query: str, terms: list[str]) -> None:
    assert split_terms(query) == terms


@pytest.mark.parametrize(
    ("word", "singular"),
    [
        ("dogs", "dog"), ("knives", "knife"), ("buses", "bus"), ("boxes", "box"), ("watches", "watch"),
        ("dresses", "dress"), ("berries", "berry"), ("glasses", "glasses"), ("scissors", "scissors"),
        ("sheep", "sheep"), ("bus", "bus"), ("glass", "wine glass"), ("cars", "car"), ("phone", "cell phone"),
    ],
)
def test_singular(word: str, singular: str) -> None:
    assert _singular(word) == singular


def test_literal_concepts_for_people() -> None:
    plain = literal_concept("people")
    assert (plain.prompts, plain.kind, plain.verify) == (["person"], "object", False)
    kids = literal_concept("kids")
    assert kids.kind == "person_attribute" and kids.verify and kids.question == "Is this person a child?"
    menacing = literal_concept("menacing people")
    assert menacing.prompts == ["person"] and menacing.verify and "menacing people" in (menacing.question or "")
    multiword = literal_concept("cell phones")
    assert multiword.prompts == ["cell phone"] and not multiword.verify


def test_offline_plan_lexicon_expansion() -> None:
    plan = QueryPlanner().offline_plan("dangerous things")
    assert plan.source == "lexicon" and plan.expanded
    (objects,) = plan.concepts  # an object category: no person concept
    assert "knife" in objects.prompts and "gun" in objects.prompts and objects.verify
    objects, people = QueryPlanner().offline_plan("suspicious things").concepts  # "suspicious" also covers people
    assert people.kind == "person_attribute" and people.prompts == ["person"] and people.verify


def test_offline_plan_person_words_override_lexicon() -> None:
    # "armed" is a weapon alias, but "armed men" is about people
    (concept,) = QueryPlanner().offline_plan("armed men").concepts
    assert concept.kind == "person_attribute" and concept.prompts == ["person"]


def test_offline_plan_literal_and_dedupe() -> None:
    plan = QueryPlanner().offline_plan("dogs, cats, dogs")
    assert plan.source == "literal" and not plan.expanded
    assert [c.name for c in plan.concepts] == ["dogs", "cats"]
    assert plan.prompts == ["dog", "cat"]


def test_needs_llm() -> None:
    assert not needs_llm(["dogs", "red car"])
    assert needs_llm(["dangerous things"])
    assert needs_llm(["weapons"])
    assert needs_llm(["people wearing hats"])
    assert needs_llm(["menacing people"])


async def test_plan_default_and_literal_modes() -> None:
    planner = QueryPlanner()
    default = await planner.plan(None)
    assert default.source == "default" and "person" in default.prompts and default.query is None
    assert (await planner.plan("   ")).source == "default"
    literal = await planner.plan("weapons", expand=False, llm=FakeLLM("{}"))
    assert literal.source == "literal" and literal.prompts == ["weapon"]


async def test_plan_auto_uses_llm_only_for_abstract_queries() -> None:
    reply = json.dumps({"concepts": [
        {"name": "dangerous things", "kind": "object", "prompts": ["Kitchen Knife", "handgun", "handgun", " "], "verify": True,
         "question": "Is this dangerous?"},
        {"name": "menacing people", "kind": "person_attribute", "prompts": ["man", "woman"], "verify": False},
        {"name": "dogs", "kind": "object", "prompts": ["dogs"], "verify": True},
        "garbage",
    ]})
    llm = FakeLLM(f"Sure! ```json\n{reply}\n```")
    planner = QueryPlanner()
    concrete = await planner.plan("dogs", llm=llm)
    assert concrete.source == "literal" and llm.calls == 0

    plan = await planner.plan("dangerous things, menacing people", llm=llm)
    assert plan.source == "llm" and plan.expanded and llm.calls == 1
    danger, menacing, dogs = plan.concepts
    assert danger.prompts == ["kitchen knife", "handgun"] and danger.verify
    assert menacing.prompts == ["person"] and menacing.verify and "menacing people" in (menacing.question or "")
    assert dogs.prompts == ["dog"] and not dogs.verify  # verify without a question is dropped

    again = await planner.plan("Dangerous Things, Menacing People ", llm=llm)
    assert again is plan and llm.calls == 1  # cached per (query, llm)


async def test_plan_llm_prompt_budget() -> None:
    concepts = [{"name": f"c{i}", "prompts": [f"thing {i} {j}" for j in range(15)]} for i in range(5)]
    plan = await QueryPlanner().plan("lots of stuff", expand=True, llm=FakeLLM(json.dumps({"concepts": concepts})))
    assert len(plan.prompts) == 40
    assert [len(c.prompts) for c in plan.concepts] == [15, 15, 10]


@pytest.mark.parametrize("reply", ["not json at all", json.dumps({"concepts": []}), RuntimeError("LLM down")])
async def test_plan_falls_back_to_lexicon(reply: str | Exception) -> None:
    llm = FakeLLM(reply)
    plan = await QueryPlanner().plan("weapons", expand=True, llm=llm)
    assert plan.source == "lexicon" and "gun" in plan.prompts and llm.calls == 1


async def test_plan_accepts_top_level_list() -> None:
    plan = await QueryPlanner().plan("ties", expand=True, llm=FakeLLM(json.dumps([{"name": "ties", "prompts": ["tie"]}])))
    assert plan.source == "llm" and plan.prompts == ["tie"]
