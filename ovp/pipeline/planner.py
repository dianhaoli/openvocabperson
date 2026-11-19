"""Query planning: free text → concrete detector prompts (+ VLM verification questions).

Examples
* ``"people, dogs"``            → literal prompts ``person``, ``dog``
* ``"dangerous things"``        → knife, gun, rifle, scissors, fire, ... (+ VLM verification)
* ``"menacing people"``         → detect ``person``, then ask the VLM "Does this person appear menacing?"
* ``"all living things"``       → person, dog, cat, bird, horse, plant, tree, ...

Uses the configured LLM when available (``expand=True`` or ``"auto"`` for abstract queries) and
falls back to a built-in lexicon so the feature still works fully offline.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Optional, Union

from ..models.detectors import COCO_CLASSES, COCO_SYNONYMS
from ..models.llm import ChatModel, extract_json
from ..types import Concept, QueryPlan

log = logging.getLogger("ovp.plan")

LEXICON: dict[str, dict[str, Any]] = {
    "dangerous": {
        "prompts": ["knife", "gun", "pistol", "rifle", "scissors", "axe", "machete", "sword", "baseball bat", "hammer",
                    "fire", "flame", "explosive", "syringe", "broken glass", "chainsaw"],
        "question": "Is this a potentially dangerous object (weapon, sharp, burning, or otherwise hazardous)?",
        "people": True,
    },
    "weapon": {
        "prompts": ["gun", "pistol", "handgun", "rifle", "shotgun", "knife", "machete", "sword", "axe", "baseball bat", "crowbar", "taser"],
        "question": "Is this a weapon or an object being used as a weapon?",
    },
    "sharp": {
        "prompts": ["knife", "scissors", "razor blade", "box cutter", "broken glass", "needle", "syringe", "axe", "saw", "sword", "screwdriver"],
        "question": "Is this a sharp or pointed object?",
    },
    "hot": {
        "prompts": ["fire", "flame", "candle", "stove", "oven", "grill", "iron", "kettle", "lighter", "cigarette", "hot pan", "fireplace", "heater"],
        "question": "Is this object hot, burning, or a heat source?",
    },
    "fire": {"prompts": ["fire", "flame", "smoke", "candle", "torch", "lighter", "burning object"], "question": "Is there fire, flame or smoke here?"},
    "alive": {
        "prompts": ["person", "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe", "fish",
                    "insect", "animal", "potted plant", "tree", "flower"],
        "question": "Is this a living thing (person, animal or plant)?",
    },
    "animal": {
        "prompts": ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe", "fish", "rabbit", "deer", "monkey", "insect", "animal"],
        "question": "Is this an animal?",
    },
    "vehicle": {
        "prompts": ["car", "truck", "bus", "motorcycle", "bicycle", "van", "train", "boat", "airplane", "scooter"],
        "question": None,
    },
    "electronic": {
        "prompts": ["cell phone", "laptop", "tv", "monitor", "keyboard", "camera", "tablet", "headphones", "remote", "game controller"],
        "question": None,
    },
    "food": {"prompts": ["food", "pizza", "sandwich", "fruit", "vegetable", "cake", "donut", "bread", "drink", "bottle", "cup"], "question": "Is this food or drink?"},
    "suspicious": {
        "prompts": ["person", "unattended bag", "backpack", "suitcase", "mask", "weapon"],
        "question": "Does this look suspicious (e.g. concealed face, unattended bag, weapon, forced entry)?",
        "people": True,
    },
    "valuable": {"prompts": ["wallet", "purse", "handbag", "jewelry", "watch", "laptop", "cell phone", "cash", "camera"], "question": None},
    "drug": {"prompts": ["syringe", "pills", "pill bottle", "powder", "smoking pipe", "cigarette", "joint"], "question": "Is this drug related paraphernalia?"},
}

LEXICON_ALIASES = {
    "danger": "dangerous", "hazard": "dangerous", "hazardous": "dangerous", "threat": "dangerous", "threatening": "dangerous", "unsafe": "dangerous",
    "weapons": "weapon", "armed": "weapon", "firearm": "weapon", "firearms": "weapon",
    "pointy": "sharp", "cutting": "sharp", "heat": "hot", "burning": "fire", "flames": "fire",
    "living": "alive", "life": "alive", "organisms": "alive", "animals": "animal", "pets": "animal", "pet": "animal", "wildlife": "animal",
    "vehicles": "vehicle", "transport": "vehicle", "transportation": "vehicle", "electronics": "electronic", "devices": "electronic", "gadgets": "electronic",
    "drugs": "drug", "narcotics": "drug", "valuables": "valuable",
}

PERSON_WORDS = {"people", "person", "persons", "man", "men", "woman", "women", "someone", "guy", "guys", "kid", "kids", "child", "children",
                "boy", "boys", "girl", "girls", "adult", "adults", "teen", "teens", "pedestrian", "pedestrians", "individual", "individuals",
                "human", "humans", "worker", "workers", "officer", "officers", "player", "players", "crowd", "anyone", "everyone", "who"}

ABSTRACT_WORDS = {"thing", "things", "stuff", "item", "items", "object", "objects", "anything", "everything", "all", "any", "something",
                  "suspicious", "dangerous", "unusual", "weird", "interesting", "important", "related"}


IRREGULAR = {"knives": "knife", "leaves": "leaf", "wolves": "wolf", "shelves": "shelf", "mice": "mouse", "geese": "goose",
             "teeth": "tooth", "feet": "foot", "children": "child", "men": "man", "women": "woman", "glasses": "glasses",
             "scissors": "scissors", "pants": "pants", "jeans": "jeans", "shorts": "shorts", "sunglasses": "sunglasses",
             "fish": "fish", "sheep": "sheep", "deer": "deer", "bus": "bus", "buses": "bus", "boxes": "box", "series": "series"}


def _singular(word: str) -> str:
    w = word.strip()
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in COCO_CLASSES or w in COCO_SYNONYMS:
        return COCO_SYNONYMS.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def split_terms(query: str) -> list[str]:
    parts = re.split(r"[,;\n]|\band\b|\bor\b|\+", query, flags=re.I)
    terms = []
    for p in parts:
        t = p.strip().lower()
        prev = None
        while prev != t:
            prev = t
            t = re.sub(r"^(find|show|show me|search for|search|look for|detect|locate|segment|get|highlight|count|me|of|all|every|each|any|the|a|an)\s+", "", t).strip()
        t = t.strip(" .!?\"'")
        if t:
            terms.append(t)
    return terms


def _lexicon_key(term: str) -> Optional[str]:
    for w in re.findall(r"[a-z]+", term):
        if w in LEXICON:
            return w
        if w in LEXICON_ALIASES:
            return LEXICON_ALIASES[w]
    return None


def _person_term(term: str) -> bool:
    words = set(re.findall(r"[a-z]+", term))
    return bool(words & PERSON_WORDS)


SIMPLE_PEOPLE = {"people", "person", "persons", "humans", "human", "everyone", "anyone"}
PERSON_SUBTYPES = {"man": "man", "men": "man", "woman": "woman", "women": "woman", "kid": "child", "kids": "child",
                   "child": "child", "children": "child", "boy": "boy", "boys": "boy", "girl": "girl", "girls": "girl",
                   "adult": "adult", "adults": "adult", "teen": "teenager", "teens": "teenager"}


def literal_concept(term: str) -> Concept:
    if term in SIMPLE_PEOPLE:
        return Concept(name=term, prompts=["person"])
    if term in PERSON_SUBTYPES:
        return Concept(name=term, prompts=["person"], kind="person_attribute", verify=True,
                       question=f"Is this person a {PERSON_SUBTYPES[term]}?")
    if _person_term(term):
        return Concept(name=term, prompts=["person"], kind="person_attribute", verify=True,
                       question=f"Does this person match the description '{term}'?")
    words = term.split()
    phrase = " ".join(_singular(w) if i == len(words) - 1 else w for i, w in enumerate(words))
    return Concept(name=term, prompts=[COCO_SYNONYMS.get(phrase, phrase)])


def lexicon_concept(term: str, key: str) -> Concept:
    entry = LEXICON[key]
    q = entry["question"]
    return Concept(name=term, prompts=list(entry["prompts"]), kind="object", verify=q is not None, question=q)


def needs_llm(terms: list[str]) -> bool:
    for t in terms:
        words = re.findall(r"[a-z]+", t)
        if len(words) > 2 or set(words) & ABSTRACT_WORDS or _lexicon_key(t):
            return True
        if _person_term(t) and len(words) > 1:
            return True
    return False


PLANNER_SYSTEM = """You turn a user's visual search request into instructions for an open-vocabulary object detector (YOLOE / OWLv2 / Grounding DINO) plus an optional vision-language-model (VLM) check.

Return JSON: {"concepts": [{"name": str, "kind": "object" | "person_attribute", "prompts": [str], "verify": bool, "question": str | null}]}

Rules:
- One concept per distinct thing the user asked for, named with the user's words.
- "prompts" are short concrete noun phrases a detector can see (e.g. "kitchen knife", "handgun", "rifle", "candle"). Expand abstract categories into 4-15 specific, visually distinct objects. Never use adjectives alone. Use singular nouns.
- If the concept is about people with some property (behaviour, clothing, emotion, role, age, e.g. "menacing people", "people wearing red", "kids", "police officers"), use kind "person_attribute" with prompts ["person"] and a precise yes/no question about one person crop.
- If the request also implies people (e.g. "dangerous things" includes menacing or armed people), add a separate person_attribute concept for that.
- Set verify=true when detector prompts could produce false positives for the user's intent (abstract, subjective, or broad categories) and write "question" as a yes/no question about a single image crop. Use verify=false for plain concrete objects ("dogs", "cars", "people").
- Do NOT expand concrete objects the user named ("ties" → ["tie"], "red cars" → ["car"] with verify=true and question "Is this car red?"). Only expand categories.
- Interpret categories the way a safety / security analyst would: "dangerous things" means weapons (gun, knife, rifle, machete, axe, baseball bat), sharp objects, fire/flames, explosives, hazardous tools — not wild animals unless the user mentions animals. "Alive / living things" means people, animals and plants.
- At most 40 prompts in total. Output JSON only."""


MAX_PLAN_PROMPTS = 40


class QueryPlanner:
    def __init__(self, max_cache: int = 256):
        self._cache: "OrderedDict[tuple, QueryPlan]" = OrderedDict()
        self.max_cache = max_cache

    def default_plan(self) -> QueryPlan:
        return QueryPlan(query=None, concepts=[Concept(name="default", prompts=list(COCO_CLASSES))], expanded=False, source="default")

    def offline_plan(self, query: str) -> QueryPlan:
        terms = split_terms(query)
        concepts: list[Concept] = []
        used_lexicon = False
        for t in terms:
            key = _lexicon_key(t)
            if key and not (_person_term(t) and key not in ("dangerous", "suspicious")):
                concepts.append(lexicon_concept(t, key))
                used_lexicon = True
                if LEXICON[key].get("people"):
                    concepts.append(Concept(name=f"{t} (people)", prompts=["person"], kind="person_attribute", verify=True,
                                            question=f"Does this person appear {key} (e.g. armed, aggressive or threatening)?"))
            else:
                concepts.append(literal_concept(t))
        return QueryPlan(query=query, concepts=_dedupe(concepts), expanded=used_lexicon, source="lexicon" if used_lexicon else "literal")

    async def plan(self, query: Optional[str], expand: Union[bool, str] = "auto", llm: Optional[ChatModel] = None) -> QueryPlan:
        if not query or not query.strip():
            return self.default_plan()
        query = query.strip()
        terms = split_terms(query)
        use_llm = llm is not None and (expand is True or (expand == "auto" and needs_llm(terms)))
        if expand is False or not use_llm:
            plan = self.offline_plan(query)
            if expand is False:
                # literal: no lexicon expansion, but keep person-attribute handling
                plan = QueryPlan(query=query, concepts=_dedupe([literal_concept(t) for t in terms]), expanded=False, source="literal")
            return plan
        key = (query.lower(), llm.spec)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            text = await llm.complete(
                [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": f"Request: {query}"}],
                max_tokens=900,
                json_mode=True,
            )
            plan = self._parse(query, extract_json(text))
        except Exception as e:
            log.warning("LLM planning failed (%s); using offline lexicon", e)
            return self.offline_plan(query)
        self._cache[key] = plan
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return plan

    @staticmethod
    def _parse(query: str, data: Any) -> QueryPlan:
        items = data.get("concepts", []) if isinstance(data, dict) else data
        concepts: list[Concept] = []
        total = 0
        for it in items or []:
            if not isinstance(it, dict) or total >= MAX_PLAN_PROMPTS:
                continue
            kind = "person_attribute" if it.get("kind") == "person_attribute" else "object"
            prompts = [str(p).strip().lower() for p in (it.get("prompts") or []) if str(p).strip()]
            if kind == "person_attribute":
                prompts = ["person"]
            prompts = list(dict.fromkeys(COCO_SYNONYMS.get(p, p) for p in prompts))[: MAX_PLAN_PROMPTS - total]
            if not prompts:
                continue
            total += len(prompts)
            question = it.get("question") or None
            verify = bool(it.get("verify")) and bool(question)
            if kind == "person_attribute" and not question:
                question = f"Does this person match the description '{it.get('name', query)}'?"
                verify = True
            concepts.append(Concept(name=str(it.get("name") or query), prompts=prompts, kind=kind, verify=verify or kind == "person_attribute", question=question))
        if not concepts:
            raise ValueError("planner returned no concepts")
        return QueryPlan(query=query, concepts=_dedupe(concepts), expanded=True, source="llm")


def _dedupe(concepts: list[Concept]) -> list[Concept]:
    seen: dict[tuple, Concept] = {}
    for c in concepts:
        k = (c.name, c.kind)
        if k not in seen:
            seen[k] = c
    return list(seen.values())
