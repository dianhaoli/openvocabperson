"""Docs and UI stay in sync with the code they describe (config, thresholds, model cache, API)."""

import re
from pathlib import Path

from ovp.config import REID_THRESHOLDS
from ovp.models import registry
from ovp.storage.db import DEFAULT_MEDIA_STATUSES, MEDIA_STATUSES

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def env_vars() -> set[str]:
    return set(re.findall(r'_env(?:_int|_float|_bool|_list)?\(\s*"([A-Z_]+)"', read("ovp/config.py")))


def test_every_setting_is_documented():
    names = env_vars()
    assert "OVP_CONF" in names and "DATABASE_URL" in names
    for doc in ("README.md", ".env.example"):
        text = read(doc)
        assert not [n for n in names if n not in text], doc


def test_reid_thresholds_match_config():
    readme, details = read("README.md"), read("docs/DETAILS.md")
    for kind, name in (("osnet", "OSNet"), ("clip", "CLIP"), ("color", "colour")):
        t = REID_THRESHOLDS[kind]
        assert f"{name} {t.match:.2f}/{t.review:.2f}" in readme
        assert re.search(rf"\| {t.match:.2f} \| {t.review:.2f} \|", details), kind


def test_model_cache_cap_and_weight_suffixes_documented():
    for doc in ("README.md", "docs/DETAILS.md", "docs/API.md"):
        text = read(doc)
        assert re.search(rf"(at most|up to) {registry.MAX_EXTRA_MODELS}\b", text, re.I), doc
        assert all(f"`{suffix}`" in text for suffix in registry.WEIGHT_SUFFIXES), doc


def test_reid_spaces_and_migration_documented():
    for doc in ("README.md", "docs/DETAILS.md"):
        text = read(doc)
        assert "reid:color:v2" in text and "reid:clip:" in text and "python -m ovp reembed" in text, doc


def test_media_status_filter_documented_and_used_by_ui():
    api = read("docs/API.md")
    assert f"status={','.join(DEFAULT_MEDIA_STATUSES)}" in api
    assert all(f"`{s}`" in api for s in MEDIA_STATUSES)
    assert "failed_media_ids" in api
    library = read("frontend/src/components/library/LibraryTab.tsx")
    assert all(f"'{s}'" in library for s in MEDIA_STATUSES)


def test_no_stale_tracking_or_conf_docs():
    details = read("docs/DETAILS.md")
    assert "cannot track" not in details
    assert '"conf": 0.25' not in read("docs/API.md")


def test_analyze_tab_conf_defaults_to_auto():
    tab = read("frontend/src/components/analyze/AnalyzeTab.tsx")
    assert "useState<number | null>(null)" in tab
    assert "if (conf != null) options.conf = conf;" in tab


def test_scan_recheck_unlink_documented():
    details = read("docs/DETAILS.md")
    assert "rejected person sighting is unlinked from its person" in details
