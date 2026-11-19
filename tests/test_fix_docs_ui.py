"""Docs and packaging stay in sync with the code: every environment variable that
ovp/config.py reads is documented, and the requirements carry the undeclared imports."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _config_env_vars() -> set[str]:
    src = (ROOT / "ovp" / "config.py").read_text()
    names = set(re.findall(r'_env\w*\(\s*"([A-Z0-9_]+)"', src))
    assert len(names) > 30  # the regex still matches how config.py reads variables
    return names


def test_every_setting_is_in_the_readme_table():
    readme = (ROOT / "README.md").read_text()
    table = readme[readme.index("## Configuration") : readme.index("## Limitations")]
    missing = {n for n in _config_env_vars() if f"`{n}`" not in table}
    assert not missing, f"README configuration table lacks {sorted(missing)}"


def test_every_setting_is_in_env_example():
    example = (ROOT / ".env.example").read_text()
    missing = {n for n in _config_env_vars() if not re.search(rf"^#?\s*{n}=", example, re.M)}
    # GOOGLE_API_KEY is an alias, mentioned next to GEMINI_API_KEY
    missing.discard("GOOGLE_API_KEY")
    assert "GOOGLE_API_KEY" in example
    assert not missing, f".env.example lacks {sorted(missing)}"


def test_api_doc_lists_media_statuses_and_scan_summary():
    api = (ROOT / "docs" / "API.md").read_text()
    assert '"processing" | "done" | "error" | "cancelled"' in api
    for key in ("scanned", "media_with_hits", "accepted", "rejected", "updated"):
        assert f'"{key}"' in api
    assert "in memory" in api.lower()
    assert "OVP_IMPORT_ROOTS" in api and "OVP_MAX_UPLOAD_MB" in api


def _requirement_names(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"([A-Za-z0-9_.\-\[\]]+)\s*(.*)", line)
        if m and not line.startswith("-r"):
            out[m.group(1).lower()] = m.group(2)
    return out


def test_torchreid_import_dependencies_are_required():
    reqs = _requirement_names(ROOT / "requirements.txt")
    assert "torchreid" in reqs
    assert "scipy" in reqs and "tensorboard" in reqs


def test_local_vlm_needs_recent_transformers():
    spec = _requirement_names(ROOT / "requirements-local.txt")["transformers"]
    major, minor = map(int, re.search(r">=\s*(\d+)\.(\d+)", spec).groups())  # type: ignore[union-attr]
    assert (major, minor) >= (4, 50)  # AutoModelForImageTextToText


def test_dockerfiles_build_the_ui_with_the_same_node():
    def node(p: Path) -> str:
        return re.search(r"FROM (node:\S+)", p.read_text()).group(1)  # type: ignore[union-attr]

    assert node(ROOT / "Dockerfile") == node(ROOT / "frontend" / "Dockerfile")
