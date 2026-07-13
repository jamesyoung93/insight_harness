"""Copy audit: the rendered UI must never narrate itself, reference its own
development phasing, or address the builder instead of the user.

Scans every UI module (app.py and views/, if present) for banned strings.
Developer-facing files (README, harness internals other than user-rendered
refusal text, the data generator) are deliberately out of scope.
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

BANNED = [
    "Alpha",
    "Beta",
    "V1",
    "placeholder",
    "demo",
    "stand-in",
    "swap in",
    "benchmark",
    "production version",
]


def ui_files():
    files = [REPO / "app.py"]
    files += sorted((REPO / "views").glob("*.py")) if (REPO / "views").is_dir() else []
    return files


def string_literals(path):
    """Every string constant in the file — covers labels, captions, markdown,
    f-string fragments, docstrings. Identifier/kwarg names are excluded, which
    is the point: the audit targets what a user could ever read."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


@pytest.mark.parametrize("path", ui_files(), ids=lambda p: p.name)
def test_no_fourth_wall_breaks(path):
    hits = []
    for lineno, value in string_literals(path):
        for term in BANNED:
            if term.lower() in value.lower():
                snippet = value.strip().replace("\n", " ")[:90]
                hits.append(f"{path.name}:{lineno} contains {term!r}: {snippet!r}")
    assert not hits, "\n".join(hits)


def test_refusal_copy_is_product_voiced():
    """Refusal reasons render directly to users; they must never mention
    development phases or scope roadmaps."""
    text = (REPO / "harness" / "triage.py").read_text(encoding="utf-8")
    for term in ("Alpha", "Beta", "scope expansion", "later iteration"):
        assert term not in text, f"triage.py contains roadmap language: {term!r}"
