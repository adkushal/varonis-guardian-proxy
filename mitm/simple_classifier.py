"""Simple heuristic classifier for prompt categories 1–3 (assignment requirement)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Exact reason phrases required by the assignment wording.
REASON_VIOLENCE = "Description of violent acts"
REASON_ILLEGAL = "Inquiries on how to perform an illegal activity"
REASON_SEXUAL = "Any sexual content"
REASON_TOXIC = "The prompt is considered toxic."


@dataclass(frozen=True)
class Classification:
    category: Optional[str]  # violence | illegal | sexual | None
    block_message: Optional[str]


def _blocked(reason: str, category: str) -> Classification:
    return Classification(
        category=category,
        block_message=f"The prompt was blocked because it contained {reason}",
    )


# Generic verbs like "kill" or "shoot" only count as violence when aimed at a
# person, otherwise "kill a process" / "shoot photos" would be blocked.
_PERSON = (
    r"(?:him|her|them|me|you|us|someone|somebody|anyone|everyone|people|"
    r"(?:a|the|my|his|her|their)\s+(?:person|man|woman|child|kid|baby|wife|husband|"
    r"boss|neighbou?r|teacher|cop|officer|family|friend|coworker|classmates?))"
)

# Ordered: first match wins. Patterns end in \w* where suffixes are allowed
# ("porn" → "pornographic", "drug" → "drugs").
_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "violence",
        REASON_VIOLENCE,
        re.compile(
            r"\b(?:"
            rf"(?:kill|murder|stab|shoot|strangle|beat\s+up|assault|attack|hurt|torture)\w*\s+{_PERSON}\b|"
            r"murder\w*|behead\w*|massacre\w*|mass\s+shooting\w*|"
            r"(?:build|make)\s+(?:a\s+)?(?:bomb|pipe\s+bomb|explosive)\w*"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "illegal",
        REASON_ILLEGAL,
        re.compile(
            r"\b(?:"
            r"how\s+(?:to|do\s+i|can\s+i|would\s+i)\s+"
            r"(?:hack|steal|launder|smuggle|forge|shoplift|counterfeit|pick\s+a\s+lock)\w*|"
            r"launder\w*\s+money|"
            r"commit\w*\s+(?:fraud|theft|burglary|arson|identity\s+theft)\w*|"
            r"(?:make|cook|synthesi[sz]e)\s+(?:meth|fentanyl|counterfeit)\w*|"
            r"bypass\w*\s+(?:security|alarm)\w*|"
            r"illegal\s+(?:drug|activit|weapon|gun|firearm)\w*|"
            r"buy\w*\s+(?:drugs|cocaine|heroin|meth|fentanyl|stolen)\w*|"
            r"evad\w*\s+tax\w*|tax\s+evasion"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "sexual",
        REASON_SEXUAL,
        re.compile(
            r"\b(?:"
            r"porn\w*|explicit\s+sex\w*|sex(?:ual)?\s+(?:act|content|scene|stor)\w*|"
            r"nudes?\b|naked\s+(?:photo|pic|image)\w*|erotic\w*|nsfw\b|xxx\b"
            r")",
            re.IGNORECASE,
        ),
    ),
]


def classify_prompt(text: str) -> Classification:
    """Return a named category hit, or empty classification if none match."""
    if not text or not text.strip():
        return Classification(category=None, block_message=None)
    for category, reason, pattern in _RULES:
        if pattern.search(text):
            return _blocked(reason, category)
    return Classification(category=None, block_message=None)
