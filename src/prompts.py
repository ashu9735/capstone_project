"""Prompt loader. Prompts are versioned files on disk, not strings in code."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.config import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "prompts"
_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str

    def render(self, **values: object) -> str:
        text = self.template
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text


@lru_cache(maxsize=32)
def load_prompt(relative_path: str) -> Prompt:
    path = PROMPT_DIR / relative_path
    raw = path.read_text(encoding="utf-8")
    version, name = "unversioned", path.stem
    match = _FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "version":
                version = value
            elif key == "name":
                name = value
        raw = raw[match.end() :]
    return Prompt(name=name, version=version, template=raw.strip())
