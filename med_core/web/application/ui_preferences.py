"""Filesystem-backed UI preference store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_UI_PREFERENCES = {
    "history_display_mode": "friendly",
    "language": "zh",
    "theme_mode": "auto",
}


class UIPreferencesStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return dict(DEFAULT_UI_PREFERENCES)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return dict(DEFAULT_UI_PREFERENCES)
        if not isinstance(payload, dict):
            return dict(DEFAULT_UI_PREFERENCES)
        return {
            "history_display_mode": payload.get(
                "history_display_mode",
                DEFAULT_UI_PREFERENCES["history_display_mode"],
            ),
            "language": payload.get("language", DEFAULT_UI_PREFERENCES["language"]),
            "theme_mode": payload.get("theme_mode", DEFAULT_UI_PREFERENCES["theme_mode"]),
        }

    def save(self, preferences: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "history_display_mode": preferences.get(
                "history_display_mode",
                DEFAULT_UI_PREFERENCES["history_display_mode"],
            ),
            "language": preferences.get("language", DEFAULT_UI_PREFERENCES["language"]),
            "theme_mode": preferences.get("theme_mode", DEFAULT_UI_PREFERENCES["theme_mode"]),
        }
        self.path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized

    def reset(self) -> dict[str, Any]:
        return self.save(dict(DEFAULT_UI_PREFERENCES))
