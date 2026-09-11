"""Strings the desktop window shows outside the page: the quit dialog and pywebview's own dialogs/menus.

The page translates itself (frontend/src/i18n). These few host-side strings
start in English; the languages plan adds the other six.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ShellStrings:
    quit_title: str
    quit_body: str                                   # carries a {count} placeholder
    localization: Dict[str, str] = field(default_factory=dict)   # pywebview localization keys


ENGLISH = ShellStrings(
    quit_title="Quit OpenDPD Studio?",
    quit_body="{count} experiment(s) are running. Quit OpenDPD Studio and stop them?",
    localization={
        "global.quit": "Quit",
        "global.cancel": "Cancel",
        "global.ok": "OK",
        "global.saveFile": "Save file",
        "global.quitConfirmation": "Do you really want to quit?",
    },
)


def shell_strings(language: Optional[str] = None) -> ShellStrings:
    """Strings for a UI language code; English until the other languages exist."""
    return ENGLISH
