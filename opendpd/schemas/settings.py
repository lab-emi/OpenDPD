"""Per-workspace preferences of the workbench (never scientific): what the GUI remembers between launches."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from .common import StrictModel

UILanguage = Literal["en", "fr", "de", "es", "zh", "ja", "ko"]
UI_LANGUAGES: Tuple[str, ...] = ("en", "fr", "de", "es", "zh", "ja", "ko")


class WorkspaceSettings(StrictModel):
    """Stored as ``<workspace>/settings.json``; ``language`` None means "follow the browser or system language"."""

    language: Optional[UILanguage] = None
