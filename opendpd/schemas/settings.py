"""Per-workspace preferences of the workbench (never scientific): what the GUI remembers between launches."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from .common import StrictModel

UILanguage = Literal["en", "nl", "zh", "fr", "de", "it", "ja", "ko", "es"]
UI_LANGUAGES: Tuple[str, ...] = ("en", "nl", "zh", "fr", "de", "it", "ja", "ko", "es")


class WorkspaceSettings(StrictModel):
    """Stored as ``<workspace>/settings.json``; ``language`` None uses the English UI default."""

    language: Optional[UILanguage] = None
