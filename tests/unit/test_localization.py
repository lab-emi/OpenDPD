"""Presentation-only localization must retain scientific values and command text."""
import re

import pytest

from opendpd.schemas import UI_LANGUAGES, WorkspaceSettings
from opendpd.studio.localization import catalogue, language_tag, localize
from opendpd.studio.strings import shell_strings


@pytest.mark.parametrize("language", UI_LANGUAGES)
def test_catalogue_and_native_language_are_complete(language):
    en, translated = catalogue("en"), catalogue(language)
    assert translated.keys() == en.keys()
    for key, value in translated.items():
        assert value.strip(), key
        assert sorted(re.findall(r"\{(\w+)\}", value)) == sorted(re.findall(r"\{(\w+)\}", en[key])), key
    assert WorkspaceSettings(language=language).language == language
    assert "{count}" in shell_strings(language).quit_body
    assert language_tag(language) == ("zh-CN" if language == "zh" else language)


@pytest.mark.parametrize("language", UI_LANGUAGES)
def test_diagnostic_values_and_commands_are_preserved(language):
    text = localize("cross-correlation peaks at +1.25 samples (correlation 0.98)", language)
    assert "+1.25" in text and "0.98" in text
    assert "{" not in text
    if language != "en":
        assert localize("Input and output are time aligned", language) != "Input and output are time aligned"
    command = "opendpd run --config train.json --workspace <workspace>"
    assert localize(command, language) == command
    assert localize("my-capture-01", language) == "my-capture-01"
