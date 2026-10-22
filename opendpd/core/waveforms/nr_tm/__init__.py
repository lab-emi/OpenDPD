"""3GPP TS 38.141-1 section 4.9.2 NR FR1 standard test models (baseband), single and multi-carrier.

Public API (everything else is internal):

* ``validate_recipe(recipe)``   raise ``ValueError`` for a recipe the standard or the generator limits do not allow
* ``derive(recipe)``            structural ``GeneratorConfig`` fields + ``NrTmSummary`` (+ the internal plan)
* ``resolve(recipe, **edits)``  recipe + complete validated ``GeneratorConfig`` + summary
* ``catalog()``                 the models, parameter ranges and fixed parameters for the UI
* ``recipe_label(recipe)``      readable name (``NR-FR1-TM3.1a · 100 MHz · 30 kHz · FDD``) for pickers and dataset captures
* ``synthesize_nr_tm(config)``  complex64 IQ of exact length + ``GeneratorAnalysis`` (same contract as ``generator.synthesize``)
* ``SOURCE_FILES``              every source file of this package (hashed into the signal id of test-model configurations)
"""
from __future__ import annotations

from pathlib import Path

from .catalog import catalog     # the function intentionally takes the place of the submodule of the same name
from .label import recipe_label
from .plan import STRUCTURAL_FIELDS, NrTmDerived, derive, resolve, validate_recipe

SOURCE_FILES: tuple[Path, ...] = tuple(sorted(Path(__file__).resolve().parent.glob("*.py")))


def synthesize_nr_tm(config):
    """complex64 IQ of exact length + GeneratorAnalysis for a GeneratorConfig that carries an nr_test_model recipe."""
    from .synth import synthesize_nr_tm as synthesize   # scipy is imported only when a signal is generated

    return synthesize(config)


__all__ = ["STRUCTURAL_FIELDS", "SOURCE_FILES", "NrTmDerived", "catalog", "derive", "recipe_label", "resolve", "synthesize_nr_tm",
           "validate_recipe"]
