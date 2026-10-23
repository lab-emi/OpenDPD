"""The readable name of a test-model signal (pickers, dataset captures, collections)."""
import pytest

from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier


def recipe(*cells, **kwargs):
    return NrTestModelConfig(carriers=[NrTmCarrier(scs_khz=s, bandwidth_mhz=b) for s, b in cells], **kwargs)


@pytest.mark.parametrize("r,label", [
    (recipe((30, 100), model="NR-FR1-TM3.1a"), "NR-FR1-TM3.1a · 100 MHz · 30 kHz · FDD"),
    (recipe((15, 5), model="NR-FR1-TM1.1", duplex="tdd"), "NR-FR1-TM1.1 · 5 MHz · 15 kHz · TDD"),
    (recipe((30, 100), (30, 100), model="NR-FR1-TM3.1a"), "NR-FR1-TM3.1a · 2 × 100 MHz · 30 kHz · FDD"),
    (recipe((15, 20), (30, 100), model="NR-FR1-TM3.1"), "NR-FR1-TM3.1 · 20+100 MHz · 15/30 kHz · FDD"),
    (recipe((30, 100), (15, 20), model="NR-FR1-TM3.1"), "NR-FR1-TM3.1 · 100+20 MHz · 15/30 kHz · FDD"),
    (recipe((30, 20), model="NR-FR1-TM1.1", layers=2), "NR-FR1-TM1.1 · 20 MHz · 30 kHz · FDD · 2 layers"),
    (recipe((15, 20), model="NR-N-TM", nb_iot={"position": "both", "boost_db": 6}), "NR-N-TM · 20 MHz · 15 kHz · FDD"),
])
def test_label_reads_like_the_recipe(r, label):
    assert nr_tm.recipe_label(r) == label


def test_the_label_depends_on_the_recipe_only_and_stays_short_for_the_largest_recipe():
    base = recipe((30, 20))
    assert nr_tm.recipe_label(base) == nr_tm.recipe_label(base.model_copy(update={"data_convention": "matlab", "subframes": 3}))
    cells = ([(15, 3), (30, 100), (60, 100)] * 6)[:16]            # the label does not depend on whether the combination resolves
    widest = NrTestModelConfig.model_construct(model="NR-FR1-TM3.1b", duplex="tdd", layers=1,
                                               carriers=[NrTmCarrier(scs_khz=s, bandwidth_mhz=b) for s, b in cells])
    label = nr_tm.recipe_label(widest)
    assert len(label) < 100 and label.startswith("NR-FR1-TM3.1b · 3+100+100+3+") and label.endswith("MHz · 15/30/60 kHz · TDD")      # DatasetCapture.label allows 180
