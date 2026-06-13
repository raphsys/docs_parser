import pytest

from pagereconstruct import PageReconstructInputAdapter, PageReconstructInputError
from tests.pagereconstruct._fixtures import translated_input_data


def test_missing_reconstruction_units_raises():
    data = translated_input_data()
    del data["views"]["reconstruction_units"]
    with pytest.raises(PageReconstructInputError):
        PageReconstructInputAdapter().normalize(data)


def test_normalize_exposes_four_views():
    out = PageReconstructInputAdapter().normalize(translated_input_data())
    assert out["translated_units"] and out["preservation_plan"] and out["exclusion_plan"]
    assert out["units"] and out["page"]["geometry"]["unit"] == "pt"


def test_non_dict_raises():
    with pytest.raises(TypeError):
        PageReconstructInputAdapter().normalize([])
