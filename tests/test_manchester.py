import pytest

from shared.manchester import esi_to_color, has_critical_vitals

NORMAL = dict(
    oxygen_saturation=98,
    systolic_pressure=120,
    heart_rate=80,
    respiratory_rate=16,
    temperature_celsius=36.8,
)


def test_normal_vitals_are_not_critical():
    assert not has_critical_vitals(**NORMAL)


def test_missing_vitals_are_not_critical():
    assert not has_critical_vitals(
        oxygen_saturation=None,
        systolic_pressure=None,
        heart_rate=None,
        respiratory_rate=None,
        temperature_celsius=None,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"oxygen_saturation": 89},
        {"systolic_pressure": 79},
        {"heart_rate": 151},
        {"heart_rate": 39},
        {"respiratory_rate": 31},
        {"respiratory_rate": 7},
        {"temperature_celsius": 40.6},
        {"temperature_celsius": 34.9},
    ],
)
def test_out_of_range_vitals_are_critical(override):
    assert has_critical_vitals(**{**NORMAL, **override})


def test_zero_saturation_is_critical():
    assert has_critical_vitals(**{**NORMAL, "oxygen_saturation": 0})


@pytest.mark.parametrize(
    ("esi", "color"),
    [(1, "Vermelho"), (2, "Laranja"), (3, "Amarelo"), (4, "Verde"), (5, "Azul")],
)
def test_esi_maps_to_manchester_color(esi, color):
    assert esi_to_color(esi, critical=False) == color


def test_critical_vitals_force_red_regardless_of_esi():
    assert esi_to_color(5, critical=True) == "Vermelho"
