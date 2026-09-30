ESI_TO_COLOR = {
    1: "Vermelho",
    2: "Laranja",
    3: "Amarelo",
    4: "Verde",
    5: "Azul",
}

DEFAULT_COLOR = "Amarelo"
CRITICAL_COLOR = ESI_TO_COLOR[1]


def _outside(value: float | None, low: float = float("-inf"), high: float = float("inf")) -> bool:
    return value is not None and (value < low or value > high)


def has_critical_vitals(
    *,
    oxygen_saturation: float | None,
    systolic_pressure: float | None,
    heart_rate: float | None,
    respiratory_rate: float | None,
    temperature_celsius: float | None,
) -> bool:
    return (
        _outside(oxygen_saturation, low=90)
        or _outside(systolic_pressure, low=80)
        or _outside(heart_rate, 40, 150)
        or _outside(respiratory_rate, 8, 30)
        or _outside(temperature_celsius, 35.0, 40.5)
    )


def esi_to_color(esi: int, *, critical: bool) -> str:
    if critical:
        return CRITICAL_COLOR
    return ESI_TO_COLOR.get(esi, DEFAULT_COLOR)
