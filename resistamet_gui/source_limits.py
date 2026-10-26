"""Per-model source/measure operating envelopes for the 2400 family.

``instrument.ModelSpec`` only carries the absolute maxima (e.g. 2410:
1100 V / 1.05 A), but none of these instruments can deliver both at once.
Each datasheet specifies a small set of rectangular operating regions
("±21 V @ ±1.05 A, ±1100 V @ ±21 mA"); a (V, I) point is reachable when it
lies inside at least one of them. This module encodes those rectangles and
answers the two questions the sweep UI needs:

* how far may the source go at all (``max_source``), and
* given the largest |source| of a sweep, how large may the compliance of
  the measured function be (``max_compliance``).

DC values from the Keithley datasheets; pulse-only regions (2430) are
deliberately left out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Region:
    """One rectangular operating region: |V| <= v_max and |I| <= i_max."""
    v_max: float
    i_max: float


@dataclass(frozen=True)
class Envelope:
    model: str
    regions: tuple[Region, ...]

    @property
    def v_max(self) -> float:
        return max(r.v_max for r in self.regions)

    @property
    def i_max(self) -> float:
        return max(r.i_max for r in self.regions)

    def contains(self, v: float, i: float) -> bool:
        return any(abs(v) <= r.v_max and abs(i) <= r.i_max for r in self.regions)

    def max_source(self, source_func: str) -> float:
        """Largest programmable |source| for 'VOLT' or 'CURR'."""
        return self.v_max if source_func == 'VOLT' else self.i_max

    def max_compliance(self, source_func: str, source_abs_max: float) -> float:
        """Largest compliance of the *measured* function that is still
        reachable while sourcing up to ``source_abs_max``.

        Returns 0.0 if the source value itself is outside the envelope.
        """
        s = abs(source_abs_max)
        if source_func == 'VOLT':
            fits = [r.i_max for r in self.regions if s <= r.v_max]
        else:
            fits = [r.v_max for r in self.regions if s <= r.i_max]
        return max(fits) if fits else 0.0

    def describe(self) -> str:
        parts = [f"±{_fmt(r.v_max, 'V')} @ ±{_fmt(r.i_max, 'A')}" for r in self.regions]
        return f"Keithley {self.model}: " + ", ".join(parts)


def _fmt(value: float, unit: str) -> str:
    for scale, prefix in ((1.0, ''), (1e-3, 'm'), (1e-6, 'µ')):
        if abs(value) >= scale:
            return f"{value / scale:g}{prefix}{unit}"
    return f"{value:g}{unit}"


_ENVELOPES: dict[str, Envelope] = {
    "2400": Envelope("2400", (Region(21.0, 1.05), Region(210.0, 0.105))),
    "2401": Envelope("2401", (Region(21.0, 1.05),)),
    "2410": Envelope("2410", (Region(21.0, 1.05), Region(1100.0, 0.021))),
    "2420": Envelope("2420", (Region(21.0, 3.15), Region(63.0, 1.05))),
    "2425": Envelope("2425", (Region(21.0, 3.15), Region(105.0, 1.05))),
    "2430": Envelope("2430", (Region(21.0, 3.15), Region(105.0, 1.05))),
    "2440": Envelope("2440", (Region(10.5, 5.25), Region(42.0, 1.05))),
    "2450": Envelope("2450", (Region(21.0, 1.05), Region(210.0, 0.105))),
}

# Used when no instrument has ever been identified: the region every
# 2400-family member can deliver, so nothing the UI allows can be rejected
# by whichever instrument turns out to be connected.
FALLBACK_ENVELOPE = Envelope("unknown", (Region(10.5, 1.05),))


def envelope_for_model(model: Optional[str]) -> Optional[Envelope]:
    """Envelope for a model number like "2410", or None if unknown."""
    if not model:
        return None
    return _ENVELOPES.get(str(model))
