"""Timing, load and energy planning for the I-V sweep.

The user thinks about a sweep in terms of how long the sample is under
load, not NPLC. Per point::

    apply S_k -> settle -> measure -> apply S_k+1 ...
    |<------ on-time ------>|

* settle  = ``sweep_delay`` (source delay; excluded from the measurement)
* measure = integration window; NPLC is derived from it

Energy limit (point-by-point only): at most ``sweep_energy_limit`` J in any
``sweep_energy_window`` s. Before each point the runner admits it only if
the energy of the last window plus the worst case of that point
(|S_k| · compliance · ``point_time_bound_s``) fits the budget; otherwise it
waits at 0 until enough energy has aged out. The reading itself is bounded
by the bus timeout ``point_time_bound_s``, so the worst case holds even if
the sample suddenly conducts up to compliance. Energy is counted when a
point ends and ages out one window later — later than it was delivered,
i.e. on the safe side.

The same plan is used by the GUI (preview: duration, duty, worst-case
energy) and by the worker (derived NPLC, validation before output-on), so
the numbers the user sees are the ones that run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .source_limits import Envelope
from .timing import _PER_READING_OVERHEAD_S, _per_conversion_overhead_s

NPLC_MIN = 0.01
NPLC_MAX = 10.0
DEFAULT_LINE_FREQ_HZ = 50.0
# Reading buffer of the 2400 family; a hardware sweep cannot exceed it.
HARDWARE_SWEEP_MAX_POINTS = 2500
# A hardware sweep is one :READ? that must return within the bus timeout;
# linux-gpib's longest finite timeout is 1000 s.
HARDWARE_SWEEP_MAX_S = 900.0

# Bus/loop time per point while the source is applied in point-by-point
# mode (write source + :READ? round-trip). Rough lab-rig estimate, same
# order as timing._PER_READING_OVERHEAD_S per transaction.
_LIVE_ON_OVERHEAD_S = 2 * _PER_READING_OVERHEAD_S

# linux-gpib (pyvisa-py) supports only these timeouts and rounds a request
# *down* to the next one; other VISA backends take any value.
GPIB_TIMEOUTS_S = (10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 10e-3, 30e-3,
                   0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
# Headroom of the per-point timeout over the nominal reading time.
_POINT_TIMEOUT_FACTOR = 1.5


def gpib_timeout_at_least(seconds: float) -> float:
    """Smallest bus timeout that linux-gpib will honour unchanged and that
    is >= ``seconds``."""
    for t in GPIB_TIMEOUTS_S:
        if t >= seconds:
            return t
    return GPIB_TIMEOUTS_S[-1]


def sweep_values(start: float, stop: float, step: float) -> list[float]:
    """Linear staircase from start to stop (inclusive), same point count
    as ``Keithley2400.setup_sweep``."""
    if step <= 0 or start == stop:
        return [start]
    n = int(round(abs(stop - start) / step)) + 1
    if n == 1:
        return [start]
    return [start + (stop - start) * k / (n - 1) for k in range(n)]


def conversions_per_reading(auto_zero: str, filter_enabled: bool,
                            filter_type: str, filter_count: int) -> int:
    """A/D integrations per returned reading (same convention as timing.py)."""
    az = 3 if str(auto_zero).lower() == 'on' else 1
    fc = max(1, int(filter_count)) if (filter_enabled and str(filter_type).lower().startswith('rep')) else 1
    return az * fc


@dataclass(frozen=True)
class SweepPlan:
    source_func: str                 # 'VOLT' or 'CURR'
    forward: tuple[float, ...]       # one direction, in run order
    up_down: bool
    compliance: float
    settle_s: float
    meas_request_s: float
    energy_limit_j: float            # 0 = no limit
    energy_window_s: float
    point_by_point: bool
    auto_zero: str
    conversions: int
    line_freq_hz: float
    nplc: float
    warnings: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------ building

    @classmethod
    def from_settings(cls, m: dict, line_freq_hz: Optional[float] = None) -> 'SweepPlan':
        f = float(line_freq_hz or m.get('line_frequency_hz') or DEFAULT_LINE_FREQ_HZ)
        src = 'VOLT' if m.get('sweep_source', 'voltage') == 'voltage' else 'CURR'
        start = float(m.get('sweep_start', 0.0))
        stop = float(m.get('sweep_stop', 1.0))
        direction = m.get('sweep_direction', 'up')
        if direction == 'down':
            start, stop = stop, start
        values = sweep_values(start, stop, float(m.get('sweep_step', 0.05)))

        conv = conversions_per_reading(
            m.get('auto_zero', 'on'), bool(m.get('filter_enabled', False)),
            m.get('filter_type', 'repeat'), int(m.get('filter_count', 1)),
        )
        warnings = []
        if m.get('sweep_meas_time') is not None:
            meas = max(0.0, float(m['sweep_meas_time']))
            nplc_raw = meas * f / conv
            nplc = min(NPLC_MAX, max(NPLC_MIN, nplc_raw))
            if nplc != nplc_raw:
                warnings.append(
                    f"Measure time {meas * 1e3:g} ms not reachable with "
                    f"{conv} conversion(s)/reading — using NPLC {nplc:g}"
                )
        else:
            nplc = float(m.get('nplc', 1.0))
            meas = nplc * conv / f

        return cls(
            source_func=src, forward=tuple(values),
            up_down=(direction == 'up_down'),
            compliance=float(m.get('sweep_compliance', 0.1)),
            settle_s=max(0.0, float(m.get('sweep_delay', 0.0))),
            meas_request_s=meas,
            energy_limit_j=max(0.0, float(m.get('sweep_energy_limit', 0.0))),
            energy_window_s=max(0.0, float(m.get('sweep_energy_window', 10.0))),
            point_by_point=bool(m.get('sweep_point_by_point', False)),
            auto_zero=str(m.get('auto_zero', 'on')),
            conversions=conv, line_freq_hz=f, nplc=nplc,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------- derived

    @property
    def all_values(self) -> tuple[float, ...]:
        if self.up_down:
            return self.forward + tuple(reversed(self.forward))
        return self.forward

    @property
    def n_points(self) -> int:
        return len(self.all_values)

    @property
    def source_abs_max(self) -> float:
        return max(abs(v) for v in self.forward)

    @property
    def limited(self) -> bool:
        return self.energy_limit_j > 0

    @property
    def meas_s(self) -> float:
        """Effective measurement window incl. A/D settling overhead
        (same model as timing.estimate_max_sample_rate_hz)."""
        az_factor = 3 if self.auto_zero.lower() == 'on' else 1
        filter_n = self.conversions // az_factor
        per_conv = _per_conversion_overhead_s(self.auto_zero)
        return (az_factor * self.nplc / self.line_freq_hz + per_conv) * filter_n

    @property
    def on_time_s(self) -> float:
        """Nominal time the sample sees each level (settle + measure + bus)."""
        t = self.settle_s + self.meas_s
        if self.point_by_point:
            t += _LIVE_ON_OVERHEAD_S
        return t

    @property
    def point_time_bound_s(self) -> float:
        """Hard upper bound on one point: the :READ? bus timeout."""
        return gpib_timeout_at_least(self.on_time_s * _POINT_TIMEOUT_FACTOR)

    def point_worst_energy_j(self, level: float) -> float:
        """Most energy one point can deliver before its reading times out."""
        return abs(level) * self.compliance * self.point_time_bound_s

    @property
    def total_min_s(self) -> float:
        """Duration if the limiter never has to wait."""
        return self.n_points * self.on_time_s

    @property
    def total_s(self) -> float:
        """Duration upper bound: limiter simulated with every point at its
        worst-case admission energy and drawing compliance for its
        nominal on-time."""
        if not (self.point_by_point and self.limited) or self.energy_window_s <= 0:
            return self.total_min_s
        from collections import deque
        t, hist, in_win = 0.0, deque(), 0.0
        for v in self.all_values:
            need = self.point_worst_energy_j(v)
            while True:
                while hist and hist[0][0] <= t:
                    in_win -= hist.popleft()[1]
                if in_win + need <= self.energy_limit_j or not hist:
                    break
                t = hist[0][0]
            t += self.on_time_s
            e = abs(v) * self.compliance * self.on_time_s
            hist.append((t + self.energy_window_s, e))
            in_win += e
        return t

    @property
    def duty(self) -> float:
        """Fraction of the run under load (worst case)."""
        return self.total_min_s / self.total_s if self.total_s > 0 else 1.0

    @property
    def peak_power_w(self) -> float:
        """Upper bound: largest |source| times compliance of the other quantity."""
        return self.source_abs_max * self.compliance

    @property
    def worst_case_energy_j(self) -> float:
        """Σ |S_k| · compliance · on-time — what the sample can absorb at most."""
        return sum(abs(v) for v in self.all_values) * self.compliance * self.on_time_s

    # ---------------------------------------------------------- validation

    def errors(self, envelope: Optional[Envelope]) -> list[str]:
        """Reasons the sweep must not run (empty list = ok)."""
        errs = []
        unit_s, unit_c = ('V', 'A') if self.source_func == 'VOLT' else ('A', 'V')
        if envelope is not None:
            smax = envelope.max_source(self.source_func)
            if self.source_abs_max > smax:
                errs.append(
                    f"|source| {self.source_abs_max:g} {unit_s} exceeds the "
                    f"{envelope.model} limit of {smax:g} {unit_s}"
                )
            else:
                cmax = envelope.max_compliance(self.source_func, self.source_abs_max)
                if self.compliance > cmax:
                    errs.append(
                        f"Compliance {self.compliance:g} {unit_c} not reachable "
                        f"at {self.source_abs_max:g} {unit_s} on the "
                        f"{envelope.model} (max {cmax:g} {unit_c})"
                    )
        if self.limited:
            if not self.point_by_point:
                errs.append("The energy limit needs point-by-point (Live) mode")
            elif self.energy_window_s <= 0:
                errs.append("Energy window must be > 0 s")
            else:
                worst = self.point_worst_energy_j(self.source_abs_max)
                if worst > self.energy_limit_j:
                    errs.append(
                        f"One point at {self.source_abs_max:g} {unit_s} may deliver up to "
                        f"{worst:.3g} J (compliance × {self.point_time_bound_s:g} s) — more "
                        f"than the {self.energy_limit_j:g} J limit; lower the compliance"
                    )
        if not self.point_by_point and self.total_s / (2 if self.up_down else 1) > HARDWARE_SWEEP_MAX_S:
            errs.append(
                f"Hardware sweep would take {format_duration(self.total_s)} — longer than "
                f"the bus timeout allows ({format_duration(HARDWARE_SWEEP_MAX_S)} per "
                f"direction); enable point-by-point mode"
            )
        if not self.point_by_point and len(self.forward) > HARDWARE_SWEEP_MAX_POINTS:
            errs.append(
                f"{len(self.forward)} points exceed the instrument buffer "
                f"({HARDWARE_SWEEP_MAX_POINTS}) — enable point-by-point mode"
            )
        return errs


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"
