"""I-V sweep execution for :class:`workers.MeasurementWorker`.

Two strategies, selected by ``SweepPlan.point_by_point``:

* **hardware** — the instrument runs the staircase from its trigger model
  and returns all readings with one ``:READ?``. Fast, but nothing is seen
  until the end and the run cannot be aborted mid-sweep.
* **point-by-point** — software sets each level and reads it. Every point
  is plotted and exported as it arrives, an abort takes effect after the
  current point, and an energy limit (J per time window) can be enforced:
  a point is only started when the worst case it can deliver still fits
  the window budget, otherwise the source waits at 0 (see sweep_plan).
  The measure range is chosen by software (``_Ranger``) instead of the
  instrument's autorange, which hunts when the reading sits at a range
  boundary (seen on a 2410 at a TVS knee: one reading took >10 s).

Both write the same CSV rows: point, V, I, compliance, power, total energy,
energy in the current window, time waited at 0 before the point,
programmed setpoint, measure range (point-by-point only). Energy is |V·I| times the time the level was applied —
measured per point in point-by-point mode (from writing the level to the
end of its reading), the plan's estimate in hardware mode.
"""
from __future__ import annotations

import time
from collections import deque

import pyvisa

from .sweep_plan import SweepPlan

_STAT_BIT_COMPLIANCE = 1 << 3
_STAT_BIT_RANGE_COMPLIANCE = 1 << 16


def _parse_triples(response: str):
    """Yield (V, I, STAT) from a VOLT,CURR,STAT :READ? response."""
    parts = [p.strip() for p in response.split(',') if p.strip()]
    for k in range(0, len(parts), 3):
        try:
            v = float(parts[k])
            i = float(parts[k + 1]) if k + 1 < len(parts) else float('nan')
            stat = int(float(parts[k + 2])) if k + 2 < len(parts) else 0
        except (ValueError, IndexError):
            v, i, stat = float('nan'), float('nan'), 0
        yield v, i, stat


def _comp(stat: int) -> str:
    return 'COMP' if (stat & _STAT_BIT_COMPLIANCE) else 'OK'


class _EnergyWindow:
    """Energy delivered in the last ``window_s`` seconds. Each point's
    energy is booked at its end and expires one window later."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._hist: deque[tuple[float, float]] = deque()
        self._sum = 0.0

    def add(self, t_end: float, energy_j: float) -> None:
        self._hist.append((t_end + self.window_s, energy_j))
        self._sum += energy_j

    def energy(self, now: float) -> float:
        while self._hist and self._hist[0][0] <= now:
            self._sum -= self._hist.popleft()[1]
        return max(self._sum, 0.0) if self._hist else 0.0

    def next_expiry(self) -> float:
        return self._hist[0][0] if self._hist else 0.0


class _Recorder:
    """Accumulates power/energy and writes export rows."""

    def __init__(self, worker, plan: SweepPlan):
        self.worker = worker
        self.plan = plan
        self.index = 0
        self.energy_j = 0.0                       # total delivered
        self.window = _EnergyWindow(plan.energy_window_s)
        self.waited_s = 0.0
        self.timeouts = 0

    def book(self, energy_j: float) -> None:
        """Count energy delivered now (total and limiter window)."""
        self.energy_j += energy_j
        self.window.add(time.monotonic(), energy_j)

    def add(self, trace: int, v: float, i: float, comp: str, on_time_s: float,
            setpoint: float, wait_s: float = 0.0, energy_j: float = None,
            meas_range: float = float('nan')) -> None:
        """Record one point. ``energy_j`` overrides |V·I|·on-time (used when
        the reading is missing and the worst case has to be booked)."""
        p = abs(v * i)
        if energy_j is None:
            energy_j = p * on_time_s if p == p else 0.0   # NaN -> 0
        self.book(energy_j)
        in_window = self.window.energy(time.monotonic())
        try:
            self.worker.exporter.write_row(
                [self.index, v, i, comp, p, self.energy_j, in_window, wait_s, setpoint,
                 meas_range])
        except Exception:
            pass
        self.worker.sweep_point.emit(trace, v, i, comp, p, self.energy_j, in_window)
        self.index += 1


def run_sweep(worker, plan: SweepPlan) -> None:
    """Run the configured sweep; errors are reported via worker signals."""
    try:
        if plan.point_by_point:
            _run_point_by_point(worker, plan)
        else:
            _run_hardware(worker, plan)
    except Exception as e:
        worker.error_occurred.emit(f"Sweep error: {str(e)}")
    finally:
        try:
            worker.keithley.write(":OUTP OFF")
        except Exception:
            pass


def emulate_autozero_once(worker) -> None:
    """Firmware without ``:SYST:AZER:STAT ONCE`` (e.g. 2410 C17) leaves
    auto-zero ON (3 integrations per conversion). Take one reading with
    auto-zero ON so zero/reference are fresh, then switch it OFF — the
    same thing ONCE does. Point-by-point mode only (output on, fixed
    source mode); a hardware sweep keeps auto-zero ON."""
    if not getattr(worker, '_azer_once_unsupported', False):
        return
    k = worker.keithley
    k.query(":READ?")
    k.write(":SYST:AZER:STAT OFF")


# ------------------------------------------------------------------ hardware

def configure_hardware(keithley, plan: SweepPlan) -> int:
    points = keithley.setup_sweep(
        plan.source_func, plan.forward[0], plan.forward[-1],
        abs(plan.forward[1] - plan.forward[0]) if len(plan.forward) > 1 else 0.0,
        plan.compliance, plan.nplc, plan.settle_s,
    )
    keithley.write(":SOUR:DEL:AUTO OFF")
    keithley.write(f":SOUR:DEL {plan.settle_s}")
    keithley.write(":TRIG:DEL 0")
    return points


def _run_hardware(worker, plan: SweepPlan) -> None:
    k = worker.keithley
    rec = _Recorder(worker, plan)
    traces = [plan.forward]
    if plan.up_down:
        traces.append(tuple(reversed(plan.forward)))
    if k.dev:
        # x4: on firmware without auto-zero ONCE the hardware sweep runs
        # with auto-zero ON (3 integrations instead of 1).
        k.dev.timeout = max(10000, int(len(plan.forward) * plan.on_time_s * 4000) + 10000)

    total = 0
    for trace_no, values in enumerate(traces):
        if trace_no > 0:
            worker.status_update.emit("Running reverse sweep...")
            k.write(f":SOUR:{plan.source_func}:START {values[0]}")
            k.write(f":SOUR:{plan.source_func}:STOP {values[-1]}")
        k.write(":OUTP ON")
        response = k.query(":READ?").strip()
        k.write(":OUTP OFF")
        vs, is_, comps = [], [], []
        for n, (v, i, stat) in enumerate(_parse_triples(response)):
            c = _comp(stat)
            vs.append(v); is_.append(i); comps.append(c)
            setpoint = values[n] if n < len(values) else float('nan')
            rec.add(trace_no, v, i, c, plan.on_time_s, setpoint)
        worker.sweep_complete.emit(vs, is_, comps)
        total += len(vs)
    worker.status_update.emit(
        f"Sweep complete: {total} points acquired, ≈{rec.energy_j:.3g} J into the sample"
    )


# ------------------------------------------------------------- point by point

def _configure_point_by_point(keithley, plan: SweepPlan) -> None:
    src = plan.source_func
    meas, meas_sense = ('CURR', "'CURR:DC'") if src == 'VOLT' else ('VOLT', "'VOLT:DC'")
    keithley.write(":SENS:FUNC:CONC OFF")
    keithley.write(f":SOUR:FUNC {src}")
    keithley.write(f":SOUR:{src}:MODE FIX")
    # One fixed range for the whole sweep: no range changes (and glitches)
    # between points.
    keithley.write(f":SOUR:{src}:RANG {plan.source_abs_max}")
    keithley.write(f":SOUR:{src} 0")
    keithley.write(f":SENS:FUNC {meas_sense}")
    keithley.write(f":SENS:{meas}:PROT {plan.compliance}")
    keithley.write(f":SENS:{meas}:NPLC {plan.nplc}")
    keithley.write(":TRIG:COUN 1")
    keithley.write(":TRIG:DEL 0")
    keithley.write(":SOUR:DEL:AUTO OFF")
    keithley.write(f":SOUR:DEL {plan.settle_s}")
    keithley.write(":FORM:ELEM VOLT,CURR,STAT")


def _instrument_errors(keithley) -> list[str]:
    """Drain the instrument error queue (best effort)."""
    errs = []
    for _ in range(10):
        try:
            e = keithley.query(":SYST:ERR?").strip()
        except Exception:
            break
        if not e or e.startswith(('0,', '+0,')):
            break
        errs.append(e)
    return errs


class _Ranger:
    """Software selection of a fixed measure range.

    Before each point the range is set from the previous reading: up when
    it would need more than 1/UP_HEADROOM of the range, down only when it
    is below DOWN_FRACTION of it (hysteresis, no hunting). A reading in
    range compliance (STAT bit 16, the instrument limits the current to
    the range) or at full scale is repeated one range higher. The range
    never exceeds the one that holds the compliance limit — the
    instrument refuses that anyway (error 824).
    """
    UP_HEADROOM = 1.5
    DOWN_FRACTION = 0.05
    FULL_SCALE = 0.98

    def __init__(self, keithley, plan: SweepPlan):
        self.k = keithley
        self.meas = 'CURR' if plan.source_func == 'VOLT' else 'VOLT'
        self.compliance = plan.compliance
        self.k.write(f":SENS:{self.meas}:RANG:AUTO OFF")
        self.max_range = self._set(self.compliance)
        self.range = self._set(0.0)
        self.min_range = self.range
        self.changes = 0

    def _set(self, request: float) -> float:
        self.k.write(f":SENS:{self.meas}:RANG {min(abs(request), self.compliance)}")
        return float(self.k.query(f":SENS:{self.meas}:RANG?"))

    @staticmethod
    def measured(plan: SweepPlan, v: float, i: float) -> float:
        return abs(i) if plan.source_func == 'VOLT' else abs(v)

    def prepare(self, last_abs: float) -> None:
        """Pick the range for the next point from the last reading."""
        if last_abs != last_abs:  # NaN: keep
            return
        want = last_abs * self.UP_HEADROOM
        if want > self.range and self.range < self.max_range:
            new = self._set(want)
        elif want < self.DOWN_FRACTION * self.range and self.range > self.min_range:
            new = self._set(want)
        else:
            return
        if new != self.range:
            self.range = new
            self.changes += 1

    def overflow(self, value_abs: float, stat: int) -> bool:
        """Reading limited by the range (not by the compliance)?"""
        if self.range >= self.max_range:
            return False
        return bool(stat & _STAT_BIT_RANGE_COMPLIANCE) or value_abs >= self.FULL_SCALE * self.range

    def up(self) -> bool:
        if self.range >= self.max_range:
            return False
        self.range = self._set(self.range * self.UP_HEADROOM)
        self.changes += 1
        return True


class _PointTimeout(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def _read_point(keithley, index: int, setpoint: float, plan: SweepPlan) -> str:
    """:READ? one point; on a timeout clear the bus and explain what happened."""
    try:
        return keithley.query(":READ?").strip()
    except pyvisa.errors.VisaIOError as e:
        if e.error_code != pyvisa.constants.StatusCode.error_timeout:
            raise
        try:
            keithley.dev.clear()
        except Exception:
            pass
        errs = _instrument_errors(keithley)
        unit = 'V' if plan.source_func == 'VOLT' else 'A'
        raise _PointTimeout(
            f"No reading at point {index} (setpoint {setpoint:g} {unit}) within "
            f"{plan.point_time_bound_s:g} s. Instrument errors: "
            f"{'; '.join(errs) if errs else 'none'}"
        ) from e


def _sleep_until(worker, deadline: float) -> None:
    while worker.running:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.02))


def _admit(worker, plan: SweepPlan, rec: _Recorder, level: float, at_zero) -> float:
    """Wait (at 0) until the worst case of the next point fits the energy
    budget. Returns the time waited."""
    if not plan.limited:
        return 0.0
    need = plan.point_worst_energy_j(level)
    t0 = time.monotonic()
    waited = False
    while worker.running:
        now = time.monotonic()
        if rec.window.energy(now) + need <= plan.energy_limit_j:
            break
        if rec.window.next_expiry() <= now:
            # Nothing left to age out: the point alone exceeds the budget
            # (SweepPlan.errors() rejects that before the run).
            break
        at_zero()
        waited = True
        _sleep_until(worker, rec.window.next_expiry())
    return time.monotonic() - t0 if waited else 0.0


def _run_point_by_point(worker, plan: SweepPlan) -> None:
    k = worker.keithley
    src = plan.source_func
    _configure_point_by_point(k, plan)
    rec = _Recorder(worker, plan)
    ranger = _Ranger(k, plan)
    traces = [plan.forward]
    if plan.up_down:
        traces.append(tuple(reversed(plan.forward)))

    k.write(":OUTP ON")
    if k.dev:
        k.dev.timeout = 10000
    emulate_autozero_once(worker)
    if k.dev:
        # Bounds the time (and so the energy) of every reading; linux-gpib
        # honours this value exactly (see gpib_timeout_at_least).
        k.dev.timeout = plan.point_time_bound_s * 1000

    state = {'level': 0.0}   # level currently applied

    def at_zero():
        if state['level'] != 0.0:
            k.write(f":SOUR:{src} 0")
            state['level'] = 0.0

    aborted = False
    last_abs = 0.0
    for trace_no, values in enumerate(traces):
        if trace_no > 0:
            worker.status_update.emit("Running reverse sweep...")
        vs, is_, comps = [], [], []
        for s in values:
            if not worker.running:
                aborted = True
                break
            ranger.prepare(last_abs)
            waited = 0.0
            t0 = t_read = time.monotonic()
            try:
                while True:   # one reading, repeated one range up on overflow
                    w = _admit(worker, plan, rec, s, at_zero)
                    waited += w
                    if not worker.running:
                        aborted = True
                        break
                    if state['level'] != s:
                        k.write(f":SOUR:{src} {s}")
                        state['level'] = s
                        t0 = time.monotonic()
                    t_read = time.monotonic()
                    response = _read_point(k, rec.index, s, plan)
                    v, i, stat = next(_parse_triples(response),
                                      (float('nan'), float('nan'), 0))
                    m = _Ranger.measured(plan, v, i)
                    if ranger.overflow(m, stat) and ranger.up():
                        # Book what this reading delivered, measure again.
                        rec.book(abs(v * i) * (time.monotonic() - t0))
                        t0 = time.monotonic()
                        continue
                    break
            except _PointTimeout as e:
                rec.waited_s += waited
                at_zero()
                dt = time.monotonic() - t_read
                rec.timeouts += 1
                worker.status_update.emit(f"Warning: {e} — point skipped")
                # Unknown current: book the worst case.
                rec.add(trace_no, float('nan'), float('nan'), 'TIMEOUT', dt, s, waited,
                        energy_j=abs(s) * plan.compliance * dt, meas_range=ranger.range)
                last_abs = ranger.range  # start the next point higher
                continue
            if aborted:
                break
            rec.waited_s += waited
            on_time = time.monotonic() - t0
            c = _comp(stat)
            vs.append(v); is_.append(i); comps.append(c)
            rec.add(trace_no, v, i, c, on_time, s, waited, meas_range=ranger.range)
            last_abs = m
        if vs:
            worker.sweep_complete.emit(vs, is_, comps)
        if aborted:
            break
    k.write(f":SOUR:{src} 0")
    k.write(":OUTP OFF")
    state_txt = "aborted" if aborted else "complete"
    extra = f", {rec.timeouts} timed out" if rec.timeouts else ""
    worker.status_update.emit(
        f"Sweep {state_txt}: {rec.index} points{extra}, {rec.energy_j:.3g} J into the "
        f"sample, {rec.waited_s:.1f} s waited for the energy limit, "
        f"{ranger.changes} range changes"
    )
