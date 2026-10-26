"""Tests for the sweep extensions: operating envelopes, sweep planning
(time under load, energy limit) and the point-by-point / hardware runners."""
from __future__ import annotations

import os
import time

import pytest
import pyvisa

from resistamet_gui.source_limits import FALLBACK_ENVELOPE, envelope_for_model
from resistamet_gui.sweep_plan import (
    HARDWARE_SWEEP_MAX_POINTS, SweepPlan, format_duration, gpib_timeout_at_least,
    sweep_values,
)


# ------------------------------------------------------------------ envelope

class TestEnvelope:
    def test_2410_high_voltage_region_limits_current(self):
        env = envelope_for_model("2410")
        assert env.max_compliance('VOLT', 43.0) == pytest.approx(0.021)
        assert env.max_compliance('VOLT', 20.0) == pytest.approx(1.05)
        assert env.max_compliance('CURR', 0.1) == pytest.approx(21.0)
        assert env.max_compliance('CURR', 0.02) == pytest.approx(1100.0)

    def test_source_outside_envelope_gives_zero(self):
        env = envelope_for_model("2410")
        assert env.max_compliance('VOLT', 1200.0) == 0.0
        assert env.max_compliance('CURR', 2.0) == 0.0

    def test_contains(self):
        env = envelope_for_model("2400")
        assert env.contains(200.0, 0.1)
        assert not env.contains(200.0, 0.5)

    def test_unknown_model(self):
        assert envelope_for_model("9999") is None
        assert envelope_for_model(None) is None
        # Fallback must fit inside every known model.
        for model in ("2400", "2401", "2410", "2420", "2425", "2430", "2440", "2450"):
            env = envelope_for_model(model)
            for r in FALLBACK_ENVELOPE.regions:
                assert env.contains(r.v_max, r.i_max), model


# ---------------------------------------------------------------------- plan

def _m(**kw):
    m = dict(
        sweep_source='voltage', sweep_start=0.0, sweep_stop=1.0, sweep_step=0.25,
        sweep_compliance=0.01, sweep_direction='up', sweep_delay=0.005,
        sweep_meas_time=0.02, sweep_point_by_point=False,
        auto_zero='once', filter_enabled=False, filter_type='repeat', filter_count=1,
    )
    m.update(kw)
    return m


class TestSweepPlan:
    def test_sweep_values_inclusive(self):
        v = sweep_values(41.0, 43.0, 0.001)
        assert len(v) == 2001
        assert v[0] == 41.0 and v[-1] == 43.0

    def test_down_and_up_down(self):
        p = SweepPlan.from_settings(_m(sweep_direction='down'))
        assert p.forward[0] == 1.0 and p.forward[-1] == 0.0
        p = SweepPlan.from_settings(_m(sweep_direction='up_down'))
        assert p.n_points == 10
        assert p.all_values[4] == 1.0 and p.all_values[5] == 1.0

    def test_nplc_derived_from_measure_time(self):
        # 20 ms at 50 Hz, AZ once, no filter -> 1 NPLC
        p = SweepPlan.from_settings(_m(), 50.0)
        assert p.nplc == pytest.approx(1.0)
        # AZ on (3 conversions) x filter 5 -> 15 conversions
        p = SweepPlan.from_settings(_m(auto_zero='on', filter_enabled=True, filter_count=5), 50.0)
        assert p.conversions == 15
        assert p.nplc == pytest.approx(1.0 / 15)

    def test_unreachable_measure_time_is_clamped_with_warning(self):
        p = SweepPlan.from_settings(_m(sweep_meas_time=1.0), 50.0)
        assert p.nplc == 10.0
        assert p.warnings

    def test_without_measure_time_uses_nplc(self):
        m = _m(); del m['sweep_meas_time']; m['nplc'] = 2.0
        assert SweepPlan.from_settings(m, 50.0).nplc == 2.0

    def test_timing_and_worst_case_energy(self):
        p = SweepPlan.from_settings(_m(), 50.0)
        assert p.on_time_s == pytest.approx(p.settle_s + p.meas_s)
        assert p.total_s == pytest.approx(5 * p.on_time_s) == p.total_min_s
        # Σ|V| = 0+0.25+0.5+0.75+1 = 2.5
        assert p.worst_case_energy_j == pytest.approx(2.5 * 0.01 * p.on_time_s)
        assert p.peak_power_w == pytest.approx(0.01)

    def test_gpib_timeout_rounding(self):
        assert gpib_timeout_at_least(0.15) == 0.3
        assert gpib_timeout_at_least(0.3) == 0.3
        assert gpib_timeout_at_least(2.0) == 3.0
        p = SweepPlan.from_settings(_m(sweep_point_by_point=True), 50.0)
        assert p.point_time_bound_s >= 1.5 * p.on_time_s

    def test_energy_limit_slows_worst_case_duration(self):
        free = SweepPlan.from_settings(_m(sweep_point_by_point=True, sweep_stop=42.0,
                                          sweep_step=0.1, sweep_compliance=0.02), 50.0)
        lim = SweepPlan.from_settings(_m(sweep_point_by_point=True, sweep_stop=42.0,
                                         sweep_step=0.1, sweep_compliance=0.02,
                                         sweep_energy_limit=1.0, sweep_energy_window=10.0), 50.0)
        assert lim.limited and not free.limited
        assert lim.errors(None) == []
        assert lim.total_min_s == pytest.approx(free.total_s)
        # worst case: Σ E_point / (1 J / 10 s) at least
        assert lim.total_s >= lim.worst_case_energy_j / 0.1 * 0.9
        assert lim.total_s > 3 * lim.total_min_s

    def test_energy_limit_errors(self):
        # single point could exceed the budget
        p = SweepPlan.from_settings(_m(sweep_point_by_point=True, sweep_stop=42.0, sweep_step=1.0,
                                       sweep_compliance=1.0, sweep_energy_limit=1.0))
        assert any("lower the compliance" in e for e in p.errors(None))
        # hardware sweep cannot enforce it
        p = SweepPlan.from_settings(_m(sweep_energy_limit=1.0))
        assert any("point-by-point" in e for e in p.errors(None))

    def test_errors_against_envelope(self):
        env = envelope_for_model("2410")
        p = SweepPlan.from_settings(_m(sweep_start=41, sweep_stop=43, sweep_step=0.5,
                                       sweep_compliance=0.1))
        assert any("not reachable" in e for e in p.errors(env))
        p = SweepPlan.from_settings(_m(sweep_start=41, sweep_stop=43, sweep_step=0.5,
                                       sweep_compliance=0.02))
        assert p.errors(env) == []
        p = SweepPlan.from_settings(_m(sweep_stop=1200, sweep_step=100))
        assert any("exceeds" in e for e in p.errors(env))

    def test_hardware_buffer_limit(self):
        m = _m(sweep_stop=3.0, sweep_step=0.001)
        assert len(SweepPlan.from_settings(m).forward) > HARDWARE_SWEEP_MAX_POINTS
        assert SweepPlan.from_settings(m).errors(None)
        assert SweepPlan.from_settings(dict(m, sweep_point_by_point=True)).errors(None) == []

    def test_long_hardware_sweep_is_refused(self):
        # 2101 points (fits the buffer) x ~0.5 s > 900 s
        p = SweepPlan.from_settings(_m(sweep_stop=42.0, sweep_step=0.02, sweep_delay=0.3,
                                       sweep_meas_time=0.2))
        assert any("bus timeout" in e for e in p.errors(None))
        p = SweepPlan.from_settings(_m(sweep_stop=42.0, sweep_step=0.02, sweep_delay=0.3,
                                       sweep_meas_time=0.2, sweep_point_by_point=True))
        assert p.errors(None) == []

    def test_format_duration(self):
        assert format_duration(0.25) == "250 ms"
        assert format_duration(12.34) == "12.3 s"
        assert format_duration(216) == "3 min 36 s"
        assert format_duration(7300) == "2 h 01 min"


# ------------------------------------------------------------------- runners

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from resistamet_gui._simulator import FakeKeithley, FakeResourceManager  # noqa: E402
from resistamet_gui.workers import MeasurementWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    # A full QApplication (not QCoreApplication) so GUI test modules can
    # run in the same process in any order.
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_sleep_inhibitor(monkeypatch):
    from resistamet_gui import system_utils
    monkeypatch.setattr(system_utils.SleepInhibitor, "inhibit", lambda self, reason="": True)
    monkeypatch.setattr(system_utils.SleepInhibitor, "uninhibit", lambda self: True)


@pytest.fixture
def rm2410(monkeypatch):
    rm = FakeResourceManager(model="2410")
    monkeypatch.setattr(pyvisa, "ResourceManager", lambda *a, **k: rm)
    return rm


@pytest.fixture
def scpi_log(monkeypatch):
    log = []
    orig = FakeKeithley.write

    def write(self, cmd):
        log.append(cmd)
        return orig(self, cmd)
    monkeypatch.setattr(FakeKeithley, "write", write)
    return log


def _settings(tmp_path, **kw):
    m = {
        "sampling_rate": 10.0, "nplc": 1.0, "settling_time": 0.0,
        "gpib_address": "GPIB0::24::INSTR", "stop_on_compliance": False,
        "auto_zero": "once", "filter_enabled": False, "filter_type": "repeat",
        "filter_count": 1,
        "sweep_source": "voltage", "sweep_start": 0.0, "sweep_stop": 0.5,
        "sweep_step": 0.125, "sweep_compliance": 0.01, "sweep_delay": 0.0,
        "sweep_meas_time": 0.001, "sweep_direction": "up",
        "sweep_point_by_point": True,
    }
    m.update(kw)
    return {
        "measurement": m,
        "display": {"enable_plot": False, "plot_update_interval": 100, "buffer_size": 100},
        "file": {"auto_save_interval": 60, "data_directory": str(tmp_path / "data")},
        "output": {"format": "csv", "compression": "never", "compression_threshold_mb": 5},
    }


def _run(qapp, worker, timeout_s=10.0, on_point=None):
    got = {"points": [], "complete": [], "errors": []}
    worker.sweep_point.connect(lambda *a: (got["points"].append(a), on_point and on_point(worker, got)))
    worker.sweep_complete.connect(lambda v, i, c: got["complete"].append(list(v)))
    worker.error_occurred.connect(got["errors"].append)
    worker.start()
    deadline = time.time() + timeout_s
    while worker.isRunning() and time.time() < deadline:
        qapp.processEvents(); time.sleep(0.005)
    worker.wait(3000)
    for _ in range(5):
        qapp.processEvents(); time.sleep(0.01)
    return got


class TestEnergyWindow:
    def test_energy_ages_out_after_window(self):
        from resistamet_gui.sweep_runner import _EnergyWindow
        w = _EnergyWindow(10.0)
        w.add(0.0, 0.4); w.add(5.0, 0.5)
        assert w.energy(9.9) == pytest.approx(0.9)
        assert w.energy(10.0) == pytest.approx(0.5)
        assert w.next_expiry() == 15.0
        assert w.energy(15.0) == 0.0


class _Clock:
    """Virtual time for the runner: sleep advances it, readings take time."""
    def __init__(self):
        self.t = 1000.0
    def monotonic(self):
        return self.t
    def sleep(self, s):
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    from resistamet_gui import sweep_runner
    c = _Clock()
    monkeypatch.setattr(sweep_runner, "time", c)
    return c


class TestPointByPoint:
    def test_live_points_and_energy(self, qapp, rm2410, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(tmp_path)))
        assert got["errors"] == []
        assert len(got["points"]) == 5
        assert len(got["complete"]) == 1
        energies = [p[5] for p in got["points"]]
        assert energies == sorted(energies) and energies[-1] > 0
        # 100 Ω DUT: P = V²/R at 0.5 V
        assert got["points"][-1][4] == pytest.approx(0.25 / 100, rel=0.05)

    def test_staircase_without_limit(self, qapp, rm2410, scpi_log, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(tmp_path)))
        assert got["errors"] == []
        levels = [c for c in scpi_log if c.startswith(":SOUR:VOLT ")]
        # initial 0, the four non-zero levels (0 V is already applied), final 0
        assert levels == [":SOUR:VOLT 0"] + [f":SOUR:VOLT {v}" for v in
                                             (0.125, 0.25, 0.375, 0.5)] + [":SOUR:VOLT 0"]

    def test_energy_limit_holds_in_every_window(self, qapp, rm2410, scpi_log, clock,
                                                monkeypatch, tmp_path):
        orig = FakeKeithley.query

        def query(self, cmd):
            if cmd.startswith(":READ?"):
                clock.t += 0.1                      # each reading takes 100 ms
            return orig(self, cmd)
        monkeypatch.setattr(FakeKeithley, "query", query)
        limit, window = 1.8e-3, 1.0
        s = _settings(tmp_path, sweep_start=0.4, sweep_stop=0.5, sweep_step=0.025,
                      sweep_meas_time=0.09, sweep_energy_limit=limit,
                      sweep_energy_window=window)
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", s))
        assert got["errors"] == []
        assert len(got["points"]) == 5
        assert all(p[6] <= limit for p in got["points"])
        # the limiter had to wait at 0 at least once ...
        levels = [c for c in scpi_log if c.startswith(":SOUR:VOLT ")]
        assert levels.count(":SOUR:VOLT 0") > 2
        # ... so the run took longer than one window of virtual time
        # (5 readings alone would be 0.5 s)
        assert clock.t - 1000.0 > window

    def test_autozero_once_emulated_on_old_firmware(self, qapp, rm2410, scpi_log,
                                                    monkeypatch, tmp_path):
        orig = FakeKeithley.write

        def write(self, cmd):
            orig(self, cmd)
            if cmd == ":SYST:AZER:STAT ONCE":
                self._queue_error(-102, "Syntax error")
        monkeypatch.setattr(FakeKeithley, "write", write)
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(tmp_path)))
        assert got["errors"] == []
        assert ":SYST:AZER:STAT OFF" in scpi_log
        assert scpi_log.index(":OUTP ON") < scpi_log.index(":SYST:AZER:STAT OFF")

    def test_up_down_two_traces(self, qapp, rm2410, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u",
                                           _settings(tmp_path, sweep_direction="up_down")))
        assert [p[0] for p in got["points"]] == [0] * 5 + [1] * 5
        assert len(got["complete"]) == 2

    def test_abort_stops_after_current_point(self, qapp, rm2410, monkeypatch, tmp_path):
        orig = FakeKeithley.query

        def slow_query(self, cmd):
            if cmd.startswith(":READ?"):
                time.sleep(0.01)
            return orig(self, cmd)
        monkeypatch.setattr(FakeKeithley, "query", slow_query)

        def stop_at_2(worker, got):
            if len(got["points"]) == 2:
                worker.stop_measurement()
        got = _run(qapp, MeasurementWorker(
            "sweep", "s", "u", _settings(tmp_path, sweep_step=0.005)), on_point=stop_at_2)
        assert 2 <= len(got["points"]) < 50

    def test_envelope_violation_blocks_start(self, qapp, rm2410, scpi_log, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(
            tmp_path, sweep_start=41.0, sweep_stop=43.0, sweep_step=1.0,
            sweep_compliance=0.1)))
        assert got["errors"] and "not reachable" in got["errors"][0]
        assert ":OUTP ON" not in scpi_log


class TestSoftwareRanging:
    def test_ranges_up_on_overflow_and_measures_correctly(self, qapp, rm2410, scpi_log, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(tmp_path)))
        assert got["errors"] == []
        assert ":SENS:CURR:RANG:AUTO OFF" in scpi_log
        assert ":SENS:CURR:RANG:AUTO ON" not in scpi_log
        # 100 Ω: every reported current must be the true V/R (no range-limited reading)
        for p in got["points"]:
            assert p[2] == pytest.approx(p[1] / 100.0, rel=1e-6, abs=1e-12)
        assert all(p[3] == 'OK' for p in got["points"])

    def test_never_requests_range_above_compliance(self, qapp, rm2410, scpi_log, tmp_path):
        _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(tmp_path)))
        reqs = [float(c.split()[1]) for c in scpi_log if c.startswith(":SENS:CURR:RANG ")]
        assert reqs and max(reqs) <= 0.01

    def test_hysteresis_no_hunting(self):
        from resistamet_gui.sweep_runner import _Ranger

        class K:
            ranges = (1.05e-6, 1.05e-5, 1.05e-4, 1.05e-3, 1.05e-2)
            def __init__(self): self.r, self.writes = 1.05e-6, 0
            def write(self, c):
                if ":RANG " in c:
                    self.writes += 1
                    v = float(c.split()[1])
                    self.r = next(x for x in self.ranges if v <= x)
            def query(self, c): return repr(self.r)
        plan = SweepPlan.from_settings(_m(sweep_point_by_point=True))
        k = K(); r = _Ranger(k, plan)
        r.prepare(10.3e-6)                 # TVS knee: right at the 10 µA boundary
        assert r.range == pytest.approx(1.05e-4)
        n = k.writes
        for cur in (9.8e-6, 10.4e-6, 9.9e-6, 10.6e-6, 1.2e-5):   # jitter around it
            r.prepare(cur)
        assert r.range == pytest.approx(1.05e-4) and k.writes == n   # no hunting
        r.prepare(3e-6)                    # below 5 % of the range -> down one range
        assert r.range == pytest.approx(1.05e-5)


class TestTimeout:
    def test_timed_out_point_is_skipped_and_sweep_continues(self, qapp, rm2410, scpi_log,
                                                            monkeypatch, tmp_path):
        orig = FakeKeithley.query
        done = {"raised": False}

        def query(self, cmd):
            # first reading at the 0.25 V level times out
            if cmd.startswith(":READ?") and self.state["sour_volt"] == 0.25 and not done["raised"]:
                done["raised"] = True
                raise pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_timeout)
            return orig(self, cmd)
        monkeypatch.setattr(FakeKeithley, "query", query)
        monkeypatch.setattr(FakeKeithley, "clear", lambda self: None, raising=False)
        status = []
        w = MeasurementWorker("sweep", "s", "u", _settings(tmp_path))
        w.status_update.connect(status.append)
        got = _run(qapp, w)
        assert got["errors"] == []
        assert [p[3] for p in got["points"]] == ['OK', 'OK', 'TIMEOUT', 'OK', 'OK']
        assert any("point 2 (setpoint 0.25 V)" in m for m in status)
        # the level is dropped to 0 right after the timeout
        k = scpi_log.index(":SOUR:VOLT 0.25")
        assert ":SOUR:VOLT 0" in scpi_log[k + 1:k + 6]
        assert not any(c.startswith(":SOUR:VOLT 0.") for c in scpi_log[k + 1:scpi_log.index(":SOUR:VOLT 0", k)])


class TestHardwareSweep:
    def test_settle_is_not_overridden(self, qapp, rm2410, scpi_log, tmp_path):
        got = _run(qapp, MeasurementWorker("sweep", "s", "u", _settings(
            tmp_path, sweep_point_by_point=False, sweep_delay=0.003)))
        assert got["errors"] == []
        assert len(got["complete"]) == 1 and len(got["points"]) == 5
        assert ":SOUR:DEL 0.003" in scpi_log
        last_auto = max((k for k, c in enumerate(scpi_log) if c == ":SOUR:DEL:AUTO ON"), default=-1)
        assert last_auto < scpi_log.index(":SOUR:DEL 0.003")
        assert ":OUTP ON" in scpi_log
