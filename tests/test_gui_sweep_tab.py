"""GUI tests for the sweep-tab extensions (limits, plan preview, settings).

Named test_gui_* so it runs next to test_gui_smoke, before any module
creates a QCoreApplication.
"""
import os
import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from resistamet_gui.ui.main_window import ResistanceMeterApp
    monkeypatch.setattr(ResistanceMeterApp, 'select_user', lambda self: None)
    w = ResistanceMeterApp()
    w.config_manager.add_user("u")
    w.current_user = "u"
    w.user_settings = w.config_manager.get_user_settings("u")
    w.update_ui_from_settings()
    yield w
    w.close()


def test_unknown_instrument_uses_conservative_limits(win):
    t = win.tab_sweep
    assert "not identified" in t.sweep_limits_label.text()
    t.sweep_stop.setValue(43.0)
    assert t.sweep_stop.value() == pytest.approx(10.5)


def test_identified_2410_limits_and_persistence(win):
    t = win.tab_sweep
    win._remember_instrument("2410", 50.0)
    assert "2410" in t.sweep_limits_label.text()
    assert win.config_manager.get_machine_setting("instrument_model") == "2410"
    t.sweep_stop.setValue(43.0)
    assert t.sweep_stop.value() == pytest.approx(43.0)
    # Current source: compliance is a voltage up to 1100 V, source up to 1.05 A
    t.sweep_source.setCurrentText("current")
    assert t.sweep_compliance._max == pytest.approx(1100.0)
    assert t.sweep_start._max == pytest.approx(1.05)


def test_idn_parsing_remembers_model(win):
    msg = win._remember_instrument_from_idn(
        "KEITHLEY INSTRUMENTS INC.,MODEL 2410,0643894,C34", 50.0)
    assert "1100V" in msg
    assert win._known_model == "2410"


def test_plan_preview_flags_unreachable_compliance(win):
    t = win.tab_sweep
    win._remember_instrument("2410")
    t.sweep_source.setCurrentText("voltage")
    t.sweep_start.setValue(41.0); t.sweep_stop.setValue(43.0)
    t.sweep_step.setValue(0.001); t.sweep_compliance.setValue(0.1)
    txt = t.sweep_plan_label.text()
    assert "Total" in txt and "energy" in txt
    assert "not reachable" in txt
    t.sweep_compliance.setValue(0.02)
    assert "not reachable" not in t.sweep_plan_label.text()


def test_gathered_settings_are_in_seconds(win):
    t = win.tab_sweep
    t.sweep_delay.setValue(5.0); t.sweep_meas_time.setValue(20.0)
    t.sweep_energy_limit.setValue(1.0); t.sweep_energy_window.setValue(10.0)
    t.sweep_point_by_point.setChecked(True)
    m = win.gather_settings_for_mode('sweep')['measurement']
    assert m['sweep_delay'] == pytest.approx(0.005)
    assert m['sweep_meas_time'] == pytest.approx(0.02)
    assert m['sweep_energy_limit'] == 1.0 and m['sweep_energy_window'] == 10.0
    assert m['sweep_point_by_point'] is True
    assert m['nplc'] == pytest.approx(t.sweep_nplc.value())


def test_energy_limit_preview(win):
    t = win.tab_sweep
    win._remember_instrument("2410")
    t.sweep_source.setCurrentText("voltage")
    t.sweep_start.setValue(0.0); t.sweep_stop.setValue(42.0); t.sweep_step.setValue(0.1)
    t.sweep_compliance.setValue(0.02)
    t.sweep_point_by_point.setChecked(True)
    t.sweep_energy_limit.setValue(1.0)
    txt = t.sweep_plan_label.text()
    assert "Limit" in txt and "worst case per point" in txt and "…" in txt
    t.sweep_point_by_point.setChecked(False)
    assert "needs point-by-point" in t.sweep_plan_label.text()


def test_swap_axes_and_engineering_ticks(win):
    t = win.tab_sweep
    c = t.iv_canvas
    ax = c.axes
    c.plot_sweep([0, 1, 2], [0, 1e-3, 2e-3])
    assert ax.get_xlabel() == "Voltage" and ax.get_ylabel() == "Current"
    assert ax.yaxis.get_major_formatter()(1e-3) == "1 mA"
    t.sweep_swap_axes.setChecked(True)               # existing trace is swapped
    assert ax.get_xlabel() == "Current" and ax.get_ylabel() == "Voltage"
    assert list(c._lines[0].get_xdata()) == [0, 1e-3, 2e-3]
    assert ax.xaxis.get_major_formatter()(2e-3) == "2 mA"
    c.clear_plot()                                   # survives a new run
    c.start_live_trace()
    c.append_live_point(5.0, 1e-6)
    assert list(c._lines[0].get_xdata()) == [1e-6]
    assert ax.get_xlabel() == "Current"
    t.sweep_swap_axes.setChecked(False)
    assert list(c._lines[0].get_xdata()) == [5.0]
    assert ax.get_xlabel() == "Voltage"
