"""I-V sweep tab: load-oriented timing, instrument limits, live plot.

Mixed into ``ResistanceMeterApp``. Kept in its own module so the upstream
``main_window.py`` only carries small hooks into it.

* Timing is entered as settle / measure (ms); NPLC is derived and shown
  read-only. An energy limit (J per s window) protects the sample.
* Source and compliance limits follow the identified instrument's
  operating envelope (``source_limits``). The model is remembered per
  machine, so the limits are right before the first connection of a
  session.
* A preview shows points, time under load, duty, total duration and the
  worst-case power/energy; impossible combinations are shown in red and
  block the start.
* In point-by-point mode the plot and readout grow with every point; the
  energy limiter holds the source at 0 whenever the next point could
  exceed the budget.
* Plot options: swap X/Y; ticks in engineering units.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QHBoxLayout, QLabel, QMessageBox

from ..constants import DEFAULT_SETTINGS
from ..instrument import parse_model_from_idn
from ..source_limits import FALLBACK_ENVELOPE, Envelope, envelope_for_model
from ..sweep_plan import DEFAULT_LINE_FREQ_HZ, SweepPlan, format_duration
from .widgets import EngineeringSpinBox, NoScrollSpinBox, format_engineering

_TRACE_COLORS = ['blue', 'red', 'green', 'orange', 'purple']


def _trace_label(n: int) -> str:
    return 'Forward' if n == 1 else 'Reverse' if n == 2 else f'Trace {n}'


def _set_box_range(box, lo: float, hi: float) -> None:
    """Re-range an EngineeringSpinBox and clamp its current value."""
    box._min, box._max = lo, hi
    box.setValue(box.value())


class SweepTabMixin:

    # ------------------------------------------------------ instrument memory

    def _init_instrument_memory(self) -> None:
        self._known_model: Optional[str] = self.config_manager.get_machine_setting('instrument_model')
        self._line_freq_hz: float = float(
            self.config_manager.get_machine_setting('line_frequency_hz', DEFAULT_LINE_FREQ_HZ))

    def _remember_instrument(self, model: Optional[str], line_freq_hz: Optional[float] = None) -> None:
        """Store the identified model (and line frequency) for this machine
        and re-apply the sweep limits."""
        if model and envelope_for_model(model):
            self._known_model = model
            self.config_manager.set_machine_setting('instrument_model', model)
        if line_freq_hz:
            self._line_freq_hz = float(line_freq_hz)
            self.config_manager.set_machine_setting('line_frequency_hz', self._line_freq_hz)
        if hasattr(self, 'tab_sweep'):
            self._update_sweep_limits()

    def _remember_instrument_from_idn(self, idn: str, line_freq_hz: Optional[float] = None) -> str:
        """Parse an *IDN? reply, remember it, return a one-line limit summary."""
        spec = parse_model_from_idn(idn)
        self._remember_instrument(spec.model if spec else None, line_freq_hz)
        env = envelope_for_model(spec.model) if spec else None
        return env.describe() if env else "Model not in limits table — using conservative limits"

    def _sweep_envelope(self) -> Envelope:
        return envelope_for_model(self._known_model) or FALLBACK_ENVELOPE

    # ---------------------------------------------------------------- widgets

    def _build_sweep_timing_rows(self, widget, param_layout) -> None:
        defaults = DEFAULT_SETTINGS['measurement']

        def ms_box(value_s, minimum, tip):
            box = NoScrollSpinBox(decimals=1, minimum=minimum, maximum=60000.0,
                                  singleStep=1.0, suffix=" ms")
            box.setValue(value_s * 1e3)
            box.setToolTip(tip)
            box.valueChanged.connect(lambda _v: self._update_sweep_plan_preview())
            return box

        widget.sweep_delay = ms_box(
            defaults['sweep_delay'], 0.0,
            "Settle time after each level is applied, before the measurement\n"
            "starts. Not part of the measurement, but the sample is under load.")
        widget.sweep_meas_time = ms_box(
            defaults['sweep_meas_time'], 0.1,
            "Integration window per point. NPLC is derived from it using the\n"
            "auto-zero and filter settings (each multiplies the A/D conversions).")
        param_layout.addRow(*self._form_pair(
            "Settle", widget.sweep_delay,
            "Measure", widget.sweep_meas_time,
        ))

        widget.sweep_energy_limit = EngineeringSpinBox(
            unit='J', minimum=0.0, maximum=1e6, default=defaults['sweep_energy_limit'])
        widget.sweep_energy_limit.setToolTip(
            "At most this energy into the sample within any window of the\n"
            "given length. Before each point the worst case (|level| ×\n"
            "compliance × reading timeout) is checked against the budget; if\n"
            "it does not fit, the source waits at 0 until older energy has\n"
            "aged out. 0 = no limit. Live mode only.\nAccepts: 50mJ, 1J, …")
        widget.sweep_energy_limit.valueChanged.connect(lambda _v: self._update_sweep_plan_preview())
        widget.sweep_energy_window = NoScrollSpinBox(
            decimals=1, minimum=0.1, maximum=3600.0, singleStep=1.0, suffix=" s")
        widget.sweep_energy_window.setValue(defaults['sweep_energy_window'])
        widget.sweep_energy_window.setToolTip("Length of the sliding window for the energy limit.")
        widget.sweep_energy_window.valueChanged.connect(lambda _v: self._update_sweep_plan_preview())
        param_layout.addRow(*self._form_pair(
            "Max. energy", widget.sweep_energy_limit,
            "per", widget.sweep_energy_window,
        ))

        widget.sweep_nplc = NoScrollSpinBox(decimals=2, minimum=0.01, maximum=10.0)
        widget.sweep_nplc.setReadOnly(True)
        widget.sweep_nplc.setButtonSymbols(QAbstractSpinBox.NoButtons)
        widget.sweep_nplc.setToolTip("Derived from the measure time — not editable.")

        widget.sweep_point_by_point = QCheckBox("Live (point-by-point)")
        widget.sweep_point_by_point.setChecked(bool(defaults['sweep_point_by_point']))
        widget.sweep_point_by_point.setToolTip(
            "Checked: software steps every point — live plot, power and energy,\n"
            "abort after the current point, no point limit.\n"
            "Unchecked: instrument hardware sweep — faster, results at the end,\n"
            "max 2500 points.")
        widget.sweep_point_by_point.toggled.connect(lambda _c: self._update_sweep_plan_preview())
        param_layout.addRow(*self._form_pair(
            "NPLC", widget.sweep_nplc,
            "Mode", widget.sweep_point_by_point,
        ))

        widget.sweep_limits_label = QLabel()
        widget.sweep_limits_label.setWordWrap(True)
        widget.sweep_limits_label.setStyleSheet("color: #555;")
        param_layout.addRow("Limits", widget.sweep_limits_label)
        widget.sweep_plan_label = QLabel()
        widget.sweep_plan_label.setWordWrap(True)
        widget.sweep_plan_label.setTextFormat(Qt.RichText)
        param_layout.addRow("Plan", widget.sweep_plan_label)

        widget.sweep_compliance.valueChanged.connect(lambda _v: self._update_sweep_plan_preview())
        widget.sweep_direction.currentTextChanged.connect(lambda _t: self._update_sweep_points())

    def _build_sweep_plot_options(self, widget, plot_layout) -> None:
        """Option row above the I-V plot."""
        widget.sweep_swap_axes = QCheckBox("Swap X/Y axes")
        widget.sweep_swap_axes.setToolTip(
            "Plot the measured quantity on X and the source on Y\n"
            "(e.g. voltage over current). Applies to the current plot too.")
        widget.sweep_swap_axes.toggled.connect(widget.iv_canvas.set_swapped)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(widget.sweep_swap_axes)
        plot_layout.addLayout(row)

    # ----------------------------------------------------------------- limits

    def _update_sweep_limits(self, w=None) -> None:
        w = w or self.tab_sweep
        env = self._sweep_envelope()
        if w.sweep_source.currentText() == 'voltage':
            smax, step_min, cmin, cmax = env.v_max, 1e-6, 1e-7, env.i_max
        else:
            smax, step_min, cmin, cmax = env.i_max, 1e-9, 1e-3, env.v_max
        _set_box_range(w.sweep_start, -smax, smax)
        _set_box_range(w.sweep_stop, -smax, smax)
        _set_box_range(w.sweep_step, step_min, 2 * smax)
        _set_box_range(w.sweep_compliance, cmin, cmax)
        if env is FALLBACK_ENVELOPE:
            txt = ("Instrument not identified yet — conservative limits. "
                   "Press Test Connection.")
        else:
            txt = env.describe()
        w.sweep_limits_label.setText(txt)
        self._update_sweep_plan_preview(w)

    # ------------------------------------------------------------------- plan

    def _sweep_settings_from_ui(self, w=None) -> dict:
        """Sweep keys as the worker expects them (seconds, not ms)."""
        w = w or self.tab_sweep
        return {
            'sweep_delay': w.sweep_delay.value() / 1e3,
            'sweep_meas_time': w.sweep_meas_time.value() / 1e3,
            'sweep_energy_limit': w.sweep_energy_limit.value(),
            'sweep_energy_window': w.sweep_energy_window.value(),
            'sweep_point_by_point': w.sweep_point_by_point.isChecked(),
        }

    def _sweep_plan_from_ui(self, w=None) -> SweepPlan:
        w = w or self.tab_sweep
        base = (self.user_settings or DEFAULT_SETTINGS)['measurement']
        m = {
            'auto_zero': base.get('auto_zero', 'once'),
            'filter_enabled': base.get('filter_enabled', False),
            'filter_type': base.get('filter_type', 'repeat'),
            'filter_count': base.get('filter_count', 1),
            'sweep_source': w.sweep_source.currentText(),
            'sweep_start': w.sweep_start.value(),
            'sweep_stop': w.sweep_stop.value(),
            'sweep_step': w.sweep_step.value(),
            'sweep_compliance': w.sweep_compliance.value(),
            'sweep_direction': w.sweep_direction.currentText(),
        }
        m.update(self._sweep_settings_from_ui(w))
        return SweepPlan.from_settings(m, self._line_freq_hz)

    def _sweep_plan_errors(self, plan: SweepPlan) -> list[str]:
        return plan.errors(self._sweep_envelope())

    def _update_sweep_plan_preview(self, w=None) -> None:
        w = w or getattr(self, 'tab_sweep', None)
        if w is None or not hasattr(w, 'sweep_plan_label'):
            return
        plan = self._sweep_plan_from_ui(w)
        w.sweep_nplc.setValue(plan.nplc)
        if plan.total_s - plan.total_min_s < 0.05 * plan.total_s:
            total = f"≈ <b>{format_duration(plan.total_s)}</b>"
        else:
            total = (f"<b>{format_duration(plan.total_min_s)} … "
                     f"{format_duration(plan.total_s)}</b>")
        lines = [
            f"Under load {plan.on_time_s * 1e3:.1f} ms/point · duty ≥ {plan.duty * 100:.0f} %",
            f"Total {total} ({plan.n_points} points, {plan.conversions} conv./reading, "
            f"{plan.line_freq_hz:g} Hz)",
            f"Worst case: peak {format_engineering(plan.peak_power_w, 'W', 3)}, "
            f"energy ≤ <b>{format_engineering(plan.worst_case_energy_j, 'J', 3)}</b>",
        ]
        if plan.limited:
            top = plan.point_worst_energy_j(plan.source_abs_max)
            lines.append(
                f"Limit {format_engineering(plan.energy_limit_j, 'J', 3)} per "
                f"{plan.energy_window_s:g} s — worst case per point ≤ "
                f"{format_engineering(top, 'J', 3)} (timeout {plan.point_time_bound_s:g} s)")
        for warn in plan.warnings:
            lines.append(f"<span style='color:#b36b00'>{warn}</span>")
        for err in self._sweep_plan_errors(plan):
            lines.append(f"<span style='color:red'><b>{err}</b></span>")
        w.sweep_plan_label.setText("<br>".join(lines))

    def _sweep_preflight_ok(self) -> bool:
        errs = self._sweep_plan_errors(self._sweep_plan_from_ui())
        if errs:
            QMessageBox.warning(self, "Sweep not possible", "\n".join(errs))
            return False
        return True

    # ------------------------------------------------------------- live data

    def _on_sweep_started(self, w, settings: dict) -> None:
        m = settings['measurement']
        self._sweep_live = bool(m.get('sweep_point_by_point', False))
        self._sweep_live_trace = -1
        self._sweep_live_count = 0
        plan = SweepPlan.from_settings(m, self._line_freq_hz)
        self._sweep_live_total = plan.n_points
        self._sweep_window_s = plan.energy_window_s
        self._sweep_comp_count = 0
        self._sweep_energy_j = 0.0
        self._set_sweep_axis_labels(w)

    def _set_sweep_axis_labels(self, w) -> None:
        if w.sweep_source.currentText() == 'voltage':
            w.iv_canvas.set_labels('Voltage (V)', 'Current (A)', 'I-V Characteristic')
        else:
            w.iv_canvas.set_labels('Current (A)', 'Voltage (V)', 'V-I Characteristic')

    def _sweep_xy(self, w, voltages, currents):
        """Source quantity on x, measured quantity on y."""
        if w.sweep_source.currentText() == 'voltage':
            return voltages, currents
        return currents, voltages

    def on_sweep_point(self, trace: int, v: float, i: float, comp: str,
                       power: float, energy: float, segment: float) -> None:
        w = self.tab_sweep
        self._sweep_live_count += 1
        self._sweep_energy_j = energy
        if comp != 'OK':
            self._sweep_comp_count += 1
        if not self._sweep_live:
            # Hardware sweep: points arrive in one burst at the end and are
            # plotted by on_sweep_complete.
            return
        if trace != self._sweep_live_trace:
            self._sweep_live_trace = trace
            n = trace + 1
            w.iv_canvas.start_live_trace(_trace_label(n), _TRACE_COLORS[(n - 1) % len(_TRACE_COLORS)])
        x, y = self._sweep_xy(w, v, i)
        w.iv_canvas.append_live_point(x, y)
        flag = f"  ⚠ {comp}" if comp != 'OK' else ""
        w.live_readout.setText(
            f"{self._sweep_live_count}/{self._sweep_live_total} | "
            f"{format_engineering(v, 'V', 4)}  {format_engineering(i, 'A', 4)} | "
            f"P {format_engineering(power, 'W', 3)} | "
            f"E {format_engineering(energy, 'J', 3)} "
            f"(last {self._sweep_window_s:g} s: {format_engineering(segment, 'J', 3)}){flag}"
        )

    def _sweep_summary_text(self) -> str:
        """Readout after a sweep (or a finished trace): totals incl. energy."""
        n, comp = self._sweep_live_count, self._sweep_comp_count
        txt = f"{n} points"
        if comp:
            txt += f" | {comp} in compliance"
        return txt + f" | E {format_engineering(self._sweep_energy_j, 'J', 3)}"
