"""
Unit tests for the PID Simulator project.

Covers:
  - InternalPIDController: all equation types, controller actions, anti-windup, reset
  - FOPDTModel: self-regulating and integrating process dynamics, dead-time behavior
  - CommResult / InternalMathComm: interface contract verification

Run with:  pytest tests/test_pid_simulator.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from pid_controller import InternalPIDController
from comms_manager import CommResult, InternalMathComm

# FOPDTModel lives inside the main .pyw file — we import it directly.
import importlib
import sys

# Import FOPDTModel from the .pyw file
spec = importlib.util.spec_from_file_location(
    "simulator",
    "PythonCLX_PIDSimulator.pyw",
    submodule_search_locations=[],
)
# We only need FOPDTModel and constants — avoid launching the GUI
# by patching customtkinter before importing the module
import unittest.mock as _m
_ctk_mock = _m.MagicMock()
with _m.patch.dict(sys.modules, {"customtkinter": _ctk_mock}):
    simulator = importlib.util.module_from_spec(spec)
    # Patch the __main__ guard so importing doesn't launch the app
    with _m.patch.object(simulator, "__name__", "simulator"):
        spec.loader.exec_module(simulator)

FOPDTModel = simulator.FOPDTModel
MIN_TIME_CONSTANT = simulator.MIN_TIME_CONSTANT
SCAN_PERIOD_S = simulator.SCAN_PERIOD_S


# ===========================================================================
# InternalPIDController Tests
# ===========================================================================
class TestInternalPIDControllerInit:
    """Verify default state after construction."""

    def test_default_equation_type(self):
        pid = InternalPIDController()
        assert pid.eq_type == "Parallel"

    def test_default_action(self):
        pid = InternalPIDController()
        assert pid.action == "Reverse Acting"

    def test_default_gains(self):
        pid = InternalPIDController()
        assert pid.kp == 1.0
        assert pid.ki == 0.0
        assert pid.kd == 0.0

    def test_default_output_limits(self):
        pid = InternalPIDController()
        assert pid.cv_min == 0.0
        assert pid.cv_max == 100.0


class TestPIDReset:
    """Verify reset clears accumulated state."""

    def test_reset_clears_integral(self):
        pid = InternalPIDController()
        pid.ki = 1.0
        pid.calculate(50.0, 40.0, 0.1)  # accumulate some integral
        pid.reset()
        assert pid.integral_error == 0.0
        assert pid.cv == 0.0
        assert pid.last_error == 0.0
        assert pid.last_last_error == 0.0


class TestParallelPID:
    """Parallel equation: CV = Kp*e + Ki*∫e*dt + Kd*de/dt."""

    @pytest.fixture
    def pid(self):
        p = InternalPIDController()
        p.eq_type = "Parallel"
        return p

    def test_proportional_only(self, pid):
        pid.kp = 2.0
        pid.ki = 0.0
        pid.kd = 0.0
        cv = pid.calculate(sp=50.0, pv=40.0, dt=0.1)
        # error = 50 - 40 = 10, CV = 2.0 * 10 = 20.0
        assert cv == pytest.approx(20.0)

    def test_integral_accumulates(self, pid):
        pid.kp = 0.0
        pid.ki = 1.0
        pid.kd = 0.0
        # First scan: integral = 10 * 0.1 = 1.0
        cv1 = pid.calculate(50.0, 40.0, 0.1)
        assert cv1 == pytest.approx(1.0)
        # Second scan: integral = 1.0 + 10 * 0.1 = 2.0
        cv2 = pid.calculate(50.0, 40.0, 0.1)
        assert cv2 == pytest.approx(2.0)

    def test_derivative_responds_to_error_change(self, pid):
        pid.kp = 0.0
        pid.ki = 0.0
        pid.kd = 1.0
        # First scan: de/dt = (10 - 0) / 0.1 = 100
        cv = pid.calculate(50.0, 40.0, 0.1)
        assert cv == pytest.approx(100.0)  # clamped to cv_max
        # Second scan: same error → de/dt = 0
        pid.cv = 0.0  # reset CV manually to isolate derivative
        cv2 = pid.calculate(50.0, 40.0, 0.1)
        assert cv2 == pytest.approx(0.0)

    def test_zero_error_gives_zero_output(self, pid):
        pid.kp = 5.0
        pid.ki = 0.0
        pid.kd = 0.0
        cv = pid.calculate(50.0, 50.0, 0.1)
        assert cv == pytest.approx(0.0)

    def test_negative_error(self, pid):
        pid.kp = 1.0
        cv = pid.calculate(40.0, 50.0, 0.1)
        # error = 40 - 50 = -10
        assert cv == pytest.approx(0.0)  # clamped to cv_min


class TestSeriesPID:
    """Series (interacting) equation:
    CV = Kp*e + Kp*Ki*∫e + Kp*Kd*de/dt + Kp*Ki*Kd*e."""

    @pytest.fixture
    def pid(self):
        p = InternalPIDController()
        p.eq_type = "Series"
        return p

    def test_proportional_only(self, pid):
        pid.kp = 2.0
        pid.ki = 0.0
        pid.kd = 0.0
        cv = pid.calculate(50.0, 40.0, 0.1)
        # All Ki/Kd terms zero out, CV = Kp * e = 20
        assert cv == pytest.approx(20.0)

    def test_with_all_gains(self, pid):
        pid.kp = 1.0
        pid.ki = 0.5
        pid.kd = 0.2
        cv = pid.calculate(50.0, 40.0, 0.1)
        # error = 10
        # p_term = 1.0 * 10 = 10
        # integral = 10 * 0.1 = 1.0 → i_term = 1.0 * 0.5 * 1.0 = 0.5
        # derivative = (10 - 0) / 0.1 = 100 → d_term = 1.0 * 0.2 * 100 = 20
        # interact = 1.0 * (0.5 * 0.2) * 10 = 1.0
        # Total = 10 + 0.5 + 20 + 1.0 = 31.5, clamped to 100
        assert cv == pytest.approx(31.5)


class TestVelocityPIDE:
    """PIDE velocity form: CV accumulates ΔCV each scan."""

    @pytest.fixture
    def pid(self):
        p = InternalPIDController()
        p.eq_type = "PIDE"
        return p

    def test_cv_accumulates(self, pid):
        pid.kp = 1.0
        pid.ki = 0.0
        pid.kd = 0.0
        # First scan: de = 10 - 0 = 10, dp = 10
        cv1 = pid.calculate(50.0, 40.0, 0.1)
        assert cv1 == pytest.approx(10.0)
        # Second scan: same error → de = 0 → dp = 0 → cv stays at 10
        cv2 = pid.calculate(50.0, 40.0, 0.1)
        assert cv2 == pytest.approx(10.0)


class TestControllerAction:
    """Direct vs Reverse acting changes error sign."""

    def test_reverse_acting_default(self):
        pid = InternalPIDController()
        pid.kp = 1.0
        # error = SP - PV = 50 - 40 = +10 → positive CV
        cv = pid.calculate(50.0, 40.0, 0.1)
        assert cv > 0

    def test_direct_acting_flips_error(self):
        pid = InternalPIDController()
        pid.action = "Direct Acting"
        pid.kp = 1.0
        # error = PV - SP = 40 - 50 = -10 → negative (clamped to 0)
        cv = pid.calculate(50.0, 40.0, 0.1)
        assert cv == pytest.approx(0.0)  # clamped at cv_min

    def test_direct_acting_positive_when_pv_above_sp(self):
        pid = InternalPIDController()
        pid.action = "Direct Acting"
        pid.kp = 1.0
        # error = PV - SP = 60 - 50 = +10 → positive CV
        cv = pid.calculate(50.0, 60.0, 0.1)
        assert cv == pytest.approx(10.0)


class TestAntiWindup:
    """CV clamping and integral back-calculation."""

    def test_cv_clamped_at_max(self):
        pid = InternalPIDController()
        pid.kp = 20.0  # error=10 → CV=200, should clamp to 100
        cv = pid.calculate(50.0, 40.0, 0.1)
        assert cv == pytest.approx(100.0)

    def test_cv_clamped_at_min(self):
        pid = InternalPIDController()
        pid.kp = 20.0
        cv = pid.calculate(40.0, 50.0, 0.1)
        # error = -10 → CV = -200, clamped to 0
        assert cv == pytest.approx(0.0)

    def test_integral_backcalculation_on_clamp(self):
        pid = InternalPIDController()
        pid.kp = 0.0
        pid.ki = 1000.0  # huge Ki to force saturation
        pid.calculate(50.0, 40.0, 0.1)
        # integral would be 10*0.1 = 1.0, CV = 1000 → clamped at 100
        # Back-calc should have reduced integral_error
        assert pid.cv == pytest.approx(100.0)
        assert pid.integral_error == pytest.approx(0.0)


# ===========================================================================
# FOPDTModel Tests
# ===========================================================================
class TestFOPDTModelSelfRegulating:
    """Self-regulating process: output settles to steady state."""

    def test_steady_state_with_constant_cv(self):
        """A constant CV should drive PV toward Gain*CV + Bias."""
        cv_history = [50.0] * 200
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Self-Regulating")
        pv = 0.0
        # Run many iterations toward steady state
        for i in range(199):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
        # Steady state for FOPDT: PV → Gain * CV + Bias = 1.0 * 50 + 0 = 50
        assert pv == pytest.approx(50.0, abs=1.0)

    def test_dead_time_delays_response(self):
        """With dead time, PV shouldn't respond until after the delay."""
        cv_history = [0.0] * 10 + [50.0] * 50
        model = FOPDTModel(cv_history, (1.0, 5.0, 10.0, 0.0), "Self-Regulating")
        pv = 0.0
        # At step 5 (before dead time of 10), PV should still be ~0
        for i in range(5):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
        assert abs(pv) < 1.0  # still near zero

    def test_zero_cv_returns_to_bias(self):
        """With CV=0, PV should decay toward bias."""
        cv_history = [0.0] * 200
        model = FOPDTModel(cv_history, (1.0, 5.0, 0.0, 20.0), "Self-Regulating")
        pv = 50.0  # start above bias
        for i in range(199):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
        assert pv == pytest.approx(20.0, abs=1.0)

    def test_min_time_constant_safeguard(self):
        """TimeConstant of 0 shouldn't cause divide-by-zero."""
        cv_history = [50.0] * 10
        model = FOPDTModel(cv_history, (1.0, 0.0, 0.0, 0.0), "Self-Regulating")
        # Should not raise — MIN_TIME_CONSTANT prevents /0
        pv = model.update(0.0, [0, 1])
        assert np.isfinite(pv[0])


class TestFOPDTModelIntegrating:
    """Integrating process: output ramps relative to balance point."""

    def test_ramps_with_cv_above_balance(self):
        """PV should increase continuously when CV > balance_point."""
        cv_history = [10.0] * 50
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Integrating", balance_point=0.0)
        pv = 0.0
        previous_pv = 0.0
        for i in range(20):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
            assert pv > previous_pv  # always increasing
            previous_pv = pv

    def test_cv_at_balance_holds_steady(self):
        """With CV == balance_point, integrating process shouldn't move."""
        cv_history = [50.0] * 50
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Integrating", balance_point=50.0)
        pv = 25.0
        for i in range(20):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
        assert pv == pytest.approx(25.0, abs=0.01)

    def test_cv_below_balance_ramps_down(self):
        """CV < balance_point should cause PV to decrease (PV floors at 0)."""
        cv_history = [40.0] * 50
        model = FOPDTModel(cv_history, (0.1, 10.0, 0.0, 0.0), "Integrating", balance_point=50.0)
        pv = 500.0  # high start so we have room to ramp down
        for i in range(5):
            ts = [i, i + 1]
            new_pv = model.update(pv, ts)[0]
            assert new_pv < pv
            pv = new_pv


class TestFOPDTModelBalancePoint:
    """Balance point — the CV threshold where PV reverses direction."""

    def test_above_balance_ramps_up(self):
        """CV=80, balance=50 → PV should increase."""
        cv_history = [80.0] * 50
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Integrating", balance_point=50.0)
        pv = 25.0
        for i in range(10):
            ts = [i, i + 1]
            new_pv = model.update(pv, ts)[0]
            assert new_pv > pv
            pv = new_pv

    def test_below_balance_ramps_down(self):
        """CV=20, balance=50 → PV should decrease (PV floors at 0)."""
        cv_history = [20.0] * 50
        model = FOPDTModel(cv_history, (0.1, 10.0, 0.0, 0.0), "Integrating", balance_point=50.0)
        pv = 500.0  # high start so we have room to ramp down
        for i in range(5):
            ts = [i, i + 1]
            new_pv = model.update(pv, ts)[0]
            assert new_pv < pv
            pv = new_pv

    def test_at_balance_holds_steady(self):
        """CV == balance_point → PV should not change."""
        cv_history = [50.0] * 50
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Integrating", balance_point=50.0)
        pv = 75.0
        for i in range(20):
            ts = [i, i + 1]
            pv = model.update(pv, ts)[0]
        assert pv == pytest.approx(75.0, abs=0.01)

    def test_default_balance_is_50(self):
        """Default balance_point should be 50.0."""
        model = FOPDTModel([0.0], (1.0, 10.0, 0.0, 0.0), "Integrating")
        assert model.balance_point == 50.0


class TestFOPDTModelEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_cv_history_uses_zero(self):
        """If ts is before dead time, um should be 0."""
        cv_history = [50.0]
        model = FOPDTModel(cv_history, (1.0, 10.0, 100.0, 0.0), "Self-Regulating")
        # Dead time 100 means at ts=0, um=0
        pv = model.update(0.0, [0, 1])
        assert np.isfinite(pv[0])

    def test_cv_index_beyond_history_uses_last(self):
        """When dead-time adjusted index exceeds history, use last CV value."""
        cv_history = [10.0, 20.0, 30.0]
        model = FOPDTModel(cv_history, (1.0, 10.0, 0.0, 0.0), "Self-Regulating")
        # At ts=50, int(50-0)=50 >= len(3), should use cv_history[-1] = 30
        pv = model.update(0.0, [50, 51])
        assert np.isfinite(pv[0])


# ===========================================================================
# CommResult / InternalMathComm Tests
# ===========================================================================
class TestCommResult:
    """CommResult data container."""

    def test_stores_value_and_status(self):
        r = CommResult(42.5, "Success")
        assert r.Value == 42.5
        assert r.Status == "Success"

    def test_none_value(self):
        r = CommResult(None, "Error")
        assert r.Value is None
        assert r.Status == "Error"


class TestInternalMathComm:
    """InternalMathComm no-op backend."""

    def test_pre_flight_always_passes(self):
        comm = InternalMathComm()
        ok, err = comm.pre_flight([])
        assert ok is True
        assert err == ""

    def test_write_pv_returns_success(self):
        comm = InternalMathComm()
        result = comm.write_pv("any_tag", 42.0)
        assert result.Status == "Success"
        assert result.Value == 42.0

    def test_read_returns_zeros(self):
        comm = InternalMathComm()
        results = comm.read_cv_sp("cv", "sp")
        assert len(results) == 2
        assert results[0].Value == 0.0
        assert results[1].Value == 0.0

    def test_close_does_not_raise(self):
        comm = InternalMathComm()
        comm.close()  # should not raise
