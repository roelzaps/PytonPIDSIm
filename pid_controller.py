class InternalPIDController:
    """PID controller supporting Parallel, Series, and PIDE (velocity) equation types."""

    def __init__(self):
        self.eq_type = "Parallel"
        self.action = "Reverse Acting"
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0
        self.last_time = None

        # State variables
        self.last_error = 0.0
        self.last_last_error = 0.0
        self.integral_error = 0.0
        self.cv = 0.0
        self.cv_min = 0.0
        self.cv_max = 100.0

    def reset(self) -> None:
        self.last_error = 0.0
        self.last_last_error = 0.0
        self.integral_error = 0.0
        self.cv = 0.0
        self.last_time = None

    def calculate(self, sp: float, pv: float, dt: float) -> float:
        """Compute the next CV output given setpoint, process value, and time step.

        Args:
            sp: Setpoint value.
            pv: Current process value.
            dt: Time step in seconds.

        Returns:
            Controller output (CV), clamped to [cv_min, cv_max].
        """
        if self.action == "Direct Acting":
            error = pv - sp
        else:
            error = sp - pv

        if self.eq_type == "Parallel":
            self._calculate_parallel(error, dt)
        elif self.eq_type == "Series":
            self._calculate_series(error, dt)
        elif self.eq_type == "PIDE":
            self._calculate_velocity(error, dt)

        self._apply_anti_windup(error, dt)

        # Update state for next scan
        self.last_last_error = self.last_error
        self.last_error = error

        return self.cv

    def _calculate_parallel(self, error: float, dt: float) -> None:
        self.integral_error += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0

        self.cv = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * derivative
        )

    def _calculate_series(self, error: float, dt: float) -> None:
        """Interacting (Series) PID: CV = Kp * [e + Ki*∫e + Kd*de/dt + Ki*Kd*e]."""
        self.integral_error += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0

        self.cv = (
            self.kp * error
            + self.kp * self.ki * self.integral_error
            + self.kp * self.kd * derivative
            + self.kp * (self.ki * self.kd) * error
        )

    def _calculate_velocity(self, error: float, dt: float) -> None:
        """Velocity (PIDE) form: ΔCV accumulated each scan."""
        de = error - self.last_error
        d2e = error - 2 * self.last_error + self.last_last_error

        dp = self.kp * de
        di = self.ki * error * dt
        dd = (self.kd * d2e) / dt if dt > 0 else 0.0

        self.cv += dp + di + dd

    def _apply_anti_windup(self, error: float, dt: float) -> None:
        """Clamp CV and back-calculate integral to prevent windup."""
        if self.cv > self.cv_max:
            self.cv = self.cv_max
            if self.eq_type in ("Parallel", "Series"):
                self.integral_error -= error * dt
        elif self.cv < self.cv_min:
            self.cv = self.cv_min
            if self.eq_type in ("Parallel", "Series"):
                self.integral_error -= error * dt
