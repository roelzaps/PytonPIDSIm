class InternalPIDController:
    def __init__(self):
        self.eq_type = "Parallel" # "Parallel", "Series", "PIDE"
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0
        self.last_time = None
        
        # State variables
        self.last_error = 0.0
        self.last_last_error = 0.0 # for PIDE velocity form
        self.integral_error = 0.0
        self.cv = 0.0
        self.cv_min = 0.0
        self.cv_max = 100.0

    def reset(self):
        self.last_error = 0.0
        self.last_last_error = 0.0
        self.integral_error = 0.0
        self.cv = 0.0
        self.last_time = None

    def calculate(self, sp, pv, dt):
        error = sp - pv
        
        if self.eq_type == "Parallel":
            self.integral_error += error * dt
            derivative_error = (error - self.last_error) / dt if dt > 0 else 0.0
            
            p_term = self.kp * error
            i_term = self.ki * self.integral_error
            d_term = self.kd * derivative_error
            
            self.cv = p_term + i_term + d_term
            
        elif self.eq_type == "Series":
            # Interacting PID. Assuming Ki and Kd are 1/tau_i and tau_d
            # Output = Kp * [ e + Ki * int(e) + Kd * de/dt + (Ki*Kd)*e ]
            self.integral_error += error * dt
            derivative_error = (error - self.last_error) / dt if dt > 0 else 0.0
            
            p_term = self.kp * error
            i_term = self.kp * self.ki * self.integral_error
            d_term = self.kp * self.kd * derivative_error
            interact_term = self.kp * (self.ki * self.kd) * error
            
            self.cv = p_term + i_term + d_term + interact_term
            
        elif self.eq_type == "PIDE":
            # Velocity Form
            # dCV = Kp*[de] + Ki*[e]*dt + Kd*[e_n - 2*e_{n-1} + e_{n-2}]/dt
            # Here, the CV accumulates the change.
            de = error - self.last_error
            d2e = error - 2*self.last_error + self.last_last_error
            
            dp = self.kp * de
            di = self.ki * error * dt
            dd = (self.kd * d2e) / dt if dt > 0 else 0.0
            
            self.cv += dp + di + dd

        # Anti-windup clamping
        if self.cv > self.cv_max:
            self.cv = self.cv_max
            # Reset integral accumulation if clamped
            if self.eq_type in ["Parallel", "Series"]:
                self.integral_error -= error * dt
        elif self.cv < self.cv_min:
            self.cv = self.cv_min
            if self.eq_type in ["Parallel", "Series"]:
                self.integral_error -= error * dt

        # Update states
        self.last_last_error = self.last_error
        self.last_error = error
        
        return self.cv
