import threading
import time
import random
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

from comms_manager import PyLogixComm, ModbusTCPComm, OpcUaComm, InternalMathComm
from pid_controller import InternalPIDController

# ---------------------------------------------------------------------------
# Module-Level Constants
# ---------------------------------------------------------------------------
SCAN_PERIOD_S = 0.1
SCANS_PER_SECOND = int(1 / SCAN_PERIOD_S)
MIN_TIME_CONSTANT = 0.001
SCANS_PER_MINUTE = SCANS_PER_SECOND * 60
WINDOW_EDGE_OFFSET = 7
TASKBAR_HEIGHT = 73
DEFAULT_MODBUS_PORT = 502


# ---------------------------------------------------------------------------
# Periodic Interval Thread
# ---------------------------------------------------------------------------
class PeriodicInterval(threading.Thread):
    """Executes a task function at a fixed periodic interval in a daemon thread."""

    def __init__(self, task_function, period):
        super().__init__()
        self.daemon = True
        self.task_function = task_function
        self.period = period
        self.i = 0
        self.t0 = time.time()
        self.stop_event = threading.Event()
        self.locker = threading.Lock()
        self.start()

    def sleep(self):
        self.i += 1
        delta = self.t0 + self.period * self.i - time.time()
        if delta > 0:
            time.sleep(delta)

    def run(self):
        while not self.stop_event.is_set():
            with self.locker:
                self.task_function()
            self.sleep()

    def stop(self):
        self.stop_event.set()


# ---------------------------------------------------------------------------
# FOPDT Process Model
# ---------------------------------------------------------------------------
class FOPDTModel:
    """First-Order Plus Dead-Time process model with self-regulating and integrating modes."""

    def __init__(self, cv_history, model_data, process_type="Self-Regulating"):
        self.cv_history = cv_history
        self.Gain, self.TimeConstant, self.DeadTime, self.Bias = model_data
        self.process_type = process_type

    def calc(self, PV, ts):
        if (ts - self.DeadTime) <= 0:
            um = 0
        elif int(ts - self.DeadTime) >= len(self.cv_history):
            um = self.cv_history[-1]
        else:
            um = self.cv_history[int(ts - self.DeadTime)]

        if self.process_type == "Integrating":
            dydt = self.Gain * um
        else:
            tc = max(self.TimeConstant, MIN_TIME_CONSTANT)
            dydt = (-(PV - self.Bias) + self.Gain * um) / tc
        return dydt

    def update(self, PV, ts):
        y = odeint(self.calc, PV, ts)
        return y[-1]


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class PIDSimForCLX:
    """PID Simulator with configurable process model and multiple communication backends."""

    def __init__(self):
        self.comm = None
        self.internal_pid = InternalPIDController()
        self._initial_pv = 0.0
        self.reset()
        self._build_gui()

        model_data = (
            float(self.model_gain.get()),
            float(self.model_tc.get()),
            float(self.model_dt.get()),
            float(self.model_bias.get()),
        )
        self.process = FOPDTModel(self.cv_list, model_data, self.process_type_var.get())

    def reset(self):
        self.scan_count = 0
        self.pv_list = []
        self.cv_list = []
        self.sp_list = []
        self.looper = None
        self.anim = None

    # ------------------------------------------------------------------
    # GUI Construction — each section is its own builder method
    # ------------------------------------------------------------------
    def _build_gui(self):
        self._init_window()
        row = 0
        row = self._build_mode_section(row)
        row = self._build_comms_section(row)
        row = self._build_tags_section(row)
        row = self._build_pid_section(row)
        row = self._build_model_section(row)
        row = self._build_buttons(row)
        self._build_status_bar(row)
        self.on_mode_change(self.mode_var.get())

    def _init_window(self):
        self.root = ctk.CTk()
        self.root.title("PID Simulator & Process Model")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.root.resizable(True, True)
        self.root.geometry(
            f"{int(self.screen_width / 2)}x{self.screen_height - TASKBAR_HEIGHT}"
            f"+{-WINDOW_EDGE_OFFSET}+0"
        )
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkScrollableFrame(self.root)
        self.main_frame.pack(expand=True, fill=ctk.BOTH, padx=10, pady=10)

        self.pv_text = ctk.StringVar(value="0.0")
        self.cv_text = ctk.StringVar(value="0.0")
        self.sp_text = ctk.StringVar(value="0.0")
        self.gui_status = ctk.StringVar(value="Ready")

    def _build_mode_section(self, row):
        ctk.CTkLabel(
            self.main_frame, text="Controller Mode:", font=("Arial", 14, "bold")
        ).grid(row=row, column=0, pady=5, sticky=ctk.W)

        self.mode_var = ctk.StringVar(value="PyLogix")
        self.mode_dropdown = ctk.CTkOptionMenu(
            self.main_frame, variable=self.mode_var, width=250,
            values=["PyLogix", "Modbus", "OPC UA", "Internal PID"],
            command=self.on_mode_change,
        )
        self.mode_dropdown.grid(row=row, column=1, pady=5, sticky=ctk.W)
        return row + 1

    def _build_comms_section(self, row):
        ctk.CTkLabel(
            self.main_frame, text="Communication Settings", font=("Arial", 14, "bold")
        ).grid(row=row, column=0, pady=10, sticky=ctk.W)
        row += 1

        ctk.CTkLabel(self.main_frame, text="IP / URL:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.ip = ctk.CTkEntry(self.main_frame, width=250)
        self.ip.insert(0, "192.168.123.100")
        self.ip.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        ctk.CTkLabel(self.main_frame, text="Slot / Port / Unit ID:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.slot = ctk.CTkEntry(self.main_frame, width=100)
        self.slot.insert(0, "2")
        self.slot.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        ctk.CTkLabel(self.main_frame, text="Modbus Scaling Factors (SP, PV, CV):").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        scale_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        scale_frame.grid(row=row, column=1, pady=2, sticky=ctk.W)

        self.scale_sp = ctk.CTkEntry(scale_frame, width=50)
        self.scale_sp.insert(0, "1.0")
        self.scale_sp.pack(side=ctk.LEFT, padx=(0, 5))

        self.scale_pv = ctk.CTkEntry(scale_frame, width=50)
        self.scale_pv.insert(0, "1.0")
        self.scale_pv.pack(side=ctk.LEFT, padx=5)

        self.scale_cv = ctk.CTkEntry(scale_frame, width=50)
        self.scale_cv.insert(0, "10.0")
        self.scale_cv.pack(side=ctk.LEFT, padx=5)
        return row + 1

    def _build_tags_section(self, row):
        ctk.CTkLabel(
            self.main_frame, text="Tags / Addresses / Holding Registers",
            font=("Arial", 14, "bold"),
        ).grid(row=row, column=0, pady=10, sticky=ctk.W)
        row += 1

        tag_defs = [("SP", "sptag", "PID_SP"), ("PV", "pvtag", "PID_PV"), ("CV", "cvtag", "PID_CV")]
        text_vars = {"SP": self.sp_text, "PV": self.pv_text, "CV": self.cv_text}

        for name, attr_name, default in tag_defs:
            ctk.CTkLabel(self.main_frame, text=f"{name} Tag/Addr/Reg:").grid(
                row=row, column=0, pady=2, sticky=ctk.W
            )
            entry = ctk.CTkEntry(self.main_frame, width=250)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=2, sticky=ctk.W)
            setattr(self, attr_name, entry)

            lbl = ctk.CTkLabel(self.main_frame, textvariable=text_vars[name])
            lbl.grid(row=row, column=2, padx=20, sticky=ctk.W)
            row += 1

        return row

    def _build_pid_section(self, row):
        ctk.CTkLabel(
            self.main_frame, text="Internal PID Settings", font=("Arial", 14, "bold")
        ).grid(row=row, column=0, pady=10, sticky=ctk.W)
        row += 1

        # Equation type
        ctk.CTkLabel(self.main_frame, text="Equation Type:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.eq_type_var = ctk.StringVar(value="Parallel")
        self.eq_dropdown = ctk.CTkOptionMenu(
            self.main_frame, variable=self.eq_type_var, width=250,
            values=["Parallel", "Series", "PIDE"], command=self.on_eq_change,
        )
        self.eq_dropdown.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        self.eq_display_var = ctk.StringVar(value="CV = Kp*e + Ki*∫e dt + Kd*de/dt")
        ctk.CTkLabel(self.main_frame, text="Formula:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        ctk.CTkLabel(
            self.main_frame, textvariable=self.eq_display_var, text_color="gray"
        ).grid(row=row, column=1, columnspan=2, pady=2, sticky=ctk.W)
        row += 1

        # Controller action
        ctk.CTkLabel(self.main_frame, text="Controller Action:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.controller_action_var = ctk.StringVar(value="Reverse Acting")
        self.controller_action_dropdown = ctk.CTkOptionMenu(
            self.main_frame, variable=self.controller_action_var, width=250,
            values=["Reverse Acting", "Direct Acting"],
            command=self.on_controller_action_change,
        )
        self.controller_action_dropdown.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        self.action_hint_var = ctk.StringVar(
            value="(e = SP − PV: CV increases when PV is below SP)"
        )
        ctk.CTkLabel(
            self.main_frame, textvariable=self.action_hint_var,
            text_color="gray", font=("Arial", 11, "italic"),
        ).grid(row=row, column=0, columnspan=3, pady=2, sticky=ctk.W)
        row += 1

        # Internal setpoint
        ctk.CTkLabel(self.main_frame, text="Internal SP:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.internal_sp = ctk.CTkEntry(self.main_frame, width=100)
        self.internal_sp.insert(0, "50.0")
        self.internal_sp.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        # PID tuning parameters
        pid_params = ctk.CTkFrame(self.main_frame)
        pid_params.grid(row=row, column=0, columnspan=3, pady=2, sticky=ctk.W)
        ctk.CTkLabel(pid_params, text="Kp:").pack(side=ctk.LEFT, padx=5)
        self.pid_kp = ctk.CTkEntry(pid_params, width=60)
        self.pid_kp.insert(0, "1.0")
        self.pid_kp.pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(pid_params, text="Ki (1/s):").pack(side=ctk.LEFT, padx=5)
        self.pid_ki = ctk.CTkEntry(pid_params, width=60)
        self.pid_ki.insert(0, "0.1")
        self.pid_ki.pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(pid_params, text="Kd (s):").pack(side=ctk.LEFT, padx=5)
        self.pid_kd = ctk.CTkEntry(pid_params, width=60)
        self.pid_kd.insert(0, "0.0")
        self.pid_kd.pack(side=ctk.LEFT, padx=5)
        return row + 1

    def _build_model_section(self, row):
        ctk.CTkLabel(
            self.main_frame, text="Process Model (FOPDT)", font=("Arial", 14, "bold")
        ).grid(row=row, column=0, pady=10, sticky=ctk.W)
        row += 1

        # Process type selection
        ctk.CTkLabel(self.main_frame, text="Process Type:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        self.process_type_var = ctk.StringVar(value="Self-Regulating")
        self.process_type_dropdown = ctk.CTkOptionMenu(
            self.main_frame, variable=self.process_type_var, width=250,
            values=["Self-Regulating", "Integrating"],
            command=self.on_process_type_change,
        )
        self.process_type_dropdown.grid(row=row, column=1, pady=2, sticky=ctk.W)
        row += 1

        self.process_type_hint_var = ctk.StringVar(
            value="(Output settles to a steady-state for a constant CV)"
        )
        ctk.CTkLabel(
            self.main_frame, textvariable=self.process_type_hint_var,
            text_color="gray", font=("Arial", 11, "italic"),
        ).grid(row=row, column=0, columnspan=3, pady=2, sticky=ctk.W)
        row += 1

        # Model parameters
        model_params = [
            ("Gain", "model_gain", "1.45"),
            ("Time Constant (s)", "model_tc", "62.3"),
            ("Dead Time (s)", "model_dt", "10.1"),
            ("Bias", "model_bias", "13.5"),
            ("Noise Amplitude (EU)", "model_noise", "0.05"),
        ]
        for name, attr_name, default in model_params:
            ctk.CTkLabel(self.main_frame, text=f"{name}:").grid(
                row=row, column=0, pady=2, sticky=ctk.W
            )
            entry = ctk.CTkEntry(self.main_frame, width=100)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=2, sticky=ctk.W)
            if attr_name == "model_noise":
                ctk.CTkLabel(
                    self.main_frame, text="(Adds ± random Noise to PV)",
                    text_color="gray", font=("Arial", 11, "italic"),
                ).grid(row=row, column=2, padx=5, pady=2, sticky=ctk.W)
            setattr(self, attr_name, entry)
            row += 1

        return row

    def _build_buttons(self, row):
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=3, pady=20)

        self.button_start = ctk.CTkButton(
            btn_frame, text="Start Simulator", command=self.start
        )
        self.button_start.pack(side=ctk.LEFT, padx=10)

        self.button_stop = ctk.CTkButton(
            btn_frame, text="Stop Simulator", command=self.stop, state=ctk.DISABLED
        )
        self.button_stop.pack(side=ctk.LEFT, padx=10)

        self.button_livetrend = ctk.CTkButton(
            btn_frame, text="Show Trend", command=self.reopen_live_trend, state=ctk.DISABLED
        )
        self.button_livetrend.pack(side=ctk.LEFT, padx=10)
        return row + 1

    def _build_status_bar(self, row):
        ctk.CTkLabel(self.main_frame, text="Status:").grid(
            row=row, column=0, pady=2, sticky=ctk.W
        )
        ctk.CTkLabel(
            self.main_frame, textvariable=self.gui_status, wraplength=400
        ).grid(row=row, column=1, columnspan=2, pady=2, sticky=ctk.W)

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    def on_eq_change(self, eq_type):
        formulas = {
            "Parallel": "CV = Kp*e + Ki*∫e dt + Kd*de/dt",
            "Series": "CV = Kp*[ e + Ki*∫e dt + Kd*de/dt + (Ki*Kd)*e ]",
            "PIDE": "ΔCV = Kp*Δe + Ki*e*dt + Kd*Δ²e/dt",
        }
        self.eq_display_var.set(formulas.get(eq_type, ""))

    def on_controller_action_change(self, action):
        if action == "Reverse Acting":
            self.action_hint_var.set("(e = SP − PV: CV increases when PV is below SP)")
        else:
            self.action_hint_var.set("(e = PV − SP: CV increases when PV is above SP)")

    def on_process_type_change(self, process_type):
        is_integrating = (process_type == "Integrating")
        if is_integrating:
            self.process_type_hint_var.set(
                "(Output ramps continuously for a constant CV — e.g. level, position)"
            )
            self.model_tc.configure(state=ctk.DISABLED)
            self.model_bias.configure(state=ctk.DISABLED)
        else:
            self.process_type_hint_var.set(
                "(Output settles to a steady-state for a constant CV)"
            )
            self.model_tc.configure(state=ctk.NORMAL)
            self.model_bias.configure(state=ctk.NORMAL)

    def on_mode_change(self, mode):
        is_internal = (mode == "Internal PID")
        state_int = ctk.NORMAL if is_internal else ctk.DISABLED
        state_ext = ctk.DISABLED if is_internal else ctk.NORMAL
        is_modbus = (mode == "Modbus")
        state_modbus = ctk.NORMAL if is_modbus else ctk.DISABLED

        self.eq_dropdown.configure(state=state_int)
        self.controller_action_dropdown.configure(state=state_int)
        self.internal_sp.configure(state=state_int)
        self.pid_kp.configure(state=state_int)
        self.pid_ki.configure(state=state_int)
        self.pid_kd.configure(state=state_int)

        self.ip.configure(state=state_ext)
        self.slot.configure(state=state_ext)
        self.cvtag.configure(state=state_ext)
        self.sptag.configure(state=state_ext)
        self.scale_sp.configure(state=state_modbus)
        self.scale_pv.configure(state=state_modbus)
        self.scale_cv.configure(state=state_modbus)

    # ------------------------------------------------------------------
    # Thread-Safe GUI Helper
    # ------------------------------------------------------------------
    def _set_gui_var(self, var, value):
        """Schedule a StringVar update on the main tkinter thread."""
        self.root.after_idle(var.set, str(value))

    # ------------------------------------------------------------------
    # Simulation Lifecycle
    # ------------------------------------------------------------------
    def _setup_comms(self):
        mode = self.mode_var.get()
        if mode == "PyLogix":
            self.comm = PyLogixComm()
            self.comm.setup(self.ip.get(), self.slot.get())
        elif mode == "Modbus":
            self.comm = ModbusTCPComm()
            try:
                port = int(self.slot.get())
            except ValueError:
                port = DEFAULT_MODBUS_PORT
                self._set_gui_var(self.gui_status, f"Invalid port — defaulting to {DEFAULT_MODBUS_PORT}")
            try:
                s_sp = float(self.scale_sp.get())
            except ValueError:
                s_sp = 1.0
            try:
                s_pv = float(self.scale_pv.get())
            except ValueError:
                s_pv = 1.0
            try:
                s_cv = float(self.scale_cv.get())
            except ValueError:
                s_cv = 1.0
            self.comm.setup(self.ip.get(), 1, port, s_cv, s_sp, s_pv)
        elif mode == "OPC UA":
            self.comm = OpcUaComm()
            self.comm.setup(self.ip.get())
        elif mode == "Internal PID":
            self.comm = InternalMathComm()
            self.internal_pid.reset()
            self.internal_pid.eq_type = self.eq_type_var.get()
            self.internal_pid.action = self.controller_action_var.get()
            self.internal_pid.kp = float(self.pid_kp.get())
            self.internal_pid.ki = float(self.pid_ki.get())
            self.internal_pid.kd = float(self.pid_kd.get())

    def _pre_flight_checks(self):
        self._setup_comms()

        tags_to_test = []
        if self.mode_var.get() != "Internal PID":
            tags_to_test = [self.cvtag.get(), self.sptag.get(), self.pvtag.get()]

        success, err = self.comm.pre_flight(tags_to_test)
        if not success:
            raise Exception(err)

        self._initial_pv = float(self.model_bias.get())
        self.reset()
        self.gui_status.set("Running...")
        self.process.Gain = float(self.model_gain.get())
        self.process.TimeConstant = max(float(self.model_tc.get()) * SCANS_PER_SECOND, MIN_TIME_CONSTANT)
        self.process.DeadTime = float(self.model_dt.get()) * SCANS_PER_SECOND
        self.process.Bias = float(self.model_bias.get())
        self.process.process_type = self.process_type_var.get()
        self.process.cv_history = self.cv_list

        # Lock UI during simulation
        self.button_stop.configure(state=ctk.NORMAL)
        self.button_start.configure(state=ctk.DISABLED)
        self.button_livetrend.configure(state=ctk.DISABLED)

    def start(self):
        try:
            self._pre_flight_checks()
        except Exception as e:
            self.gui_status.set(f"Setup Error: {e}")
        else:
            self.looper = PeriodicInterval(self._scan_cycle, SCAN_PERIOD_S)
            self._show_live_trend()

    def _scan_cycle(self):
        """Single scan cycle — runs in the PeriodicInterval background thread."""
        try:
            is_internal = (self.mode_var.get() == "Internal PID")

            # Read SP and CV from source
            if is_internal:
                ext_cv = 0.0
                ext_sp = float(self.internal_sp.get())
                self._set_gui_var(self.cv_text, "Internal")
                self._set_gui_var(self.sp_text, ext_sp)
            else:
                reads = self.comm.read_cv_sp(self.cvtag.get(), self.sptag.get())
                if (getattr(reads[0], "Status", "Error") != "Success"
                        or getattr(reads[1], "Status", "Error") != "Success"):
                    raise Exception(
                        f"Read Error: {getattr(reads[0], 'Status', 'Err')} / "
                        f"{getattr(reads[1], 'Status', 'Err')}"
                    )
                ext_cv = float(reads[0].Value or 0)
                ext_sp = float(reads[1].Value or 0)
                self._set_gui_var(self.cv_text, round(ext_cv, 3))
                self._set_gui_var(self.sp_text, round(ext_sp, 3))

            # Determine current PV (first scan uses model bias)
            current_pv = self.pv_list[-1] if self.pv_list else self._initial_pv

            # Compute CV
            if is_internal:
                cv_to_use = self.internal_pid.calculate(ext_sp, current_pv, SCAN_PERIOD_S)
                self._set_gui_var(self.cv_text, round(cv_to_use, 3))
            else:
                cv_to_use = ext_cv

            # Update history (O(1) list append instead of O(n) np.append)
            self.cv_list.append(cv_to_use)
            self.sp_list.append(ext_sp)
            self.process.cv_history = self.cv_list

            # Run process model
            ts = [self.scan_count, self.scan_count + 1]
            pv_calc = self.process.update(current_pv, ts)

            # Apply noise
            try:
                noise_amp = float(self.model_noise.get())
            except ValueError:
                noise_amp = 0.0
            noise = random.uniform(-noise_amp, noise_amp) if noise_amp > 0 else 0.0
            new_pv = pv_calc[0] + noise
            self.pv_list.append(new_pv)

            # Write PV to comms backend
            write_res = self.comm.write_pv(self.pvtag.get(), new_pv)
            if getattr(write_res, "Status", "Error") == "Success":
                self._set_gui_var(self.pv_text, round(new_pv, 2))
            else:
                self._set_gui_var(
                    self.gui_status,
                    f"Write Error: {getattr(write_res, 'Status', 'Unknown')}",
                )

        except Exception as e:
            self._set_gui_var(self.gui_status, f"Loop Error: {e}")
        else:
            self.scan_count += 1

    def stop(self):
        try:
            self.button_start.configure(state=ctk.NORMAL)
            self.button_stop.configure(state=ctk.DISABLED)
            if self.anim and len(plt.get_fignums()) > 0:
                self.anim.pause()
                self.anim = None
            if self.looper:
                self.looper.stop()
                self.looper = None
            time.sleep(SCAN_PERIOD_S)
            if self.comm:
                self.comm.close()
            plt.close("all")
        except Exception as e:
            self.gui_status.set(f"Stop Error: {e}")

    # ------------------------------------------------------------------
    # Live Trend Plotting
    # ------------------------------------------------------------------
    def _show_live_trend(self):
        fig = plt.figure()
        self.ax = plt.axes()
        (sp_line,) = self.ax.plot([], [], lw=2, color="Red", label="SP")
        (cv_line,) = self.ax.plot([], [], lw=2, color="Green", label="CV")
        (pv_line,) = self.ax.plot([], [], lw=2, color="Blue", label="PV")

        def init():
            sp_line.set_data([], [])
            pv_line.set_data([], [])
            cv_line.set_data([], [])
            plt.ylabel("EU")
            plt.xlabel("Time (min)")
            plt.suptitle("Live Data")
            plt.legend(loc="upper right")

        def animate(i):
            try:
                # Snapshot lists and trim to common length to avoid
                # race conditions with the scan thread
                n = min(len(self.sp_list), len(self.cv_list), len(self.pv_list))
                if n == 0:
                    return
                sp_arr = np.array(self.sp_list[:n])
                cv_arr = np.array(self.cv_list[:n])
                pv_arr = np.array(self.pv_list[:n])
                x = np.arange(n) / SCANS_PER_MINUTE
                sp_line.set_data(x, sp_arr)
                cv_line.set_data(x, cv_arr)
                pv_line.set_data(x, pv_arr)
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception as e:
                self._set_gui_var(self.gui_status, f"Plot Error: {e}")

        self.anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=60, interval=1000
        )

        mngr = plt.get_current_fig_manager()
        mngr.window.geometry(
            f"{int(self.screen_width / 2)}x{self.screen_height - TASKBAR_HEIGHT}"
            f"+{int(self.screen_width / 2) - WINDOW_EDGE_OFFSET + 1}+0"
        )
        plt.gcf().canvas.mpl_connect("close_event", self._on_plot_close)
        plt.show()

    def _on_plot_close(self, event):
        if self.looper:
            self.button_livetrend.configure(state=ctk.NORMAL)

    def reopen_live_trend(self):
        self.button_livetrend.configure(state=ctk.DISABLED)
        if not plt.get_fignums():
            self._show_live_trend()


if __name__ == "__main__":
    gui_app = PIDSimForCLX()
    gui_app.root.mainloop()
