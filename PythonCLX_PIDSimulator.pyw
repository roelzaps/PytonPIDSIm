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

    def __init__(self, cv_history, model_data, process_type="Self-Regulating", balance_point=50.0):
        self.cv_history = cv_history
        self.Gain, self.TimeConstant, self.DeadTime, self.Bias = model_data
        self.process_type = process_type
        self.balance_point = balance_point

    def calc(self, PV, ts):
        if (ts - self.DeadTime) <= 0:
            # No CV has reached the process yet — use neutral value
            um = self.balance_point if self.process_type == "Integrating" else 0
        elif int(ts - self.DeadTime) >= len(self.cv_history):
            um = self.cv_history[-1]
        else:
            um = self.cv_history[int(ts - self.DeadTime)]

        if self.process_type == "Integrating":
            dydt = self.Gain * (um - self.balance_point)
        else:
            tc = max(self.TimeConstant, MIN_TIME_CONSTANT)
            dydt = (-(PV - self.Bias) + self.Gain * um) / tc
        return dydt

    def update(self, PV, ts):
        y = odeint(self.calc, PV, ts)
        return np.maximum(y[-1], 0.0)


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
        self.comms_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.comms_frame.grid(row=row, column=0, columnspan=3, sticky=ctk.W + ctk.E)

        ctk.CTkLabel(
            self.comms_frame, text="Communication Settings", font=("Arial", 14, "bold")
        ).grid(row=0, column=0, pady=10, sticky=ctk.W)

        self.ip_label = ctk.CTkLabel(self.comms_frame, text="IP / URL:")
        self.ip_label.grid(row=1, column=0, pady=2, sticky=ctk.W)
        self.ip = ctk.CTkEntry(self.comms_frame, width=250)
        self.ip.insert(0, "192.168.123.100")
        self.ip.grid(row=1, column=1, pady=2, sticky=ctk.W)

        self.slot_label = ctk.CTkLabel(self.comms_frame, text="Slot:")
        self.slot_label.grid(row=2, column=0, pady=2, sticky=ctk.W)
        self.slot = ctk.CTkEntry(self.comms_frame, width=100)
        self.slot.insert(0, "2")
        self.slot.grid(row=2, column=1, pady=2, sticky=ctk.W)

        # Modbus scaling (in its own sub-frame for easy hide/show)
        self.modbus_frame = ctk.CTkFrame(self.comms_frame, fg_color="transparent")
        self.modbus_frame.grid(row=3, column=0, columnspan=3, sticky=ctk.W)
        ctk.CTkLabel(self.modbus_frame, text="Scaling Factors (SP, PV, CV):").grid(
            row=0, column=0, pady=2, sticky=ctk.W
        )
        scale_inner = ctk.CTkFrame(self.modbus_frame, fg_color="transparent")
        scale_inner.grid(row=0, column=1, pady=2, sticky=ctk.W)

        self.scale_sp = ctk.CTkEntry(scale_inner, width=50)
        self.scale_sp.insert(0, "1.0")
        self.scale_sp.pack(side=ctk.LEFT, padx=(0, 5))

        self.scale_pv = ctk.CTkEntry(scale_inner, width=50)
        self.scale_pv.insert(0, "1.0")
        self.scale_pv.pack(side=ctk.LEFT, padx=5)

        self.scale_cv = ctk.CTkEntry(scale_inner, width=50)
        self.scale_cv.insert(0, "10.0")
        self.scale_cv.pack(side=ctk.LEFT, padx=5)
        return row + 1

    def _build_tags_section(self, row):
        self.tags_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.tags_frame.grid(row=row, column=0, columnspan=3, sticky=ctk.W + ctk.E)

        self.tags_header = ctk.CTkLabel(
            self.tags_frame, text="Tags / Addresses",
            font=("Arial", 14, "bold"),
        )
        self.tags_header.grid(row=0, column=0, pady=10, sticky=ctk.W)

        tag_defs = [("SP", "sptag", "PID_SP"), ("PV", "pvtag", "PID_PV"), ("CV", "cvtag", "PID_CV")]
        text_vars = {"SP": self.sp_text, "PV": self.pv_text, "CV": self.cv_text}
        self._tag_labels = []

        for i, (name, attr_name, default) in enumerate(tag_defs, start=1):
            lbl = ctk.CTkLabel(self.tags_frame, text=f"{name} Tag:")
            lbl.grid(row=i, column=0, pady=2, sticky=ctk.W)
            self._tag_labels.append(lbl)

            entry = ctk.CTkEntry(self.tags_frame, width=250)
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=2, sticky=ctk.W)
            setattr(self, attr_name, entry)

            val_lbl = ctk.CTkLabel(self.tags_frame, textvariable=text_vars[name])
            val_lbl.grid(row=i, column=2, padx=20, sticky=ctk.W)

        return row + 1

    def _build_pid_section(self, row):
        self.pid_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pid_frame.grid(row=row, column=0, columnspan=3, sticky=ctk.W + ctk.E)
        parent = self.pid_frame  # alias for readability

        ctk.CTkLabel(
            parent, text="Internal PID Settings", font=("Arial", 14, "bold")
        ).grid(row=0, column=0, pady=10, sticky=ctk.W)
        row += 1

        r = 1  # row counter within pid_frame

        # Equation type
        ctk.CTkLabel(parent, text="Equation Type:").grid(
            row=r, column=0, pady=2, sticky=ctk.W
        )
        self.eq_type_var = ctk.StringVar(value="Parallel")
        self.eq_dropdown = ctk.CTkOptionMenu(
            parent, variable=self.eq_type_var, width=250,
            values=["Parallel", "Series", "PIDE"], command=self.on_eq_change,
        )
        self.eq_dropdown.grid(row=r, column=1, pady=2, sticky=ctk.W)
        r += 1

        self.eq_display_var = ctk.StringVar(value="CV = Kp*e + Ki*∫e dt + Kd*de/dt")
        ctk.CTkLabel(parent, text="Formula:").grid(
            row=r, column=0, pady=2, sticky=ctk.W
        )
        ctk.CTkLabel(
            parent, textvariable=self.eq_display_var, text_color="gray"
        ).grid(row=r, column=1, columnspan=2, pady=2, sticky=ctk.W)
        r += 1

        # Controller action
        ctk.CTkLabel(parent, text="Controller Action:").grid(
            row=r, column=0, pady=2, sticky=ctk.W
        )
        self.controller_action_var = ctk.StringVar(value="Reverse Acting")
        self.controller_action_dropdown = ctk.CTkOptionMenu(
            parent, variable=self.controller_action_var, width=250,
            values=["Reverse Acting", "Direct Acting"],
            command=self.on_controller_action_change,
        )
        self.controller_action_dropdown.grid(row=r, column=1, pady=2, sticky=ctk.W)
        r += 1

        self.action_hint_var = ctk.StringVar(
            value="(e = SP − PV: CV increases when PV is below SP)"
        )
        ctk.CTkLabel(
            parent, textvariable=self.action_hint_var,
            text_color="gray", font=("Arial", 11, "italic"),
        ).grid(row=r, column=0, columnspan=3, pady=2, sticky=ctk.W)
        r += 1

        # Internal setpoint
        ctk.CTkLabel(parent, text="Internal SP:").grid(
            row=r, column=0, pady=2, sticky=ctk.W
        )
        self.internal_sp = ctk.CTkEntry(parent, width=100)
        self.internal_sp.insert(0, "50.0")
        self.internal_sp.grid(row=r, column=1, pady=2, sticky=ctk.W)
        r += 1

        # Auto/Manual mode
        ctk.CTkLabel(parent, text="PID Mode:").grid(
            row=r, column=0, pady=2, sticky=ctk.W
        )
        pid_mode_frame = ctk.CTkFrame(parent, fg_color="transparent")
        pid_mode_frame.grid(row=r, column=1, columnspan=2, pady=2, sticky=ctk.W)
        self.pid_mode_var = ctk.StringVar(value="Auto")
        self.pid_auto_rb = ctk.CTkRadioButton(
            pid_mode_frame, text="Auto", variable=self.pid_mode_var,
            value="Auto", command=self._on_pid_mode_change,
        )
        self.pid_auto_rb.pack(side=ctk.LEFT, padx=(0, 15))
        self.pid_manual_rb = ctk.CTkRadioButton(
            pid_mode_frame, text="Manual", variable=self.pid_mode_var,
            value="Manual", command=self._on_pid_mode_change,
        )
        self.pid_manual_rb.pack(side=ctk.LEFT)
        r += 1

        # Manual CV entry (hidden by default)
        self.manual_cv_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.manual_cv_frame.grid(row=r, column=0, columnspan=3, pady=2, sticky=ctk.W)
        ctk.CTkLabel(self.manual_cv_frame, text="Manual CV (%):").grid(
            row=0, column=0, pady=2, sticky=ctk.W
        )
        self.manual_cv_entry = ctk.CTkEntry(self.manual_cv_frame, width=100)
        self.manual_cv_entry.insert(0, "0.0")
        self.manual_cv_entry.grid(row=0, column=1, pady=2, padx=5, sticky=ctk.W)
        ctk.CTkLabel(
            self.manual_cv_frame, text="(Step CV to observe open-loop response)",
            text_color="gray", font=("Arial", 11, "italic"),
        ).grid(row=0, column=2, padx=5, pady=2, sticky=ctk.W)
        self.manual_cv_frame.grid_remove()  # hidden in Auto mode
        r += 1

        # PID tuning parameters
        self.pid_tuning_frame = ctk.CTkFrame(parent)
        self.pid_tuning_frame.grid(row=r, column=0, columnspan=3, pady=2, sticky=ctk.W)
        ctk.CTkLabel(self.pid_tuning_frame, text="Kp:").pack(side=ctk.LEFT, padx=5)
        self.pid_kp = ctk.CTkEntry(self.pid_tuning_frame, width=60)
        self.pid_kp.insert(0, "1.0")
        self.pid_kp.pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(self.pid_tuning_frame, text="Ki (1/s):").pack(side=ctk.LEFT, padx=5)
        self.pid_ki = ctk.CTkEntry(self.pid_tuning_frame, width=60)
        self.pid_ki.insert(0, "0.1")
        self.pid_ki.pack(side=ctk.LEFT, padx=5)
        ctk.CTkLabel(self.pid_tuning_frame, text="Kd (s):").pack(side=ctk.LEFT, padx=5)
        self.pid_kd = ctk.CTkEntry(self.pid_tuning_frame, width=60)
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

        # Balance point frame (integrating mode only)
        self.balance_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.balance_frame.grid(row=row, column=0, columnspan=3, sticky=ctk.W)
        ctk.CTkLabel(self.balance_frame, text="Balance Point (CV):").grid(
            row=0, column=0, pady=2, sticky=ctk.W
        )
        self.model_balance = ctk.CTkEntry(self.balance_frame, width=100)
        self.model_balance.insert(0, "50.0")
        self.model_balance.grid(row=0, column=1, pady=2, sticky=ctk.W)
        ctk.CTkLabel(
            self.balance_frame, text="(CV value where PV holds steady)",
            text_color="gray", font=("Arial", 11, "italic"),
        ).grid(row=0, column=2, padx=5, pady=2, sticky=ctk.W)
        self.balance_frame.grid_remove()  # hidden by default (Self-Regulating)
        row += 1

        # Shared model parameters (always visible)
        shared_params = [
            ("Gain", "model_gain", "1.45"),
            ("Dead Time (s)", "model_dt", "10.1"),
            ("Noise Amplitude (EU)", "model_noise", "0.05"),
        ]
        for name, attr_name, default in shared_params:
            ctk.CTkLabel(self.main_frame, text=f"{name}:").grid(
                row=row, column=0, pady=2, sticky=ctk.W
            )
            entry = ctk.CTkEntry(self.main_frame, width=100)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=2, sticky=ctk.W)
            if attr_name == "model_noise":
                ctk.CTkLabel(
                    self.main_frame, text="(Adds \u00b1 random Noise to PV)",
                    text_color="gray", font=("Arial", 11, "italic"),
                ).grid(row=row, column=2, padx=5, pady=2, sticky=ctk.W)
            setattr(self, attr_name, entry)
            row += 1

        # Self-regulating-only parameters (Time Constant + Bias)
        self.self_reg_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.self_reg_frame.grid(row=row, column=0, columnspan=3, sticky=ctk.W)
        sr_params = [
            ("Time Constant (s)", "model_tc", "62.3"),
            ("Bias", "model_bias", "13.5"),
        ]
        for i, (name, attr_name, default) in enumerate(sr_params):
            ctk.CTkLabel(self.self_reg_frame, text=f"{name}:").grid(
                row=i, column=0, pady=2, sticky=ctk.W
            )
            entry = ctk.CTkEntry(self.self_reg_frame, width=100)
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=2, sticky=ctk.W)
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
            self.self_reg_frame.grid_remove()
            self.balance_frame.grid()
        else:
            self.process_type_hint_var.set(
                "(Output settles to a steady-state for a constant CV)"
            )
            self.self_reg_frame.grid()
            self.balance_frame.grid_remove()

    def _on_pid_mode_change(self):
        """Toggle between Auto (PID calculates CV) and Manual (user sets CV)."""
        is_manual = (self.pid_mode_var.get() == "Manual")
        if is_manual:
            self.manual_cv_frame.grid()
            # Pre-fill manual CV with current PID output
            try:
                current_cv = float(self.cv_text.get())
            except ValueError:
                current_cv = 0.0
            self._set_entry_text(self.manual_cv_entry, str(round(current_cv, 2)))
        else:
            self.manual_cv_frame.grid_remove()
            # Bumpless transfer: seed PID state with current manual CV
            try:
                manual_cv = float(self.manual_cv_entry.get())
            except ValueError:
                manual_cv = 0.0
            self.internal_pid.cv = manual_cv
            self.internal_pid.integral_error = 0.0

    def _set_entry_text(self, entry, text):
        """Clear and set text in a CTkEntry widget."""
        entry.delete(0, ctk.END)
        entry.insert(0, text)

    def on_mode_change(self, mode):
        is_internal = (mode == "Internal PID")
        is_modbus = (mode == "Modbus")
        is_opcua = (mode == "OPC UA")

        # Show/hide entire sections
        if is_internal:
            self.comms_frame.grid_remove()
            self.tags_frame.grid_remove()
            self.pid_frame.grid()
        else:
            self.comms_frame.grid()
            self.tags_frame.grid()
            self.pid_frame.grid_remove()

        # Modbus-specific: show scaling, update labels & defaults
        if is_modbus:
            self.modbus_frame.grid()
            self.slot_label.configure(text="Port:")
            self.ip_label.configure(text="IP Address:")
            self._set_entry_text(self.ip, "127.0.0.1")
            self._set_entry_text(self.slot, str(DEFAULT_MODBUS_PORT))
            self.tags_header.configure(text="Holding Registers")
            for lbl, name in zip(self._tag_labels, ["SP", "PV", "CV"]):
                lbl.configure(text=f"{name} Register:")
        else:
            self.modbus_frame.grid_remove()

        # OPC UA-specific
        if is_opcua:
            self.slot_label.configure(text="")
            self.slot.grid_remove()
            self.ip_label.configure(text="Server URL:")
            self._set_entry_text(self.ip, "opc.tcp://localhost:4840")
            self.tags_header.configure(text="OPC UA Node IDs")
            for lbl, name in zip(self._tag_labels, ["SP", "PV", "CV"]):
                lbl.configure(text=f"{name} Node ID:")
        else:
            self.slot.grid()

        # PyLogix-specific
        if mode == "PyLogix":
            self.slot_label.configure(text="Slot:")
            self.ip_label.configure(text="IP Address:")
            self._set_entry_text(self.ip, "192.168.123.100")
            self._set_entry_text(self.slot, "2")
            self.tags_header.configure(text="PLC Tags")
            for lbl, name in zip(self._tag_labels, ["SP", "PV", "CV"]):
                lbl.configure(text=f"{name} Tag:")

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
        self.process.balance_point = float(self.model_balance.get()) if self.process_type_var.get() == "Integrating" else 0.0
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
                if self.pid_mode_var.get() == "Manual":
                    try:
                        cv_to_use = float(self.manual_cv_entry.get())
                    except ValueError:
                        cv_to_use = 0.0
                    cv_to_use = max(0.0, min(100.0, cv_to_use))
                else:
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
        fig, self.ax = plt.subplots()
        (sp_line,) = self.ax.plot([], [], lw=2, color="Red", label="SP")
        (cv_line,) = self.ax.plot([], [], lw=2, color="Green", label="CV")
        (pv_line,) = self.ax.plot([], [], lw=2, color="Blue", label="PV")

        # Crosshair cursor elements
        vline = self.ax.axvline(x=0, color="gray", ls="--", lw=0.8, visible=False)
        hline = self.ax.axhline(y=0, color="gray", ls="--", lw=0.8, visible=False)
        cursor_annot = self.ax.annotate(
            "", xy=(0, 0), fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#222222", ec="gray", alpha=0.9),
            color="white", visible=False,
            xytext=(15, 15), textcoords="offset points",
        )

        # Snap-to-line dot markers (colored to match each line)
        snap_sp, = self.ax.plot([], [], 'o', color='Red', ms=8, zorder=5, visible=False)
        snap_pv, = self.ax.plot([], [], 'o', color='Blue', ms=8, zorder=5, visible=False)
        snap_cv, = self.ax.plot([], [], 'o', color='Green', ms=8, zorder=5, visible=False)
        snap_dots = [snap_sp, snap_pv, snap_cv]

        # View / scroll state
        TREND_WINDOW_MIN = 5.0   # visible x-axis window width in minutes
        self._trend_paused = False
        self._trend_follow = True  # auto-scroll to latest data
        self._trend_x_offset = 0.0  # manual scroll offset (minutes)
        self._cursor_x_data = None
        self._cursor_sp_data = None
        self._cursor_cv_data = None
        self._cursor_pv_data = None

        def _minutes_to_hhmmss(minutes):
            total_secs = int(minutes * 60)
            h = total_secs // 3600
            m = (total_secs % 3600) // 60
            s = total_secs % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        def init():
            sp_line.set_data([], [])
            pv_line.set_data([], [])
            cv_line.set_data([], [])
            plt.ylabel("EU")
            plt.xlabel("Time (min)")
            plt.suptitle("Live Data")
            plt.legend(loc="upper right")

            # Controls help text in the plot
            help_text = (
                "Controls:\n"
                "  Space ···· Pause / Resume\n"
                "  ← / → ··· Scroll time\n"
                "  Home ····· Jump to live\n"
                "  End ······ Jump to start\n"
                "  Scroll ··· Zoom in / out\n"
                "  Hover ···· Inspect values"
            )
            self.ax.text(
                0.01, 0.02, help_text, transform=self.ax.transAxes,
                fontsize=7, fontfamily="monospace", verticalalignment="bottom",
                color="#888888", alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a1a", ec="#444444", alpha=0.5),
            )

        def animate(i):
            if self._trend_paused:
                return
            try:
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

                # Cache for cursor lookup
                self._cursor_x_data = x
                self._cursor_sp_data = sp_arr
                self._cursor_cv_data = cv_arr
                self._cursor_pv_data = pv_arr

                # Rolling X-axis window
                x_max = x[-1]
                if self._trend_follow:
                    x_right = x_max
                else:
                    x_right = self._trend_x_offset
                x_left = max(0.0, x_right - TREND_WINDOW_MIN)
                self.ax.set_xlim(x_left, x_left + TREND_WINDOW_MIN)

                # Y-axis: autoscale only to visible data
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=True)
            except Exception as e:
                self._set_gui_var(self.gui_status, f"Plot Error: {e}")

        def on_mouse_move(event):
            if (event.inaxes != self.ax or self._cursor_x_data is None
                    or len(self._cursor_x_data) == 0):
                vline.set_visible(False)
                hline.set_visible(False)
                cursor_annot.set_visible(False)
                for dot in snap_dots:
                    dot.set_visible(False)
                fig.canvas.draw_idle()
                return

            # Find nearest data index
            idx = int(np.searchsorted(self._cursor_x_data, event.xdata, side="left"))
            idx = np.clip(idx, 0, len(self._cursor_x_data) - 1)

            x_val = self._cursor_x_data[idx]
            sp_val = self._cursor_sp_data[idx]
            cv_val = self._cursor_cv_data[idx]
            pv_val = self._cursor_pv_data[idx]

            # Update crosshair
            vline.set_xdata([x_val])
            hline.set_ydata([event.ydata])
            vline.set_visible(True)
            hline.set_visible(True)

            # Update annotation
            time_str = _minutes_to_hhmmss(x_val)
            cursor_annot.set_text(
                f"Time: {time_str}\n"
                f"SP:   {sp_val:>8.2f}\n"
                f"PV:   {pv_val:>8.2f}\n"
                f"CV:   {cv_val:>8.2f}"
            )
            cursor_annot.xy = (x_val, event.ydata)
            cursor_annot.set_visible(True)

            # Update snap dots on each line
            snap_sp.set_data([x_val], [sp_val])
            snap_pv.set_data([x_val], [pv_val])
            snap_cv.set_data([x_val], [cv_val])
            for dot in snap_dots:
                dot.set_visible(True)

            # Flip annotation side near right edge
            xlim = self.ax.get_xlim()
            if x_val > (xlim[0] + xlim[1]) * 0.7:
                cursor_annot.set_position((-130, 15))
            else:
                cursor_annot.set_position((15, 15))

            fig.canvas.draw_idle()

        def _get_x_max():
            """Return the latest time value, or 0 if no data."""
            if self._cursor_x_data is not None and len(self._cursor_x_data) > 0:
                return float(self._cursor_x_data[-1])
            return 0.0

        def on_key_press(event):
            scroll_step = TREND_WINDOW_MIN * 0.25  # 25% of window per press
            if event.key == " ":
                self._trend_paused = not self._trend_paused
                status = "PAUSED" if self._trend_paused else "Live Data"
                plt.suptitle(status, color="orange" if self._trend_paused else "black")
            elif event.key == "left":
                # Scroll backward
                if self._trend_follow:
                    self._trend_x_offset = _get_x_max()
                self._trend_follow = False
                self._trend_x_offset = max(
                    TREND_WINDOW_MIN,
                    self._trend_x_offset - scroll_step,
                )
                plt.suptitle("Scrolling ◀", color="#2196F3")
            elif event.key == "right":
                # Scroll forward
                x_max = _get_x_max()
                self._trend_x_offset = min(
                    x_max,
                    self._trend_x_offset + scroll_step,
                )
                if self._trend_x_offset >= x_max:
                    self._trend_follow = True
                    plt.suptitle(
                        "PAUSED" if self._trend_paused else "Live Data",
                        color="orange" if self._trend_paused else "black",
                    )
                else:
                    plt.suptitle("Scrolling ▶", color="#2196F3")
            elif event.key == "home":
                # Jump to live
                self._trend_follow = True
                plt.suptitle(
                    "PAUSED" if self._trend_paused else "Live Data",
                    color="orange" if self._trend_paused else "black",
                )
            elif event.key == "end":
                # Jump to beginning
                self._trend_follow = False
                self._trend_x_offset = TREND_WINDOW_MIN
                plt.suptitle("Scrolling ◀", color="#2196F3")
            fig.canvas.draw_idle()

        def on_scroll(event):
            """Mouse scroll to zoom the time window."""
            nonlocal TREND_WINDOW_MIN
            if event.inaxes != self.ax:
                return
            if event.button == 'up':
                TREND_WINDOW_MIN = max(0.5, TREND_WINDOW_MIN * 0.8)  # zoom in
            elif event.button == 'down':
                TREND_WINDOW_MIN = min(60.0, TREND_WINDOW_MIN * 1.25)  # zoom out
            fig.canvas.draw_idle()

        self.anim = animation.FuncAnimation(
            fig, animate, init_func=init, interval=1000,
            save_count=1, cache_frame_data=False,
        )

        # Connect interactive events
        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

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
