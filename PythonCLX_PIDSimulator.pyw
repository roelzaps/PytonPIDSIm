import threading
import time
import random
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

# Import our new modules
from comms_manager import PyLogixComm, ModbusTCPComm, OpcUaComm, InternalMathComm
from pid_controller import InternalPIDController

class PeriodicInterval(threading.Thread):
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

class FOPDTModel(object):
    def __init__(self, CV, ModelData):
        self.CV = CV
        self.Gain, self.TimeConstant, self.DeadTime, self.Bias = ModelData

    def calc(self, PV, ts):
        if (ts - self.DeadTime) <= 0:
            um = 0
        elif int(ts - self.DeadTime) >= len(self.CV):
            um = self.CV[-1]
        else:
            um = self.CV[int(ts - self.DeadTime)]
        # Add safeguard to avoid divide by zero if TC=0
        tc = max(self.TimeConstant, 0.001)
        dydt = (-(PV - self.Bias) + self.Gain * um) / tc
        return dydt

    def update(self, PV, ts):
        y = odeint(self.calc, PV, ts)
        return y[-1]

class PIDSimForCLX(object):
    def __init__(self):
        self.comm = None
        self.internal_pid = InternalPIDController()
        self.reset()
        self.gui_setup()
        
        # default FOPDT
        model = (float(self.model_gain.get()), float(self.model_tc.get()), float(self.model_dt.get()), float(self.model_bias.get()))
        self.process = FOPDTModel(self.CV, model)

    def reset(self):
        self.scan_count = 0
        self.PV = np.zeros(0)
        self.CV = np.zeros(0)
        self.SP = np.zeros(0)
        self.looper = None
        self.anim = None

    def gui_setup(self):
        self.root = ctk.CTk()
        self.root.title("PID Simulator & Process Model")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.offset = 7
        self.toolbar = 73

        self.root.resizable(True, True)
        self.root.geometry(f"{int(self.screen_width/2)}x{self.screen_height-self.toolbar}+{-self.offset}+0")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkScrollableFrame(self.root)
        self.main_frame.pack(expand=True, fill=ctk.BOTH, padx=10, pady=10)

        # Variables
        self.pv_text = ctk.StringVar(value="0.0")
        self.cv_text = ctk.StringVar(value="0.0")
        self.sp_text = ctk.StringVar(value="0.0")
        self.gui_status = ctk.StringVar(value="Ready")
        
        row_idx = 0
        
        # --- MODE SELECTION ---
        ctk.CTkLabel(self.main_frame, text="Controller Mode:", font=("Arial", 14, "bold")).grid(row=row_idx, column=0, pady=5, sticky=ctk.W)
        self.mode_var = ctk.StringVar(value="PyLogix")
        self.mode_dropdown = ctk.CTkOptionMenu(self.main_frame, variable=self.mode_var, width=250,
                                               values=["PyLogix", "Modbus", "OPC UA", "Internal PID"],
                                               command=self.on_mode_change)
        self.mode_dropdown.grid(row=row_idx, column=1, pady=5, sticky=ctk.W)
        row_idx += 1
        
        # --- COMMS SETTINGS ---
        ctk.CTkLabel(self.main_frame, text="Communication Settings", font=("Arial", 14, "bold")).grid(row=row_idx, column=0, pady=10, sticky=ctk.W)
        row_idx += 1
        
        ctk.CTkLabel(self.main_frame, text="IP / URL:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        self.ip = ctk.CTkEntry(self.main_frame, width=250)
        self.ip.insert(0, "192.168.123.100")
        self.ip.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
        row_idx += 1
        
        ctk.CTkLabel(self.main_frame, text="Slot / Port / Unit ID:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        self.slot = ctk.CTkEntry(self.main_frame, width=100)
        self.slot.insert(0, "2")
        self.slot.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
        row_idx += 1
        ctk.CTkLabel(self.main_frame, text="Modbus Scaling Factors (SP, PV, CV):").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        
        scale_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        scale_frame.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
        
        self.scale_sp = ctk.CTkEntry(scale_frame, width=50)
        self.scale_sp.insert(0, "1.0")
        self.scale_sp.pack(side=ctk.LEFT, padx=(0, 5))
        
        self.scale_pv = ctk.CTkEntry(scale_frame, width=50)
        self.scale_pv.insert(0, "1.0")
        self.scale_pv.pack(side=ctk.LEFT, padx=5)
        
        self.scale_cv = ctk.CTkEntry(scale_frame, width=50)
        self.scale_cv.insert(0, "10.0")
        self.scale_cv.pack(side=ctk.LEFT, padx=5)
        row_idx += 1
        
        # --- TAGS / ADDRESSES / HOLDING REGISTERS ---
        ctk.CTkLabel(self.main_frame, text="Tags / Addresses / Holding Registers", font=("Arial", 14, "bold")).grid(row=row_idx, column=0, pady=10, sticky=ctk.W)
        row_idx += 1
        
        for name, var_bind, default in [("SP", "sptag", "PID_SP"), ("PV", "pvtag", "PID_PV"), ("CV", "cvtag", "PID_CV")]:
            ctk.CTkLabel(self.main_frame, text=f"{name} Tag/Addr/Reg:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
            entry = ctk.CTkEntry(self.main_frame, width=250)
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
            setattr(self, var_bind, entry)
            # Live value
            if name == "SP": lbl = ctk.CTkLabel(self.main_frame, textvariable=self.sp_text)
            elif name == "PV": lbl = ctk.CTkLabel(self.main_frame, textvariable=self.pv_text)
            else: lbl = ctk.CTkLabel(self.main_frame, textvariable=self.cv_text)
            lbl.grid(row=row_idx, column=2, padx=20, sticky=ctk.W)
            row_idx += 1
            
        # --- INTERNAL PID SETTINGS ---
        ctk.CTkLabel(self.main_frame, text="Internal PID Settings", font=("Arial", 14, "bold")).grid(row=row_idx, column=0, pady=10, sticky=ctk.W)
        row_idx += 1
        
        ctk.CTkLabel(self.main_frame, text="Equation Type:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        self.eq_type_var = ctk.StringVar(value="Parallel")
        self.eq_dropdown = ctk.CTkOptionMenu(self.main_frame, variable=self.eq_type_var, width=250, values=["Parallel", "Series", "PIDE"], command=self.on_eq_change)
        self.eq_dropdown.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
        row_idx += 1
        
        self.eq_display_var = ctk.StringVar(value="CV = Kp*e + Ki*∫e dt + Kd*de/dt")
        ctk.CTkLabel(self.main_frame, text="Formula:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, textvariable=self.eq_display_var, text_color="gray").grid(row=row_idx, column=1, columnspan=2, pady=2, sticky=ctk.W)
        row_idx += 1
        
        ctk.CTkLabel(self.main_frame, text="Internal SP:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        self.internal_sp = ctk.CTkEntry(self.main_frame, width=100)
        self.internal_sp.insert(0, "50.0")
        self.internal_sp.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
        row_idx += 1
        
        pid_params = ctk.CTkFrame(self.main_frame)
        pid_params.grid(row=row_idx, column=0, columnspan=3, pady=2, sticky=ctk.W)
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
        row_idx += 1

        # --- MODEL SETTINGS ---
        ctk.CTkLabel(self.main_frame, text="Process Model (FOPDT)", font=("Arial", 14, "bold")).grid(row=row_idx, column=0, pady=10, sticky=ctk.W)
        row_idx += 1
        
        for name, var_bind, default in [("Gain", "model_gain", "1.45"), ("Time Constant (s)", "model_tc", "62.3"), 
                                        ("Dead Time (s)", "model_dt", "10.1"), ("Bias", "model_bias", "13.5")]:
            ctk.CTkLabel(self.main_frame, text=f"{name}:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
            entry = ctk.CTkEntry(self.main_frame, width=100)
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, pady=2, sticky=ctk.W)
            setattr(self, var_bind, entry)
            row_idx += 1

        # --- BUTTONS ---
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.grid(row=row_idx, column=0, columnspan=3, pady=20)
        row_idx += 1
        
        self.button_start = ctk.CTkButton(btn_frame, text="Start Simulator", command=self.start)
        self.button_start.pack(side=ctk.LEFT, padx=10)
        self.button_stop = ctk.CTkButton(btn_frame, text="Stop Simulator", command=self.stop, state=ctk.DISABLED)
        self.button_stop.pack(side=ctk.LEFT, padx=10)
        self.button_livetrend = ctk.CTkButton(btn_frame, text="Show Trend", command=self.show_live_trend, state=ctk.DISABLED)
        self.button_livetrend.pack(side=ctk.LEFT, padx=10)

        # Status
        ctk.CTkLabel(self.main_frame, text="Status:").grid(row=row_idx, column=0, pady=2, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, textvariable=self.gui_status, wraplength=400).grid(row=row_idx, column=1, columnspan=2, pady=2, sticky=ctk.W)
        
        self.on_mode_change(self.mode_var.get())

    def on_eq_change(self, eq_type):
        if eq_type == "Parallel":
            self.eq_display_var.set("CV = Kp*e + Ki*∫e dt + Kd*de/dt")
        elif eq_type == "Series":
            self.eq_display_var.set("CV = Kp*[ e + Ki*∫e dt + Kd*de/dt + (Ki*Kd)*e ]")
        elif eq_type == "PIDE":
            self.eq_display_var.set("ΔCV = Kp*Δe + Ki*e*dt + Kd*Δ²e/dt")

    def on_mode_change(self, mode):
        # Enable/Disable specific fields depending on mode
        is_internal = (mode == "Internal PID")
        state_int = ctk.NORMAL if is_internal else ctk.DISABLED
        state_ext = ctk.DISABLED if is_internal else ctk.NORMAL
        is_modbus = (mode == "Modbus")
        state_modbus = ctk.NORMAL if is_modbus else ctk.DISABLED
        
        self.eq_dropdown.configure(state=state_int)
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
        

    def setup_comms(self):
        mode = self.mode_var.get()
        if mode == "PyLogix":
            self.comm = PyLogixComm()
            self.comm.setup(self.ip.get(), self.slot.get())
        elif mode == "Modbus":
            self.comm = ModbusTCPComm()
            # Try to parse port, fallback 502
            try: port = int(self.slot.get())
            except: port = 502
            try: s_sp = float(self.scale_sp.get())
            except: s_sp = 1.0
            try: s_pv = float(self.scale_pv.get())
            except: s_pv = 1.0
            try: s_cv = float(self.scale_cv.get())
            except: s_cv = 1.0
            self.comm.setup(self.ip.get(), 1, port, s_cv, s_sp, s_pv)
        elif mode == "OPC UA":
            self.comm = OpcUaComm()
            self.comm.setup(self.ip.get())
        elif mode == "Internal PID":
            self.comm = InternalMathComm()
            
            # Setup internal PID parameters
            self.internal_pid.reset()
            self.internal_pid.eq_type = self.eq_type_var.get()
            self.internal_pid.kp = float(self.pid_kp.get())
            self.internal_pid.ki = float(self.pid_ki.get())
            self.internal_pid.kd = float(self.pid_kd.get())

    def pre_flight_checks(self):
        self.setup_comms()
        
        tags_to_test = []
        if self.mode_var.get() != "Internal PID":
            tags_to_test = [self.cvtag.get(), self.sptag.get(), self.pvtag.get()]
            
        success, err = self.comm.pre_flight(tags_to_test)
        if not success:
            raise Exception(err)

        self.reset()
        self.gui_status.set("Running...")
        self.process.Gain = float(self.model_gain.get())
        self.process.TimeConstant = max(float(self.model_tc.get()) * 10, 0.001)  # Due to sample rate 0.1s
        self.process.DeadTime = float(self.model_dt.get()) * 10
        self.process.Bias = float(self.model_bias.get())
        
        # Lock UI
        self.button_stop.configure(state=ctk.NORMAL)
        self.button_start.configure(state=ctk.DISABLED)
        self.button_livetrend.configure(state=ctk.DISABLED)
        
    def start(self):
        try:
            self.pre_flight_checks()
        except Exception as e:
            self.gui_status.set(f"Setup Error: {e}")
        else:
            self.looper = PeriodicInterval(self.thread_get_data, 0.1)
            self.live_trend()

    def thread_get_data(self):
        try:
            is_internal = (self.mode_var.get() == "Internal PID")
            
            # Read variables
            if is_internal:
                ext_cv = 0.0
                ext_sp = float(self.internal_sp.get())
                self.cv_text.set("Internal")
                self.sp_text.set(str(ext_sp))
            else:
                reads = self.comm.read_cv_sp(self.cvtag.get(), self.sptag.get())
                if getattr(reads[0], "Status", "Error") != "Success" or getattr(reads[1], "Status", "Error") != "Success":
                    raise Exception(f"Read Error: {getattr(reads[0], 'Status', 'Err')} / {getattr(reads[1], 'Status', 'Err')}")
                ext_cv = float(reads[0].Value or 0)
                ext_sp = float(reads[1].Value or 0)
                self.cv_text.set(str(round(ext_cv, 3)))
                self.sp_text.set(str(round(ext_sp, 3)))
                
            # Process calculations
            # Determine previous PV for PID
            current_pv = self.PV[-1] if len(self.PV) > 0 else float(self.model_bias.get())
            
            # Use Internal PID CV or External CV
            if is_internal:
                cv_to_use = self.internal_pid.calculate(ext_sp, current_pv, 0.1)
                self.cv_text.set(str(round(cv_to_use, 3)))
            else:
                cv_to_use = ext_cv
                
            self.CV = np.append(self.CV, cv_to_use)
            self.SP = np.append(self.SP, ext_sp)
            self.process.CV = self.CV
            
            ts = [self.scan_count, self.scan_count + 1]
            pv_calc = self.process.update(current_pv, ts)
            
            noise = (random.randint(0, 10) / 100) - 0.05
            new_pv = pv_calc[0] + noise
            self.PV = np.append(self.PV, new_pv)
            
            # Write PV
            write_res = self.comm.write_pv(self.pvtag.get(), new_pv)
            if getattr(write_res, "Status", "Error") == "Success":
                 self.pv_text.set(str(round(new_pv, 2)))
            else:
                 self.gui_status.set(f"Write Error: {getattr(write_res, 'Status', 'Unknown')}")

        except Exception as e:
            self.gui_status.set(f"Loop Error: {e}")

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
            time.sleep(0.1)
            if self.comm:
               self.comm.close()
            plt.close("all")
        except Exception as e:
            self.gui_status.set(f"Stop Error: {e}")

    def live_trend(self):
        fig = plt.figure()
        self.ax = plt.axes()
        (SP,) = self.ax.plot([], [], lw=2, color="Red", label="SP")
        (CV,) = self.ax.plot([], [], lw=2, color="Green", label="CV")
        (PV,) = self.ax.plot([], [], lw=2, color="Blue", label="PV")

        def init():
            SP.set_data([], [])
            PV.set_data([], [])
            CV.set_data([], [])
            plt.ylabel("EU")
            plt.xlabel("Time (min)")
            plt.suptitle("Live Data")
            plt.legend(loc="upper right")

        def animate(i):
            try:
                x = np.arange(len(self.SP), dtype=int)
                x = x / 600
                SP.set_data(x, self.SP)
                CV.set_data(x, self.CV)
                PV.set_data(x, self.PV)
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception as e:
                self.gui_status.set(f"Plot Error: {e}")

        self.anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60, interval=1000)

        mngr = plt.get_current_fig_manager()
        mngr.window.geometry(f"{int(self.screen_width/2)}x{self.screen_height-self.toolbar}+{int(self.screen_width/2)-self.offset+1}+0")
        plt.gcf().canvas.mpl_connect("close_event", self.on_plot_close)
        plt.show()

    def on_plot_close(self, event):
        if self.looper:
            self.button_livetrend.configure(state=ctk.NORMAL)

    def show_live_trend(self):
        self.button_livetrend.configure(state=ctk.DISABLED)
        open_plots = plt.get_fignums()
        if len(open_plots) == 0:
            self.live_trend()

if __name__ == "__main__":
    gui_app = PIDSimForCLX()
    gui_app.root.mainloop()
