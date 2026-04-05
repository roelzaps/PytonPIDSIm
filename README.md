# PID Simulator & Process Model

A desktop PID simulator with a built-in FOPDT process model, supporting multiple industrial communication protocols and an internal software PID controller. Built with Python and CustomTkinter.

![PID_Sim](https://github.com/Destination2Unknown/PythonCLX_PIDSimulator/assets/92536730/1e441359-a999-41a2-b821-1571d1576793)

---

## Features

### Communication Modes

| Mode | Description |
|---|---|
| **PyLogix** | Read/write tags to Allen-Bradley PLCs via EtherNet/IP |
| **Modbus TCP** | Communicate with Modbus TCP devices using holding registers with configurable scaling factors |
| **OPC UA** | Connect to OPC UA servers using node IDs |
| **Internal PID** | Standalone software PID controller — no external hardware required |

The GUI automatically adapts when switching modes:
- **PyLogix**: Shows IP, Slot, and PLC Tag fields
- **Modbus**: Shows IP (`127.0.0.1`), Port (`502`), Register fields, and Scaling Factors
- **OPC UA**: Shows Server URL and Node ID fields (Slot hidden)
- **Internal PID**: Hides all comms/tag fields, shows PID tuning controls

### Internal PID Controller

- **Three equation types**: Parallel, Series, and PIDE (Velocity)
- **Controller action**: Reverse Acting or Direct Acting
- **Anti-windup**: Output clamping with integral back-calculation
- **Auto/Manual mode**: Toggle with radio buttons
  - **Auto**: PID calculates CV from SP and PV
  - **Manual**: User directly enters CV (%) to observe open-loop process response
  - **Bumpless transfer**: Switching from Manual → Auto seeds the PID to avoid output bumps

### Process Model (FOPDT)

| Parameter | Self-Regulating | Integrating |
|---|---|---|
| **Gain** | ✅ | ✅ |
| **Dead Time** | ✅ | ✅ |
| **Noise Amplitude** | ✅ | ✅ |
| **Time Constant** | ✅ | Hidden |
| **Bias** | ✅ | Hidden |
| **Balance Point (CV)** | Hidden | ✅ |

- **Self-Regulating**: Output settles to steady-state for a constant CV (e.g., temperature, pressure)
- **Integrating**: Output ramps continuously — direction reverses around the Balance Point (e.g., level, position)
- **Non-negative constraint**: Process value (PV) is clamped to ≥ 0.0

### Live Trend

![image](https://user-images.githubusercontent.com/92536730/154958077-527e4e79-6add-4fdc-bab9-9b7ee979cb87.png)

Real-time plotting of SP, PV, and CV with interactive features:

| Control | Action |
|---|---|
| **Spacebar** | Pause / Resume the trend |
| **Mouse hover** | Crosshair cursor with colored snap dots on each line |
| **Tooltip** | Displays time (hh:mm:ss), SP, PV, and CV at the cursor position |
| **← Left Arrow** | Scroll backward in time |
| **→ Right Arrow** | Scroll forward in time |
| **Home** | Jump to live (latest data) |
| **End** | Jump to beginning |
| **Mouse Scroll Up** | Zoom in (narrow the time window, min 30s) |
| **Mouse Scroll Down** | Zoom out (widen the time window, max 60 min) |

- Default view window: **5 minutes** (rolling)
- Auto-follows latest data; scrolling switches to manual mode
- Title indicates state: `Live Data`, `PAUSED`, or `Scrolling ◀/▶`

---

## Project Structure

```
PytonPIDSIm/
├── PythonCLX_PIDSimulator.pyw   # Main application (GUI + process model + scan loop)
├── pid_controller.py            # Internal PID controller class
├── comms_manager.py             # Communication backends (PyLogix, Modbus, OPC UA, Internal)
├── tests/
│   └── test_pid_simulator.py    # Unit tests (38 tests)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Destination2Unknown/PythonCLX_PIDSimulator.git
cd PythonCLX_PIDSimulator

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `customtkinter` | Modern GUI framework |
| `numpy` | Numerical computation |
| `matplotlib` | Live trend plotting |
| `scipy` | ODE solver for process model |
| `pylogix` | Allen-Bradley PLC communication |
| `pymodbus` | Modbus TCP communication |
| `asyncua` | OPC UA client |

---

## Usage

### Launch the Simulator

```bash
python PythonCLX_PIDSimulator.pyw
```

### Quick Start (Internal PID — no hardware needed)

1. Select **Internal PID** from the Controller Mode dropdown
2. Set PID tuning: `Kp=1.0`, `Ki=0.1`, `Kd=0.0`
3. Set Internal SP to `50.0`
4. Click **Start Simulator**
5. The trend window opens automatically — watch the PV track the SP

### Open-Loop Step Test

1. Select **Internal PID** → switch PID Mode to **Manual**
2. Set Manual CV to `0.0`, click **Start Simulator**
3. Change Manual CV to `50.0` to step the output
4. Observe the open-loop process response on the trend
5. Switch back to **Auto** for bumpless transfer to closed-loop

### External PLC Mode

1. Select **PyLogix**, **Modbus**, or **OPC UA**
2. Enter the device IP/URL, Slot/Port, and tag names or register addresses
3. Click **Start Simulator**
4. The simulator reads SP and CV from the PLC/device and writes PV back

---

## Running Tests

```bash
python -m pytest tests/test_pid_simulator.py -v
```

All **38 tests** cover:
- PID controller (Parallel, Series, PIDE equations)
- Controller action (Reverse, Direct)
- Anti-windup clamping and back-calculation
- FOPDT model (Self-Regulating and Integrating)
- Balance point behavior
- Communication result objects

---

## How It Works

```
┌─────────────┐     CV      ┌──────────────┐     PV
│  PID         │──────────►│  FOPDT Model   │──────────►
│  Controller  │◄──────────│  (+ Dead Time) │
│  (or PLC)   │     PV      │  (+ Noise)     │
└─────────────┘             └──────────────┘
       ▲
       │ SP (setpoint)
```

- **Scan rate**: 100ms (10 scans/second)
- **Dead time**: Implemented via CV history buffer
- **ODE solver**: `scipy.integrate.odeint`
- **Thread safety**: Dedicated scan thread with lock; GUI updates via `StringVar.set()`

---

## License

MIT License — see [LICENSE](LICENSE) for details.
