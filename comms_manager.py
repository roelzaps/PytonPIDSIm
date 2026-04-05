from abc import ABC, abstractmethod

from pylogix import PLC
from pymodbus.client import ModbusTcpClient


class CommResult:
    """Standard result container for communication read/write operations."""

    def __init__(self, value, status):
        self.Value = value
        self.Status = status


class CommManager(ABC):
    """Abstract base class for all communication backends.

    Subclasses must implement setup, pre_flight, read_cv_sp, write_pv, and close.
    """

    def __init__(self):
        self.ip_address = ""
        self.port = 502
        self.slot = 0
        self.timeout = 10.0

    @abstractmethod
    def setup(self, ip, slot=None, port=None):
        ...

    @abstractmethod
    def pre_flight(self, tags):
        ...

    @abstractmethod
    def read_cv_sp(self, cv_tag, sp_tag):
        ...

    @abstractmethod
    def write_pv(self, pv_tag, value):
        ...

    @abstractmethod
    def close(self):
        ...


class PyLogixComm(CommManager):
    """Allen-Bradley PLC communication via PyLogix."""

    def __init__(self):
        super().__init__()
        self.plc = PLC()

    def setup(self, ip, slot=0, port=None):
        self.plc.IPAddress = ip
        self.plc.ProcessorSlot = int(slot)
        self.plc.SocketTimeout = 10.0

    def pre_flight(self, tags):
        ret = self.plc.Read(tags)
        if any(x.Value is None for x in ret):
            if any(x.Status != "Success" for x in ret):
                return False, f"Error: {[x.Status for x in ret]}"
            return False, "Check PLC and Tag Configuration"
        self.plc.SocketTimeout = 0.5
        return True, ""

    def read_cv_sp(self, cv_tag, sp_tag):
        ret = self.plc.Read([cv_tag, sp_tag])
        return [CommResult(x.Value, x.Status) for x in ret]

    def write_pv(self, pv_tag, value):
        ret = self.plc.Write(pv_tag, value)
        return CommResult(ret.Value, ret.Status)

    def close(self):
        self.plc.Close()


class ModbusTCPComm(CommManager):
    """Modbus TCP communication via pymodbus."""

    def __init__(self):
        super().__init__()
        self.client = None
        self.scale_cv = 1.0
        self.scale_sp = 1.0
        self.scale_pv = 1.0

    def setup(self, ip, slot=1, port=502, scale_cv=1.0, scale_sp=1.0, scale_pv=1.0):
        self.ip_address = ip
        self.port = int(port)
        self.unit_id = int(slot)
        self.scale_cv = float(scale_cv) if float(scale_cv) != 0 else 1.0
        self.scale_sp = float(scale_sp) if float(scale_sp) != 0 else 1.0
        self.scale_pv = float(scale_pv) if float(scale_pv) != 0 else 1.0
        self.client = ModbusTcpClient(self.ip_address, port=self.port, timeout=10.0)

    def pre_flight(self, tags):
        if not self.client.connect():
            return False, "Failed to connect to Modbus TCP Server"
        self.client.timeout = 0.5
        return True, ""

    def _parse_tag(self, tag):
        """Convert a Modbus register tag to a zero-based address."""
        try:
            val = int(tag)
            if val >= 400001:
                return val - 400001
            elif val >= 40001:
                return val - 40001
            return val
        except ValueError:
            return 0

    def read_cv_sp(self, cv_tag, sp_tag):
        if not self.client:
            return [CommResult(None, "Not Connected"), CommResult(None, "Not Connected")]

        cv_addr = self._parse_tag(cv_tag)
        sp_addr = self._parse_tag(sp_tag)

        cv_res = self.client.read_holding_registers(address=cv_addr, count=1, device_id=self.unit_id)
        if cv_res.isError():
            cv_ret = CommResult(None, "Modbus Error")
        else:
            cv_ret = CommResult(cv_res.registers[0] / self.scale_cv, "Success")

        sp_res = self.client.read_holding_registers(address=sp_addr, count=1, device_id=self.unit_id)
        if sp_res.isError():
            sp_ret = CommResult(None, "Modbus Error")
        else:
            sp_ret = CommResult(sp_res.registers[0] / self.scale_sp, "Success")

        return [cv_ret, sp_ret]

    def write_pv(self, pv_tag, value):
        pv_addr = self._parse_tag(pv_tag)
        val_int = int(value * self.scale_pv)
        res = self.client.write_register(address=pv_addr, value=val_int, device_id=self.unit_id)
        if res.isError():
            return CommResult(None, "Modbus Write Error")
        return CommResult(value, "Success")

    def close(self):
        if self.client:
            self.client.close()


import asyncio
from asyncua import Client, ua


class OpcUaComm(CommManager):
    """OPC UA communication via asyncua (sync wrapper)."""

    def __init__(self):
        super().__init__()
        self.client = None
        self.loop = asyncio.new_event_loop()

    def _run_sync(self, coro):
        return self.loop.run_until_complete(coro)

    def setup(self, ip, slot=None, port=None):
        self.client = Client(url=ip, timeout=10.0)

    def pre_flight(self, tags):
        try:
            self._run_sync(self.client.connect())
            return True, ""
        except Exception as e:
            return False, str(e)

    def read_cv_sp(self, cv_tag, sp_tag):
        try:
            cv_node = self.client.get_node(cv_tag)
            sp_node = self.client.get_node(sp_tag)
            cv_val = self._run_sync(cv_node.read_value())
            sp_val = self._run_sync(sp_node.read_value())
            return [CommResult(cv_val, "Success"), CommResult(sp_val, "Success")]
        except Exception as e:
            return [CommResult(None, str(e)), CommResult(None, str(e))]

    def write_pv(self, pv_tag, value):
        try:
            pv_node = self.client.get_node(pv_tag)
            data_type = self._run_sync(pv_node.read_data_type_as_variant_type())
            dv = ua.DataValue(ua.Variant(value, data_type))
            self._run_sync(pv_node.write_value(dv))
            return CommResult(value, "Success")
        except Exception as e:
            return CommResult(None, str(e))

    def close(self):
        try:
            if self.client:
                self._run_sync(self.client.disconnect())
        except Exception:
            pass


class InternalMathComm(CommManager):
    """No-op comms backend for Internal PID mode — maintains interface compatibility."""

    def setup(self, ip=None, slot=None, port=None):
        pass

    def pre_flight(self, tags):
        return True, ""

    def read_cv_sp(self, cv_tag, sp_tag):
        return [CommResult(0.0, "Success"), CommResult(0.0, "Success")]

    def write_pv(self, pv_tag, value):
        return CommResult(value, "Success")

    def close(self):
        pass
