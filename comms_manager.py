from pylogix import PLC
from pymodbus.client import ModbusTcpClient

# For Modbus, we'll try to use standard 16-bit registers as requested
# We will just write/read the integer values.

class CommResult:
    def __init__(self, value, status):
        self.Value = value
        self.Status = status

class CommManager:
    def __init__(self):
        self.ip_address = ""
        self.port = 502
        self.slot = 0
        self.timeout = 10.0
        
    def setup(self, ip, slot=None, port=None):
        pass

    def read_cv_sp(self, cv_tag, sp_tag):
        # returns [CommResult, CommResult]
        pass
        
    def write_pv(self, pv_tag, value):
        # returns CommResult
        pass
        
    def close(self):
        pass


class PyLogixComm(CommManager):
    def __init__(self):
        super().__init__()
        self.plc = PLC()
        
    def setup(self, ip, slot=0, port=None):
        self.plc.IPAddress = ip
        self.plc.ProcessorSlot = int(slot)
        self.plc.SocketTimeout = 10.0
        
    def pre_flight(self, tags):
        # test read
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
    def __init__(self):
        super().__init__()
        self.client = None
        self.scale_cv = 1.0
        self.scale_sp = 1.0
        self.scale_pv = 1.0
        
    def setup(self, ip, slot=1, port=502, scale_cv=1.0, scale_sp=1.0, scale_pv=1.0):
        # slot acts as slave id/unit id in Modbus
        self.ip_address = ip
        self.port = int(port)
        self.unit_id = int(slot)
        self.scale_cv = float(scale_cv) if float(scale_cv) != 0 else 1.0
        self.scale_sp = float(scale_sp) if float(scale_sp) != 0 else 1.0
        self.scale_pv = float(scale_pv) if float(scale_pv) != 0 else 1.0
        self.client = ModbusTcpClient(self.ip_address, port=self.port, timeout=10.0)
        
    def pre_flight(self, tags):
        # Tags for modbus will be parsed as addresses, e.g. "40001" or just "0"
        if not self.client.connect():
            return False, "Failed to connect to Modbus TCP Server"
        self.client.timeout = 0.5
        return True, ""

    def _parse_tag(self, tag):
        # Handle 4x holding register definitions seamlessly
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
        # Modbus value needs to be scaled before it is added to holding register
        val_int = int(value * self.scale_pv)
        res = self.client.write_register(address=pv_addr, value=val_int, device_id=self.unit_id)
        if res.isError():
            return CommResult(None, "Modbus Write Error")
        # modbus doesn't return the written value like PyLogix, so we return the intended value
        return CommResult(value, "Success")
        
    def close(self):
        if self.client:
            self.client.close()


import asyncio
from asyncua import Client, ua

# Asyncua requires an event loop, since our get_data runs synchronously in a thread,
# we need to run asyncio event loop in the background or use synchronous wrappers.

class OpcUaComm(CommManager):
    def __init__(self):
        super().__init__()
        self.client = None
        self.loop = asyncio.new_event_loop()
        
    def _run_sync(self, coro):
        return self.loop.run_until_complete(coro)

    def setup(self, ip, slot=None, port=None):
        # IP will be the full OPC UA endpoint URL e.g., opc.tcp://127.0.0.1:4840
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
        except:
            pass


class InternalMathComm(CommManager):
    # This dummy comms manager is used when Internal Math PID is selected, 
    # to maintain compatibility with the UI. However, CV calculation is handled natively.
    def setup(self, ip, slot=None, port=None):
        pass
        
    def pre_flight(self, tags):
        return True, ""

    def read_cv_sp(self, cv_tag, sp_tag):
        # We don't read CV/SP from an external client if we are the controller. 
        # Well, we *could* read SP. But the internal PID handles this.
        return [CommResult(0.0, "Success"), CommResult(0.0, "Success")]

    def write_pv(self, pv_tag, value):
        return CommResult(value, "Success")
        
    def close(self):
        pass
