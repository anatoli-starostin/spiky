"""Sudoless M1 temperature monitor (macmon). One background `macmon pipe` process; latest sample cached."""
import json
import subprocess
import threading
import time


class Thermal:
    def __init__(self, interval_ms=2000, log_path=None):
        self.latest = {"cpu_temp": None, "gpu_temp": None, "sys_power": None, "t": None}
        self.max_cpu_temp = 0.0
        self._log = open(log_path, "a") if log_path else None
        self._proc = subprocess.Popen(["macmon", "pipe", "-i", str(interval_ms)],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self._proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            temp = d.get("temp") or {}
            s = {"cpu_temp": temp.get("cpu_temp_avg"), "gpu_temp": temp.get("gpu_temp_avg"),
                 "sys_power": d.get("sys_power"), "t": d.get("timestamp")}
            self.latest = s
            if s["cpu_temp"] is not None:
                self.max_cpu_temp = max(self.max_cpu_temp, s["cpu_temp"])
            if self._log:
                self._log.write(json.dumps(s) + "\n")
                self._log.flush()

    def cpu_temp(self):
        return self.latest["cpu_temp"]

    def wait_below(self, limit, why="", poll=10):
        """Block until cpu temp < limit. Returns seconds waited."""
        t0 = time.time()
        while self.cpu_temp() is None:
            time.sleep(1)
        announced = False
        while self.cpu_temp() > limit:
            if not announced:
                print(f"[thermal] {why} cpu {self.cpu_temp():.1f}C > {limit}C, cooling...", flush=True)
                announced = True
            time.sleep(poll)
        waited = time.time() - t0
        if announced:
            print(f"[thermal] resumed at {self.cpu_temp():.1f}C after {waited:.0f}s", flush=True)
        return waited

    @staticmethod
    def on_ac():
        """True when the Mac draws from the power adapter (pmset). Unknown -> True."""
        try:
            return "Battery Power" not in subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                                         text=True, timeout=10).stdout
        except Exception:
            return True

    def wait_power(self, why="", poll=60):
        """Block while running on battery (laptop unplugged). Returns seconds waited."""
        t0 = time.time()
        if self.on_ac():
            return 0.0
        print(f"[power] {why} on battery, paused until the adapter is connected...", flush=True)
        while not self.on_ac():
            time.sleep(poll)
        print(f"[power] resumed after {time.time() - t0:.0f}s", flush=True)
        return time.time() - t0

    def close(self):
        self._proc.terminate()
        if self._log:
            self._log.close()
