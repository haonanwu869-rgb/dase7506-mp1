"""Run the unchanged evaluator and record native Windows process peak working set."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import runpy
import sys
import time


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


if __name__ == '__main__':
    command = list(sys.argv)
    output = Path(sys.argv[sys.argv.index('--output')+1])
    started = time.perf_counter()
    sys.argv[0] = 'evaluate.py'
    runpy.run_path(str(Path(__file__).with_name('evaluate.py')), run_name='__main__')
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    metadata = dict(command=command, process_seconds=time.perf_counter()-started,
                    peak_working_set_bytes=counters.PeakWorkingSetSize,
                    peak_pagefile_bytes=counters.PeakPagefileUsage)
    output.with_suffix('.resources.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)
