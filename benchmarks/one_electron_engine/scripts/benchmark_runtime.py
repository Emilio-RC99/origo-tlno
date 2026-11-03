#!/usr/bin/env python3
"""Production benchmark handles and physical-core placement."""
import ctypes
import os
from pathlib import Path
from benchmark_molecule_suite import tlno_copy


class TlnoArm:
    def __init__(self,lib,path,basis,screened,spherical,scope,epsilon):
        self.lib,self.screened,self.epsilon = lib,screened,epsilon
        self.prepare()
        self.handle = lib.tlno_onee_create_from_inp(str(path).encode(), basis.encode())
        if not self.handle: raise RuntimeError(f"TLNO creation failed: {path}/{basis}")
        try:
            suffix = "_sph" if spherical else ""
            self.nbf = getattr(lib,"tlno_onee_nbf"+suffix)(self.handle)
            if self.nbf <= 0:
                raise RuntimeError(f"Invalid TLNO basis size: {self.nbf}")
            self.function = getattr(lib,"tlno_onee_eval_" + scope + suffix)
            self.copy_function = getattr(lib,"tlno_onee_copy" + suffix)
            if scope == "cached": lib.tlno_onee_prepare_plan(self.handle)
        except Exception:
            self.close()
            raise
    
    def prepare(self):
        self.lib.tlno_onee_set_screening(int(self.screened))
        if self.screened and self.epsilon:
            self.lib.tlno_onee_set_group_pair_decay_eps(ctypes.c_double(self.epsilon))
            self.lib.tlno_onee_set_nuclear_screen_eps(ctypes.c_double(self.epsilon))

    def run(self): self.function(self.handle)
    def copy(self): return tlno_copy(self.lib,self.handle,self.nbf,self.copy_function)
    def close(self):
        if self.handle:
            self.lib.tlno_onee_destroy(self.handle)
            self.handle = None


def physical_cpus():
    allowed = os.sched_getaffinity(0)
    cores = {}
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topology/"physical_package_id").read_text().strip(),
                   (topology/"core_id").read_text().strip())
        except OSError:
            # Affinity still bounds placement on hosts without sysfs topology.
            key = ("unknown", cpu)
        cores.setdefault(key,cpu)
    cpus = list(cores.values())
    return cpus[1:]+cpus[:1] if len(cpus)>1 else cpus

    