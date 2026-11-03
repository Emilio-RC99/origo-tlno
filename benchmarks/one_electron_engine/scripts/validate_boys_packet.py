#!/usr/bin/env python3
"""Validate every packet lane against 60-digit incomplete-gamma moments."""
import argparse
import functools
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import mpmath as mp

BENCH = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--probe', type = Path, default = BENCH/'build'/'boys_packet_probe')
    parser.add_argument('--out', type = Path, default = BENCH/'results'/'validation'/'boys_packets.json')
    args = parser.parse_args()
    mp.mp.dps = 60
    rng = random.Random(2419)
    xs=[0.,1e-16,.005-1e-12,.005,.005+1e-12,11.99999999,12.,
        39.995,39.99999999,40.,40.00000001,100.,1e4]
    xs += [40 + 2 * n + d for n in range(9) for d in (-1e-9, 0., 1e-9)]
    xs += [rng.uniform(0, 60) for _ in range(100)]
    xs += [10**rng.uniform(-14,4) for _ in range(100)]
    cases=[]
    for width in (1,2,4,8):
        for n in range(9):
            for i,x in enumerate(xs):
                lanes=[x if i % 2 else xs[(i + 31 * l) % len(xs)] for l in range(width)]
                cases.append((width, n, (.01,.5,3.,80.)[i % 4], lanes))
    payload=''.join(f'{w} {n} {p:.17g} '+' '.join(f'{x:.17g}' for x in lanes)+'\n'
                    for w,n,p,lanes in cases)
    completed=subprocess.run([str(args.probe.resolve())],input=payload,text=True,
                             capture_output=True,check=True)
    lines=completed.stdout.splitlines()
    if len(lines)!=len(cases): raise RuntimeError('Incomplete packet output')

    @functools.lru_cache(None)
    def reference(n, x):
        if x==0: return 1/mp.mpf(2*n+1)
        order=n+mp.mpf('.5');z=mp.mpf(x)
        return mp.gammainc(order,0,z)/(2*z**order)

    worst = 0.; count = 0; worst_case = None
    for (width, n, p, xs), line in zip(cases, lines):
        values = list(map(float, line.split()))
        if len(values) != (n + 1) * width: raise RuntimeError('Incomplete packet row')
        for order in range(n + 1):
            for lane, x in enumerate(xs):
                ref = float(reference(order, x) * (-2 * mp.mpf(p))**order)
                got = values[order * width + lane]
                relative = abs(got - ref)/abs(ref)
                if not math.isfinite(got) or relative>2e-13:
                    raise RuntimeError((width, n, order, p, lane, x, got, ref, relative))
                count += 1
                if relative > worst:
                    worst = relative; worst_case = dict(width = width, max_n = n, order = order,
                        p = p, lane = lane, x = x, got = got, reference = ref)
    result = dict(status = 'PASS', packets = len(cases), values = count, max_relative_error = worst,
                worst_case = worst_case, reference = 'mpmath incomplete gamma, 60 digits',
                probe_sha256 = hashlib.sha256(args.probe.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(result, indent = 2) + '\n')
    print(json.dumps(result, indent = 2))


if __name__ == '__main__': main()
