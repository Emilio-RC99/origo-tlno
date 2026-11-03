#!/usr/bin/env python3
"""Interleaved one-electron S/T/V timings adapted from the production D3 driver.

Use --plan to inspect the workload without loading numerical libraries. Actual
measurements require --out and validate finite, symmetric matrices against an
independent Libcint reference before recording any timing observations.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from benchmark_molecule_suite import (
    BASIS_PAIRS, DEFAULT_LIB, load_tlno, make_libcint, pin_threads,
    pyscf_basis_spec, read_geometry, resolve_molecules,
)
from benchmark_runtime import TlnoArm, physical_cpus
from benchmark_statistics import TimedRunner, stable_seed, time_interleaved


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(args, parser):
    try:
        molecules = resolve_molecules(args.molecules)
    except ValueError as error:
        parser.error(str(error))
    bases = [name.strip() for name in args.basis.split(',') if name.strip()]
    if not molecules or not bases or len(bases) != len(set(bases)):
        parser.error('Select at least one molecule and unique basis names')
    if not set(bases) <= BASIS_PAIRS.keys():
        parser.error('Unknown basis; choose from ' + ', '.join(BASIS_PAIRS))
    return molecules, bases


def summarize(output):
    grouped = {}
    for path in sorted(output.glob('run-*.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = row['molecule'], row['basis']
                item = grouped.setdefault(key, {'molecule': key[0], 'basis': key[1],
                    'ao_count': int(row['ao_count']), 'reference': row['reference'],
                    'samples': {}, 'error': 0.0})
                for column, values in row.items():
                    if column.endswith('_samples_s'):
                        arm = column.removesuffix('_samples_s')
                        item['samples'].setdefault(arm, []).extend(json.loads(values))
                item['error'] = max(item['error'], float(row['external_max_abs']))
    rows = []
    for item in grouped.values():
        row = {'molecule': item['molecule'], 'basis': item['basis'],
               'ao_count': item['ao_count'], 'reference': item['reference'],
               'external_max_abs': item['error']}
        medians = {arm: statistics.median(values) for arm, values in item['samples'].items()}
        for arm, median in medians.items():
            row[arm + '_median_s'] = median
            row[arm + '_observations'] = len(item['samples'][arm])
            if arm != 'reference':
                row['ratio_reference_over_' + arm] = medians['reference'] / median
        rows.append(row)
    with (output / 'pooled.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {'status': 'PASS', 'cases': len(rows), 'reference': rows[0]['reference'],
        'maximum_absolute_matrix_error': max(row['external_max_abs'] for row in rows),
        'primary_statistic': 'median', 'timing_order': 'randomized-interleaved'}
    for arm in grouped[next(iter(grouped))]['samples']:
        if arm != 'reference':
            ratios = [row['ratio_reference_over_' + arm] for row in rows]
            aggregate['geometric_mean_reference_over_' + arm] = math.exp(
                statistics.fmean(math.log(value) for value in ratios))
    (output / 'summary.json').write_text(json.dumps(aggregate, indent=2) + '\n')


def environment():
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pyscf')}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'libc': platform.libc_ver()}


def build_information(library):
    path = library.parent / 'onee_build_info.txt'
    if path.is_file():
        return {'available': True, 'path': str(path), 'sha256': sha(path),
                'text': path.read_text()}
    return {'available': False, 'path': str(path),
            'reason': 'build metadata unavailable for this library'}


def measure(args, molecules, bases):

    import numpy as np
    
    library = args.lib.resolve()
    native = args.native.resolve() if args.native else None
    lib = load_tlno(library)
    set_threads = lib.omp_set_num_threads
    set_threads.argtypes = [ctypes.c_int]; set_threads.restype = None
    label = 'provided-native-adapter' if native else 'pyscf-wheel'
    scopes = {'on': [('candidate', True)], 'off': [('candidate', False)],
              'both': [('screened', True), ('unscreened', False)]}[args.screening]
    rows = []
    partial = None if args.validate_only else args.out / f'run-{args.worker_run}.partial.jsonl'
    started = datetime.now(timezone.utc).isoformat()
    with (nullcontext(None) if args.validate_only else partial.open('x')) as journal:
        for path in molecules:
            charge, multiplicity, atoms = read_geometry(path)
            for basis in bases:
                arms = {}
                try:
                    # Raw provider parameters and shell order are exported once,
                    # outside all setup/evaluation timings.
                    basis_spec = pyscf_basis_spec(basis, lib = lib, atoms = atoms)
                    basis_hash = hashlib.sha256(json.dumps(basis_spec,
                        sort_keys = True, separators=(',', ':')).encode()).hexdigest()
                    for name, screened in scopes:
                        arms[name] = TlnoArm(lib, path, basis, screened,
                            args.representation == 'spherical', args.scope, args.decay_eps)
                    n, setup, evaluate, matrices = make_libcint(atoms, basis_spec,
                        charge, multiplicity, args.threads, native, 0.,
                        args.representation == 'spherical')
                    if any(arm.nbf != n for arm in arms.values()):
                        raise RuntimeError(f'AO count mismatch: {path.stem}/{basis}')
                    prepared = setup() if args.scope == 'cached' else None
                    def reference_run():
                        evaluate(prepared if prepared is not None else setup())
                    def prepare_candidate(arm):
                        def prepare():
                            set_threads(args.threads); arm.prepare()
                        return prepare
                    runners = {name: TimedRunner(prepare_candidate(arm), arm.run)
                               for name, arm in arms.items()}
                    runners['reference'] = TimedRunner(lambda: set_threads(args.threads), reference_run)
                    for runner in runners.values():
                        runner.prepare(); runner.run()
                    scale = 1 / np.sqrt(np.diag(matrices[0]))
                    reference = [scale[:, None] * matrix * scale[None, :] for matrix in matrices]
                    values = {name: arm.copy() for name, arm in arms.items()}
                    if not all(np.isfinite(matrix).all()
                               for matrix in reference):
                        raise RuntimeError('Nonfinite reference S/T/V output')
                    errors = {}
                    for name, outputs in values.items():
                        if not all(np.isfinite(matrix).all() and np.array_equal(matrix, matrix.T)
                                   for matrix in outputs):
                            raise RuntimeError(f'nonfinite or asymmetric {name} matrices')
                        errors[name] = [float(np.max(np.abs(x - y))) for x, y in zip(outputs, reference)]
                    error = max(value for differences in errors.values() for value in differences)
                    if error > args.matrix_gate:
                        raise RuntimeError(f'External matrix gate failed: {path.stem}/{basis}: {errors}')
                    screening_delta = max(float(np.max(np.abs(x - y))) for x, y in
                        zip(values['screened'], values['unscreened'])) if args.screening == 'both' else None
                    del values, reference
                    if args.validate_only:
                        rows.append({'molecule': path.stem, 'basis': basis, 'ao_count': n,
                            'representation': args.representation, 'scope': args.scope,
                            'threads': args.threads, 'screening': args.screening,
                            'external_max_abs': error, 'operator_errors_S_T_V': errors,
                            'screened_vs_unscreened_max_abs': screening_delta,
                            'basis_sha256': basis_hash})
                        continue
                    pilot, _ = time_interleaved(runners, repeats = 3, warmups = 1,
                        seed=stable_seed(args.seed, args.worker_run, path.stem, basis, 'pilot'))
                    loops = max(1, min(1024, math.ceil(args.sample_ms / 1000 /
                        min(statistics.median(samples) for samples in pilot.values()))))
                    def batch(runner):
                        def run():
                            for _ in range(loops):
                                runner.run()
                        return TimedRunner(runner.prepare, run)
                    seed = stable_seed(args.seed, args.worker_run, path.stem, basis)
                    samples, order = time_interleaved({name: batch(runner) for name, runner in runners.items()},
                        repeats=args.repeats, warmups=args.warmup, seed=seed)
                    samples = {name: [value / loops for value in observations]
                               for name, observations in samples.items()}
                    row = {'molecule': path.stem, 'basis': basis, 'ao_count': n,
                        'representation': args.representation, 'scope': args.scope,
                        'threads': args.threads, 'screening': args.screening,
                        'run_id': args.worker_run, 'repeats': args.repeats, 'warmups': args.warmup,
                        'batch_calls': loops, 'sample_ms': args.sample_ms, 'seed': seed,
                        'reference': label, 'external_max_abs': error,
                        'operator_errors_S_T_V': json.dumps(errors),
                        'screened_vs_unscreened_max_abs': screening_delta,
                        'measurement_order': '|'.join(order), 'basis_parameters': 'candidate-raw-export',
                        'basis_sha256': basis_hash}
                    for name, observations in samples.items():
                        row[name + '_samples_s'] = json.dumps(observations)
                        row[name + '_median_s'] = statistics.median(observations)
                    for name in arms:
                        row['ratio_reference_over_' + name] = row['reference_median_s'] / row[name + '_median_s']
                    rows.append(row); journal.write(json.dumps(row) + '\n'); journal.flush()
                    print(f'{len(rows)}/{len(molecules)*len(bases)} {path.stem}/{basis}: gate={error:.3e}', flush=True)
                finally:
                    for arm in arms.values():
                        arm.close()
    if args.validate_only:
        report = {'status': 'PASS', 'mode': 'validation-only', 'cases': len(rows),
            'reference': label, 'matrix_max_abs_gate': args.matrix_gate,
            'maximum_absolute_matrix_error': max(row['external_max_abs'] for row in rows),
            'results': rows, 'environment': environment(), 'library_sha256': sha(library),
            'build_information': build_information(library),
            'molecule_hashes': {path.name: sha(path) for path in molecules},
            'native_adapter_sha256': sha(native) if native else None,
            'native_commit_supplied': args.native_commit}
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    target = args.out / f'run-{args.worker_run}.csv'
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {'status': 'PASS', 'cases': len(rows), 'run_id': args.worker_run,
        'reference': label, 'native_commit_supplied': args.native_commit,
        'environment': environment(), 'affinity': sorted(os.sched_getaffinity(0)),
        'library_sha256': sha(library), 'molecule_hashes': {path.name: sha(path) for path in molecules},
        'build_information': build_information(library),
        'native_adapter_sha256': sha(native) if native else None,
        'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    if native is None:
        from pyscf.gto import moleintor
        adapter = Path(moleintor.libcgto._name).resolve()
        metadata['wheel_adapter'] = str(adapter)
        metadata['wheel_adapter_sha256'] = sha(adapter)
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', '--candidate', type=Path, default=DEFAULT_LIB)
    parser.add_argument('--molecules', default='all', help='all, comma-separated input stems, or input paths')
    parser.add_argument('--basis', default=','.join(BASIS_PAIRS), help='comma-separated short basis names')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--scope', choices=('full', 'cached'), default='full')
    parser.add_argument('--representation', choices=('cartesian', 'spherical'), default='spherical')
    parser.add_argument('--screening', choices=('on', 'off', 'both'), default='on')
    parser.add_argument('--decay-eps', type=float, default=0., help='screened threshold override; zero keeps defaults')
    parser.add_argument('--matrix-gate', type=float, default=2e-8)
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--sample-ms', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--cpus', help='comma-separated allowed logical CPUs; default one per physical core')
    parser.add_argument('--native', type=Path, help='optional external native Libcint adapter; default is PySCF wheel')
    parser.add_argument('--native-commit', help='source commit supplied for native adapter; recorded without ELF verification')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='inspect workload without loading libraries or timing')
    mode.add_argument('--validate-only', action='store_true',
        help='run the benchmark matrix gates once, without pilots, warmups or timing')
    parser.add_argument('--out', type=Path,
        help='new directory for timings; optional validation.json directory with --validate-only')
    parser.add_argument('--worker-run', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.threads <= 3:
        parser.error('threads must be 1..3')
    if (args.processes < 1 or args.repeats < 1 or args.warmup < 0 or
            args.sample_ms <= 0 or not math.isfinite(args.sample_ms)):
        parser.error('processes, repeats and sample-ms must be positive; warmup must be nonnegative')
    if args.matrix_gate <= 0 or args.decay_eps < 0 or not math.isfinite(args.matrix_gate + args.decay_eps):
        parser.error('matrix-gate must be positive and decay-eps nonnegative, both finite')
    molecules, bases = selection(args, parser)
    plan = {'cases': len(molecules)*len(bases), 'processes': args.processes,
        'rounds_per_process': args.repeats, 'warmups': args.warmup,
        'molecules': [path.name for path in molecules], 'bases': bases,
        'library': str(args.lib.resolve()), 'reference': 'provided-native-adapter' if args.native else 'pyscf-wheel',
        'threads': args.threads, 'scope': args.scope, 'representation': args.representation,
        'screening': args.screening, 'basis_parameters': 'candidate-raw-export',
        'sample_ms': args.sample_ms, 'matrix_max_abs_gate': args.matrix_gate}
    if args.plan:
        print(json.dumps(plan, indent=2)); return
    if args.out is None and not args.validate_only:
        parser.error('--out is required to measure; use --plan to inspect')
    if args.native and not args.native_commit:
        parser.error('--native requires --native-commit')
    for path in (args.lib, args.native):
        if path is not None and not path.is_file():
            parser.error(f'missing library: {path}')
    if not hasattr(os, 'sched_getaffinity'):
        parser.error('physical-core placement requires Linux CPU affinity support')
    if args.cpus:
        try:
            cpus = [int(cpu) for cpu in args.cpus.split(',')]
        except ValueError:
            parser.error('--cpus must contain comma-separated integer CPU IDs')
    else:
        cpus = physical_cpus()[:args.threads]
    if len(set(cpus)) != args.threads or not set(cpus) <= os.sched_getaffinity(0):
        parser.error('select exactly one allowed CPU per requested thread')
    os.sched_setaffinity(0, set(cpus))
    pin_threads(args.threads)
    if args.out is not None:
        args.out = args.out.resolve()
    if args.validate_only:
        measure(args, molecules, bases); return
    if args.worker_run:
        measure(args, molecules, bases); return
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for run in range(1, args.processes + 1):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker-run', str(run)]
        subprocess.run(command, check=True)
    summarize(args.out)
    print(f'PASS: {plan["cases"]} cases, {args.processes} processes; outputs={args.out}')


if __name__ == '__main__':
    main()