#!/usr/bin/env python3
"""Fresh 55-class NativePacket<W> lane-occupancy causal campaign.

For each canonical s--f class, compare exactly W mathematical shell quartets:
  * full: W independent quartets in one native SIMD invocation;
  * one-repeated: one live quartet per invocation, invoked W times.

W is read from the benchmark binary (4 for AVX2, 8 for AVX-512).  The two arms
must agree on route selection, counts, and a seven-component normalized
fingerprint before their timing ratio is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYMBOL = {0: "s", 1: "p", 2: "d", 3: "f"}
PAIR_TYPES = [
    (0, 0),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2), (3, 3),
]
FP_FIELDS = (
    "fp_checksum_per_quartet",
    "fp_abs_sum_per_quartet",
    "fp_square_sum_per_quartet",
    "fp_indexed_1_per_quartet",
    "fp_indexed_2_per_quartet",
    "fp_indexed_3_per_quartet",
    "fp_max_abs",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair_label(pair: tuple[int, int]) -> str:
    return SYMBOL[pair[0]] + SYMBOL[pair[1]]


def canonical_classes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bra_index, bra in enumerate(PAIR_TYPES):
        for ket in PAIR_TYPES[:bra_index + 1]:
            la, lb = bra
            lc, ld = ket
            rows.append({
                "class": f"{pair_label(bra)}|{pair_label(ket)}",
                "la": la, "lb": lb, "lc": lc, "ld": ld,
                "nroots": (la + lb + lc + ld) // 2 + 1,
                "n_spherical": (
                    (2 * la + 1) * (2 * lb + 1) *
                    (2 * lc + 1) * (2 * ld + 1)
                ),
            })
    if len(rows) != 55:
        raise RuntimeError("canonical s--f census is not 55")
    return rows


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([^=\s]+)=(.*)$", line)
        if match:
            out[match.group(1)] = match.group(2).strip()
    return out


def controlled_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "TRUE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "BLIS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def git_value(root: Path, *args: str) -> str:
    p = subprocess.run(
        ["git", *args], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("repository root not found")


def parse_provenance_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in out:
            out[key] = value.strip()
    return out


def machine_metadata() -> dict[str, str]:
    info: dict[str, str] = {}
    proc = subprocess.run(
        ["lscpu"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_vendor": info.get("Vendor ID", "unavailable"),
        "cpu_family": info.get("CPU family", "unavailable"),
        "cpu_model_id": info.get("Model", "unavailable"),
        "cpu_model_name": info.get("Model name", "unavailable"),
        "threads_per_core": info.get("Thread(s) per core", "unavailable"),
        "cores_per_socket": info.get("Core(s) per socket", "unavailable"),
        "sockets": info.get("Socket(s)", "unavailable"),
    }


def run_point(
    binary: Path,
    row: dict[str, Any],
    mode: str,
    nprim: int,
    target_integrals: int,
    repeats: int,
    warmup: int,
    cpu: int,
    timeout: int,
    raw_dir: Path,
) -> dict[str, Any]:
    # Both arms execute W quartets per iteration; scale work by spherical output.
    probe = subprocess.run(
        [str(binary), "--class", str(row["class"]), "--mode", mode,
         "--nprim", str(nprim), "--iterations", "1",
         "--repeats", "1", "--warmup", "0"],
        env=controlled_env(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False, timeout=timeout)
    if probe.returncode != 0:
        raise RuntimeError(
            f"probe failed for {row['class']} {mode}:\n{probe.stdout[-4000:]}")
    probe_kv = parse_kv(probe.stdout)
    lanes = int(probe_kv["native_simd_lanes"])
    per_iteration = max(1, lanes * int(row["n_spherical"]))
    iterations = max(
        1, min(2_000_000, math.ceil(target_integrals / per_iteration)))

    command = [
        "taskset", "-c", str(cpu), str(binary),
        "--class", str(row["class"]), "--mode", mode,
        "--nprim", str(nprim), "--iterations", str(iterations),
        "--repeats", str(repeats), "--warmup", str(warmup),
    ]
    p = subprocess.run(
        command, env=controlled_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=timeout)
    safe = str(row["class"]).replace("|", "_")
    raw = raw_dir / f"{safe}_{mode}.log"
    raw.write_text(p.stdout, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(
            f"{row['class']} {mode} exited {p.returncode}; see {raw}")
    kv = parse_kv(p.stdout)

    required = {
        "engine", "mode", "class", "native_simd_lanes",
        "live_lanes_per_invocation", "route_calls_per_iteration",
        "quartets_per_iteration", "spherical_per_quartet",
        "primitive_quartets_per_quartet", "used_native_specialized",
        "iterations", "repeats", "warmup", "ns_per_quartet_median",
        "ns_per_spherical_integral_median", "checksum_per_quartet",
        "fp_count", "fp_count_expected", "count_gate",
        "timed_vs_fp_checksum_gate", *FP_FIELDS,
    }
    missing = sorted(required - kv.keys())
    if missing:
        raise RuntimeError(f"{row['class']} {mode}: missing {missing}")
    if kv["engine"] != "tlno_native_lane_ablation":
        raise RuntimeError("wrong benchmark engine")
    if kv["mode"] != mode or kv["class"] != str(row["class"]):
        raise RuntimeError("benchmark identity mismatch")
    if kv["count_gate"] != "pass" or kv["timed_vs_fp_checksum_gate"] != "pass":
        raise RuntimeError(f"{row['class']} {mode}: self gate failed")
    if int(kv["fp_count"]) != int(kv["fp_count_expected"]):
        raise RuntimeError(f"{row['class']} {mode}: count mismatch")
    if int(kv["quartets_per_iteration"]) != lanes:
        raise RuntimeError(
            f"{row['class']} {mode}: expected exactly W={lanes} quartets")

    expected_live = lanes if mode == "full" else 1
    expected_calls = 1 if mode == "full" else lanes
    if int(kv["live_lanes_per_invocation"]) != expected_live:
        raise RuntimeError(f"{row['class']} {mode}: live-lane contract failed")
    if int(kv["route_calls_per_iteration"]) != expected_calls:
        raise RuntimeError(f"{row['class']} {mode}: invocation contract failed")

    out: dict[str, Any] = {
        **row,
        "mode": mode,
        "native_simd_lanes": lanes,
        "live_lanes_per_invocation": int(kv["live_lanes_per_invocation"]),
        "route_calls_per_iteration": int(kv["route_calls_per_iteration"]),
        "quartets_per_iteration": int(kv["quartets_per_iteration"]),
        "spherical_per_quartet": int(kv["spherical_per_quartet"]),
        "primitive_quartets_per_quartet":
            int(kv["primitive_quartets_per_quartet"]),
        "used_native_specialized": int(kv["used_native_specialized"]),
        "iterations": int(kv["iterations"]),
        "repeats": int(kv["repeats"]),
        "warmup": int(kv["warmup"]),
        "ns_per_quartet_median": float(kv["ns_per_quartet_median"]),
        "ns_per_spherical_integral_median":
            float(kv["ns_per_spherical_integral_median"]),
        "checksum_per_quartet": float(kv["checksum_per_quartet"]),
        "fp_count": int(kv["fp_count"]),
        "raw_log": str(raw),
    }
    for field in FP_FIELDS:
        out[field] = float(kv[field])
    return out


def fingerprint_gate(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    per_element_abs: float = 2.0e-12,
    rel: float = 5.0e-10,
) -> dict[str, Any]:
    if lhs["fp_count"] != rhs["fp_count"]:
        return {
            "status": "fail", "worst_field": "fp_count",
            "worst_scaled": math.inf,
        }
    logical_count = max(
        1.0, float(lhs["fp_count"]) / float(lhs["quartets_per_iteration"]))
    worst_scaled = 0.0
    worst_field = ""
    for field in FP_FIELDS:
        a, b = float(lhs[field]), float(rhs[field])

        if not (math.isfinite(a) and math.isfinite(b)):
            return {
                "status": "fail",
                "worst_field": field,
                "worst_scaled": math.inf,
            }
        
        abs_budget = per_element_abs * (
            1.0 if field == "fp_max_abs" else logical_count)
        allowed = abs_budget + rel * max(abs(a), abs(b))
        scaled = abs(a - b) / allowed
        if scaled > worst_scaled:
            worst_scaled = scaled
            worst_field = field
    return {
        "status": "pass" if worst_scaled <= 1.0 else "fail",
        "worst_field": worst_field,
        "worst_scaled": worst_scaled,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def geomean(values: list[float]) -> float:
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise RuntimeError("invalid timing ratio")
    return math.exp(statistics.fmean(math.log(v) for v in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-lanes", type=int, choices=(4, 8), required=True)
    parser.add_argument(
        "--build-provenance", type=Path, default=None,
        help=("key=value build provenance; defaults to <binary>.provenance.txt "
              "when that file exists"))
    parser.add_argument("--nprim", type=int, default=3)
    parser.add_argument("--target-integrals", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--cpu", type=int,
        default=min(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else 0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing out-dir: {args.out_dir}")
    if min(args.nprim, args.target_integrals, args.repeats, args.timeout) <= 0:
        raise SystemExit("positive campaign sizes required")
    if args.warmup < 0 or args.cpu < 0:
        raise SystemExit("warmup/cpu must be nonnegative")
    if hasattr(os, "sched_getaffinity") and args.cpu not in os.sched_getaffinity(0):
        raise SystemExit(f"cpu {args.cpu} outside affinity mask")

    args.out_dir.mkdir(parents=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir()
    root = repo_root(Path(__file__))
    build_provenance = (
        args.build_provenance.resolve()
        if args.build_provenance is not None
        else Path(str(binary) + ".provenance.txt")
    )
    build_meta = parse_provenance_file(build_provenance)
    generated_utc = datetime.now(timezone.utc).isoformat()
    git_head = git_value(root, "rev-parse", "HEAD")
    git_status = "dirty" if git_value(root, "status", "--porcelain") else "clean"
    csv_meta: dict[str, Any] = {
        "generated_utc": generated_utc,
        "git_head": git_head,
        "git_status": git_status,
        **machine_metadata(),
        "cpu_pin": args.cpu,
        "threads": 1,
        "binary_sha256": sha256(binary),
        "compiler": build_meta.get("compiler", "UNRECORDED"),
        "compiler_version": build_meta.get("compiler_version", "UNRECORDED"),
        "compile_flags": build_meta.get("compile_flags", "UNRECORDED"),
        "isa_label": "w4-avx2" if args.expected_lanes == 4 else "w8-avx512",
    }
    provenance = {
        "schema": "tlno_native_lane_ablation_55_v2",
        **csv_meta,
        "binary": str(binary),
        "build_provenance": str(build_provenance),
        "build_provenance_sha256": (
            sha256(build_provenance) if build_provenance.is_file()
            else "UNRECORDED"
        ),
        "expected_lanes": args.expected_lanes,
        "classes": 55,
        "nprim": args.nprim,
        "target_integrals": args.target_integrals,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "contrast": "W_live_lanes_one_call_vs_1_live_lane_W_calls",
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    curves: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, row in enumerate(canonical_classes()):
        order = ("full", "one-repeated") if index % 2 == 0 else ("one-repeated", "full")
        points: dict[str, dict[str, Any]] = {}
        print(f"[{index+1:02d}/55] {row['class']} order={','.join(order)}", flush=True)
        for mode in order:
            point = run_point(
                binary, row, mode, args.nprim, args.target_integrals,
                args.repeats, args.warmup, args.cpu, args.timeout, raw_dir)
            if point["native_simd_lanes"] != args.expected_lanes:
                raise RuntimeError(
                    f"binary reports W={point['native_simd_lanes']}, "
                    f"expected {args.expected_lanes}")
            point.update(csv_meta)
            curves.append(point)
            points[mode] = point
            print(
                f"  {mode:12s} {point['ns_per_quartet_median']:.6g} ns/quartet "
                f"special={point['used_native_specialized']}",
                flush=True)

        full = points["full"]
        one = points["one-repeated"]
        if full["used_native_specialized"] != one["used_native_specialized"]:
            failures.append(f"{row['class']}: route-selection mismatch")
        for field in ("spherical_per_quartet", "primitive_quartets_per_quartet"):
            if full[field] != one[field]:
                failures.append(f"{row['class']}: {field} mismatch")
        gate = fingerprint_gate(full, one)
        if gate["status"] != "pass":
            failures.append(
                f"{row['class']}: fingerprint {gate['worst_field']} "
                f"scaled={gate['worst_scaled']:.6e}")
        summary.append({
            **row,
            **csv_meta,
            "native_simd_lanes": args.expected_lanes,
            "used_native_specialized": full["used_native_specialized"],
            "full_ns_per_quartet": full["ns_per_quartet_median"],
            "one_repeated_ns_per_quartet": one["ns_per_quartet_median"],
            "lane_pack_speedup":
                one["ns_per_quartet_median"] / full["ns_per_quartet_median"],
            "fingerprint_gate": gate["status"],
            "fingerprint_worst_field": gate["worst_field"],
            "fingerprint_worst_scaled": gate["worst_scaled"],
        })

    write_csv(args.out_dir / "lane_ablation_curves.csv", curves)
    write_csv(args.out_dir / "lane_ablation_55.csv", summary)
    gm = geomean([float(r["lane_pack_speedup"]) for r in summary])
    wins = sum(float(r["lane_pack_speedup"]) > 1.0 for r in summary)
    gate_passes = sum(r["fingerprint_gate"] == "pass" for r in summary)
    text = "\n".join([
        f"native_simd_lanes={args.expected_lanes}",
        "classes=55",
        f"fingerprint_passes={gate_passes}",
        f"fingerprint_failures={55-gate_passes}",
        f"geomean_lane_pack_speedup={gm:.9g}",
        f"full_width_faster_classes={wins}/55",
        f"failures={len(failures)}",
        *[f"failure={x}" for x in failures],
    ]) + "\n"
    (args.out_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    if failures:
        raise SystemExit(f"{len(failures)} lane-ablation gates failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
