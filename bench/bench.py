#!/usr/bin/env python3
"""Steady-state decode speed of one llama-bench configuration.

The model (16 GB) does not fit in this machine's RAM (15 GB) together with
everything else, so the experts the CPU reads come off the SSD the first time
they are touched. Measured on Ornith-1.5-27B-A3B with 28 CPU layers, six
128-token repetitions ran at 4.9, 10.7, 16.7, 19.6, 22.1 and 21.0 tok/s while
20.8 GB were read from disk. A single short run therefore measures the disk,
not the engine. This script runs enough repetitions to get past that, reports
the median of the last four, and prints how much was read from disk so a
number that is still disk-bound can be recognised as such.

usage: bench.py [--reps N] [--label TEXT] -- <llama-bench args>
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, 'llama.cpp', 'build', 'bin', 'llama-bench')
RESULTS = os.path.join(ROOT, 'bench', 'results.jsonl')


def vmstat():
    out = {}
    with open('/proc/vmstat') as f:
        for line in f:
            k, v = line.split()
            if k in ('pgmajfault', 'pgpgin'):
                out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--tail', type=int, default=4, help='repetitions the steady-state median is taken over')
    ap.add_argument('--label', default='')
    ap.add_argument('args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.args[1:] if a.args and a.args[0] == '--' else a.args

    cmd = [BENCH, *args, '-r', str(a.reps), '-o', 'jsonl']
    before = vmstat()
    t0 = time.time()
    # -v prints the model's vocabulary, which is not always valid UTF-8 (LFM2.5's is not): replace, don't crash
    proc = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    wall = time.time() - t0
    after = vmstat()

    hybrid_log = [l for l in proc.stderr.splitlines() if 'llama_moe_cache' in l]
    rows = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith('{')]
    if proc.returncode != 0 or not rows:
        print(proc.stderr[-3000:], file=sys.stderr)
        sys.exit(1)

    disk_mib = (after['pgpgin'] - before['pgpgin']) / 1024
    faults = after['pgmajfault'] - before['pgmajfault']
    for r in rows:
        samples = r['samples_ts']
        steady = statistics.median(samples[-a.tail:])
        test = f"pp{r['n_prompt']}" if r['n_prompt'] else f"tg{r['n_gen']}"
        print(f"{a.label:24s} {test:7s} steady {steady:6.2f} tok/s   samples {[round(x, 1) for x in samples]}")
        with open(RESULTS, 'a') as f:
            f.write(json.dumps({
                'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'label': a.label, 'args': args, 'test': test,
                'steady_ts': steady, 'samples_ts': samples, 'disk_read_mib': round(disk_mib),
                'major_faults': faults, 'wall_s': round(wall, 1), 'moe_cache_log': hybrid_log,
            }) + '\n')
    print(f"{'':24s} disk read {disk_mib:,.0f} MiB, {faults:,} major faults, {wall:.0f} s")
    for l in hybrid_log:
        print('   ', l.strip())


if __name__ == '__main__':
    main()
