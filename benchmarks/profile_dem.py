#!/usr/bin/env python3
"""
Profile dem_pix on synthetic inputs.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np

from demregpy.demmap import dem_pix


def build_inputs(nt: int, nf: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    logt = np.linspace(5.7, 7.2, nt)
    dlogt = np.full(nt, logt[1] - logt[0])
    rmatrix = rng.random((nt, nf)) + 0.1
    dnin = rng.random(nf) + 1.0
    ednin = np.full(nf, 0.1)
    glc = np.zeros(nf)
    return dnin, ednin, rmatrix, logt, dlogt, glc


def build_batch(npix: int, nt: int, nf: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    logt = np.linspace(5.7, 7.2, nt)
    dlogt = np.full(nt, logt[1] - logt[0])
    rmatrix = rng.random((nt, nf)) + 0.1
    dnin = rng.random((npix, nf)) + 1.0
    ednin = np.full((npix, nf), 0.1)
    glc = np.zeros(nf)
    return dnin, ednin, rmatrix, logt, dlogt, glc


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile dem_pix on synthetic inputs.")
    parser.add_argument("--nt", type=int, default=40, help="Number of temperature bins.")
    parser.add_argument("--nf", type=int, default=6, help="Number of filters.")
    parser.add_argument("--npix", type=int, default=1000, help="Pixels per batch run.")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="batch",
        help="Profile a single dem_pix call or a batch loop.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["cumtime", "tottime", "calls", "ncalls"],
        help="Sort key for pstats output.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Number of rows to print.")
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Optional path to dump raw cProfile stats.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        dnin, ednin, rmatrix, logt, dlogt, glc = build_inputs(args.nt, args.nf, args.seed)

        def run():
            dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, warn=False)

        label = f"dem_pix profile nt={args.nt} nf={args.nf}"
    else:
        dnin, ednin, rmatrix, logt, dlogt, glc = build_batch(
            args.npix, args.nt, args.nf, args.seed
        )

        def run():
            for i in range(args.npix):
                dem_pix(dnin[i], ednin[i], rmatrix, logt, dlogt, glc, warn=False)

        label = f"dem_pix profile batch npix={args.npix} nt={args.nt} nf={args.nf}"

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.dump))

    print(label)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
