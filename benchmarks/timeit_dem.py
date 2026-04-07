#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time dem_pix end-to-end on synthetic inputs.
"""

from __future__ import annotations

import argparse
import timeit

import numpy as np

from demregpy.demmap import dem_pix


def build_inputs(nt: int, nf: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    logt = np.linspace(5.7, 7.2, nt)
    dlogt = np.full(nt, (logt[1] - logt[0]))
    rmatrix = rng.random((nt, nf)) + 0.1
    dnin = rng.random(nf) + 1.0
    ednin = np.full(nf, 0.1)
    glc = np.zeros(nf)
    return dnin, ednin, rmatrix, logt, dlogt, glc


def build_batch(npix: int, nt: int, nf: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    logt = np.linspace(5.7, 7.2, nt)
    dlogt = np.full(nt, (logt[1] - logt[0]))
    rmatrix = rng.random((nt, nf)) + 0.1
    dnin = rng.random((npix, nf)) + 1.0
    ednin = np.full((npix, nf), 0.1)
    glc = np.zeros(nf)
    return dnin, ednin, rmatrix, logt, dlogt, glc


def main() -> int:
    parser = argparse.ArgumentParser(description="Time dem_pix on synthetic inputs.")
    parser.add_argument("--nt", type=int, default=40, help="Number of temperature bins.")
    parser.add_argument("--nf", type=int, default=6, help="Number of filters.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of timing repeats.")
    parser.add_argument("--number", type=int, default=3, help="Calls per repeat.")
    parser.add_argument("--npix", type=int, default=200, help="Pixels per run (batch loop).")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="batch",
        help="Time a single dem_pix call or a batch loop.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        dnin, ednin, rmatrix, logt, dlogt, glc = build_inputs(args.nt, args.nf)

        def run():
            dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, warn=False)
    else:
        dnin, ednin, rmatrix, logt, dlogt, glc = build_batch(
            args.npix, args.nt, args.nf
        )

        def run():
            for i in range(args.npix):
                dem_pix(dnin[i], ednin[i], rmatrix, logt, dlogt, glc, warn=False)

    results = timeit.repeat(run, repeat=args.repeat, number=args.number)
    best = min(results) / args.number
    avg = (sum(results) / len(results)) / args.number

    if args.mode == "single":
        label = f"dem_pix nt={args.nt} nf={args.nf}"
    else:
        label = f"dem_pix batch npix={args.npix} nt={args.nt} nf={args.nf}"
    print(label)
    print(f"best: {best:.6f}s  avg: {avg:.6f}s  repeats={args.repeat} number={args.number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
