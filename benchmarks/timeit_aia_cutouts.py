#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark dn2dem on saved AIA cutouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import timeit

import numpy as np
import scipy.io as io
from sunpy.map import Map

from demregpy import dn2dem
from demregpy.tresp import aia_tresp


DATA_DIR = Path(__file__).resolve().parents[1] / "demregpy" / "tests" / "data" / "aia"
WAVES = [94, 131, 171, 193, 211, 335]
TIME_TAG = "2014-01-01T00-00-00"


def load_cutouts():
    files = [DATA_DIR / f"aia_synoptic_{TIME_TAG}_{w:03d}.fits" for w in WAVES]
    maps = [Map(p) for p in files]
    maps = sorted(maps, key=lambda x: x.wavelength)
    return maps


def load_response():
    trin = io.readsav(aia_tresp)
    tresp_logt = np.array(trin["logt"])
    nt = len(tresp_logt)
    nf = len(trin["tr"][:])
    trmatrix = np.zeros((nt, nf))
    for i in range(nf):
        trmatrix[:, i] = trin["tr"][i]
    return trmatrix, tresp_logt


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dn2dem on AIA cutouts.")
    parser.add_argument("--size", type=int, default=32, help="Square subregion size.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeats for timeit.")
    parser.add_argument("--number", type=int, default=1, help="Calls per repeat.")
    args = parser.parse_args()

    maps = load_cutouts()
    trmatrix, tresp_logt = load_response()

    nx = ny = args.size
    data = np.zeros((nx, ny, len(maps)))
    for j, m in enumerate(maps):
        data[:, :, j] = m.data[:nx, :ny]
    data[data < 0] = 0.0
    edata = 0.1 * data + 1e-8

    logtemps = np.linspace(5.7, 7.1, num=17)
    temps = 10 ** logtemps

    def run():
        dn2dem(data, edata, trmatrix, tresp_logt, temps, nmu=40, warn=False)

    results = timeit.repeat(run, repeat=args.repeat, number=args.number)
    best = min(results) / args.number
    avg = (sum(results) / len(results)) / args.number

    print(f"dn2dem AIA cutouts size={args.size}x{args.size}")
    print(f"best: {best:.6f}s  avg: {avg:.6f}s  repeats={args.repeat} number={args.number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
