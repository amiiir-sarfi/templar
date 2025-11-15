#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import numpy as np


def main():
    root = Path("/root/datasets/dclm_tokenized")
    for npy in sorted(root.glob("train_*.npy")):
        m = re.fullmatch(r"train_(\d{6})\.npy", npy.name)
        if not m:
            continue
        idx = m.group(1)
        out_path = npy.with_name(f"sample_ids_{idx}.bin")

        arr = np.load(npy, mmap_mode="r")  # memory-friendly
        total = arr.size
        n = total // 2048  # one entry per 2048 elements

        zeros = np.arange(n, dtype=np.int32)
        zeros.tofile(out_path)

        rem = total % 2048
        print(f"Wrote {out_path.name}: {n} zeros (from {total} elems, {rem} leftover).")
        # break


if __name__ == "__main__":
    main()
