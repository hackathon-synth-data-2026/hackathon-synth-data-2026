"""Post-hoc clinical consistency repair for synthetic ICU summary rows.

Usage: python repair.py in.parquet out.parquet
Rules enforced (they hold in >99.9% of real rows):
  lowest <= first, last, median <= highest for each vital
  diastolic <= systolic pressure for each summary
  GCS within 3..15
"""
import sys
import pandas as pd

VITALS = ["DiasABP", "GCS", "Glucose", "HR", "MAP", "NIDiasABP", "NIMAP",
          "NISysABP", "RespRate", "SaO2", "SysABP", "Temp"]
SUFFIX = ["first", "last", "lowest", "highest", "median"]
PRESSURE_PAIRS = [("DiasABP", "SysABP"), ("NIDiasABP", "NISysABP")]

d = pd.read_parquet(sys.argv[1])


def fix_gcs():
    for s in SUFFIX:
        d[f"GCS_{s}"] = d[f"GCS_{s}"].clip(3, 15)


def fix_blocks():
    for v in VITALS:
        block = d[[f"{v}_{s}" for s in SUFFIX]]
        lo, hi = block.min(axis=1), block.max(axis=1)
        d[f"{v}_lowest"] = lo.where(d[f"{v}_lowest"].notna())
        d[f"{v}_highest"] = hi.where(d[f"{v}_highest"].notna())
        for s in ["first", "last", "median"]:
            c = f"{v}_{s}"
            d[c] = d[c].clip(lower=d[f"{v}_lowest"], upper=d[f"{v}_highest"]).where(d[c].notna())


def fix_pressures():
    for dia, sys_ in PRESSURE_PAIRS:
        for s in SUFFIX:
            a, b = f"{dia}_{s}", f"{sys_}_{s}"
            swap = (d[a] > d[b]).fillna(False)
            d.loc[swap, [a, b]] = d.loc[swap, [b, a]].values


fix_gcs()
for _ in range(3):
    fix_blocks()
    fix_pressures()
fix_blocks()

d.to_parquet(sys.argv[2], index=False)
