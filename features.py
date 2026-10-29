"""Feature construction shared by train.py and explain.py.

Each zone normalises its own load with its own training-period mean/std, so no zone
needs another zone's raw scale. The only cross-zone input is neighbor_lagH: the mean
of the neighbours' *normalised* load at the forecast origin (one-hop aggregation).
"""
import os

import numpy as np
import pandas as pd

from config import HORIZON, TRAIN, neighbours

HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL_FEATURES = ["lagH", "lag24", "lag168", "hod_sin", "hod_cos", "is_weekend", "is_holiday"]
GRAPH_FEATURE = "neighbor_lagH"


def load_clean():
    d = pd.read_csv(os.path.join(HERE, "data", "clean.csv"), parse_dates=["ts"])
    has_temp = d["temperature_c"].notna().any()
    return d, bool(has_temp)


def build(d, has_temp):
    """Return one frame per zone with features, normalised target and denorm stats."""
    H = HORIZON
    wide = d.pivot(index="ts", columns="zone", values="load_mw").asfreq("h")
    tr = wide.loc[TRAIN[0]:TRAIN[1]]
    mu, sd = tr.mean(), tr.std()
    z = (wide - mu) / sd                               # per-zone normalised load
    nb = neighbours()
    feats = LOCAL_FEATURES + (["temperature_c"] if has_temp else [])

    cal = d.set_index(["zone", "ts"])
    out = {}
    for zone in wide.columns:
        f = pd.DataFrame(index=wide.index)
        f["y"] = z[zone]                               # target at time t (normalised)
        f["lagH"] = z[zone].shift(H)                   # latest load known at forecast origin
        f["lag24"] = z[zone].shift(24)
        f["lag168"] = z[zone].shift(168)
        hod = f.index.hour
        f["hod_sin"] = np.sin(2 * np.pi * hod / 24)
        f["hod_cos"] = np.cos(2 * np.pi * hod / 24)
        f["is_weekend"] = (f.index.dayofweek >= 5).astype(int)
        c = cal.loc[zone].reindex(f.index)
        f["is_holiday"] = c["is_holiday"].fillna(0).astype(int)
        if has_temp:
            t = c["temperature_c"]
            tmu, tsd = t.loc[TRAIN[0]:TRAIN[1]].mean(), t.loc[TRAIN[0]:TRAIN[1]].std()
            f["temperature_c"] = (t - tmu) / tsd
        f[GRAPH_FEATURE] = z[nb[zone]].shift(H).mean(axis=1, skipna=False)
        f["load_mw"] = wide[zone]
        f = f.dropna(subset=["y"] + feats + [GRAPH_FEATURE])
        out[zone] = {"frame": f, "mu": float(mu[zone]), "sd": float(sd[zone])}
    return out, feats


def split(frame, period):
    return frame.loc[period[0]:period[1]]
