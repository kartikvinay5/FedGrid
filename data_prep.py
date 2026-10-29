"""Load ICED 'Yearly Demand Profile' exports, clean them, and add calendar/weather columns.

Usage: python data_prep.py
Reads  data/raw/*.xlsx (and data/weather.csv if present)
Writes data/clean.csv, outputs/data_quality.json
"""
import glob
import json
import os

import holidays
import numpy as np
import pandas as pd

from config import HOLIDAY_SUBDIV, WEATHER_POINTS, ZONES

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_TO_CODE = {v: k for k, v in ZONES.items()}


def load_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "raw", "*.xlsx"))):
        d = pd.read_excel(f)
        d = d[d["State"].astype(str).str.contains(r" - 20\d\d$")]  # drop footer rows
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d[["state", "year"]] = d["State"].str.rsplit(" - ", n=1, expand=True)
    d["ts"] = pd.to_datetime(d["Date"] + "-" + d["year"], format="%d-%b %I%p-%Y")
    d["zone"] = d["state"].map(NAME_TO_CODE)
    d["load_mw"] = pd.to_numeric(d["Hourly Demand Met (in MW)"], errors="coerce")
    d = d[["zone", "ts", "load_mw"]].drop_duplicates(["zone", "ts"])
    return d.sort_values(["zone", "ts"]).reset_index(drop=True)


def flag_copied_days(d):
    """A day whose 24 hourly values exactly repeat an earlier day of the same zone is a
    source copy (ICED re-uses months), so it is set to NaN rather than trusted."""
    d = d.copy()
    d["date"] = d["ts"].dt.date
    copied = {}
    for z, g in d.groupby("zone"):
        seen, bad = set(), []
        for day, x in g.groupby("date"):
            sig = tuple(np.round(x["load_mw"].values, 2))
            if len(sig) == 24 and sig in seen:
                bad.append(day)
            seen.add(sig)
        copied[z] = bad
        d.loc[(d["zone"] == z) & d["date"].isin(bad), "load_mw"] = np.nan
    return d.drop(columns="date"), copied


def add_calendar(d):
    years = sorted(d["ts"].dt.year.unique())
    d["is_holiday"] = 0
    for z, sub in HOLIDAY_SUBDIV.items():
        cal = holidays.India(subdiv=sub, years=years)
        m = d["zone"] == z
        d.loc[m, "is_holiday"] = d.loc[m, "ts"].dt.date.map(lambda x: int(x in cal))
    return d


def add_weather(d):
    path = os.path.join(HERE, "data", "weather.csv")
    if not os.path.exists(path):
        d["temperature_c"] = np.nan
        return d, False
    # Open-Meteo multi-location CSV: a location table, a blank line, then hourly rows
    # with a location_id column in the order the coordinates were requested.
    lines = open(path, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("location_id,time"))
    w = pd.read_csv(path, skiprows=start)
    w["ts"] = pd.to_datetime(w["time"])
    codes = list(WEATHER_POINTS)
    w["zone"] = w["location_id"].map(lambda i: codes[int(i)])
    tcol = [c for c in w.columns if c.startswith("temperature_2m")][0]
    w = w.rename(columns={tcol: "temperature_c"})[["zone", "ts", "temperature_c"]]
    return d.merge(w, on=["zone", "ts"], how="left"), True


def main():
    d = load_raw()
    d, copied = flag_copied_days(d)
    d = add_calendar(d)
    d, has_weather = add_weather(d)
    d.to_csv(os.path.join(HERE, "data", "clean.csv"), index=False)

    quality = {
        "rows": int(len(d)),
        "range": [str(d["ts"].min()), str(d["ts"].max())],
        "weather": has_weather,
        "per_zone": {
            z: {
                "hours": int(len(g)),
                "copied_days_removed": len(copied[z]),
                "copied_months": sorted({str(x)[:7] for x in copied[z]}),
                "mean_mw": round(float(g["load_mw"].mean()), 1),
                "min_mw": round(float(g["load_mw"].min()), 1),
                "max_mw": round(float(g["load_mw"].max()), 1),
                "holiday_days": int(g.loc[g["is_holiday"] == 1, "ts"].dt.date.nunique()),
            }
            for z, g in d.groupby("zone")
        },
    }
    json.dump(quality, open(os.path.join(HERE, "outputs", "data_quality.json"), "w"), indent=2)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
