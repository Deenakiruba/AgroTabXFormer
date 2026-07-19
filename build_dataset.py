"""
Build the modelling dataset from raw NASA POWER daily records.

Produces one observation per sowing event (district x season x year):
  - tabular predictors known on the day of sowing
  - a 170 x 4 growing-season sequence (Tmax, Tmin, GDD, cumulative GDD)
  - target: harvest day, the first day on which accumulated GDD reaches AGDD_REQ

Accumulated GDD and harvest day define the target only and are never used
as model inputs.

Usage:  python -m src.build_dataset
"""
import os
import numpy as np
import pandas as pd

from .config import (RAW_DIR, CLEAN_DIR, OUT_DIR, DISTRICT_FILES, DISTRICTS,
                     POWER_COLS, T_BASE, AGDD_REQ, HORIZON, MIN_DAYS, SEASONS)


def read_power_csv(path):
    """Read a NASA POWER export, skipping the variable-length header."""
    with open(path) as f:
        lines = f.readlines()
    try:
        hdr = next(i for i, l in enumerate(lines) if l.lstrip().startswith("YEAR"))
    except StopIteration:
        hdr = next(i for i, l in enumerate(lines) if "-END HEADER-" in l) + 1
    return pd.read_csv(path, skiprows=hdr)


def build_clean(district):
    """Clean one district: mask fill values, interpolate, derive Avg_Temp and GDD."""
    raw = read_power_csv(os.path.join(RAW_DIR, DISTRICT_FILES[district]))
    raw[POWER_COLS] = raw[POWER_COLS].mask(raw[POWER_COLS] <= -900, np.nan)

    d = pd.DataFrame()
    d["Date"] = (pd.to_datetime(raw["YEAR"].astype(str), format="%Y")
                 + pd.to_timedelta(raw["DOY"] - 1, unit="D"))
    d["Year"], d["DOY"] = raw["YEAR"], raw["DOY"]
    d["Max_Temp"], d["Min_Temp"] = raw["T2M_MAX"], raw["T2M_MIN"]
    d["Humidity"], d["Rainfall"] = raw["RH2M"], raw["PRECTOTCORR"]
    d["Soil_Moisture"] = raw["GWETTOP"]

    for c in ["Max_Temp", "Min_Temp", "Humidity", "Rainfall", "Soil_Moisture"]:
        d[c] = d[c].interpolate(limit_direction="both")

    d["Avg_Temp"] = (d["Max_Temp"] + d["Min_Temp"]) / 2.0
    d["GDD"] = np.maximum(0.0, d["Avg_Temp"] - T_BASE)

    # GDD is an affine transform of Avg_Temp above T_BASE; SDs must match.
    assert np.isclose(d["GDD"].std(), d["Avg_Temp"].std(), rtol=1e-3), \
        f"{district}: GDD formula error"

    return d.sort_values("Date").reset_index(drop=True)


def harvest_day(gdd, start_idx, req=AGDD_REQ, min_d=MIN_DAYS, max_d=HORIZON):
    """First day on which accumulated GDD reaches `req`, subject to a minimum duration."""
    seg = gdd[start_idx:start_idx + max_d]
    if len(seg) < min_d:
        return None
    acc = np.cumsum(seg)
    ok = np.where((acc >= req) & (np.arange(1, len(acc) + 1) >= min_d))[0]
    return int(ok[0]) + 1 if len(ok) else None


def build_dataset(clean):
    """One row per sowing event, plus the matching growing-season sequence."""
    rows, seqs = [], []
    for district in DISTRICTS:
        d = clean[district]
        pos = {t: i for i, t in enumerate(d["Date"])}
        gdd = d["GDD"].values

        for season, (month, day) in SEASONS.items():
            for year in sorted(d["Year"].unique()):
                try:
                    sow = pd.Timestamp(year=int(year), month=month, day=day)
                except ValueError:
                    continue
                if sow not in pos:
                    continue

                i = pos[sow]
                h = harvest_day(gdd, i)
                if h is None:
                    continue

                hist = d.iloc[max(0, i - 30):i]          # 30 days before sowing
                rows.append(dict(
                    District=district, Season=season,
                    Sowing_Year=year, Sowing_DOY=sow.dayofyear,
                    SM_sow=d.loc[i, "Soil_Moisture"], RH_sow=d.loc[i, "Humidity"],
                    Tmax_sow=d.loc[i, "Max_Temp"], Tmin_sow=d.loc[i, "Min_Temp"],
                    Rain_30d=hist["Rainfall"].sum(), Tavg_30d=hist["Avg_Temp"].mean(),
                    GDD_30d=hist["GDD"].mean(), SM_30d=hist["Soil_Moisture"].mean(),
                    RH_30d=hist["Humidity"].mean(),
                    Harvest_Day=h,
                ))

                seg = d.iloc[i:i + HORIZON]
                arr = np.zeros((HORIZON, 4), np.float32)
                n = len(seg)
                arr[:n, 0] = seg["Max_Temp"]
                arr[:n, 1] = seg["Min_Temp"]
                arr[:n, 2] = seg["GDD"]
                arr[:n, 3] = np.cumsum(seg["GDD"]) / 1000.0
                seqs.append(arr)

    return pd.DataFrame(rows), np.array(seqs, np.float32)


def main():
    clean = {}
    for district in DISTRICTS:
        d = build_clean(district)
        clean[district] = d
        d.to_csv(os.path.join(CLEAN_DIR, f"Clean_{district}.csv"), index=False)
        print(f"{district:<12} n={len(d)}  GDD {d.GDD.min():.2f}-{d.GDD.max():.2f}")

    df, seq = build_dataset(clean)
    df.to_csv(os.path.join(OUT_DIR, "model_dataset.csv"), index=False)
    np.save(os.path.join(OUT_DIR, "seq.npy"), seq)

    assert "Acc_GDD" not in df.columns, "leaked feature present"
    print(f"\nsowing events: {len(df)}   sequences: {seq.shape}")
    print(f"harvest day: mean {df.Harvest_Day.mean():.1f}  sd {df.Harvest_Day.std():.1f}  "
          f"range {df.Harvest_Day.min()}-{df.Harvest_Day.max()}")


if __name__ == "__main__":
    main()
