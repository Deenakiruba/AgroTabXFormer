"""
Statistical analyses reported in the manuscript.

  - Nonlinearity: BDS and Terasvirta tests on deseasonalised AR residuals
  - Spatial/seasonal effects: two-way factorial ANOVA, Tukey HSD, Levene, Welch
  - Model comparison: Diebold-Mariano test on pooled out-of-fold predictions
  - Uncertainty: bootstrap confidence intervals
  - Error propagation: Monte Carlo perturbation of the temperature input

Usage:  python -m src.statistics
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.ar_model import ar_select_order, AutoReg

from .config import OUT_DIR, CLEAN_DIR, DISTRICTS, AGDD_REQ, MIN_DAYS


# ---------------------------------------------------------------- nonlinearity
def deseasonalise(d, column):
    """Remove the day-of-year climatological mean."""
    clim = d.groupby("DOY")[column].transform("mean")
    return (d[column] - clim).values


def ar_residuals(x, max_lags=60):
    """Fit AR(p) with p chosen by AIC; return residuals and the order."""
    sel = ar_select_order(x, maxlag=max_lags, ic="aic", old_names=False)
    p = max(sel.ar_lags) if sel.ar_lags else 1
    res = AutoReg(x, lags=p, old_names=False).fit()
    return res.resid, p


def nonlinearity_table(variables=("Rainfall", "Avg_Temp")):
    """AR order and residual SD per district and variable, for BDS input."""
    rows = []
    for district in DISTRICTS:
        d = pd.read_csv(os.path.join(CLEAN_DIR, f"Clean_{district}.csv"))
        for var in variables:
            x = deseasonalise(d, var)
            resid, p = ar_residuals(x)
            rows.append(dict(District=district, Series=var, AR_order=p,
                             resid_sd=round(float(np.std(resid, ddof=1)), 4),
                             at_ceiling=(p >= 60)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ANOVA
def spatial_seasonal_anova(df):
    """Two-way factorial ANOVA with interaction, plus Tukey, Levene and Welch."""
    model = ols("Harvest_Day ~ C(District) * C(Season)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    tukey = pairwise_tukeyhsd(df.Harvest_Day, df.District, alpha=0.05)
    groups = [g.Harvest_Day.values for _, g in df.groupby("District")]
    levene = stats.levene(*groups)
    welch = stats.f_oneway(*groups)

    return dict(anova=anova.round(3), r_squared=round(model.rsquared, 3),
                tukey=tukey, levene=levene, welch=welch)


# ---------------------------------------------------------------- DM test
def diebold_mariano(y, p1, p2, power=2):
    """
    Diebold-Mariano test of equal predictive accuracy, with the
    Harvey-Leybourne-Newbold small-sample correction.
    A negative statistic indicates that p1 has lower loss.
    """
    y, p1, p2 = (np.asarray(v, float).ravel() for v in (y, p1, p2))
    d = np.abs(y - p1) ** power - np.abs(y - p2) ** power
    n = len(d)
    d_bar = d.mean()
    variance = np.sum((d - d_bar) ** 2) / n
    if variance <= 0:
        return np.nan, np.nan
    stat = d_bar / np.sqrt(variance / n) * np.sqrt((n + 1) / n)
    p_value = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)


def dm_on_valid(y, p1, p2, power=2):
    """DM restricted to rows where both models produced predictions."""
    mask = np.isfinite(p1) & np.isfinite(p2)
    return diebold_mariano(y[mask], p1[mask], p2[mask], power)


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(y, p, n_boot=1000, seed=0):
    """95% bootstrap confidence intervals for RMSE and MAE."""
    mask = np.isfinite(p)
    y_v, p_v = np.asarray(y, float)[mask], np.asarray(p, float)[mask]
    rng = np.random.default_rng(seed)
    n = len(y_v)
    r, a = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r.append(np.sqrt(np.mean((y_v[idx] - p_v[idx]) ** 2)))
        a.append(np.mean(np.abs(y_v[idx] - p_v[idx])))
    return np.percentile(r, [2.5, 97.5]), np.percentile(a, [2.5, 97.5])


# ---------------------------------------------------------------- propagation
def propagate_temperature_error(seq, temp_sd, n_draws=1000, n_events=200, seed=0):
    """
    Monte Carlo: perturb the daily GDD series by a temperature error of
    `temp_sd` degrees C and re-derive harvest day. Returns the mean SD of
    the resulting harvest-day distribution.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(seq), min(n_events, len(seq)), replace=False)
    spreads = []
    for i in chosen:
        gdd = seq[i][:, 2]
        days = []
        for _ in range(n_draws):
            perturbed = np.maximum(0.0, gdd + rng.normal(0, temp_sd, size=len(gdd)))
            acc = np.cumsum(perturbed)
            hit = np.where((acc >= AGDD_REQ) &
                           (np.arange(1, len(acc) + 1) >= MIN_DAYS))[0]
            days.append(int(hit[0]) + 1 if len(hit) else len(acc))
        spreads.append(np.std(days))
    return float(np.mean(spreads))


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "model_dataset.csv"))
    seq = np.load(os.path.join(OUT_DIR, "seq.npy"))
    y = df.Harvest_Day.values.astype(float)

    print("=== Nonlinearity: AR order selection ===")
    print(nonlinearity_table().to_string(index=False))

    print("\n=== Spatial and seasonal effects ===")
    res = spatial_seasonal_anova(df)
    print(res["anova"])
    print(f"R^2 = {res['r_squared']}")
    print(f"Levene: F={res['levene'].statistic:.1f}, p={res['levene'].pvalue:.3g}")
    print(f"Welch:  F={res['welch'].statistic:.1f}, p={res['welch'].pvalue:.3g}")
    print(res["tukey"])

    print("\n=== Error propagation ===")
    for sd in (1.0, 2.0, 3.0, 4.0):
        spread = propagate_temperature_error(seq, sd)
        print(f"temperature error ±{sd:.1f} °C  ->  harvest uncertainty ±{spread:.2f} days")

    # DM and bootstrap require saved out-of-fold predictions (oof_*.npy)
    import glob
    oof_files = glob.glob(os.path.join(OUT_DIR, "oof_*.npy"))
    if oof_files:
        oof = {os.path.basename(f)[4:-4]: np.load(f) for f in oof_files}
        print("\n=== Diebold-Mariano (negative => first model better) ===")
        if "AgroTabXFormer" in oof:
            for name, pred in oof.items():
                if name == "AgroTabXFormer":
                    continue
                stat, p = dm_on_valid(y, oof["AgroTabXFormer"], pred)
                print(f"  AgroTabXFormer vs {name:<16} DM={stat:+.3f}  p={p:.4f}")

        print("\n=== Bootstrap 95% CI ===")
        for name, pred in oof.items():
            (rl, rh), (al, ah) = bootstrap_ci(y, pred)
            mask = np.isfinite(pred)
            r = np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))
            print(f"  {name:<16} RMSE={r:.2f} [{rl:.2f}, {rh:.2f}]")


if __name__ == "__main__":
    main()
