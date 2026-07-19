"""
Year-blocked rolling-origin cross-validation for all models.

Folds are blocked by sowing year so that no year contributes to both the
training and test partition of any fold. This prevents leakage between
sowing events of the same season, which share overlapping weather.

Scalers and target normalisation are fitted on training partitions only.

Usage:  python -m src.evaluate
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

from .config import OUT_DIR, FEATURES, CATEGORICALS, BEST_PARAMS, N_SPLITS, SEEDS
from . import models as M


# ---------------------------------------------------------------- metrics
def rmse(a, p):  return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))
def mae(a, p):   return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))
def bias(a, p):  return float(np.mean(np.asarray(p, float) - np.asarray(a, float)))


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return 100 * float(np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p))))


def year_blocked_split(years, n_splits=N_SPLITS):
    """Expanding-window CV that splits on years, never within a year."""
    years = np.asarray(years)
    unique_years = np.sort(np.unique(years))
    for tr_y, te_y in TimeSeriesSplit(n_splits=n_splits).split(unique_years):
        yield (np.where(np.isin(years, unique_years[tr_y]))[0],
               np.where(np.isin(years, unique_years[te_y]))[0])


# ---------------------------------------------------------------- CV routines
def climatology_cv(df, years):
    """Baseline: district x season mean harvest day, fitted on training folds."""
    rows = []
    for k, (tr, te) in enumerate(year_blocked_split(years)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        means = dtr.groupby(CATEGORICALS).Harvest_Day.mean()
        p = dte.set_index(CATEGORICALS).index.map(means).values
        p = np.where(pd.isna(p), dtr.Harvest_Day.mean(), p).astype(float)
        y = dte.Harvest_Day.values.astype(float)
        rows.append(dict(fold=k + 1, RMSE=rmse(y, p), MAE=mae(y, p),
                         SMAPE=smape(y, p), Bias=bias(y, p)))
    return pd.DataFrame(rows)


def tabular_cv(make_model, X, y, years):
    rows = []
    for k, (tr, te) in enumerate(year_blocked_split(years)):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append(dict(fold=k + 1, RMSE=rmse(y[te], p), MAE=mae(y[te], p),
                         SMAPE=smape(y[te], p), Bias=bias(y[te], p)))
    return pd.DataFrame(rows)


def sequence_cv(build_fn, X_tab, X_seq, y, years, params=BEST_PARAMS, seeds=SEEDS):
    """CV for the dual-input sequence models."""
    rows = []
    for k, (tr, te) in enumerate(year_blocked_split(years)):
        scaler = StandardScaler().fit(X_tab[tr])
        Xt_tr, Xt_te = scaler.transform(X_tab[tr]), scaler.transform(X_tab[te])

        C = X_seq.shape[-1]
        mu = X_seq[tr].reshape(-1, C).mean(0)
        sd = X_seq[tr].reshape(-1, C).std(0) + 1e-8
        Xs_tr, Xs_te = (X_seq[tr] - mu) / sd, (X_seq[te] - mu) / sd

        y_mu, y_sd = y[tr].mean(), y[tr].std()
        y_tr_scaled = (y[tr] - y_mu) / y_sd

        for seed in seeds:
            keras.backend.clear_session()
            tf.random.set_seed(seed)
            np.random.seed(seed)

            model = build_fn(Xt_tr.shape[1], d=params["d"], heads=params["heads"],
                             blocks=params["blocks"], drop=params["drop"])
            model.compile(keras.optimizers.Adam(params["lr"]), loss="mse")
            model.fit([Xs_tr, Xt_tr], y_tr_scaled, validation_split=0.15,
                      epochs=400, batch_size=params["bs"],
                      callbacks=M.callbacks(), verbose=0)

            p = model.predict([Xs_te, Xt_te], verbose=0).ravel() * y_sd + y_mu
            rows.append(dict(fold=k + 1, seed=seed,
                             RMSE=rmse(y[te], p), MAE=mae(y[te], p),
                             SMAPE=smape(y[te], p), Bias=bias(y[te], p)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def summarise(cv, name):
    """Average seeds within each fold first, then report fold mean +/- SD."""
    g = cv.groupby("fold").mean(numeric_only=True) if "seed" in cv.columns else cv
    return dict(
        Model=name,
        RMSE=f"{g.RMSE.mean():.2f} ± {g.RMSE.std():.2f}",
        MAE=f"{g.MAE.mean():.2f} ± {g.MAE.std():.2f}",
        SMAPE=f"{g.SMAPE.mean():.2f} ± {g.SMAPE.std():.2f}",
        Bias=f"{g.Bias.mean():+.2f}",
    )


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "model_dataset.csv"))
    seq = np.load(os.path.join(OUT_DIR, "seq.npy"))

    X_tab_df = pd.get_dummies(df[CATEGORICALS + FEATURES], columns=CATEGORICALS)
    X_tab = X_tab_df.values.astype(np.float32)
    y = df.Harvest_Day.values.astype(np.float32)
    years = df.Sowing_Year.values

    assert X_tab.shape[0] == seq.shape[0] == len(y) == len(df)
    assert "Harvest_Day" not in X_tab_df.columns

    for k, (tr, te) in enumerate(year_blocked_split(years)):
        overlap = set(years[tr]) & set(years[te])
        assert not overlap, f"fold {k+1} leaks years {sorted(overlap)}"
    print("fold structure verified: no year appears in both partitions\n")

    cv = {}
    cv["Climatology"]  = climatology_cv(df, years)
    cv["Ridge"]        = tabular_cv(lambda: Ridge(alpha=1.0), X_tab, y, years)
    cv["DecisionTree"] = tabular_cv(lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
                                    X_tab, y, years)
    cv["RandomForest"] = tabular_cv(lambda: RandomForestRegressor(500, min_samples_leaf=2,
                                                                  random_state=0, n_jobs=-1),
                                    X_tab, y, years)
    cv["XGBoost"]      = tabular_cv(lambda: xgb.XGBRegressor(n_estimators=600, learning_rate=0.03,
                                                             max_depth=4, subsample=0.8,
                                                             colsample_bytree=0.8, random_state=0),
                                    X_tab, y, years)

    for name, build_fn in [("LSTM", M.lstm), ("CNN", M.cnn),
                           ("Transformer", M.transformer),
                           ("AgroTabXFormer", M.agrotabxformer)]:
        print(f"fitting {name} ...")
        cv[name] = sequence_cv(build_fn, X_tab, seq, y, years)

    table = pd.DataFrame([summarise(c, n) for n, c in cv.items()])
    table.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
