# AgroTabXFormer

Code accompanying *Agroclimatic Forecasting of Rainfall and Temperature for Optimal Crop Sowing and Harvest Prediction using Artificial Intelligence*.

Predicts rice harvest day (days after sowing) from agroclimatic information available at the time of sowing, across six agroclimatic zones of Tamil Nadu, India, using NASA POWER data for 1981–2025.

## Overview

Harvest day is defined as the first day on which accumulated growing degree days (AGDD) reach the thermal requirement of the reference cultivar (CO 47, 1900 °C-days above a 10 °C base). Each growing season contributes one observation, giving 1,866 sowing events across six districts and seven sowing windows.

**AgroTabXFormer** combines two inputs: tabular predictors known at sowing (district, season, sowing date, soil moisture, temperature, humidity, pre-sowing rainfall) and a 170-day growing-season sequence of daily temperature and GDD. Attention pooling over the sequence allows the model to weight the phenologically relevant portion of the season. The baseline `transformer` model is identical except that it uses mean pooling, isolating the contribution of attention pooling.

## Repository structure

```
src/
  config.py          constants: districts, sowing windows, AGDD threshold, hyperparameters
  build_dataset.py   raw NASA POWER -> cleaned daily records -> one row per sowing event
  models.py          AgroTabXFormer, Transformer, LSTM, CNN, TabTransformer
  evaluate.py        year-blocked rolling-origin cross-validation for all models
  statistics.py      nonlinearity tests, ANOVA, Diebold-Mariano, bootstrap, error propagation
```

## Data

Daily agroclimatic data are obtained from the NASA POWER Data Access Viewer
(https://power.larc.nasa.gov/data-access-viewer) for the six district centroids,
covering 1981–2025. Variables used: `T2M_MAX`, `T2M_MIN`, `RH2M`, `GWETTOP`, `PRECTOTCORR`.

Place the raw exports in `data/raw/` using the filenames listed in `config.py`.

## Usage

```bash
export AGRO_DATA_DIR=/path/to/data     # defaults to ./data

python -m src.build_dataset            # build the modelling dataset
python -m src.evaluate                 # cross-validated model comparison
python -m src.statistics               # statistical analyses
```

## Evaluation protocol

Models are evaluated by five-fold rolling-origin (expanding-window) cross-validation.
Folds are blocked by sowing year, so no year appears in both the training and test
partition of any fold; this prevents leakage between sowing events of the same
season, which share overlapping weather. Feature scalers and target normalisation
are fitted on training partitions only.

Accumulated GDD and harvest day define the prediction target and are never used
as model inputs; every predictor is restricted to information available on the
day of sowing.

Pairwise differences in accuracy are assessed with the Diebold–Mariano test on
pooled out-of-fold predictions, with 95% bootstrap confidence intervals.

## Requirements

```
python >= 3.9
numpy, pandas, scipy
scikit-learn
xgboost
statsmodels >= 0.14
tensorflow >= 2.10          # Keras 3 recommended
matplotlib
shap                        # optional, for feature attribution
```

`models.py` is written for Keras 3 (`keras.ops`). Under Keras 2, replace
`ops.softmax`, `ops.matmul` and `ops.sum` with `tf.nn.softmax`, `tf.matmul`
and `tf.reduce_sum`.

## Citation

Citation details will be added on publication.

## License

MIT
