"""
Configuration for the AgroTabXFormer rice harvest-day prediction pipeline.

Edit DATA_DIR to point to the directory containing the raw NASA POWER CSV files.
All other paths are derived from it.
"""
import os

# ---------------------------------------------------------------- paths
DATA_DIR  = os.environ.get("AGRO_DATA_DIR", "./data")
RAW_DIR   = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
OUT_DIR   = os.path.join(DATA_DIR, "results")

for d in (CLEAN_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- districts
# One district per agroclimatic zone of Tamil Nadu.
DISTRICT_FILES = {
    "Coimbatore":  "Coimbatore_Raw.csv",    # Western Zone
    "Dharmapuri":  "Dharmapuri_Raw.csv",    # North Western Zone
    "Sivagangai":  "Sivagangai_Raw.csv",    # Southern Zone
    "Vellore":     "Vellore_Raw.csv",       # North Eastern Zone
    "Thanjavur":   "Thanjavur_Raw.csv",     # Cauvery Delta Zone
    "Kanyakumari": "Kanyakumari_Raw.csv",   # High Rainfall Zone
}
DISTRICTS = list(DISTRICT_FILES)

# ---------------------------------------------------------------- agronomy
T_BASE   = 10.0    # base temperature for GDD (degrees C)
AGDD_REQ = 1900.0  # accumulated GDD at maturity (degrees C-days), cultivar CO 47
HORIZON  = 170     # max days tracked after sowing
MIN_DAYS = 60      # minimum plausible crop duration

# Recognised rice sowing windows in Tamil Nadu: name -> (month, day)
SEASONS = {
    "Navarai":      (1, 15),
    "Sornavari":    (3, 15),
    "Kar":          (5, 15),
    "Kuruvai":      (6, 1),
    "Samba":        (8, 15),
    "Thaladi":      (10, 15),
    "Late_Thaladi": (11, 15),
}

# ---------------------------------------------------------------- variables
# NASA POWER column names
POWER_COLS = ["T2M_MAX", "T2M_MIN", "RH2M", "GWETTOP", "PRECTOTCORR"]

# Tabular predictors, all knowable on the day of sowing
FEATURES = [
    "Sowing_DOY", "SM_sow", "RH_sow", "Tmax_sow", "Tmin_sow",
    "Rain_30d", "Tavg_30d", "GDD_30d", "SM_30d", "RH_30d",
]
CATEGORICALS = ["District", "Season"]

# ---------------------------------------------------------------- model
BEST_PARAMS = {"d": 96, "blocks": 2, "heads": 8, "drop": 0.1, "lr": 3e-4, "bs": 32}
N_SPLITS = 5
SEEDS    = (0, 1, 2)
