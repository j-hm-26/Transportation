"""
freight_pipeline_v2.py
======================
Freight Rail Carload Forecasting Pipeline — v2
================================================
Targets  : originated, received, total (each modelled separately)
Models   : GAM (entity-level penalised splines), SARIMALite, FixedEffectsOLS
Holdout  : last 26 weeks reserved for final test (no leakage into CV)
CV       : 5×2 Repeated Time-Series Cross-Validation (on train window only)
New      : GAM variable-importance panel (permutation + component decomposition)
           Entity splines: separate smooth f(t) per company and per commodity

Author   : Extended pipeline v2
"""

# ══════════════════════════════════════════════════════════════════
# SECTION 1 – IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")
import os, sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Aesthetic constants ─────────────────────────────────────────
PALETTE = {
    "GAM":    "#2DBD6E",
    "SARIMA": "#E07B39",
    "OLS-FE": "#9B59B6",
}
TARGET_COLORS = {
    "originated": "#3B82C4",
    "received":   "#E07B39",
    "total":      "#2DBD6E",
}
BG_COLOR   = "#F8F9FA"
GRID_COLOR = "#E2E8F0"
ACCENT     = "#6C63FF"

OUTPUT_DIR  = "Visuals/freight_pipeline_v2"
DATA_PATH   = "Data/Weekly_Cargo_Data_2017_2026.csv"

N_REPEATS   = 5
K_FOLDS     = 2
FORECAST_H  = 13   # hold-out / forecast horizon (weeks)
SEASONAL_S  = 52
TARGETS     = ["originated", "received", "total"]
GAM_BIAS = {
    "originated": 30_000,
    "received":   15_000,
    "total":       0,      # total = originated + received, so no additional bias
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 – DATA I/O
# ══════════════════════════════════════════════════════════════════

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.columns = (df.columns.str.strip().str.lower()
                    .str.replace(" ", "_", regex=False))
    df["week"] = pd.to_datetime(df["week"])

    # Drop aggregate rows (TOTAL CARLOADS / TOTAL INTERMODAL)
    mask_total = df["commodity_group_code"].str.upper().str.contains("TOTAL", na=False)
    df = df[~mask_total].copy()

    # Drop exact duplicate rows (data artefact in 2025 release)
    df = df.drop_duplicates()

    df = df.rename(columns={
        "commodity_group_code": "code",
        "commodity_group":      "commodity_group",
    })
    df["company"] = df["company"].astype("category")
    df["code"]    = df["code"].astype("category")
    return df


def build_aggregate_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per week; columns: originated, received, total."""
    ts = (
        df.groupby("week")[["originated", "received"]]
        .sum()
        .assign(total=lambda x: x["originated"] + x["received"])
        .sort_index()
        .reset_index()
    )
    ts["week_num"]     = np.arange(len(ts), dtype=float)
    ts["week_of_year"] = ts["week"].dt.isocalendar().week.astype(int)
    ts["year"]         = ts["week"].dt.year
    return ts


def build_panel_data(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.copy()
    for col in ["originated", "received"]:
        panel[col] = panel[col].fillna(0).clip(lower=0)
    panel["total"] = panel["originated"] + panel["received"]

    for col in TARGETS:
        panel[f"log_{col}"] = np.log1p(panel[col])

    week_map = {w: i for i, w in enumerate(sorted(panel["week"].unique()))}
    panel["week_num"]     = panel["week"].map(week_map).astype(float)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["year"]         = panel["week"].dt.year
    panel["company_fe"]   = panel["company"].cat.codes
    panel["code_fe"]      = panel["code"].cat.codes
    return panel


def split_holdout(ts: pd.DataFrame, panel: pd.DataFrame,
                  horizon: int = FORECAST_H):
    """
    Reserve the *last* `horizon` weeks from ts & panel as a final hold-out.
    Returns (ts_train, ts_test, panel_train, panel_test).
    All CV and model fitting must use only the train portions.
    """
    cutoff_idx   = len(ts) - horizon
    cutoff_date  = ts["week"].iloc[cutoff_idx]

    ts_train = ts.iloc[:cutoff_idx].copy().reset_index(drop=True)
    ts_test  = ts.iloc[cutoff_idx:].copy().reset_index(drop=True)

    panel_train = panel[panel["week"] < cutoff_date].copy()
    panel_test  = panel[panel["week"] >= cutoff_date].copy()

    return ts_train, ts_test, panel_train, panel_test


# ══════════════════════════════════════════════════════════════════
# SECTION 3 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def make_fourier_features(t: np.ndarray, period: float = 52.1775,
                          n_harmonics: int = 10) -> np.ndarray:
    cols = []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def make_trend_features(t: np.ndarray, changepoints: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(t), t]
    for cp in changepoints:
        cols.append(np.maximum(0.0, t - cp))
    return np.column_stack(cols)


def make_calendar_features(df: pd.DataFrame) -> np.ndarray:
    q = df["week"].dt.quarter
    cols = [
        (q == 1).astype(float).values,
        (q == 2).astype(float).values,
        (q == 3).astype(float).values,
        (q == 4).astype(float).values,
        df["week_of_year"].isin([1, 52, 53]).astype(float).values,
    ]
    return np.column_stack(cols)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 – MODEL CLASSES
# ══════════════════════════════════════════════════════════════════

# ── 4.1  GAM with Entity-Level Splines ───────────────────────────
class GAMSpline:
    """
    Generalized Additive Model with entity-level penalised regression splines.

    Structure (aggregate time-series level):
        y(t) = f_trend(t)
             + Σ_c  f_company_c(t)   [one spline per company]
             + Σ_k  f_code_k(t)      [one spline per commodity code]
             + f_season(t)            [Fourier annual seasonality]
             + f_calendar(t)          [quarter + holiday-week indicators]
             + ε

    All smooth terms estimated jointly by Ridge (≈ smoothing penalty).

    For aggregate-level use the entity columns (company/code) can be passed
    as one-hot encoded weights in `entity_X`; for panel-level use they are
    simple indicator × spline interactions.

    Parameters
    ----------
    n_knots_trend   : knots for the global trend spline
    n_knots_entity  : knots for each company / commodity spline
    n_fourier       : Fourier harmonics
    alpha           : Ridge regularisation strength
    """

    def __init__(self, n_knots_trend: int = 15, n_knots_entity: int = 8,
                 n_fourier: int = 12, alpha: float = 1.0, bias: float = 0.0):
        self.n_knots_trend  = n_knots_trend
        self.n_knots_entity = n_knots_entity
        self.n_fourier      = n_fourier
        self.alpha          = alpha
        self.bias          = bias 

        self._spline_trend  = None   # global trend spline transformer
        self._entity_splines = {}    # dict: entity_name → SplineTransformer
        self._scaler        = StandardScaler()
        self._model         = Ridge(alpha=alpha)

        # Column-width bookkeeping for decomposition
        self._w_trend    = 0
        self._w_entity   = {}   # name → (start_col, end_col)
        self._w_fourier  = 0
        self._w_calendar = 5

        self.companies_  = []
        self.codes_      = []

    # ── design-matrix builders ────────────────────────────────────

    def _trend_block(self, t: np.ndarray) -> np.ndarray:
        return self._spline_trend.transform(t.reshape(-1, 1))

    def _entity_block(self, t: np.ndarray,
                      entity_indicators: dict) -> np.ndarray:
        """
        entity_indicators: dict of {entity_name: 1-D boolean/float array of length n}
        Returns horizontally stacked (n, Σ n_knots_entity) matrix.
        Unknown entities (not seen at fit-time) are silently ignored;
        missing entities (seen at fit-time but absent at predict-time) use zeros.
        """
        blocks = []
        n = len(t)
        for name in self._entity_keys:   # iterate over *training* keys in order
            sp        = self._entity_splines[name]
            B         = sp.transform(t.reshape(-1, 1))          # (n, n_knots_entity)
            indicator = entity_indicators.get(name, np.zeros(n)) # zeros if unseen
            blocks.append(B * indicator[:, None])
        return np.hstack(blocks) if blocks else np.zeros((n, 0))

    def _build_X(self, t: np.ndarray, cal_df: pd.DataFrame,
                 entity_indicators: dict) -> np.ndarray:
        Xt = self._trend_block(t)
        Xe = self._entity_block(t, entity_indicators)
        Xs = make_fourier_features(t, n_harmonics=self.n_fourier)
        Xc = make_calendar_features(cal_df)
        return np.hstack([Xt, Xe, Xs, Xc])

    # ── fit / predict ──────────────────────────────────────────────

    def fit(self, t: np.ndarray, y: np.ndarray,
            cal_df: pd.DataFrame, entity_indicators: dict):
        """
        Parameters
        ----------
        t                 : (n,) week indices
        y                 : (n,) target values
        cal_df            : DataFrame with week / week_of_year columns (len n)
        entity_indicators : {name: (n,) array} — 1 for that entity, else 0
                            Typically one-hot dummies for company & code.
        """
        # Global trend spline
        self._spline_trend = SplineTransformer(
            n_knots=self.n_knots_trend, degree=3,
            knots="quantile", include_bias=True,
        )
        self._spline_trend.fit(t.reshape(-1, 1))
        self._w_trend = self._spline_trend.transform(t.reshape(-1, 1)).shape[1]

        # Entity splines (one per entity, shared knot structure)
        col_cursor = self._w_trend
        for name in entity_indicators:
            sp = SplineTransformer(
                n_knots=self.n_knots_entity, degree=3,
                knots="quantile", include_bias=False,
            )
            sp.fit(t.reshape(-1, 1))
            self._entity_splines[name] = sp
            w = sp.transform(t.reshape(-1, 1)).shape[1]
            self._w_entity[name] = (col_cursor, col_cursor + w)
            col_cursor += w

        self._w_fourier = 2 * self.n_fourier

        # Store entity keys BEFORE building X (needed by _entity_block)
        self._entity_keys = list(entity_indicators.keys())

        # Build & scale design matrix
        X  = self._build_X(t, cal_df, entity_indicators)
        Xs = self._scaler.fit_transform(X)
        self._model.fit(Xs, y)
        return self

    def predict(self, t: np.ndarray, cal_df: pd.DataFrame,
                entity_indicators: dict) -> np.ndarray:
        X  = self._build_X(t, cal_df, entity_indicators)
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs) + self.bias

    def predict_components(self, t: np.ndarray, cal_df: pd.DataFrame,
                           entity_indicators: dict) -> dict:
        """
        Returns dict with keys:
          'trend', 'entity_<name>' (one per entity), 'seasonality', 'calendar'
        Each value is a (n,) array.
        """
        X  = self._build_X(t, cal_df, entity_indicators)
        Xs = self._scaler.transform(X)
        c  = self._model.coef_
        ic = self._model.intercept_

        comps = {}

        # Trend
        e0 = self._w_trend
        comps["trend"] = Xs[:, :e0] @ c[:e0] + ic + self.bias

        # Entity splines
        for name, (s, e) in self._w_entity.items():
            comps[f"entity_{name}"] = Xs[:, s:e] @ c[s:e]

        # Seasonality
        s_start = e0 + sum(e - s for s, e in self._w_entity.values())
        s_end   = s_start + self._w_fourier
        comps["seasonality"] = Xs[:, s_start:s_end] @ c[s_start:s_end]

        # Calendar
        cal_start = s_end
        comps["calendar"] = Xs[:, cal_start:] @ c[cal_start:]

        return comps

    def permutation_importance(self, t: np.ndarray, y: np.ndarray,
                               cal_df: pd.DataFrame,
                               entity_indicators: dict,
                               n_repeats: int = 8,
                               seed: int = 42) -> pd.DataFrame:
        """
        Permutation-based feature-group importance.
        Shuffles each logical group of columns in the scaled X matrix,
        measures increase in RMSE, then restores.

        Groups: 'trend', each entity name, 'seasonality', 'calendar'
        """
        rng = np.random.default_rng(seed)
        X   = self._build_X(t, cal_df, entity_indicators)
        Xs  = self._scaler.transform(X)
        c   = self._model.coef_

        base_rmse = np.sqrt(mean_squared_error(y, self._model.predict(Xs)))

        # Build column-group map
        col_groups = {}
        col_groups["trend"] = list(range(self._w_trend))
        for name, (s, e) in self._w_entity.items():
            col_groups[name] = list(range(s, e))
        s_start = self._w_trend + sum(e - s for s, e in self._w_entity.values())
        s_end   = s_start + self._w_fourier
        col_groups["seasonality"] = list(range(s_start, s_end))
        col_groups["calendar"]    = list(range(s_end, Xs.shape[1]))

        records = []
        for grp_name, cols in col_groups.items():
            deltas = []
            for _ in range(n_repeats):
                Xp = Xs.copy()
                perm = rng.permutation(len(y))
                Xp[:, cols] = Xp[perm][:, cols]
                rmse_perm = np.sqrt(mean_squared_error(y, self._model.predict(Xp)))
                deltas.append(rmse_perm - base_rmse)
            records.append({
                "group":          grp_name,
                "importance_mean": np.mean(deltas),
                "importance_std":  np.std(deltas),
                "base_rmse":       base_rmse,
            })
        return pd.DataFrame(records).sort_values("importance_mean", ascending=False)


# ── 4.2  SARIMALite ──────────────────────────────────────────────
class SARIMALite:
    """
    SARIMA(p,1,q)x(P,1,Q)[S] via conditional MLE (scipy L-BFGS-B).
    Double-differences the series, estimates via Gaussian log-likelihood,
    forecasts recursively, then inverts differencing.
    """
    def __init__(self, p=2, q=2, P=1, Q=1, S=52):
        self.p, self.q, self.P, self.Q, self.S = p, q, P, Q, S
        self.params_ = None
        self.resid_  = None

    @staticmethod
    def _diff(x, d=1):
        for _ in range(d): x = np.diff(x)
        return x

    def _unpack(self, params):
        p, q, P, Q = self.p, self.q, self.P, self.Q
        return (params[:p], params[p:p+q],
                params[p+q:p+q+P], params[p+q+P:p+q+P+Q])

    def _run_filter(self, params, y):
        phi, theta, Phi, Theta = self._unpack(params)
        S = self.S; n = len(y); eps = np.zeros(n)
        for t in range(n):
            hat  = sum(phi[i]   * y[t-i-1]      for i in range(self.p) if t-i-1 >= 0)
            hat -= sum(theta[j] * eps[t-j-1]     for j in range(self.q) if t-j-1 >= 0)
            hat += sum(Phi[i]   * y[t-(i+1)*S]   for i in range(self.P) if t-(i+1)*S >= 0)
            hat -= sum(Theta[j] * eps[t-(j+1)*S] for j in range(self.Q) if t-(j+1)*S >= 0)
            eps[t] = y[t] - hat
        return eps

    def _neg_loglik(self, params, y):
        sigma2 = np.exp(params[-1])
        eps    = self._run_filter(params, y)
        n      = len(y)
        return 0.5*n*np.log(2*np.pi*sigma2) + 0.5*np.sum(eps**2)/sigma2

    def fit(self, y):
        yd = self._diff(y, d=1)
        if len(yd) > self.S:
            yd = self._diff(yd, d=1)
        n_params = self.p + self.q + self.P + self.Q + 1
        x0 = np.zeros(n_params); x0[-1] = np.log(np.var(yd) + 1e-6)
        bounds = [(-0.99, 0.99)] * (n_params-1) + [(None, None)]
        res = minimize(self._neg_loglik, x0, args=(yd,),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-9})
        self.params_ = res.x; self._yd = yd; self._y = y
        self.resid_  = self._run_filter(self.params_, yd)
        return self

    def predict(self, h=1):
        phi, theta, Phi, Theta = self._unpack(self.params_)
        S = self.S
        yd_ext  = list(self._yd.copy())
        eps_ext = list(self.resid_.copy())
        preds_d = []
        for _ in range(h):
            t   = len(yd_ext)
            hat  = sum(phi[i]   * yd_ext[t-i-1]      for i in range(self.p) if t-i-1 >= 0)
            hat -= sum(theta[j] * eps_ext[t-j-1]      for j in range(self.q) if t-j-1 >= 0)
            hat += sum(Phi[i]   * yd_ext[t-(i+1)*S]   for i in range(self.P) if t-(i+1)*S >= 0)
            hat -= sum(Theta[j] * eps_ext[t-(j+1)*S]  for j in range(self.Q) if t-(j+1)*S >= 0)
            preds_d.append(hat); yd_ext.append(hat); eps_ext.append(0.0)
        recovered = np.r_[self._y[-2:],
                          np.cumsum(np.r_[self._y[-1], preds_d])[1:]]
        result = np.cumsum(np.r_[self._y[-1], np.diff(recovered)])
        return result[1:h+1]


# ── 4.3  Fixed-Effects OLS (panel) ───────────────────────────────
class FixedEffectsOLS:
    """
    Two-way Fixed-Effects OLS (company + commodity dummies) on log carloads.
    Includes linear trend, Fourier seasonality, and calendar effects.
    Supports multiple targets (originated / received / total).
    """
    def __init__(self, n_fourier=6, fit_trend=True, target="total"):
        self.n_fourier  = n_fourier
        self.fit_trend  = fit_trend
        self.target     = target
        self.model_     = LinearRegression(fit_intercept=True)
        self.scaler_    = StandardScaler()
        self.companies_ = None
        self.codes_     = None
        self.coef_df_   = None

    def _build_dummies(self, panel):
        companies = self.companies_; codes = self.codes_
        comp_dummies = np.zeros((len(panel), len(companies) - 1))
        code_dummies = np.zeros((len(panel), len(codes) - 1))
        for j, comp in enumerate(companies[1:]):
            comp_dummies[:, j] = (panel["company"] == comp).astype(float).values
        for j, code in enumerate(codes[1:]):
            code_dummies[:, j] = (panel["code"] == code).astype(float).values
        return np.hstack([comp_dummies, code_dummies])

    def _build_X(self, panel):
        t      = panel["week_num"].values
        X_seas = make_fourier_features(t, n_harmonics=self.n_fourier)
        X_fe   = self._build_dummies(panel)
        X_cal  = make_calendar_features(panel)
        blocks = [X_fe, X_seas, X_cal]
        if self.fit_trend:
            blocks = [t.reshape(-1, 1)] + blocks
        return np.hstack(blocks)

    def fit(self, panel):
        self.companies_ = list(panel["company"].cat.categories)
        self.codes_      = list(panel["code"].cat.categories)
        X  = self._build_X(panel)
        Xs = self.scaler_.fit_transform(X)
        y  = panel[f"log_{self.target}"].values
        self.model_.fit(Xs, y)
        self.coef_df_ = pd.DataFrame({
            "feature": self._make_col_names(),
            "coef":    self.model_.coef_,
        }).assign(abs_coef=lambda d: d["coef"].abs())
        return self

    def _make_col_names(self):
        names = []
        if self.fit_trend: names.append("time_trend")
        names += [f"company_{c}" for c in self.companies_[1:]]
        names += [f"code_{c}"    for c in self.codes_[1:]]
        for k in range(1, self.n_fourier + 1):
            names += [f"sin_{k}", f"cos_{k}"]
        names += ["q1", "q2", "q3", "q4", "holiday_week"]
        return names

    def predict(self, panel):
        X  = self._build_X(panel)
        Xs = self.scaler_.transform(X)
        return np.clip(np.expm1(self.model_.predict(Xs)), 0, None)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 – ENTITY INDICATORS (for GAM aggregate use)
# ══════════════════════════════════════════════════════════════════

def build_entity_indicators_agg(ts_df: pd.DataFrame,
                                 raw_df: pd.DataFrame) -> dict:
    """
    For aggregate-level GAM: build one time-series indicator per company
    and per commodity code. Each indicator is the fraction of total
    carloads from that entity in each week (soft weighting).

    Returns dict {entity_name: (n_weeks,) float array}
    """
    weeks = ts_df["week"].values
    indicators = {}

    # Company weights (fraction of weekly total)
    comp_wk = (
        raw_df.groupby(["week", "company"])["total"]
        .sum()
        .unstack("company")
        .reindex(weeks)
        .fillna(0)
    )
    comp_wk = comp_wk.div(comp_wk.sum(axis=1).replace(0, 1), axis=0)
    for col in comp_wk.columns:
        indicators[f"company_{col}"] = comp_wk[col].values

    # Commodity code weights
    code_wk = (
        raw_df.groupby(["week", "code"])["total"]
        .sum()
        .unstack("code")
        .reindex(weeks)
        .fillna(0)
    )
    code_wk = code_wk.div(code_wk.sum(axis=1).replace(0, 1), axis=0)
    for col in code_wk.columns:
        indicators[f"code_{col}"] = code_wk[col].values

    return indicators


# ══════════════════════════════════════════════════════════════════
# SECTION 6 – METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = float(np.mean(np.abs((y_true - y_pred) /
                                (np.abs(y_true) + 1.0)))) * 100.0
    r2   = r2_score(y_true, y_pred)
    total_error = np.sum(np.abs(y_true - y_pred))
    total_null_model_error = np.sum(np.abs(y_true - np.mean(y_true)))
    
    print(f"Total Absolute Error: {total_error:.2f} | "
          f"Total Null Model Error: {total_null_model_error:.2f} | "
          f"Error Ratio: {total_error / total_null_model_error:.4f}")
    return {"model": label, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE": mape, "R2": r2, "Total Error": total_error, 
            "Total Null Model Error": total_null_model_error, 
            "Error Ratio": total_error / total_null_model_error}


def summarise_cv(df_cv):
    return (df_cv.groupby(["target","model"])[["MAE","MSE","RMSE","MAPE","R2", 
                                               "Total Error", "Total Null Model Error", "Error Ratio"]]
            .agg(["mean","std"]).round(2))


# ══════════════════════════════════════════════════════════════════
# SECTION 7 – CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════

def make_time_splits(n, n_repeats=5, k_folds=2,
                     base_fracs=None, test_frac=0.12,
                     jitter_std=0.04, min_train=0.45,
                     max_train=0.85, seed=42):
    if base_fracs is None:
        base_fracs = list(np.linspace(0.55, 0.80, k_folds))
    rng = np.random.default_rng(seed)
    splits = []; fold = 0
    for _ in range(n_repeats):
        jitter = rng.normal(0, jitter_std, size=len(base_fracs))
        for bf, j in zip(base_fracs, jitter):
            frac   = float(np.clip(bf + j, min_train, max_train))
            tr_end = int(n * frac)
            te_end = min(tr_end + int(n * test_frac), n)
            if te_end - tr_end < 5:
                continue
            splits.append({"fold": fold,
                            "train_idx": np.arange(tr_end),
                            "test_idx":  np.arange(tr_end, te_end)})
            fold += 1
    return splits


def run_cv_loop(model_name, splits, fit_fn, predict_fn,
                y_global, target, verbose=True):
    records = []; fold_data = []
    for sp in splits:
        fold   = sp["fold"]
        tr_idx = sp["train_idx"]
        te_idx = sp["test_idx"]
        try:
            fitted = fit_fn(tr_idx)
            y_pred = np.clip(predict_fn(fitted, te_idx), 0, None)
        except Exception as exc:
            if verbose:
                print(f"  [{model_name}/{target}] fold {fold} FAILED: {exc}")
            continue
        y_true = y_global[te_idx]
        m = compute_metrics(y_true, y_pred, label=model_name)
        m["fold"] = fold; m["target"] = target
        records.append(m)
        fold_data.append((te_idx, y_true, y_pred))
        if verbose:
            print(f"  [{model_name}/{target}] fold {fold:2d} | "
                  f"train={len(tr_idx):4d} | test={len(te_idx):3d} | "
                  f"RMSE={m['RMSE']:>10,.0f} | MAPE={m['MAPE']:.2f}%")
    return records, fold_data


# ══════════════════════════════════════════════════════════════════
# SECTION 8 – VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════

def set_plot_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    BG_COLOR,
        "grid.color":        GRID_COLOR,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
    })


def _save(fig, fname):
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {fname}")


# ══════════════════════════════════════════════════════════════════
# SECTION 9 – EDA PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_data_overview(ts, raw):
    """Fig 01 – Data overview EDA."""
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # Top row spanning all 3 cols: three target time-series
    ax_top = fig.add_subplot(gs[0, :])
    for tgt, col in TARGET_COLORS.items():
        ax_top.plot(ts["week"], ts[tgt] / 1e6, color=col,
                    linewidth=1.4, label=tgt.capitalize(), alpha=0.85)
    ax_top.set_title("Weekly Freight Carloads — Originated / Received / Total",
                     fontsize=18, fontweight="bold")
    ax_top.set_ylabel("Carloads (millions)"); ax_top.set_xlabel("Week")
    ax_top.legend(fontsize=14)

    # Bottom-left: total carloads by company (horizontal bar, descending)
    ax2 = fig.add_subplot(gs[1, 0])
    raw_copy = raw.copy()
    raw_copy["originated"] = raw_copy["originated"].fillna(0)
    raw_copy["received"]   = raw_copy["received"].fillna(0)
    raw_copy["_tot"] = raw_copy["originated"] + raw_copy["received"]
    comp_tot = (raw_copy.groupby("company")["_tot"].sum()
                         .sort_values(ascending=True))   # ascending → largest at top of hbar
    ax2.barh(comp_tot.index, comp_tot.values / 1e6,
             color=sns.color_palette("Set2", len(comp_tot)))
    ax2.set_title("Total Carloads by Company", fontsize = 12, fontweight = "bold")
    ax2.set_xlabel("Carloads (millions)", fontsize = 12)

    # Bottom-centre: top-12 commodity groups
    ax3 = fig.add_subplot(gs[1, 1])
    comm_tot = (raw_copy.groupby("commodity_group")["_tot"].sum()
                         .sort_values(ascending=True).tail(12))
    ax3.barh(comm_tot.index, comm_tot.values / 1e6,
             color=sns.color_palette("tab20", len(comm_tot)))
    ax3.set_title("Top-12 Commodity Groups by Volume", fontsize = 12, fontweight = "bold")
    ax3.set_xlabel("Carloads (millions)", fontsize = 12)

    # Bottom-right: Originated vs Received by company — descending total,
    # CPKC bar is CPKC base + CP (striped) + KCS (dotted) stacked on top.
    ax4 = fig.add_subplot(gs[1, 2])

    # Compute originted / received per company (excluding CP & KCS for their
    # own standalone bars — they are subsumed into CPKC stack)
    orig_by_co = raw_copy.groupby("company")["originated"].sum().fillna(0)
    recv_by_co = raw_copy.groupby("company")["received"].sum().fillna(0)
    total_by_co = orig_by_co + recv_by_co

    # Sort companies by descending total (largest → leftmost on a vertical bar chart)
    companies_sorted = total_by_co.sort_values(ascending=False).index.tolist()

    # CPKC pre-merger layer volumes (CP + KCS historical)
    cp_orig   = orig_by_co.get("CP",   0)
    cp_recv   = recv_by_co.get("CP",   0)
    kcs_orig  = orig_by_co.get("KCS",  0)
    kcs_recv  = recv_by_co.get("KCS",  0)
    cpkc_orig = orig_by_co.get("CPKC", 0)
    cpkc_recv = recv_by_co.get("CPKC", 0)

    x = np.arange(len(companies_sorted))
    w = 0.38

    orig_vals = [orig_by_co.get(c, 0) / 1e6 for c in companies_sorted]
    recv_vals = [recv_by_co.get(c, 0) / 1e6 for c in companies_sorted]

    # Base originated / received bars
    ax4.bar(x - w/2, orig_vals, w, label="Originated",
            color=TARGET_COLORS["originated"], alpha=0.85)
    ax4.bar(x + w/2, recv_vals, w, label="Received",
            color=TARGET_COLORS["received"], alpha=0.85)

    # CPKC stacked layers: find its x position
    if "CPKC" in companies_sorted:
        cpkc_idx = companies_sorted.index("CPKC")

        # CP layer (striped hatching) stacked above CPKC originated bar
        ax4.bar(cpkc_idx - w/2, cp_orig / 1e6, w,
                bottom=cpkc_orig / 1e6,
                color=TARGET_COLORS["originated"], alpha=0.60,
                hatch="///", edgecolor="white", linewidth=0.6,
                label="CP (pre-merger, orig.)")
        # KCS layer (dotted hatching) stacked above CP on originated side
        ax4.bar(cpkc_idx - w/2, kcs_orig / 1e6, w,
                bottom=(cpkc_orig + cp_orig) / 1e6,
                color=TARGET_COLORS["originated"], alpha=0.35,
                hatch="...", edgecolor="white", linewidth=0.6,
                label="KCS (pre-merger, orig.)")

        # Same stacking on received side
        ax4.bar(cpkc_idx + w/2, cp_recv / 1e6, w,
                bottom=cpkc_recv / 1e6,
                color=TARGET_COLORS["received"], alpha=0.60,
                hatch="///", edgecolor="white", linewidth=0.6,
                label="CP (pre-merger, recv.)")
        ax4.bar(cpkc_idx + w/2, kcs_recv / 1e6, w,
                bottom=(cpkc_recv + cp_recv) / 1e6,
                color=TARGET_COLORS["received"], alpha=0.35,
                hatch="...", edgecolor="white", linewidth=0.6,
                label="KCS (pre-merger, recv.)")

        # Annotation above the CPKC group
        total_cpkc_stack_orig = (cpkc_orig + cp_orig + kcs_orig) / 1e6
        total_cpkc_stack_recv = (cpkc_recv + cp_recv + kcs_recv) / 1e6
        ax4.annotate("(CP+KCS)",
                     xy=(cpkc_idx, max(total_cpkc_stack_orig,
                                       total_cpkc_stack_recv)),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=10, color="#333333")

    ax4.set_xticks(x)
    ax4.set_xticklabels(companies_sorted, rotation=45, ha="right")
    ax4.set_title("Originated vs Received by Company\n(CPKC bar includes CP+KCS history)",
                  fontsize=12, fontweight = "bold")
    ax4.set_ylabel("Carloads (millions)")
    # Compact legend
    handles, labels = ax4.get_legend_handles_labels()
    # Keep only: Originated, Received, CP orig, KCS orig
    keep = [i for i, l in enumerate(labels)
            if l in ("Originated", "Received",
                     "CP (pre-merger, orig.)", "KCS (pre-merger, orig.)")]
    ax4.legend([handles[i] for i in keep],
               ["Originated", "Received", "+ CP Pre-Merger", "+ KCS Pre-Merger"],
               fontsize=12, ncol=2, loc="upper right")

    # fig.suptitle("Freight Rail Data Overview (2017–2026)",
    #              fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "fig01_data_overview.png")


def plot_seasonality_patterns(ts):
    """Fig 02 – Seasonality & trend for all three targets."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 13))
    fig.suptitle("Seasonality & Trend Patterns — All Targets",
                 fontsize=13, fontweight="bold")

    for row, tgt in enumerate(TARGETS):
        col_c = TARGET_COLORS[tgt]
        # Week-of-year profile
        ax = axes[row, 0]
        woy = ts.groupby("week_of_year")[tgt].agg(["mean","std"])
        ax.fill_between(woy.index,
                        (woy["mean"]-woy["std"])/1e6,
                        (woy["mean"]+woy["std"])/1e6,
                        alpha=0.2, color=col_c)
        ax.plot(woy.index, woy["mean"]/1e6, color=col_c, linewidth=2)
        ax.set_title(f"{tgt.capitalize()} — Weekly Profile (±1 SD)")
        ax.set_xlabel("Week of Year"); ax.set_ylabel("Carloads (M)")

        # Year-over-year overlay
        ax = axes[row, 1]
        years = sorted(ts["year"].unique())
        cmap  = plt.cm.viridis
        for yr, grp in ts.groupby("year"):
            frac = (yr - min(years)) / max(1, max(years) - min(years))
            ax.plot(grp["week_of_year"], grp[tgt]/1e6,
                    alpha=0.75, linewidth=1.1, color=cmap(frac), label=str(yr))
        ax.set_title(f"{tgt.capitalize()} — Year-over-Year")
        ax.set_xlabel("Week of Year"); ax.set_ylabel("Carloads (M)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::max(1,len(handles)//5)],
                  labels[::max(1,len(labels)//5)], fontsize=7)

        # 13-week rolling mean
        ax = axes[row, 2]
        roll = ts[tgt].rolling(13, center=True).mean()
        ax.plot(ts["week"], ts[tgt]/1e6, alpha=0.25, color="#999", linewidth=0.7)
        ax.plot(ts["week"], roll/1e6, color=col_c, linewidth=2.2,
                label="13-wk rolling mean")
        ax.set_title(f"{tgt.capitalize()} — Long-Run Trend")
        ax.set_xlabel("Week"); ax.set_ylabel("Carloads (M)"); ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig02_seasonality.png")


def plot_holdout_overview(ts_train, ts_test):
    """Fig 03 – Visualise train / hold-out split for all targets."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(f"Train / Hold-Out Split (last {FORECAST_H} weeks reserved)",
                 fontsize=13, fontweight="bold")
    for ax, tgt in zip(axes, TARGETS):
        col_c = TARGET_COLORS[tgt]
        ax.plot(ts_train["week"], ts_train[tgt]/1e6,
                color=col_c, linewidth=1.3, label="Train", alpha=0.85)
        ax.plot(ts_test["week"], ts_test[tgt]/1e6,
                color="red", linewidth=1.8, linestyle="--",
                label="Hold-out (test)", alpha=0.9)
        ax.axvline(ts_train["week"].iloc[-1], color="black",
                   linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_ylabel("Carloads (M)")
        ax.set_title(f"{tgt.capitalize()}")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig03_holdout_split.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 10 – CV PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_cv_violins(df_cv):
    """Fig 04 – Violin plots of CV metrics, one sub-figure per target."""
    metrics      = ["MAE","MSE","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]
    model_order  = [m for m in ["GAM","SARIMA","OLS-FE"]
                    if m in df_cv["model"].unique()]
    targets_here = [t for t in TARGETS if t in df_cv["target"].unique()]

    fig, axes = plt.subplots(len(targets_here), len(metrics),
                             figsize=(5*len(metrics), 5.5*len(targets_here)))
    if len(targets_here) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("5×2 Repeated K-Fold CV — Metric Distributions by Target",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, tgt in enumerate(targets_here):
        sub = df_cv[df_cv["target"] == tgt]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.violinplot(data=sub, x="model", y=metric, hue="model",
                           order=model_order, palette=PALETTE,
                           inner="box", ax=ax, linewidth=1.2, legend=False)
            sns.stripplot(data=sub, x="model", y=metric, hue="model",
                          order=model_order, palette=PALETTE,
                          size=4, jitter=True, alpha=0.5, ax=ax, legend=False)
            means = sub.groupby("model")[metric].mean()
            for xi, mdl in enumerate(model_order):
                if mdl in means.index:
                    mu = means[mdl]
                    ax.text(xi, mu, f"μ={mu:.2f}",
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE.get(mdl,"grey"), fontweight="bold")
            ax.set_title(f"{tgt.capitalize()} | {metric}", fontsize=9)
            ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, "fig04_cv_violins.png")


def plot_cv_boxplots(df_cv):
    """Fig 05 – Boxplots of CV metrics."""
    #metrics      = ["MAE","MSE","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]
    metrics      = ["Total Error", "Total Null Model Error", "Error Ratio"]
    model_order  = [m for m in ["GAM"]
                    if m in df_cv["model"].unique()]
    targets_here = [t for t in TARGETS if t in df_cv["target"].unique()]

    fig, axes = plt.subplots(len(targets_here), len(metrics),
                             figsize=(5*len(metrics), 5*len(targets_here)))
    if len(targets_here) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("5×2 Repeated K-Fold CV — Metric Boxplots",
                 fontsize=14, fontweight="bold")
    for row, tgt in enumerate(targets_here):
        sub = df_cv[df_cv["target"] == tgt]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.boxplot(data=sub, x="model", y=metric, hue="model",
                        order=model_order, palette=PALETTE,
                        width=0.45, linewidth=1.5, ax=ax, legend=False,
                        flierprops=dict(marker="o", alpha=0.5, markersize=4))
            ax.set_title(f"{tgt.capitalize()} | {metric}", fontsize=9)
            ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, "fig05_cv_boxplots.png")


def plot_metric_heatmap(df_cv):
    """Fig 06 – Heatmap of mean CV metrics (raw + normalised)."""
    metrics = ["MAE","MSE","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]
    targets_here = [t for t in TARGETS if t in df_cv["target"].unique()]

    fig, axes = plt.subplots(len(targets_here), 2,
                             figsize=(16, 4.5*len(targets_here)))
    if len(targets_here) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("CV Metric Summary — Raw Mean & Normalised Ranking",
                 fontsize=13, fontweight="bold")

    order = [m for m in ["GAM","SARIMA","OLS-FE"]
             if m in df_cv["model"].unique()]

    def fmt_annot(df):
        out = []
        for idx in df.index:
            row = []
            for col in df.columns:
                v = df.loc[idx, col]
                if col in ("MAE","MSE","RMSE","Total Error", "Total Null Model Error", "Error Ratio"): row.append(f"{v:,.0f}")
                elif col == "MAPE":              row.append(f"{v:.2f}%")
                else:                            row.append(f"{v:.3f}")
            out.append(row)
        return out

    for row, tgt in enumerate(targets_here):
        sub     = df_cv[df_cv["target"] == tgt]
        summary = sub.groupby("model")[metrics].mean().round(2)
        summary = summary.reindex([m for m in order if m in summary.index])
        norm    = summary.copy()
        for col in [c for c in metrics if c != "R2"]:
            rng = norm[col].max() - norm[col].min() + 1e-12
            norm[col] = (norm[col] - norm[col].min()) / rng
        if "R2" in norm.columns:
            rng = norm["R2"].max() - norm["R2"].min() + 1e-12
            norm["R2"] = 1.0 - (norm["R2"] - norm["R2"].min()) / rng

        sns.heatmap(summary, annot=fmt_annot(summary), fmt="",
                    cmap="YlOrRd_r", linewidths=0.5, ax=axes[row,0],
                    cbar_kws={"shrink":0.8})
        axes[row,0].set_title(f"{tgt.capitalize()} — Raw Metric Means (CV)")

        sns.heatmap(norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    linewidths=0.5, ax=axes[row,1], cbar_kws={"shrink":0.8})
        axes[row,1].set_title(f"{tgt.capitalize()} — Normalised (0.00 = best)")

    plt.tight_layout()
    _save(fig, "fig06_metric_heatmap.png")


def plot_fold_trajectory(df_cv):
    """Fig 07 – Stability of RMSE / MAPE / R2 across CV folds."""
    metrics      = ["RMSE","MAPE","R2"]
    model_order  = [m for m in ["GAM","SARIMA","OLS-FE"]
                    if m in df_cv["model"].unique()]
    targets_here = [t for t in TARGETS if t in df_cv["target"].unique()]

    fig, axes = plt.subplots(len(targets_here), len(metrics),
                             figsize=(7*len(metrics), 5*len(targets_here)))
    if len(targets_here) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Performance Stability Across CV Folds",
                 fontsize=14, fontweight="bold")

    for row, tgt in enumerate(targets_here):
        sub = df_cv[df_cv["target"] == tgt]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for mdl in model_order:
                s = sub[sub["model"] == mdl].sort_values("fold")
                ax.plot(s["fold"], s[metric], marker="o", label=mdl,
                        color=PALETTE.get(mdl,"grey"), linewidth=2, markersize=6)
            ax.set_xlabel("Fold"); ax.set_ylabel(metric)
            ax.set_title(f"{tgt.capitalize()} — {metric}")
            ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "fig07_fold_trajectory.png")


def plot_actual_vs_predicted(fold_data_all):
    """
    Fig 08 – Actual vs Predicted scatter for last CV fold.
    fold_data_all: {(model, target): list of (te_idx, y_true, y_pred)}
    """
    models       = ["GAM","SARIMA","OLS-FE"]
    targets_here = TARGETS
    n_tgt = len(targets_here)
    n_mdl = len(models)

    fig, axes = plt.subplots(n_tgt, n_mdl, figsize=(6*n_mdl, 5.5*n_tgt))
    fig.suptitle("Actual vs Predicted — Last CV Fold",
                 fontsize=13, fontweight="bold")

    for row, tgt in enumerate(targets_here):
        for col, mdl in enumerate(models):
            ax = axes[row, col]
            key = (mdl, tgt)
            if key not in fold_data_all or not fold_data_all[key]:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            _, y_te, y_pr = fold_data_all[key][-1]
            yt = y_te / 1e6; yp = y_pr / 1e6
            ax.scatter(yt, yp, color=PALETTE.get(mdl, ACCENT),
                       alpha=0.6, s=55, edgecolor="white", linewidth=0.4)
            mn = min(yt.min(), yp.min()); mx = max(yt.max(), yp.max())
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
            m = compute_metrics(y_te, y_pr, mdl)
            ax.set_title(f"{mdl} | {tgt.capitalize()}\n"
                         f"RMSE={m['RMSE']/1e3:.0f}K  R²={m['R2']:.3f}  "
                         f"MAPE={m['MAPE']:.1f}%", fontsize=8)
            ax.set_xlabel("Actual (M)"); ax.set_ylabel("Predicted (M)")
    plt.tight_layout()
    _save(fig, "fig08_actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 11 – FIT & FORECAST PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_fit_and_forecast(ts_train, ts_test, fits, fcast_dates, fcasts):
    """
    Fig 09 – In-sample fit + hold-out forecast for all targets × models.
    fits  : {(model, target): np.ndarray in-sample fitted values (train length)}
    fcasts: {(model, target): np.ndarray hold-out forecast (test length)}
    """
    # models = ["GAM","SARIMA","OLS-FE"]
    # styles = {"GAM": ("-", 2.0), "SARIMA": ("-.", 1.7), "OLS-FE": (":", 1.7)}

    models = ["GAM"]
    styles = {"GAM": ("-", 2.0)}

    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=False)
    fig.suptitle("In-Sample Fit + 26-Week Hold-Out Forecast — All Targets",
                 fontsize=16, fontweight="bold")

    for ax, tgt in zip(axes, TARGETS):
        col_c = TARGET_COLORS[tgt]
        # Actual (train)
        ax.plot(ts_train["week"], ts_train[tgt]/1e6,
                color="#333333", linewidth=1.0, label="Actual (train)",
                alpha=0.8, zorder=5)
        # Actual (test/hold-out)
        ax.plot(ts_test["week"], ts_test[tgt]/1e6,
                color="red", linewidth=2.0, linestyle="--",
                label="Actual (hold-out)", alpha=0.9, zorder=6)
        # Model fits + forecasts
        for mdl in models:
            ls, lw = styles.get(mdl, ("-", 1.5))
            key = (mdl, tgt)
            if key in fits:
                ax.plot(ts_train["week"], fits[key]/1e6,
                        color=PALETTE.get(mdl, ACCENT), linewidth=lw,
                        linestyle=ls, alpha=0.75, label=f"{mdl} (fit)")
            if key in fcasts:
                ax.plot(ts_test["week"], fcasts[key]/1e6,
                        color=PALETTE.get(mdl, ACCENT), linewidth=2.2,
                        marker="o", markersize=3, linestyle=ls,
                        label=f"{mdl} (forecast)", alpha=0.9)
        ax.axvline(ts_train["week"].iloc[-1], color="black",
                   linestyle=":", linewidth=1.2, alpha=0.5)
        ax.axhline(ts_train[tgt].mean()/1e6, color="#999999", linestyle="--", linewidth=0.8, label ="Training Data Mean")
        ax.set_title(tgt.capitalize(), fontsize=14, fontweight="bold")
        ax.set_ylabel("Carloads (M)")
        ax.legend(fontsize=12, ncol=3, loc="upper left")

    axes[-1].set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig09_fit_and_forecast.png")


def plot_forecast_comparison(ts_train, ts_test, fcasts, history_weeks=78):
    """Fig 10 – Zoomed forecast comparison with ±7% PI band."""
    # models = ["GAM","SARIMA","OLS-FE"]
    # markers = {"GAM":"o","SARIMA":"^","OLS-FE":"D"}

    models = ["GAM"]
    markers = {"GAM": "o"}

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)
    fig.suptitle(f"{FORECAST_H}-Week Hold-Out Forecast — All Targets (±7% PI)",
                 fontsize=16, fontweight="bold")

    for ax, tgt in zip(axes, TARGETS):
        # Recent actual
        ax.plot(ts_train["week"].iloc[-history_weeks:],
                ts_train[tgt].iloc[-history_weeks:]/1e6,
                color="#333", linewidth=1.8, label="Actual (train)", zorder=5)
        ax.plot(ts_test["week"], ts_test[tgt]/1e6,
                color="red", linewidth=2.0, linestyle="--",
                label="Actual (hold-out)", zorder=6)
        for mdl in models:
            key   = (mdl, tgt)
            color = PALETTE.get(mdl, ACCENT)
            mk    = markers.get(mdl, "o")
            if key not in fcasts: continue
            fcast = fcasts[key]
            ax.fill_between(ts_test["week"],
                            fcast/1e6*(1-0.07), fcast/1e6*(1+0.07),
                            alpha=0.12, color=color)
            ax.plot(ts_test["week"], fcast/1e6, color=color, linewidth=2.5,
                    marker=mk, markersize=4, label=f"{mdl} forecast")
        ax.axvline(ts_train["week"].iloc[-1], color="black",
                   linestyle=":", linewidth=1.2, alpha=0.5)
        ax.axhline(ts_train[tgt].mean()/1e6, color="#999999", linestyle="--", linewidth=0.8, label ="Training Data Mean")
        ax.set_title(tgt.capitalize()); ax.set_ylabel("Carloads (M)")
        ax.legend(fontsize=12)
    axes[-1].set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig10_forecast_comparison.png")


def plot_holdout_metrics(ts_test, fcasts):
    """Fig 11 – Hold-out evaluation metrics table and bar chart."""
    records = []
    for (mdl, tgt), fcast in fcasts.items():
        y_true = ts_test[tgt].values
        m = compute_metrics(y_true, fcast, label=mdl)
        m["target"] = tgt
        records.append(m)
    df_test = pd.DataFrame(records)

    metrics = ["RMSE","MAPE","R2"]
    targets_here = [t for t in TARGETS if t in df_test["target"].unique()]
    model_order  = [m for m in ["GAM","SARIMA","OLS-FE"]
                    if m in df_test["model"].unique()]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    fig.suptitle(f"Final Hold-Out Test Metrics (last {FORECAST_H} weeks)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(targets_here)); w = 0.25
    offsets = np.linspace(-(len(model_order)-1)*w/2,
                           (len(model_order)-1)*w/2, len(model_order))

    for ax, metric in zip(axes, metrics):
        for off, mdl in zip(offsets, model_order):
            vals = [df_test[(df_test["model"]==mdl) &
                            (df_test["target"]==tgt)][metric].values
                    for tgt in targets_here]
            vals = [v[0] if len(v) else np.nan for v in vals]
            bars = ax.bar(x + off, vals, w,
                          label=mdl, color=PALETTE.get(mdl, ACCENT), alpha=0.85)
            ax.bar_label(bars,
                         labels=[f"{v:.1f}" if metric in ("MAPE","R2")
                                 else f"{v/1e3:.0f}K" for v in vals],
                         padding=2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in targets_here])
        ax.set_title(metric); ax.legend(fontsize=8)
        if metric == "RMSE": ax.set_ylabel("RMSE (carloads)")
        elif metric == "MAPE": ax.set_ylabel("MAPE (%)")
        else: ax.set_ylabel("R²")

    plt.tight_layout()
    _save(fig, "fig11_holdout_metrics.png")
    return df_test


def plot_decomposition_comparison(ts_train, gam_models, t_train, entity_inds_train):
    """
    Fig 12 – GAM additive decomposition (trend + seasonality) per target.
    gam_models : {target: GAMSpline}
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
    fig.suptitle("GAM Additive Decomposition — Trend & Seasonality by Target",
                 fontsize=13, fontweight="bold")

    for row, tgt in enumerate(TARGETS):
        if tgt not in gam_models: continue
        gam    = gam_models[tgt]
        comps  = gam.predict_components(t_train, ts_train, entity_inds_train)
        col_c  = TARGET_COLORS[tgt]

        ax = axes[row, 0]
        ax.plot(ts_train["week"], comps["trend"]/1e6, color=col_c, linewidth=2)
        ax.set_title(f"{tgt.capitalize()} — Trend Component")
        ax.set_ylabel("Carloads (M)")

        ax = axes[row, 1]
        ax.plot(ts_train["week"], comps["seasonality"]/1e6, color=col_c,
                linewidth=1.3, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{tgt.capitalize()} — Seasonal Component")

    for ax in axes[-1, :]: ax.set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig12_decomposition.png")


def plot_residuals_panel(ts_train, residuals_dict):
    """
    Fig 13 – Residual diagnostics for all model × target combos.
    residuals_dict: {(model, target): array}
    """
    combos = list(residuals_dict.keys())
    n = len(combos)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5*n))
    fig.suptitle("Residual Diagnostics — All Models & Targets",
                 fontsize=14, fontweight="bold")
    if n == 1: axes = axes.reshape(1, -1)

    for row, (mdl, tgt) in enumerate(combos):
        resid = residuals_dict[(mdl, tgt)]
        color = PALETTE.get(mdl, ACCENT)
        ci    = 1.96 / np.sqrt(len(resid))

        ax = axes[row, 0]
        ax.plot(ts_train["week"], resid/1e3, color=color, linewidth=0.7, alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{mdl}/{tgt.capitalize()} — Residuals")
        ax.set_ylabel("Residual (K)"); ax.set_xlabel("Week")

        ax = axes[row, 1]
        sns.histplot(resid/1e3, bins=35, kde=True, color=color,
                     ax=ax, edgecolor="white")
        ax.set_title(f"{mdl}/{tgt.capitalize()} — Distribution")
        ax.set_xlabel("Residual (K)")

        ax = axes[row, 2]
        stats.probplot(resid, plot=ax)
        ax.set_title(f"{mdl}/{tgt.capitalize()} — Q-Q")
        ax.get_lines()[1].set_color("red")

        ax = axes[row, 3]
        max_lag = min(52, len(resid) // 2 - 1)
        acf_vals = np.array([1.0] + [
            float(np.corrcoef(resid[lag:], resid[:-lag])[0, 1])
            for lag in range(1, max_lag + 1)
        ])
        ax.bar(np.arange(max_lag + 1), acf_vals, color=color, alpha=0.7)
        ax.axhline( ci, color="red", linestyle="--", linewidth=1)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{mdl}/{tgt.capitalize()} — ACF")
        ax.set_xlabel("Lag (weeks)")

    plt.tight_layout()
    _save(fig, "fig13_residuals.png")


def plot_fe_coefficients(fe_models, top_n=20):
    """Fig 14 – OLS-FE top coefficients per target."""
    fig, axes = plt.subplots(1, len(TARGETS),
                             figsize=(11*len(TARGETS), max(5, top_n//2)))
    fig.suptitle(f"Top-{top_n} OLS Fixed-Effects Coefficients by Target",
                 fontsize=12, fontweight="bold")
    if len(TARGETS) == 1: axes = [axes]

    def _cat(feat):
        if feat.startswith("company_"): return "Company FE"
        if feat.startswith("code_"):    return "Commodity FE"
        if feat.startswith(("sin_","cos_")): return "Fourier"
        if feat.startswith("q") or feat == "holiday_week": return "Calendar"
        return "Trend"

    cat_pal = {"Company FE": PALETTE["SARIMA"], "Commodity FE": PALETTE["GAM"],
               "Fourier": ACCENT, "Calendar": PALETTE["OLS-FE"], "Trend": "#555"}

    for ax, tgt in zip(axes, TARGETS):
        if tgt not in fe_models: continue
        coef = fe_models[tgt].coef_df_.copy()
        top  = coef.nlargest(top_n, "abs_coef").copy()
        top["category"] = top["feature"].apply(_cat)
        sns.barplot(data=top, y="feature", x="coef",
                    hue="category", palette=cat_pal,
                    ax=ax, orient="h", legend=(tgt == TARGETS[0]))
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{tgt.capitalize()}"); ax.set_xlabel("Std Coef")
        ax.set_ylabel("")

    plt.tight_layout()
    _save(fig, "fig14_fe_coefficients.png")


def plot_gam_spline_basis(gam_models, t_train, ts_train, entity_inds_train,
                          top_n=6):
    """Fig 15 – GAM spline basis visualisation per target."""
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(18, 8*len(TARGETS)),
                             sharex="col")
    fig.suptitle("GAM — Spline Basis Functions & Trend Component",
                 fontsize=13, fontweight="bold")
    if len(TARGETS) == 1: axes = axes.reshape(1, -1)

    for row, tgt in enumerate(TARGETS):
        if tgt not in gam_models: continue
        gam   = gam_models[tgt]
        col_c = TARGET_COLORS[tgt]
        t2d   = t_train.reshape(-1, 1)
        B     = gam._spline_trend.transform(t2d)
        n_ent_cols = sum(e - s for s, e in gam._w_entity.values())
        pad   = np.zeros((len(t_train), n_ent_cols + gam._w_fourier + 5))
        Xs    = gam._scaler.transform(np.hstack([B, pad]))[:, :B.shape[1]]
        coef  = gam._model.coef_[:B.shape[1]]
        weighted = Bs = Xs * coef
        variances = np.var(weighted, axis=0)
        top_cols  = np.argsort(variances)[::-1][:top_n]

        ax = axes[row, 0]
        for ci in top_cols:
            ax.plot(ts_train["week"], weighted[:, ci], alpha=0.65, linewidth=1.1)
        ax.set_ylabel("Weighted Contribution")
        ax.set_title(f"{tgt.capitalize()} — Top-{top_n} Spline Bases (by Variance)")

        ax = axes[row, 1]
        comps = gam.predict_components(t_train, ts_train, entity_inds_train)
        ax.plot(ts_train["week"], comps["trend"]/1e6, color=col_c,
                linewidth=2.2, label="Trend")
        ax.set_ylabel("Carloads (M)")
        ax.set_title(f"{tgt.capitalize()} — Reconstructed Trend")
        ax.legend()

    for ax in axes[-1, :]: ax.set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig15_gam_spline_basis.png")


def plot_gam_variable_importance(gam_models, t_train, ts_train,
                                 entity_inds_train, y_dict):
    """
    Fig 16 – GAM Variable Importance (permutation-based) — NEW.

    Three panels, one per target:
      Left  : bar chart of permutation importance by group (ΔRMSE on shuffle)
      Right : component contribution (SD of fitted component values) as
              a proportional bar — shows which groups explain most variance
    """
    target = ["total"]

    fig, axes = plt.subplots(len(target), 2,
                             figsize=(18, 6 * len(target)))
    # fig.suptitle("GAM Variable Importance — Permutation & Component Decomposition",
    #              fontsize=18, fontweight="bold")
    if len(target) == 1: axes = axes.reshape(1, -1)

    for row, tgt in enumerate(target):
        if tgt not in gam_models: continue
        gam   = gam_models[tgt]
        y     = y_dict[tgt]
        col_c = TARGET_COLORS[tgt]

        # ── Permutation importance ───────────────────────────────
        imp_df = gam.permutation_importance(
            t_train, y, ts_train, entity_inds_train, n_repeats=10
        )

        # Collapse individual entity groups into two aggregate groups
        records_agg = []
        for _, r in imp_df.iterrows():
            grp = r["group"]
            if grp.startswith("company_"):  agg = "Company Splines"
            elif grp.startswith("code_"):   agg = "Commodity Splines"
            elif grp == "trend":            agg = "Trend Spline"
            elif grp == "seasonality":      agg = "Fourier Seasonality"
            else:                           agg = "Calendar"
            records_agg.append({"group": agg,
                                 "imp":  r["importance_mean"],
                                 "std":  r["importance_std"]})
        agg_df = (pd.DataFrame(records_agg)
                    .groupby("group")
                    .agg(imp=("imp","sum"), std=("std","mean"))
                    .reset_index()
                    .sort_values("imp", ascending=True))

        # ax = axes[row, 0]
        # colors = [PALETTE["GAM"] if v >= 0 else "#cccccc"
        #           for v in agg_df["imp"].values]
        # bars = ax.barh(agg_df["group"], agg_df["imp"], color=colors,
        #                xerr=agg_df["std"], capsize=4, alpha=0.85,
        #                error_kw=dict(elinewidth=1.2, ecolor="black"))
        # ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        # ax.set_xlabel("Mean ΔRMSE on permutation\n(higher = more important)")
        # ax.set_title(f"{tgt.capitalize()} — Permutation Importance by Group")
        # for bar, val in zip(bars, agg_df["imp"]):
        #     ax.text(max(val, 0) + agg_df["imp"].abs().max() * 0.01,
        #             bar.get_y() + bar.get_height() / 2,
        #             f"{val:,.0f}", va="center", fontsize=16, fontweight="bold",)

        # ── Component variance decomposition ────────────────────
        comps = gam.predict_components(t_train, ts_train, entity_inds_train)

        # Aggregate components into the same 5 groups
        comp_var = {}
        comp_var["Trend Spline"]      = np.std(comps.get("trend", np.zeros(len(t_train))))
        comp_var["Fourier Seasonality"] = np.std(comps.get("seasonality", np.zeros(len(t_train))))
        comp_var["Calendar"]          = np.std(comps.get("calendar", np.zeros(len(t_train))))

        comp_arr_comp = np.zeros(len(t_train))
        comp_arr_co   = np.zeros(len(t_train))
        for k, v in comps.items():
            if k.startswith("entity_company_"): comp_arr_co  += v
            elif k.startswith("entity_code_"):  comp_arr_comp += v
        comp_var["Company Splines"]   = np.std(comp_arr_co)
        comp_var["Commodity Splines"] = np.std(comp_arr_comp)

        total_var = sum(comp_var.values()) + 1e-12
        pct = {k: v / total_var * 100 for k, v in comp_var.items()}

        ax = axes[row, 1]
        group_colors = {
            "Trend Spline":       PALETTE["GAM"],
            "Company Splines":    PALETTE["SARIMA"],
            "Commodity Splines":  PALETTE["OLS-FE"],
            "Fourier Seasonality": ACCENT,
            "Calendar":           "#aaaaaa",
        }
        sorted_pct = dict(sorted(pct.items(), key=lambda x: -x[1]))
        wedges, texts, autotexts = ax.pie(
            list(sorted_pct.values()),
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            colors=[group_colors.get(k, ACCENT) for k in sorted_pct],
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        )
        for at in autotexts: at.set_fontsize(14); at.set_fontweight("bold")
        ax.legend(list(sorted_pct.keys()), loc="lower right", bbox_to_anchor=(2.25, 0.1),
                  fontsize=18, title="Component", title_fontsize=18)
        ax.set_title(f"{tgt.capitalize()} — Component Variance Share (SD-based)", fontsize=16, fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig16_gam_variable_importance.png")


def plot_gam_entity_effects(gam_models, t_train, ts_train, entity_inds_train):
    """
    Fig 17 – GAM entity-level smooth effects (company & commodity).
    Shows the fitted f_company(t) and f_code(t) contributions over time.
    """
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(20, 7 * len(TARGETS)),
                             sharex="col")
    fig.suptitle("GAM Entity-Level Smooth Effects — Company & Commodity Splines",
                 fontsize=13, fontweight="bold")
    if len(TARGETS) == 1: axes = axes.reshape(1, -1)

    for row, tgt in enumerate(TARGETS):
        if tgt not in gam_models: continue
        gam   = gam_models[tgt]
        comps = gam.predict_components(t_train, ts_train, entity_inds_train)

        # Company effects
        ax = axes[row, 0]
        comp_effects = {k: v for k, v in comps.items()
                        if k.startswith("entity_company_")}
        pal = sns.color_palette("Set2", len(comp_effects))
        for (k, v), c in zip(comp_effects.items(), pal):
            label = k.replace("entity_company_", "")
            ax.plot(ts_train["week"], v/1e3, linewidth=1.5,
                    label=label, color=c, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{tgt.capitalize()} — Company Smooth Effects f_company(t)")
        ax.set_ylabel("Contribution (K carloads)"); ax.legend(fontsize=8, ncol=2)

        # Commodity code effects
        ax = axes[row, 1]
        code_effects = {k: v for k, v in comps.items()
                        if k.startswith("entity_code_")}
        pal2 = sns.color_palette("tab20", len(code_effects))
        for (k, v), c in zip(code_effects.items(), pal2):
            label = k.replace("entity_code_", "Code ")
            ax.plot(ts_train["week"], v/1e3, linewidth=1.2,
                    label=label, color=c, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{tgt.capitalize()} — Commodity Smooth Effects f_code(t)")
        ax.set_ylabel("Contribution (K carloads)"); ax.legend(fontsize=7, ncol=3)

    for ax in axes[-1, :]: ax.set_xlabel("Week")
    plt.tight_layout()
    _save(fig, "fig17_gam_entity_effects.png")


def plot_hyperparameter_sensitivity(ts_train, y_dict, t_train,
                                    entity_inds_train, splits, target="total"):
    """Fig 18 – GAM hyperparameter grid (alpha × n_knots), CV RMSE surface."""
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
    knots  = [8, 12, 16, 20, 25]
    grid_rmse = np.full((len(alphas), len(knots)), np.nan)
    y_all = y_dict[target]

    print(f"  Computing GAM hyperparameter grid for target='{target}' …")
    for i, alpha in enumerate(alphas):
        for j, nk in enumerate(knots):
            fold_rmses = []
            for sp in splits[:6]:
                tr, te = sp["train_idx"], sp["test_idx"]
                try:
                    ent_tr = {k: v[tr] for k, v in entity_inds_train.items()}
                    ent_te = {k: v[te] for k, v in entity_inds_train.items()}
                    cal_tr = ts_train.iloc[tr].reset_index(drop=True)
                    cal_te = ts_train.iloc[te].reset_index(drop=True)
                    gam = GAMSpline(n_knots_trend=nk, n_fourier=10, alpha=alpha)
                    gam.fit(t_train[tr], y_all[tr], cal_tr, ent_tr)
                    yp = gam.predict(t_train[te], cal_te, ent_te)
                    fold_rmses.append(
                        np.sqrt(mean_squared_error(y_all[te], yp))
                    )
                except Exception:
                    pass
            if fold_rmses:
                grid_rmse[i, j] = np.mean(fold_rmses)
        print(f"    alpha={alpha} done")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid_rmse/1e3, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(knots)));  ax.set_xticklabels(knots)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel("Number of Trend Knots")
    ax.set_ylabel("Ridge α (smoothing)")
    ax.set_title(f"GAM Hyperparameter Sensitivity — CV RMSE (K) [{target}]",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE (K carloads)")
    median = np.nanmedian(grid_rmse)
    for ii in range(len(alphas)):
        for jj in range(len(knots)):
            if not np.isnan(grid_rmse[ii, jj]):
                ax.text(jj, ii, f"{grid_rmse[ii,jj]/1e3:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if grid_rmse[ii,jj] > median else "black")
    plt.tight_layout()
    _save(fig, "fig18_gam_hyperparameter.png")


def plot_scorecard(df_cv):
    """Fig 19 – Radar + RMSE/MAPE bar chart scorecard per target."""
    metrics_plot = ["MAE","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]
    model_order  = [m for m in ["GAM","SARIMA","OLS-FE"]
                    if m in df_cv["model"].unique()]
    targets_here = [t for t in TARGETS if t in df_cv["target"].unique()]

    fig, axes = plt.subplots(len(targets_here), 2,
                             figsize=(16, 7*len(targets_here)))
    fig.suptitle("Model Comparison Scorecard — 5×2 CV",
                 fontsize=14, fontweight="bold")
    if len(targets_here) == 1: axes = axes.reshape(1, -1)

    N      = len(metrics_plot)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for row, tgt in enumerate(targets_here):
        sub     = df_cv[df_cv["target"] == tgt]
        summary = sub.groupby("model")[metrics_plot].mean()
        summary = summary.reindex([m for m in model_order if m in summary.index])
        norm    = summary.copy()
        for col in [c for c in metrics_plot if c != "R2"]:
            rng = norm[col].max() - norm[col].min() + 1e-12
            norm[col] = (norm[col] - norm[col].min()) / rng
        norm["R2"] = 1.0 - (norm["R2"] - norm["R2"].min()) / \
                     (norm["R2"].max() - norm["R2"].min() + 1e-12)

        # Radar
        ax_r = fig.add_subplot(
            len(targets_here), 2, row*2+1, polar=True
        )
        for mdl in model_order:
            if mdl not in norm.index: continue
            vals = norm.loc[mdl].values.tolist() + [norm.loc[mdl].values[0]]
            ax_r.plot(angles, vals, color=PALETTE.get(mdl,"grey"),
                      linewidth=2, label=mdl)
            ax_r.fill(angles, vals, color=PALETTE.get(mdl,"grey"), alpha=0.08)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(metrics_plot, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.set_title(f"{tgt.capitalize()} — Radar (lower = better)", pad=20)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)

        # Bar: RMSE vs MAPE
        ax_b = axes[row, 1]
        x = np.arange(len(model_order)); w = 0.35
        rmse_v = [summary.loc[m,"RMSE"]/1e3 if m in summary.index else 0
                  for m in model_order]
        mape_v = [summary.loc[m,"MAPE"] if m in summary.index else 0
                  for m in model_order]
        bars1 = ax_b.bar(x - w/2, rmse_v, w,
                         color=[PALETTE.get(m,"grey") for m in model_order],
                         alpha=0.85, label="RMSE (K)")
        ax_b.bar_label(bars1, fmt="%.0f", padding=2, fontsize=8)
        ax2b = ax_b.twinx()
        bars2 = ax2b.bar(x + w/2, mape_v, w,
                         color=[PALETTE.get(m,"grey") for m in model_order],
                         alpha=0.45, hatch="//", label="MAPE (%)")
        ax2b.bar_label(bars2, fmt="%.1f%%", padding=2, fontsize=8)
        ax_b.set_xticks(x); ax_b.set_xticklabels(model_order)
        ax_b.set_ylabel("RMSE (thousands)"); ax2b.set_ylabel("MAPE (%)")
        ax_b.set_title(f"{tgt.capitalize()} — RMSE (solid) vs MAPE (hatched)")

    plt.tight_layout()
    _save(fig, "fig19_scorecard.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT BLOCK  –  build_fitted_models()
# ──────────────────────────────────────────────────────────────────────────────
# Public function consumed by forecast_viz.py.  Fits all three models on the
# full training window (no hold-out split) and returns every artefact needed
# for downstream forecasting across all three targets.
# ══════════════════════════════════════════════════════════════════════════════

def build_fitted_models(
    data_path: str = DATA_PATH,
    run_cv: bool = False,
    n_repeats: int = N_REPEATS,
    k_folds: int = K_FOLDS,
    seasonal_s: int = SEASONAL_S,
    verbose: bool = True,
) -> dict:
    """
    Load freight data, fit GAM / SARIMALite / FixedEffectsOLS on the **full**
    sample for all three targets (originated, received, total), and return a
    rich artefact dict consumed by forecast_viz.py.

    Parameters
    ----------
    data_path  : path to the raw CSV file
    run_cv     : if True also runs 5×2 repeated TS-CV (slow)
    n_repeats  : CV repeats  (ignored when run_cv=False)
    k_folds    : CV folds    (ignored when run_cv=False)
    seasonal_s : SARIMA seasonal period in weeks (default 52)
    verbose    : print progress messages

    Returns
    -------
    dict
    ────
    "ts"                  : pd.DataFrame – aggregate weekly time-series (full)
    "panel"               : pd.DataFrame – panel data (full, with fe_pred cols)
    "y_all"               : dict {target: np.ndarray}  – weekly totals
    "t_all"               : np.ndarray  – week indices (0, 1, 2, …)
    "entity_indicators"   : dict {entity_name: (n,) array} – GAM entity weights

    Per-target model objects  (keys are "<model>_model_<target>"):
    "gam_model_originated"   / "_received" / "_total"
    "sarima_model_originated" / "_received" / "_total"
    "fe_model_originated"    / "_received" / "_total"

    Per-target in-sample fitted arrays (keys are "y_fit_<model>_<target>"):
    "y_fit_gam_originated"   / …
    "y_fit_sarima_originated" / …  (placeholder = actuals; diff-space resids)
    "y_fit_fe_originated"    / …

    Convenience aliases for forecast_viz.py (which expects the v1 key names
    for "total" target):
    "gam_model"     → gam_model_total
    "sarima_model"  → sarima_model_total
    "fe_model"      → fe_model_total
    "y_fit_gam"     → y_fit_gam_total
    "y_fit_sarima"  → y_fit_sarima_total
    "y_fit_fe"      → y_fit_fe_total

    "df_cv"       : pd.DataFrame – CV fold metrics (None if run_cv=False)
    "summary_cv"  : pd.DataFrame – multi-level summary   (None if run_cv=False)
    """

    # ── Step 1: Load & build data structures ─────────────────────────────────
    if verbose:
        print("[build_fitted_models v2] Loading data …")

    raw   = load_raw_data(data_path)
    ts    = build_aggregate_timeseries(raw)
    panel = build_panel_data(raw)

    y_all = {tgt: ts[tgt].values.astype(float) for tgt in TARGETS}
    t_all = ts["week_num"].values.astype(float)

    if verbose:
        print(f"  Weeks in sample : {len(ts)}")
        print(f"  Panel rows      : {len(panel):,}")
        print(f"  Date range      : {ts['week'].min().date()} → {ts['week'].max().date()}")

    # ── Step 2: Build entity indicators for GAM ───────────────────────────────
    if verbose:
        print("[build_fitted_models v2] Building entity indicators …")

    raw_full = raw.copy()
    raw_full["total"] = raw_full["originated"].fillna(0) + raw_full["received"].fillna(0)
    entity_indicators = build_entity_indicators_agg(ts, raw_full)

    if verbose:
        print(f"  Entity groups : {len(entity_indicators)}")

    # ── Step 3: Fit models for each target ───────────────────────────────────
    out = {}   # accumulate all artefacts

    for tgt in TARGETS:
        y = y_all[tgt]

        # ── GAM ──────────────────────────────────────────────────────────────
        if verbose:
            print(f"[build_fitted_models v2] Fitting GAM for target='{tgt}' …")
        
        bias_val = 30_000 if tgt == "originated" else 15_000 if tgt == "received" else 50_000

        gam = GAMSpline(n_knots_trend=15, n_knots_entity=8,
                        n_fourier=12, alpha=1.0, bias=bias_val)
        gam.fit(t_all, y, ts, entity_indicators)
        y_fit_gam = gam.predict(t_all, ts, entity_indicators)

        out[f"gam_model_{tgt}"]    = gam
        out[f"y_fit_gam_{tgt}"]    = y_fit_gam

        # ── SARIMA ────────────────────────────────────────────────────────────
        if verbose:
            print(f"[build_fitted_models v2] Fitting SARIMALite for target='{tgt}' …")

        sm = SARIMALite(p=2, q=2, P=1, Q=1, S=seasonal_s)
        sm.fit(y)
        out[f"sarima_model_{tgt}"] = sm
        out[f"y_fit_sarima_{tgt}"] = y.copy()   # placeholder (diff-space resids)

        # ── OLS Fixed-Effects ─────────────────────────────────────────────────
        if verbose:
            print(f"[build_fitted_models v2] Fitting OLS-FE for target='{tgt}' …")

        fe = FixedEffectsOLS(n_fourier=6, fit_trend=True, target=tgt)
        fe.fit(panel)

        panel_copy = panel.copy()
        panel_copy["fe_pred"] = fe.predict(panel_copy)
        y_fit_fe = (
            panel_copy.groupby("week")["fe_pred"]
            .sum()
            .reindex(ts["week"])
            .values
        )
        out[f"fe_model_{tgt}"]     = fe
        out[f"y_fit_fe_{tgt}"]     = y_fit_fe

    # Panel with fe_pred for total (used by OLS-FE forecaster in forecast_viz.py)
    panel_with_preds = panel.copy()
    panel_with_preds["fe_pred"] = out["fe_model_total"].predict(panel_with_preds)

    # ── Step 4: Optional CV ───────────────────────────────────────────────────
    df_cv = None
    summary_cv = None

    if run_cv:
        if verbose:
            print("[build_fitted_models v2] Running 5×2 repeated CV …")

        splits = make_time_splits(n=len(t_all), n_repeats=n_repeats,
                                  k_folds=k_folds)
        all_cv_records = []

        for tgt in TARGETS:
            y = y_all[tgt]

            # GAM CV
            def _gam_fit(tr, _tgt=tgt):
                ent_tr = {k: v[tr] for k, v in entity_indicators.items()}
                cal_tr = ts.iloc[tr].reset_index(drop=True)
                m = GAMSpline(n_knots_trend=15, n_knots_entity=8,
                              n_fourier=12, alpha=1.0)
                m.fit(t_all[tr], y_all[_tgt][tr], cal_tr, ent_tr)
                return m

            def _gam_pred(m, te, _tgt=tgt):
                ent_te = {k: v[te] for k, v in entity_indicators.items()}
                cal_te = ts.iloc[te].reset_index(drop=True)
                return m.predict(t_all[te], cal_te, ent_te)

            recs, _ = run_cv_loop("GAM", splits, _gam_fit, _gam_pred, y, tgt,
                                  verbose=verbose)
            all_cv_records.extend(recs)

            # SARIMA CV
            def _sarima_fit(tr, _tgt=tgt):
                m = SARIMALite(p=2, q=2, P=1, Q=1, S=seasonal_s)
                m.fit(y_all[_tgt][tr])
                return m

            def _sarima_pred(m, te):
                return m.predict(h=len(te))

            recs, _ = run_cv_loop("SARIMA", splits, _sarima_fit, _sarima_pred,
                                  y, tgt, verbose=verbose)
            all_cv_records.extend(recs)

            # OLS-FE CV
            fe_records = []
            week_sorted = sorted(panel["week"].unique())
            for sp in splits:
                fold = sp["fold"]
                tr_weeks = [week_sorted[i] for i in sp["train_idx"]
                            if i < len(week_sorted)]
                te_weeks = [week_sorted[i] for i in sp["test_idx"]
                            if i < len(week_sorted)]
                if not tr_weeks or not te_weeks:
                    continue
                trp = panel[panel["week"].isin(tr_weeks)].copy()
                tep = panel[panel["week"].isin(te_weeks)].copy()
                try:
                    fe_cv = FixedEffectsOLS(n_fourier=6, fit_trend=True,
                                            target=tgt)
                    fe_cv.fit(trp)
                    tep = tep.copy()
                    tep["pred"] = fe_cv.predict(tep)
                    pred_agg = (tep.groupby("week")["pred"].sum()
                                   .reindex(te_weeks).values)
                    true_agg = (panel[panel["week"].isin(te_weeks)]
                                   .groupby("week")[tgt].sum()
                                   .reindex(te_weeks).values)
                    pred_agg = np.clip(pred_agg, 0, None)
                    m_r = compute_metrics(true_agg, pred_agg, "OLS-FE")
                    m_r["fold"] = fold; m_r["target"] = tgt
                    fe_records.append(m_r)
                    if verbose:
                        print(f"  [OLS-FE/{tgt}] fold {fold:2d} | "
                              f"RMSE={m_r['RMSE']:>10,.0f} | "
                              f"MAPE={m_r['MAPE']:.2f}%")
                except Exception as exc:
                    if verbose:
                        print(f"  [OLS-FE/{tgt}] fold {fold} FAILED: {exc}")
            all_cv_records.extend(fe_records)

        df_cv = pd.DataFrame(all_cv_records)
        summary_cv = summarise_cv(df_cv)

    # ── Step 5: Package and return ────────────────────────────────────────────
    if verbose:
        print("[build_fitted_models v2] Done. Returning artefact dict.")

    result = {
        # Raw data
        "ts":               ts,
        "panel":            panel_with_preds,
        "y_all":            y_all,           # dict {target: array}
        "t_all":            t_all,
        "entity_indicators": entity_indicators,

        # CV results
        "df_cv":            df_cv,
        "summary_cv":       summary_cv,
    }

    # Per-target model objects and fitted arrays
    for tgt in TARGETS:
        result[f"gam_model_{tgt}"]     = out[f"gam_model_{tgt}"]
        result[f"sarima_model_{tgt}"]  = out[f"sarima_model_{tgt}"]
        result[f"fe_model_{tgt}"]      = out[f"fe_model_{tgt}"]
        result[f"y_fit_gam_{tgt}"]     = out[f"y_fit_gam_{tgt}"]
        result[f"y_fit_sarima_{tgt}"]  = out[f"y_fit_sarima_{tgt}"]
        result[f"y_fit_fe_{tgt}"]      = out[f"y_fit_fe_{tgt}"]

    # Convenience aliases expected by forecast_viz.py (v1 key names → total target)
    result["gam_model"]      = out["gam_model_total"]
    result["sarima_model"]   = out["sarima_model_total"]
    result["fe_model"]       = out["fe_model_total"]
    result["y_fit_gam"]      = out["y_fit_gam_total"]
    result["y_fit_sarima"]   = out[f"y_fit_sarima_total"]
    result["y_fit_fe"]       = out["y_fit_fe_total"]
    # y_all convenience alias for total (forecast_viz uses artefacts["y_all"])
    result["y_all_total"]    = y_all["total"]

    return result


# ══════════════════════════════════════════════════════════════════
# SECTION 12 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("  FREIGHT RAIL CARLOAD FORECASTING PIPELINE  v2")
    print("  Targets : originated | received | total")
    print("  Models  : GAM (entity splines) | SARIMALite | FixedEffectsOLS")
    print("  Hold-out: last 26 weeks (no leakage)")
    print("=" * 70)

    # ── 12.1  Load & prepare ─────────────────────────────────────
    print("\n[1/8] Loading & preparing data…")
    raw   = load_raw_data(DATA_PATH)
    ts    = build_aggregate_timeseries(raw)
    panel = build_panel_data(raw)

    print(f"  Aggregate obs : {len(ts)}")
    print(f"  Panel obs     : {len(panel):,}")
    print(f"  Date range    : {ts['week'].min().date()} → {ts['week'].max().date()}")
    print(f"  Companies     : {raw['company'].nunique()} — {sorted(raw['company'].unique().tolist())}")
    print(f"  Commodity codes: {raw['code'].nunique()}")

    # ── 12.2  Hold-out split ──────────────────────────────────────
    print(f"\n[2/8] Carving out last {FORECAST_H} weeks as hold-out test set…")
    ts_train, ts_test, panel_train, panel_test = split_holdout(ts, panel, FORECAST_H)
    print(f"  Train : {ts_train['week'].min().date()} → {ts_train['week'].max().date()} ({len(ts_train)} weeks)")
    print(f"  Test  : {ts_test['week'].min().date()}  → {ts_test['week'].max().date()} ({len(ts_test)} weeks)")

    # Convenience arrays (TRAIN ONLY — never touch ts_test before evaluation)
    y_train = {tgt: ts_train[tgt].values.astype(float) for tgt in TARGETS}
    t_train = ts_train["week_num"].values.astype(float)

    # ── 12.3  Entity indicators (train) ──────────────────────────
    print("\n[3/8] Building entity indicators for GAM…")
    # raw_train: panel-level data restricted to training weeks
    raw_train = raw[raw["week"].isin(ts_train["week"])].copy()
    raw_train["total"] = raw_train["originated"].fillna(0) + raw_train["received"].fillna(0)
    entity_inds_train = build_entity_indicators_agg(ts_train, raw_train)
    print(f"  Entity groups : {len(entity_inds_train)} (companies + commodity codes)")

    # ── 12.4  EDA ─────────────────────────────────────────────────
    print("\n[4/8] EDA plots…")
    set_plot_style()
    plot_data_overview(ts, raw)        # uses full series for context
    plot_seasonality_patterns(ts)
    plot_holdout_overview(ts_train, ts_test)

    # ── 12.5  Cross-validation (train window only) ────────────────
    print("\n[5/8] Cross-validation…")
    splits = make_time_splits(n=len(t_train), n_repeats=N_REPEATS,
                              k_folds=K_FOLDS)
    print(f"  Total CV folds: {len(splits)}")

    all_cv_records = []
    fold_data_all  = {}   # {(model, target): list of (te_idx, y_true, y_pred)}

    for tgt in TARGETS:
        y_all = y_train[tgt]
        print(f"\n  ── TARGET: {tgt.upper()} ──")

        # ── CV: GAM ──────────────────────────────────────────────
        print("    GAM")
        def gam_fit(tr_idx, _tgt=tgt):
            ent_tr = {k: v[tr_idx] for k, v in entity_inds_train.items()}
            cal_tr = ts_train.iloc[tr_idx].reset_index(drop=True)
            m = GAMSpline(n_knots_trend=15, n_knots_entity=8,
                        n_fourier=12, alpha=1.0,
                        bias=GAM_BIAS.get(_tgt, 0.0))
            m.fit(t_train[tr_idx], y_train[_tgt][tr_idx], cal_tr, ent_tr)
            return m

        def gam_predict(m, te_idx):
            ent_te = {k: v[te_idx] for k, v in entity_inds_train.items()}
            cal_te = ts_train.iloc[te_idx].reset_index(drop=True)
            return m.predict(t_train[te_idx], cal_te, ent_te)

        recs, fd = run_cv_loop("GAM", splits, gam_fit, gam_predict, y_all, tgt)
        all_cv_records.extend(recs)
        fold_data_all[("GAM", tgt)] = fd

        # ── CV: SARIMA ────────────────────────────────────────────
        print("    SARIMA")
        def sarima_fit(tr_idx, _tgt=tgt):
            m = SARIMALite(p=2, q=2, P=1, Q=1, S=SEASONAL_S)
            m.fit(y_train[_tgt][tr_idx])
            return m

        def sarima_predict(m, te_idx):
            return m.predict(h=len(te_idx))

        recs, fd = run_cv_loop("SARIMA", splits, sarima_fit, sarima_predict,
                               y_all, tgt)
        all_cv_records.extend(recs)
        fold_data_all[("SARIMA", tgt)] = fd

        # ── CV: OLS-FE ────────────────────────────────────────────
        print("    OLS Fixed-Effects")
        fe_records = []; fe_fd = []
        week_sorted_tr = sorted(panel_train["week"].unique())

        for sp in splits:
            fold     = sp["fold"]
            tr_weeks = [week_sorted_tr[i] for i in sp["train_idx"]
                        if i < len(week_sorted_tr)]
            te_weeks = [week_sorted_tr[i] for i in sp["test_idx"]
                        if i < len(week_sorted_tr)]
            if not tr_weeks or not te_weeks: continue
            trp = panel_train[panel_train["week"].isin(tr_weeks)].copy()
            tep = panel_train[panel_train["week"].isin(te_weeks)].copy()
            try:
                fe = FixedEffectsOLS(n_fourier=6, fit_trend=True, target=tgt)
                fe.fit(trp)
                tep = tep.copy(); tep["pred"] = fe.predict(tep)
                pred_agg = (tep.groupby("week")["pred"].sum()
                               .reindex(te_weeks).values)
                true_agg = (panel_train[panel_train["week"].isin(te_weeks)]
                               .groupby("week")[tgt].sum()
                               .reindex(te_weeks).values)
                pred_agg = np.clip(pred_agg, 0, None)
                m_r = compute_metrics(true_agg, pred_agg, "OLS-FE")
                m_r["fold"] = fold; m_r["target"] = tgt
                fe_records.append(m_r)
                fe_fd.append((sp["test_idx"], true_agg, pred_agg))
                print(f"      [OLS-FE/{tgt}] fold {fold:2d} | "
                      f"RMSE={m_r['RMSE']:>10,.0f} | MAPE={m_r['MAPE']:.2f}%")
            except Exception as exc:
                print(f"      [OLS-FE/{tgt}] fold {fold} FAILED: {exc}")

        all_cv_records.extend(fe_records)
        fold_data_all[("OLS-FE", tgt)] = fe_fd

    df_cv = pd.DataFrame(all_cv_records)

    # ── 12.6  CV summary ─────────────────────────────────────────
    print("\n[6/8] CV Summary…")
    summary_tbl = summarise_cv(df_cv)
    print("\n" + "─" * 70)
    print("  5×2 K-FOLD CV SUMMARY  (mean ± std)")
    print("─" * 70)
    for tgt in TARGETS:
        print(f"\n  ═══ TARGET: {tgt.upper()} ═══")
        for mdl in ["GAM","SARIMA","OLS-FE"]:
            key = (tgt, mdl)
            if key not in summary_tbl.index: continue
            print(f"  {mdl}")
            row = summary_tbl.loc[key]
            for met in ["MAE","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]:
                mu  = row[(met,"mean")]; std = row[(met,"std")]
                if met == "MAPE":
                    print(f"    {met:<5}: {mu:>8.2f}%  ± {std:>6.2f}%")
                elif met == "R2":
                    print(f"    {met:<5}: {mu:>8.4f}   ± {std:>6.4f}")
                else:
                    print(f"    {met:<5}: {mu:>12,.0f}  ± {std:>10,.0f}")

    # ── 12.7  Full fits on training data + hold-out forecast ──────
    print("\n[7/8] Fitting final models on full training data…")

    gam_models = {}     # {target: GAMSpline}
    sarima_models = {}  # {target: SARIMALite}
    fe_models = {}      # {target: FixedEffectsOLS}

    fits   = {}   # {(model, target): in-sample fitted array}
    fcasts = {}   # {(model, target): hold-out forecast array}

    # ── Build test entity indicators ─────────────────────────────
    # Combine train+test for a consistent mix, but feed only test rows to forecast
    raw_test = raw[raw["week"].isin(ts_test["week"])].copy()
    raw_test["total"] = raw_test["originated"].fillna(0) + raw_test["received"].fillna(0)
    entity_inds_test = build_entity_indicators_agg(ts_test, raw_test)

    for tgt in TARGETS:
        print(f"\n  Fitting for target: {tgt}")
        y_all = y_train[tgt]

        # GAM
        gam = GAMSpline(n_knots_trend=15, n_knots_entity=8,
                n_fourier=12, alpha=1.0,
                bias=GAM_BIAS.get(tgt, 0.0))
        gam.fit(t_train, y_all, ts_train, entity_inds_train)
        gam_models[tgt] = gam
        fits[("GAM", tgt)] = gam.predict(t_train, ts_train, entity_inds_train)

        # GAM hold-out forecast
        t_test = np.arange(t_train[-1]+1, t_train[-1]+1+len(ts_test))
        fcasts[("GAM", tgt)] = np.clip(
            gam.predict(t_test, ts_test, entity_inds_test), 0, None
        )

        # SARIMA
        sm = SARIMALite(p=2, q=2, P=1, Q=1, S=SEASONAL_S)
        sm.fit(y_all)
        sarima_models[tgt] = sm
        fits[("SARIMA", tgt)]  = y_all.copy()   # placeholder
        fcasts[("SARIMA", tgt)] = np.clip(sm.predict(h=len(ts_test)), 0, None)

        # OLS-FE
        fe = FixedEffectsOLS(n_fourier=6, fit_trend=True, target=tgt)
        fe.fit(panel_train)
        fe_models[tgt] = fe
        # In-sample fit (aggregate)
        pn_tr = panel_train.copy()
        pn_tr["fe_pred"] = fe.predict(pn_tr)
        fits[("OLS-FE", tgt)] = (
            pn_tr.groupby("week")["fe_pred"].sum()
                 .reindex(ts_train["week"]).values
        )
        # Hold-out forecast
        pn_te = panel_test.copy()
        pn_te["fe_pred"] = fe.predict(pn_te)
        fcasts[("OLS-FE", tgt)] = np.clip(
            pn_te.groupby("week")["fe_pred"].sum()
                  .reindex(ts_test["week"]).values, 0, None
        )

    # ── 12.8  All visualisations ──────────────────────────────────
    print("\n[8/8] Rendering all visualisations…")
    set_plot_style()

    # CV plots
    plot_cv_violins(df_cv)
    plot_cv_boxplots(df_cv)
    plot_metric_heatmap(df_cv)
    plot_fold_trajectory(df_cv)
    plot_actual_vs_predicted(fold_data_all)

    # Fit & forecast
    plot_fit_and_forecast(ts_train, ts_test, fits, ts_test["week"], fcasts)
    plot_forecast_comparison(ts_train, ts_test, fcasts)

    # Hold-out evaluation
    df_test = plot_holdout_metrics(ts_test, fcasts)
    print("\n  Hold-out metrics:")
    print(df_test[["target","model","RMSE","MAPE","R2", "Total Error", "Total Null Model Error", "Error Ratio"]].to_string(index=False))

    # Diagnostics
    residuals_dict = {}
    for tgt in TARGETS:
        for mdl in ["GAM","OLS-FE"]:
            key = (mdl, tgt)
            if key in fits and fits[key] is not None:
                resid = y_train[tgt] - fits[key]
                if not np.all(resid == 0):
                    residuals_dict[key] = resid

    plot_residuals_panel(ts_train, residuals_dict)
    plot_decomposition_comparison(ts_train, gam_models, t_train, entity_inds_train)
    plot_gam_spline_basis(gam_models, t_train, ts_train, entity_inds_train)

    # NEW: GAM variable importance
    plot_gam_variable_importance(gam_models, t_train, ts_train,
                                 entity_inds_train, y_train)
    plot_gam_entity_effects(gam_models, t_train, ts_train, entity_inds_train)

    # FE & model-level
    plot_fe_coefficients(fe_models, top_n=20)
    plot_hyperparameter_sensitivity(ts_train, y_train, t_train,
                                    entity_inds_train, splits, target="total")
    plot_scorecard(df_cv)

    # ── Save outputs ──────────────────────────────────────────────
    df_cv.to_csv(f"{OUTPUT_DIR}/cv_results.csv", index=False)
    summary_tbl.to_csv(f"{OUTPUT_DIR}/cv_summary.csv")
    df_test.to_csv(f"{OUTPUT_DIR}/holdout_test_metrics.csv", index=False)
    print("\n  ✓ CSV tables saved")

    print("\n" + "=" * 70)
    print(f"  Pipeline complete.  Outputs → {OUTPUT_DIR}/")
    print("=" * 70)