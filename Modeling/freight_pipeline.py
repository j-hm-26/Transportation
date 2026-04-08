"""
freight_pipeline.py
===================
Freight Rail Carload Forecasting Pipeline
==========================================
Models: GAM (Penalized Splines), ProphetLite, SARIMALite, FixedEffectsOLS
Validation: 5×2 Repeated Time-Series Cross-Validation
Author: Extended pipeline with GAM integration
"""
#%%
# ══════════════════════════════════════════════════════════════════
# SECTION 1 – IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")
import sys, os

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
from sklearn.preprocessing import (StandardScaler, SplineTransformer,
                                   PolynomialFeatures)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Aesthetic constants ─────────────────────────────────────────
PALETTE = {
    "GAM":     "#9B59B6",
    "SARIMA":  "#E07B39",
    "Prophet": "#3B82C4",
    "OLS-FE":  "#2DBD6E",
}
BG_COLOR   = "#F8F9FA"
GRID_COLOR = "#E2E8F0"
ACCENT     = "#6C63FF"

OUTPUT_DIR = "Visuals/freight_pipeline/2017_onwards"
DATA_PATH  = "Data/Weekly_Cargo_Data_2017_2026.csv"

N_REPEATS   = 5
K_FOLDS     = 2
FORECAST_H  = 26 # 6 months ahead (26 weeks)
SEASONAL_S  = 52

MIN_YR = 2017

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 – DATA I/O
# ══════════════════════════════════════════════════════════════════

def load_raw_data(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = (df.columns.str.strip().str.lower()
                    .str.replace(" ", "_", regex=False))
    df["week"] = pd.to_datetime(df["week"])
    df = df.rename(columns={"commodity_group_code": "code",
                             "commodity_group":      "commodity_group"})
    df["company"] = df["company"].astype("category")
    df["code"]    = df["code"].astype("category")
    return df


def build_aggregate_timeseries(df):
    ts = (df.groupby("week")[["originated", "received"]]
            .sum()
            .assign(total=lambda x: x["originated"] + x["received"])
            .sort_index().reset_index())
    ts["week_num"]     = np.arange(len(ts), dtype=float)
    ts["week_of_year"] = ts["week"].dt.isocalendar().week.astype(int)
    ts["year"]         = ts["week"].dt.year
    return ts


def build_panel_data(df):
    panel = df.copy()
    panel["originated"] = panel["originated"].fillna(0)
    panel["received"]   = panel["received"].fillna(0)
    panel["total"]      = panel["originated"] + panel["received"]
    panel["total"]      = panel["total"].fillna(0).clip(lower=0)
    panel["log_total"]  = np.log1p(panel["total"])
    week_map = {w: i for i, w in enumerate(sorted(panel["week"].unique()))}
    panel["week_num"]     = panel["week"].map(week_map).astype(float)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["year"]         = panel["week"].dt.year
    panel["company_fe"]  = panel["company"].cat.codes
    panel["code_fe"]     = panel["code"].cat.codes
    return panel


# ══════════════════════════════════════════════════════════════════
# SECTION 3 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def make_fourier_features(t, period=52.1775, n_harmonics=10):
    cols = []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def make_trend_features(t, changepoints):
    cols = [np.ones_like(t), t]
    for cp in changepoints:
        cols.append(np.maximum(0.0, t - cp))
    return np.column_stack(cols)


def make_calendar_features(df):
    q    = df["week"].dt.quarter
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

# ── 4.1  GAM (Penalized Regression Splines) ──────────────────────
class GAMSpline:
    """
    Generalized Additive Model via Penalized Regression Splines.

    Structure:
        y(t) = f_trend(t) + f_season(t) + f_cal(t) + ε

    Where:
      - f_trend    : natural cubic spline basis in t
      - f_season   : Fourier series (annual periodicity)
      - f_cal      : quarter indicators + holiday weeks

    All smooth terms are jointly estimated by Ridge regression
    (L2 penalty ≈ cubic spline smoothing penalty).

    Hyperparameters
    ---------------
    n_knots_trend : int
        Number of interior knots for the trend spline.
    n_fourier : int
        Fourier harmonics for seasonal component.
    alpha : float
        Ridge regularisation (smoothing) strength.
    """

    def __init__(self, n_knots_trend=15, n_fourier=12, alpha=1.0):
        self.n_knots_trend = n_knots_trend
        self.n_fourier     = n_fourier
        self.alpha         = alpha
        self._spline_trend = None
        self._scaler       = StandardScaler()
        self._model        = Ridge(alpha=alpha)
        self._t_min        = None
        self._t_max        = None

    def _build_X(self, t, cal_df=None):
        # Trend spline
        t2d = t.reshape(-1, 1)
        X_trend = self._spline_trend.transform(t2d)
        # Seasonal Fourier
        X_seas = make_fourier_features(t, n_harmonics=self.n_fourier)
        # Calendar (if available)
        if cal_df is not None:
            X_cal = make_calendar_features(cal_df)
            return np.hstack([X_trend, X_seas, X_cal])
        return np.hstack([X_trend, X_seas])

    def fit(self, t, y, cal_df=None):
        self._t_min = t.min()
        self._t_max = t.max()
        # Fit spline transformer on training t
        self._spline_trend = SplineTransformer(
            n_knots=self.n_knots_trend,
            degree=3,
            knots="quantile",
            include_bias=True,
        )
        self._spline_trend.fit(t.reshape(-1, 1))
        self._cal_df = cal_df  # store for predict fallback

        X  = self._build_X(t, cal_df)
        Xs = self._scaler.fit_transform(X)
        self._model.fit(Xs, y)

        # Store component column widths for decomposition
        t2d = t.reshape(-1, 1)
        self._n_trend_cols  = self._spline_trend.transform(t2d).shape[1]
        self._n_fourier_cols = 2 * self.n_fourier
        return self

    def predict(self, t, cal_df=None):
        X  = self._build_X(t, cal_df)
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs)

    def predict_components(self, t):
        """Return (trend, seasonality) decomposition."""
        X  = self._build_X(t)
        Xs = self._scaler.transform(X)
        c  = self._model.coef_
        ic = self._model.intercept_
        k1 = self._n_trend_cols
        k2 = k1 + self._n_fourier_cols
        trend = Xs[:, :k1] @ c[:k1] + ic
        seas  = Xs[:, k1:k2] @ c[k1:k2]
        return trend, seas


# ── 4.2  ProphetLite (piecewise trend + Fourier) ─────────────────
class ProphetLite:
    """
    Additive decomposition model inspired by Facebook Prophet.
    y(t) = piecewise_linear_trend(t) + Fourier_seasonality(t) + ε
    Jointly estimated by Ridge regression.
    """
    def __init__(self, n_changepoints=25, n_fourier=10, alpha=0.5):
        self.n_cp  = n_changepoints
        self.n_f   = n_fourier
        self.alpha = alpha
        self.model_  = Ridge(alpha=alpha)
        self.scaler_ = StandardScaler()
        self._cp_t   = None

    def _build_X(self, t):
        X_trend = make_trend_features(t, self._cp_t)
        X_seas  = make_fourier_features(t, n_harmonics=self.n_f)
        return np.hstack([X_trend, X_seas])

    def fit(self, t, y):
        self._cp_t = np.quantile(t, np.linspace(0.05, 0.90, self.n_cp))
        self._n_trend_cols = make_trend_features(t[:1], self._cp_t).shape[1]
        X  = self._build_X(t)
        Xs = self.scaler_.fit_transform(X)
        self.model_.fit(Xs, y)
        return self

    def predict(self, t):
        X  = self._build_X(t)
        Xs = self.scaler_.transform(X)
        return self.model_.predict(Xs)

    def predict_components(self, t):
        X  = self._build_X(t)
        Xs = self.scaler_.transform(X)
        k  = self._n_trend_cols
        c  = self.model_.coef_
        trend = Xs[:, :k] @ c[:k] + self.model_.intercept_
        seas  = Xs[:, k:] @ c[k:]
        return trend, seas


# ── 4.3  SARIMALite ──────────────────────────────────────────────
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
        return 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(eps**2) / sigma2

    def fit(self, y):
        yd = self._diff(y, d=1)
        if len(yd) > self.S:
            yd = self._diff(yd, d=1)
        n_params = self.p + self.q + self.P + self.Q + 1
        x0 = np.zeros(n_params); x0[-1] = np.log(np.var(yd) + 1e-6)
        bounds = [(-0.99, 0.99)] * (n_params - 1) + [(None, None)]
        res = minimize(self._neg_loglik, x0, args=(yd,),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-9})
        self.params_ = res.x; self._yd = yd; self._y = y
        self.resid_  = self._run_filter(self.params_, yd)
        return self

    def predict(self, h=1):
        phi, theta, Phi, Theta = self._unpack(self.params_)
        S = self.S; yd_ext = list(self._yd.copy()); eps_ext = list(self.resid_.copy())
        preds_d = []
        for _ in range(h):
            t   = len(yd_ext)
            hat  = sum(phi[i]   * yd_ext[t-i-1]      for i in range(self.p) if t-i-1 >= 0)
            hat -= sum(theta[j] * eps_ext[t-j-1]     for j in range(self.q) if t-j-1 >= 0)
            hat += sum(Phi[i]   * yd_ext[t-(i+1)*S]  for i in range(self.P) if t-(i+1)*S >= 0)
            hat -= sum(Theta[j] * eps_ext[t-(j+1)*S] for j in range(self.Q) if t-(j+1)*S >= 0)
            preds_d.append(hat); yd_ext.append(hat); eps_ext.append(0.0)
        recovered = np.r_[self._y[-2:],
                          np.cumsum(np.r_[self._y[-1], preds_d])[1:]]
        result = np.cumsum(np.r_[self._y[-1], np.diff(recovered)])
        return result[1:h+1]


# ── 4.4  Fixed-Effects OLS (panel) ───────────────────────────────
class FixedEffectsOLS:
    """
    Two-way Fixed-Effects OLS (company + commodity dummies) on log carloads.
    Includes linear trend, Fourier seasonality, and calendar effects.
    """
    def __init__(self, n_fourier=6, fit_trend=True):
        self.n_fourier  = n_fourier
        self.fit_trend  = fit_trend
        self.model_     = LinearRegression(fit_intercept=True)
        self.scaler_    = StandardScaler()
        self.companies_ = None
        self.codes_     = None
        self.coef_df_   = None

    def _build_dummies(self, panel, companies, codes):
        comp_dummies = np.zeros((len(panel), len(companies) - 1))
        code_dummies = np.zeros((len(panel), len(codes) - 1))
        for j, comp in enumerate(companies[1:]):
            comp_dummies[:, j] = (panel["company"] == comp).astype(float).values
        for j, code in enumerate(codes[1:]):
            code_dummies[:, j] = (panel["code"] == code).astype(float).values
        return np.hstack([comp_dummies, code_dummies])

    def fit(self, panel):
        self.companies_ = list(panel["company"].cat.categories)
        self.codes_      = list(panel["code"].cat.categories)
        t = panel["week_num"].values
        X_seas = make_fourier_features(t, n_harmonics=self.n_fourier)
        X_fe   = self._build_dummies(panel, self.companies_, self.codes_)
        X_cal  = make_calendar_features(panel)
        blocks = [X_fe, X_seas, X_cal]
        if self.fit_trend:
            blocks = [t.reshape(-1, 1)] + blocks
        X = np.hstack(blocks)
        self.scaler_.fit(X)
        Xs = self.scaler_.transform(X)
        y = panel["log_total"].values
        self.model_.fit(Xs, y)
        col_names = self._make_col_names()
        self.coef_df_ = pd.DataFrame({
            "feature": col_names,
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
        t = panel["week_num"].values
        X_seas = make_fourier_features(t, n_harmonics=self.n_fourier)
        X_fe   = self._build_dummies(panel, self.companies_, self.codes_)
        X_cal  = make_calendar_features(panel)
        blocks = [X_fe, X_seas, X_cal]
        if self.fit_trend:
            blocks = [t.reshape(-1, 1)] + blocks
        X  = np.hstack(blocks)
        Xs = self.scaler_.transform(X)
        return np.clip(np.expm1(self.model_.predict(Xs)), 0, None)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 – METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1.0)))) * 100.0
    r2   = r2_score(y_true, y_pred)
    return {"model": label, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}


def summarise_cv(df_cv):
    return df_cv.groupby("model")[["MAE","MSE","RMSE","MAPE","R2"]].agg(["mean","std"]).round(2)


# ══════════════════════════════════════════════════════════════════
# SECTION 6 – CROSS-VALIDATION
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
                y_global, verbose=True):
    records = []; fold_data = []
    for sp in splits:
        fold   = sp["fold"]
        tr_idx = sp["train_idx"]
        te_idx = sp["test_idx"]
        try:
            fitted = fit_fn(tr_idx)
            y_pred = np.clip(predict_fn(fitted, te_idx), 0, None)
        except Exception as exc:
            if verbose: print(f"  [{model_name}] fold {fold} FAILED: {exc}")
            continue
        y_true = y_global[te_idx]
        m = compute_metrics(y_true, y_pred, label=model_name)
        m["fold"] = fold
        records.append(m)
        fold_data.append((te_idx, y_true, y_pred))
        if verbose:
            print(f"  [{model_name}] fold {fold:2d} | "
                  f"train={len(tr_idx):4d} | test={len(te_idx):3d} | "
                  f"RMSE={m['RMSE']:>10,.0f} | MAPE={m['MAPE']:.2f}%")
    return records, fold_data


# ══════════════════════════════════════════════════════════════════
# SECTION 7 – FORECASTING UTILITIES
# ══════════════════════════════════════════════════════════════════

def forecast_future(ts, model, horizon=26, model_type="prophet"):
    last_date    = ts["week"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1),
                                 periods=horizon, freq="W-WED")
    if model_type.lower() == "sarima":
        preds = model.predict(h=horizon)
    else:
        t_future = np.arange(ts["week_num"].iloc[-1] + 1,
                             ts["week_num"].iloc[-1] + 1 + horizon)
        preds = model.predict(t_future)
    return future_dates, np.clip(preds, 0, None)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 – VISUALISATION FUNCTIONS
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


# ── Fig 1: Overview EDA ──────────────────────────────────────────
def plot_data_overview(ts, raw):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1a. Full aggregate time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(ts["week"], ts["total"] / 1e6, alpha=0.25, color=ACCENT)
    ax1.plot(ts["week"], ts["total"] / 1e6, color=ACCENT, linewidth=1.2)
    ax1.set_title("Weekly Total Freight Carloads — All Companies & Commodities",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Total Carloads (millions)"); ax1.set_xlabel("Week")

    # 1b. By company
    ax2 = fig.add_subplot(gs[1, 0])
    comp_totals = (raw.groupby("company")[["originated","received"]]
                      .sum().assign(total=lambda x: x["originated"]+x["received"])
                      .sort_values("total", ascending=True))
    colors = [PALETTE.get(c, ACCENT) for c in ["OLS-FE","Prophet","SARIMA","GAM",
                                                 "OLS-FE","Prophet","SARIMA","GAM"]]
    ax2.barh(comp_totals.index, comp_totals["total"] / 1e6,
             color=sns.color_palette("muted", len(comp_totals)))
    ax2.set_title("Total Carloads by Company"); ax2.set_xlabel("Carloads (millions)")

    # 1c. By commodity group
    ax3 = fig.add_subplot(gs[1, 1])
    comm_totals = (raw.groupby("commodity_group")[["originated","received"]]
                      .sum().assign(total=lambda x: x["originated"]+x["received"])
                      .sort_values("total", ascending=True).tail(10))
    ax3.barh(comm_totals.index, comm_totals["total"] / 1e6,
             color=sns.color_palette("Set2", len(comm_totals)))
    ax3.set_title("Top-10 Commodity Groups by Volume"); ax3.set_xlabel("Carloads (millions)")

    # 1d. Year-over-year
    ax4 = fig.add_subplot(gs[1, 2])
    ts_by_year = ts.groupby("year")["total"].sum() / 1e6
    ax4.bar(ts_by_year.index, ts_by_year.values,
            color=sns.color_palette("Blues_d", len(ts_by_year)))
    ax4.set_title("Annual Total Carloads"); ax4.set_xlabel("Year")
    ax4.set_ylabel("Carloads (millions)")

    fig.suptitle(f"Freight Rail Data Overview ({MIN_YR}–2026)",
                 fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "fig01_data_overview.png")


# ── Fig 2: Seasonality patterns ──────────────────────────────────
def plot_seasonality_patterns(ts):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Seasonality & Trend Patterns", fontsize=13, fontweight="bold")

    # Week-of-year profile
    ax = axes[0]
    woy = ts.groupby("week_of_year")["total"].agg(["mean","std"])
    ax.fill_between(woy.index, (woy["mean"]-woy["std"])/1e6,
                    (woy["mean"]+woy["std"])/1e6, alpha=0.2, color=ACCENT)
    ax.plot(woy.index, woy["mean"]/1e6, color=ACCENT, linewidth=2)
    ax.set_title("Mean Weekly Profile (± 1 SD)"); ax.set_xlabel("Week of Year")
    ax.set_ylabel("Total Carloads (millions)")

    # Year-over-year overlay
    ax = axes[1]
    for yr, grp in ts.groupby("year"):
        ax.plot(grp["week_of_year"], grp["total"]/1e6,
                alpha=0.7, linewidth=1.2,
                label=str(yr), color=plt.cm.viridis((yr-MIN_YR)/6))
    ax.set_title("Year-over-Year Overlay"); ax.set_xlabel("Week of Year")
    ax.set_ylabel("Carloads (millions)"); ax.legend(fontsize=8)

    # Rolling mean / trend
    ax = axes[2]
    roll = ts["total"].rolling(13, center=True).mean()
    ax.plot(ts["week"], ts["total"]/1e6, alpha=0.3, color="#999999",
            linewidth=0.7, label="Raw")
    ax.plot(ts["week"], roll/1e6, color=PALETTE["SARIMA"],
            linewidth=2.2, label="13-wk rolling mean")
    ax.set_title("Long-Run Trend (13-wk MA)"); ax.set_xlabel("Week")
    ax.set_ylabel("Carloads (millions)"); ax.legend()

    plt.tight_layout()
    _save(fig, "fig02_seasonality.png")


# ── Fig 3: CV violin + boxplot ───────────────────────────────────
def plot_cv_violins(df_cv, metrics=None):
    if metrics is None: metrics = ["MAE","RMSE","MAPE","R2"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
    fig.suptitle("5×2 Repeated K-Fold CV — Metric Distributions",
                 fontsize=14, fontweight="bold", y=1.01)
    model_order = ["GAM","Prophet","SARIMA","OLS-FE"]
    model_order = [m for m in model_order if m in df_cv["model"].unique()]

    for ax, metric in zip(axes, metrics):
        sns.violinplot(data=df_cv, x="model", y=metric, hue="model",
                       order=model_order, palette=PALETTE,
                       inner="box", ax=ax, linewidth=1.2, legend=False)
        sns.stripplot(data=df_cv, x="model", y=metric, hue="model",
                      order=model_order, palette=PALETTE,
                      size=5, jitter=True, alpha=0.6, ax=ax, legend=False)
        means = df_cv.groupby("model")[metric].mean()
        for xi, mdl in enumerate(model_order):
            if mdl in means.index:
                mu = means[mdl]
                ax.axhline(mu, color=PALETTE.get(mdl,"grey"),
                           linestyle="--", linewidth=1, alpha=0.6)
                ax.text(xi, mu, f"μ={mu:.2f}",
                        ha="center", va="bottom", fontsize=8,
                        color=PALETTE.get(mdl,"grey"), fontweight="bold")
        ax.set_title(metric); ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, "fig03_cv_violins.png")


def plot_cv_boxplots(df_cv, metrics=None):
    if metrics is None: metrics = ["MAE","RMSE","MAPE","R2"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("5×2 Repeated K-Fold CV — Metric Boxplots",
                 fontsize=14, fontweight="bold")
    model_order = ["GAM","Prophet","SARIMA","OLS-FE"]
    model_order = [m for m in model_order if m in df_cv["model"].unique()]
    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df_cv, x="model", y=metric, hue="model",
                    order=model_order, palette=PALETTE,
                    width=0.45, linewidth=1.5, ax=ax, legend=False,
                    flierprops=dict(marker="o", alpha=0.5, markersize=4))
        ax.set_title(metric); ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, "fig04_cv_boxplots.png")


# ── Fig 4: Metric heatmap ─────────────────────────────────────────
def plot_metric_heatmap(df_cv, metrics=None):
    if metrics is None: metrics = ["MAE","MSE","RMSE","MAPE","R2"]
    summary = df_cv.groupby("model")[metrics].mean().round(2)
    norm = summary.copy()
    for col in [c for c in metrics if c != "R2"]:
        rng = norm[col].max() - norm[col].min() + 1e-12
        norm[col] = (norm[col] - norm[col].min()) / rng
    if "R2" in norm.columns:
        rng = norm["R2"].max() - norm["R2"].min() + 1e-12
        norm["R2"] = 1.0 - (norm["R2"] - norm["R2"].min()) / rng

    # Reorder rows
    order = ["GAM","Prophet","SARIMA","OLS-FE"]
    order = [m for m in order if m in summary.index]
    summary = summary.loc[order]; norm = norm.loc[order]

    fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle("CV Metric Summary — Raw Mean Values & Normalised Ranking",
                 fontsize=13, fontweight="bold")

    # Format annotations nicely
    def fmt_annot(df):
        out = []
        for idx in df.index:
            row = []
            for col in df.columns:
                v = df.loc[idx, col]
                if col in ("MAE","MSE","RMSE"):
                    row.append(f"{v:,.0f}")
                elif col == "MAPE":
                    row.append(f"{v:.2f}%")
                else:
                    row.append(f"{v:.3f}")
            out.append(row)
        return out

    annots = fmt_annot(summary)
    sns.heatmap(summary, annot=annots, fmt="", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax_r, cbar_kws={"shrink": 0.8})
    ax_r.set_title("Raw Metric Means (CV)")

    sns.heatmap(norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax_n, cbar_kws={"shrink": 0.8})
    ax_n.set_title("Normalised Score (0.00 = best)")

    plt.tight_layout()
    _save(fig, "fig05_metric_heatmap.png")


# ── Fig 5: Fold trajectory ────────────────────────────────────────
def plot_fold_trajectory(df_cv, metrics=None):
    if metrics is None: metrics = ["RMSE","MAPE","R2"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7*len(metrics), 5))
    if len(metrics) == 1: axes = [axes]
    fig.suptitle("Performance Stability Across CV Folds",
                 fontsize=14, fontweight="bold")
    model_order = ["GAM","Prophet","SARIMA","OLS-FE"]
    for ax, metric in zip(axes, metrics):
        for mdl in model_order:
            if mdl not in df_cv["model"].unique(): continue
            sub = df_cv[df_cv["model"] == mdl].sort_values("fold")
            ax.plot(sub["fold"], sub[metric],
                    marker="o", label=mdl,
                    color=PALETTE.get(mdl,"grey"),
                    linewidth=2, markersize=7)
        ax.set_xlabel("Fold Index"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} Across Folds"); ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "fig06_fold_trajectory.png")


# ── Fig 6: Actual vs predicted scatter ───────────────────────────
def plot_actual_vs_predicted(fold_data, scale=1e6):
    models = list(fold_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1: axes = [axes]
    fig.suptitle("Actual vs Predicted — Last CV Fold (all models)",
                 fontsize=13, fontweight="bold")
    for ax, mdl in zip(axes, models):
        if not fold_data[mdl]: continue
        _, y_te, y_pr = fold_data[mdl][-1]
        yt = y_te / scale; yp = y_pr / scale
        ax.scatter(yt, yp, color=PALETTE.get(mdl,"grey"),
                   alpha=0.65, s=60, edgecolor="white", linewidth=0.5)
        mn = min(yt.min(), yp.min()); mx = max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect fit")
        m = compute_metrics(y_te, y_pr, mdl)
        ax.set_title(f"{mdl}\nRMSE={m['RMSE']/1e3:.0f}K  R²={m['R2']:.3f}  MAPE={m['MAPE']:.1f}%",
                     fontsize=9)
        ax.set_xlabel(f"Actual (×{scale:.0e})")
        ax.set_ylabel(f"Predicted (×{scale:.0e})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "fig07_actual_vs_predicted.png")


# ── Fig 7: In-sample fit & forecast ──────────────────────────────
def plot_fit_and_forecast(ts, fits, fcast_dates, fcasts):
    """
    fits  : dict model→np.ndarray of in-sample fitted values
    fcasts: dict model→np.ndarray of forecast values
    """
    y_all = ts["total"].values
    fig, axes = plt.subplots(2, 1, figsize=(18, 11),
                              gridspec_kw={"height_ratios": [3, 1.4]})

    ax = axes[0]
    ax.plot(ts["week"], y_all/1e6, color="#333333",
            linewidth=0.9, label="Actual", alpha=0.8, zorder=1)
    styles = {"GAM": ("-", 1.8), "Prophet": ("--", 1.8),
              "SARIMA": ("-.", 1.6), "OLS-FE": (":", 1.6)}
    for mdl, fit in fits.items():
        ls, lw = styles.get(mdl, ("-", 1.5))
        ax.plot(ts["week"], fit/1e6, color=PALETTE.get(mdl, ACCENT),
                linewidth=lw, linestyle=ls, alpha=0.82, label=f"{mdl} (fit)")
    markers = {"GAM": "o", "Prophet": "s", "SARIMA": "^", "OLS-FE": "D"}
    for mdl, fc in fcasts.items():
        mk = markers.get(mdl, "o")
        ax.plot(fcast_dates, fc/1e6, color=PALETTE.get(mdl, ACCENT),
                linewidth=2.3, marker=mk, markersize=4, label=f"{mdl} forecast")
    last_date = ts["week"].iloc[-1]
    ax.axvline(last_date, color="black", linestyle=":", linewidth=1.3, alpha=0.5)
    ax.text(last_date, ax.get_ylim()[1]*0.97, "  Forecast →", fontsize=9, color="grey")
    ax.set_title("Total Weekly Freight Carloads — In-Sample Fit + 26-Week Forecast",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("Carloads (millions)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    # Bottom panel: residuals for GAM and Prophet
    ax2 = axes[1]
    for mdl in ["GAM", "Prophet"]:
        if mdl in fits:
            resid = (y_all - fits[mdl]) / 1e3
            ax2.plot(ts["week"], resid, color=PALETTE.get(mdl, ACCENT),
                     alpha=0.7, linewidth=0.8, label=f"{mdl} residual")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Week"); ax2.set_ylabel("Residual (thousands)")
    ax2.set_title("In-Sample Residuals (GAM & Prophet)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig08_fit_and_forecast.png")


# ── Fig 8: Decomposition comparison ──────────────────────────────
def plot_decomposition_comparison(ts, gam_model, prophet_model, t_all):
    g_trend, g_seas   = gam_model.predict_components(t_all)
    p_trend, p_seas   = prophet_model.predict_components(t_all)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig.suptitle("Additive Decomposition — GAM vs Prophet",
                 fontsize=13, fontweight="bold")

    axes[0, 0].plot(ts["week"], g_trend/1e6, color=PALETTE["GAM"], linewidth=2)
    axes[0, 0].set_title("GAM — Trend Component"); axes[0, 0].set_ylabel("Carloads (M)")

    axes[0, 1].plot(ts["week"], p_trend/1e6, color=PALETTE["Prophet"], linewidth=2)
    axes[0, 1].set_title("Prophet — Trend Component")

    axes[1, 0].plot(ts["week"], g_seas/1e6, color=PALETTE["GAM"],
                    linewidth=1.2, alpha=0.85)
    axes[1, 0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1, 0].set_title("GAM — Seasonal Component")
    axes[1, 0].set_xlabel("Week"); axes[1, 0].set_ylabel("Carloads (M)")

    axes[1, 1].plot(ts["week"], p_seas/1e6, color=PALETTE["Prophet"],
                    linewidth=1.2, alpha=0.85)
    axes[1, 1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1, 1].set_title("Prophet — Seasonal Component")
    axes[1, 1].set_xlabel("Week")

    plt.tight_layout()
    _save(fig, "fig09_decomposition.png")


# ── Fig 9: Residual diagnostics ──────────────────────────────────
def plot_residuals_panel(ts, residuals_dict):
    models = list(residuals_dict.keys())
    fig, axes = plt.subplots(len(models), 4, figsize=(20, 5*len(models)))
    fig.suptitle("Residual Diagnostics — All Models",
                 fontsize=14, fontweight="bold")
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for row, mdl in enumerate(models):
        resid = residuals_dict[mdl]
        color = PALETTE.get(mdl, ACCENT)
        n     = len(resid)
        ci    = 1.96 / np.sqrt(n)

        # Time series
        ax = axes[row, 0]
        ax.plot(ts["week"], resid/1e3, color=color, linewidth=0.7, alpha=0.75)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{mdl} — Residuals over Time")
        ax.set_ylabel("Residual (K)"); ax.set_xlabel("Week")

        # Distribution
        ax = axes[row, 1]
        sns.histplot(resid/1e3, bins=35, kde=True, color=color,
                     ax=ax, edgecolor="white")
        ax.set_title(f"{mdl} — Distribution"); ax.set_xlabel("Residual (K)")

        # Q-Q
        ax = axes[row, 2]
        stats.probplot(resid, plot=ax)
        ax.set_title(f"{mdl} — Q-Q Plot")
        ax.get_lines()[1].set_color("red")

        # ACF
        ax = axes[row, 3]
        max_lag = min(52, n // 2 - 1)
        acf_vals = np.array([1.0] + [
            float(np.corrcoef(resid[lag:], resid[:-lag])[0, 1])
            for lag in range(1, max_lag + 1)
        ])
        ax.bar(np.arange(max_lag + 1), acf_vals, color=color, alpha=0.7)
        ax.axhline(ci,  color="red", linestyle="--", linewidth=1)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{mdl} — ACF"); ax.set_xlabel("Lag (weeks)")

    plt.tight_layout()
    _save(fig, "fig10_residuals.png")


# ── Fig 10: Forecast comparison zoomed ───────────────────────────
def plot_forecast_comparison(ts, fcast_dates, forecasts,
                              history_weeks=78, ci_pct=0.07):
    y_all = ts["total"].values
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(ts["week"].iloc[-history_weeks:],
            y_all[-history_weeks:]/1e6,
            color="#333333", linewidth=1.8, label="Actual", alpha=0.9, zorder=5)
    markers = {"GAM":"o","Prophet":"s","SARIMA":"^","OLS-FE":"D"}
    for mdl, fcast in forecasts.items():
        color = PALETTE.get(mdl, ACCENT)
        mk    = markers.get(mdl, "o")
        ax.fill_between(fcast_dates,
                        fcast/1e6*(1-ci_pct), fcast/1e6*(1+ci_pct),
                        alpha=0.14, color=color)
        ax.plot(fcast_dates, fcast/1e6, color=color, linewidth=2.5,
                marker=mk, markersize=5, label=f"{mdl} forecast")
    ax.axvline(ts["week"].iloc[-1], color="black",
               linestyle=":", linewidth=1.3, alpha=0.5)
    ax.set_title(f"{FORECAST_H}-Week Ahead Freight Carload Forecast — All Models (±7% PI)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("Total Carloads (millions)")
    ax.legend(fontsize=10); plt.tight_layout()
    _save(fig, "fig11_forecast_comparison.png")


# ── Fig 11: FE Coefficients ───────────────────────────────────────
def plot_fe_coefficients(coef_df, top_n=20):
    top = coef_df.nlargest(top_n, "abs_coef").copy()
    def _cat(feat):
        if feat.startswith("company_"): return "Company FE"
        if feat.startswith("code_"):    return "Commodity FE"
        if feat.startswith(("sin_","cos_")): return "Fourier Seasonality"
        if feat.startswith("q") or feat == "holiday_week": return "Calendar"
        return "Trend"
    top["category"] = top["feature"].apply(_cat)
    cat_pal = {
        "Company FE":        PALETTE["SARIMA"],
        "Commodity FE":      PALETTE["Prophet"],
        "Fourier Seasonality": ACCENT,
        "Calendar":          PALETTE["OLS-FE"],
        "Trend":             "#555555",
    }
    fig, ax = plt.subplots(figsize=(10, max(5, top_n // 2)))
    sns.barplot(data=top, y="feature", x="coef",
                hue="category", palette=cat_pal,
                ax=ax, orient="h", legend=True)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top-{top_n} OLS Fixed-Effects Coefficients (|coef| ranked)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Standardised Coefficient"); ax.set_ylabel("Feature")
    ax.legend(title="Category", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig12_fe_coefficients.png")


# ── Fig 12: GAM spline basis visualisation ───────────────────────
def plot_gam_spline_basis(gam_model, t_all, ts, top_n=8):
    """Show the learned spline basis functions weighted by coefficient."""
    t2d = t_all.reshape(-1, 1)
    B   = gam_model._spline_trend.transform(t2d)  # (n, n_basis)
    Bs  = gam_model._scaler.transform(
        np.hstack([B, np.zeros((len(t_all),
                   gam_model._n_fourier_cols))])
    )[:, :B.shape[1]]
    coef = gam_model._model.coef_[:B.shape[1]]
    weighted = Bs * coef  # each column = basis × coefficient

    # Show top-N bases by variance of weighted contribution
    variances  = np.var(weighted, axis=0)
    top_cols   = np.argsort(variances)[::-1][:top_n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("GAM — Spline Basis Functions (Trend Component)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for ci in top_cols:
        ax.plot(ts["week"], weighted[:, ci], alpha=0.6, linewidth=1)
    ax.set_ylabel("Weighted Basis Contribution (carloads)")
    ax.set_title(f"Top-{top_n} Spline Bases by Variance")

    ax = axes[1]
    trend, seas = gam_model.predict_components(t_all)
    ax.plot(ts["week"], trend/1e6, color=PALETTE["GAM"], linewidth=2.2,
            label="Trend")
    ax.set_xlabel("Week"); ax.set_ylabel("Carloads (millions)")
    ax.set_title("Reconstructed Trend (Sum of All Spline Bases)")
    ax.legend()

    plt.tight_layout()
    _save(fig, "fig13_gam_spline_basis.png")


# ── Fig 13: Hyperparameter sensitivity ───────────────────────────
def plot_hyperparameter_sensitivity(ts, y_all, t_all, splits):
    """
    Grid of GAM alpha vs n_knots: cross-validated RMSE surface.
    """
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
    knots  = [8, 12, 16, 20, 25]
    grid_rmse = np.full((len(alphas), len(knots)), np.nan)

    print("  Computing GAM hyperparameter grid…")
    for i, alpha in enumerate(alphas):
        for j, nk in enumerate(knots):
            fold_rmses = []
            for sp in splits[:6]:   # use first 6 folds only for speed
                tr, te = sp["train_idx"], sp["test_idx"]
                try:
                    gam = GAMSpline(n_knots_trend=nk, n_fourier=10, alpha=alpha)
                    gam.fit(t_all[tr], y_all[tr])
                    yp = gam.predict(t_all[te])
                    fold_rmses.append(np.sqrt(mean_squared_error(y_all[te], yp)))
                except:
                    pass
            if fold_rmses:
                grid_rmse[i, j] = np.mean(fold_rmses)
        print(f"    alpha={alpha} done")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid_rmse/1e3, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(knots)));  ax.set_xticklabels(knots)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel("Number of Knots"); ax.set_ylabel("Ridge α (smoothing)")
    ax.set_title("GAM Hyperparameter Sensitivity — CV RMSE (thousands)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE (K carloads)")
    for i in range(len(alphas)):
        for j in range(len(knots)):
            if not np.isnan(grid_rmse[i, j]):
                ax.text(j, i, f"{grid_rmse[i,j]/1e3:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if grid_rmse[i,j] > np.nanmedian(grid_rmse) else "black")
    plt.tight_layout()
    _save(fig, "fig14_gam_hyperparameter.png")


# ── Fig 14: Summary scorecard ─────────────────────────────────────
def plot_scorecard(df_cv):
    """
    Radar / spider chart + bar chart comparing models across metrics.
    """
    metrics_plot = ["MAE","RMSE","MAPE","R2"]
    summary = df_cv.groupby("model")[metrics_plot].mean()
    order   = ["GAM","Prophet","SARIMA","OLS-FE"]
    order   = [m for m in order if m in summary.index]
    summary = summary.loc[order]

    # Normalise 0→1 (0 = best for each metric)
    norm = summary.copy()
    for col in [c for c in metrics_plot if c != "R2"]:
        rng = norm[col].max() - norm[col].min() + 1e-12
        norm[col] = (norm[col] - norm[col].min()) / rng
    norm["R2"] = 1.0 - (norm["R2"] - norm["R2"].min()) / (norm["R2"].max() - norm["R2"].min() + 1e-12)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Model Comparison Scorecard — 5×2 CV",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Radar
    ax_radar = fig.add_subplot(gs[0], polar=True)
    labels   = metrics_plot
    N        = len(labels)
    angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles  += angles[:1]

    for mdl in order:
        vals = norm.loc[mdl].values.tolist()
        vals += vals[:1]
        ax_radar.plot(angles, vals, color=PALETTE.get(mdl,"grey"),
                      linewidth=2, label=mdl)
        ax_radar.fill(angles, vals, color=PALETTE.get(mdl,"grey"), alpha=0.08)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Normalised Error Radar\n(lower = better)", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    # Bar chart of RMSE and MAPE side-by-side
    ax_bar = fig.add_subplot(gs[1])
    x = np.arange(len(order))
    w = 0.35
    rmse_vals = summary["RMSE"].values / 1e3
    mape_vals = summary["MAPE"].values
    bars1 = ax_bar.bar(x - w/2, rmse_vals, w,
                       color=[PALETTE.get(m,"grey") for m in order],
                       alpha=0.85, label="RMSE (K)")
    ax_bar.bar_label(bars1, fmt="%.0f", padding=2, fontsize=8)
    ax2b = ax_bar.twinx()
    bars2 = ax2b.bar(x + w/2, mape_vals, w,
                     color=[PALETTE.get(m,"grey") for m in order],
                     alpha=0.45, hatch="//", label="MAPE (%)")
    ax2b.bar_label(bars2, fmt="%.1f%%", padding=2, fontsize=8)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(order)
    ax_bar.set_ylabel("RMSE (thousands)"); ax2b.set_ylabel("MAPE (%)")
    ax_bar.set_title("RMSE (solid) vs MAPE (hatched)")
    plt.tight_layout()
    _save(fig, "fig15_scorecard.png")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT BLOCK
# ──────────────────────────────────────────────────────────────────────────────
# 
# PURPOSE: exposes a single public function, `build_fitted_models()`, that
#   external scripts can call after doing `import freight_pipeline as fp`.
#   The function re-runs the minimal fitting logic (no CV, no plots) and
#   returns every artefact a downstream script needs:
#       • fitted model objects              – predict on any horizon
#       • in-sample fitted value arrays     – residual / diagnostic work
#       • aggregate time-series DataFrame   – dates, actuals, week_num
#       • raw panel DataFrame               – needed by OLS-FE forecaster
#       • cross-validation results table    – if CV was already run
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
    Load the freight data and fit all four models on the full training window.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file (default: the module-level DATA_PATH constant).
    run_cv : bool
        If True, also runs the 5×2 repeated time-series CV and includes the
        resulting DataFrame in the returned dict under key ``"df_cv"``.
        CV adds significant runtime; set False for quick downstream use.
    n_repeats : int
        Number of CV repeats (only used when run_cv=True).
    k_folds : int
        Number of CV folds per repeat (only used when run_cv=True).
    seasonal_s : int
        SARIMA seasonal period in weeks (default 52).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys
    ──────────────
    "ts"            : pd.DataFrame   – aggregate weekly time-series
    "panel"         : pd.DataFrame   – full panel data (one row per
                                       company × commodity × week)
    "y_all"         : np.ndarray     – aggregate carload totals (float)
    "t_all"         : np.ndarray     – integer week index (0, 1, 2, …)

    "gam_model"     : GAMSpline      – fitted on full sample
    "prophet_model" : ProphetLite    – fitted on full sample
    "sarima_model"  : SARIMALite     – fitted on full sample
    "fe_model"      : FixedEffectsOLS – fitted on full sample

    "y_fit_gam"     : np.ndarray     – GAM in-sample fitted values
    "y_fit_prophet" : np.ndarray     – Prophet in-sample fitted values
    "y_fit_sarima"  : np.ndarray     – SARIMA in-sample (≈ actual;
                                       differenced-space residuals)
    "y_fit_fe"      : np.ndarray     – OLS-FE in-sample fitted values
                                       (aggregated from panel-level preds)

    "df_cv"         : pd.DataFrame   – CV fold metrics (None if run_cv=False)
    "summary_cv"    : pd.DataFrame   – multi-level summary table
                                       (None if run_cv=False)
    """

    # ── Step 1: Load & build data structures ─────────────────────────────────
    if verbose:
        print("[build_fitted_models] Loading data …")

    # Load raw CSV and normalise column names / types
    raw = load_raw_data(data_path)

    # Build the aggregate weekly time-series (one row per week)
    ts = build_aggregate_timeseries(raw)

    # Build the panel (one row per company × commodity × week)
    panel = build_panel_data(raw)

    # Convenience aliases used throughout the pipeline
    y_all = ts["total"].values.astype(float)   # total carloads per week
    t_all = ts["week_num"].values.astype(float) # 0-indexed week counter

    if verbose:
        print(f"  Weeks in sample : {len(ts)}")
        print(f"  Panel rows      : {len(panel):,}")
        print(f"  Date range      : {ts['week'].min().date()} → {ts['week'].max().date()}")

    # ── Step 2: Fit GAM on full sample ───────────────────────────────────────
    if verbose:
        print("[build_fitted_models] Fitting GAM …")

    # GAMSpline uses penalised cubic-spline trend + Fourier seasonality
    gam_model = GAMSpline(n_knots_trend=15, n_fourier=12, alpha=1.0)
    gam_model.fit(t_all, y_all)               # fits Ridge on full y_all

    # Produce in-sample fitted values (same t grid as training)
    y_fit_gam = gam_model.predict(t_all)

    # ── Step 3: Fit ProphetLite on full sample ────────────────────────────────
    if verbose:
        print("[build_fitted_models] Fitting ProphetLite …")

    # ProphetLite: piecewise-linear trend + Fourier seasonality, Ridge-estimated
    prophet_model = ProphetLite(n_changepoints=25, n_fourier=10, alpha=0.5)
    prophet_model.fit(t_all, y_all)

    # In-sample fitted values
    y_fit_prophet = prophet_model.predict(t_all)

    # ── Step 4: Fit SARIMALite on full sample ────────────────────────────────
    if verbose:
        print("[build_fitted_models] Fitting SARIMALite …")

    # SARIMALite: double-differenced SARIMA(2,1,2)×(1,1,1)[52] via L-BFGS-B
    sarima_model = SARIMALite(p=2, q=2, P=1, Q=1, S=seasonal_s)
    sarima_model.fit(y_all)

    # SARIMA residuals live in the differenced space; return actuals as the
    # "in-sample fitted" placeholder (consistent with the main pipeline)
    y_fit_sarima = y_all.copy()

    # ── Step 5: Fit FixedEffectsOLS on full panel ─────────────────────────────
    if verbose:
        print("[build_fitted_models] Fitting Fixed-Effects OLS …")

    # Two-way FE OLS on log-carloads; operates on the panel (not aggregate)
    fe_model = FixedEffectsOLS(n_fourier=6, fit_trend=True)
    fe_model.fit(panel)                         # fits on panel rows

    # Predict at panel level, then aggregate back to weekly totals
    panel = panel.copy()                        # avoid mutating caller's copy
    panel["fe_pred"] = fe_model.predict(panel)  # predicted log-scale, exp'd inside

    # Sum panel-level predictions to weekly aggregate (aligns with ts index)
    y_fit_fe = (
        panel
        .groupby("week")["fe_pred"]
        .sum()
        .reindex(ts["week"])   # enforce exact date alignment with ts
        .values
    )

    # ── Step 6 (optional): Cross-validate all models ──────────────────────────
    df_cv = None
    summary_cv = None

    if run_cv:
        if verbose:
            print("[build_fitted_models] Running 5×2 repeated CV …")

        # Generate time-series CV splits (non-shuffled, forward-chaining)
        splits = make_time_splits(
            n=len(y_all),
            n_repeats=n_repeats,
            k_folds=k_folds,
        )

        # ── CV: GAM ──────────────────────────────────────────────────────────
        def _gam_fit(tr):
            m = GAMSpline(n_knots_trend=15, n_fourier=12, alpha=1.0)
            m.fit(t_all[tr], y_all[tr])
            return m

        def _gam_pred(m, te):
            return m.predict(t_all[te])

        recs_gam, _ = run_cv_loop(
            "GAM", splits, _gam_fit, _gam_pred, y_all, verbose=verbose
        )

        # ── CV: ProphetLite ───────────────────────────────────────────────────
        def _prophet_fit(tr):
            m = ProphetLite(n_changepoints=20, n_fourier=10, alpha=1.0)
            m.fit(t_all[tr], y_all[tr])
            return m

        def _prophet_pred(m, te):
            return m.predict(t_all[te])

        recs_prophet, _ = run_cv_loop(
            "Prophet", splits, _prophet_fit, _prophet_pred, y_all, verbose=verbose
        )

        # ── CV: SARIMALite ────────────────────────────────────────────────────
        def _sarima_fit(tr):
            m = SARIMALite(p=2, q=2, P=1, Q=1, S=seasonal_s)
            m.fit(y_all[tr])
            return m

        def _sarima_pred(m, te):
            return m.predict(h=len(te))

        recs_sarima, _ = run_cv_loop(
            "SARIMA", splits, _sarima_fit, _sarima_pred, y_all, verbose=verbose
        )

        # ── CV: OLS Fixed-Effects (panel-level, aggregated) ───────────────────
        fe_records = []
        week_sorted = sorted(panel["week"].unique())   # ordered list of unique weeks

        for sp in splits:
            fold = sp["fold"]
            # Map integer fold indices → actual week timestamps
            tr_weeks = [week_sorted[i] for i in sp["train_idx"] if i < len(week_sorted)]
            te_weeks = [week_sorted[i] for i in sp["test_idx"]  if i < len(week_sorted)]
            if not tr_weeks or not te_weeks:
                continue  # skip degenerate folds
            tr_panel = panel[panel["week"].isin(tr_weeks)].copy()
            te_panel = panel[panel["week"].isin(te_weeks)].copy()
            try:
                fe_cv = FixedEffectsOLS(n_fourier=6, fit_trend=True)
                fe_cv.fit(tr_panel)
                te_panel = te_panel.copy()
                te_panel["pred"] = fe_cv.predict(te_panel)
                # Aggregate predicted & actual carloads to weekly totals
                pred_agg = (
                    te_panel.groupby("week")["pred"].sum().reindex(te_weeks).values
                )
                true_agg = (
                    panel[panel["week"].isin(te_weeks)]
                    .groupby("week")["total"]
                    .sum()
                    .reindex(te_weeks)
                    .values
                )
                pred_agg = np.clip(pred_agg, 0, None)
                m = compute_metrics(true_agg, pred_agg, "OLS-FE")
                m["fold"] = fold
                fe_records.append(m)
                if verbose:
                    print(
                        f"  [OLS-FE] fold {fold:2d} | "
                        f"RMSE={m['RMSE']:>10,.0f} | MAPE={m['MAPE']:.2f}%"
                    )
            except Exception as exc:
                if verbose:
                    print(f"  [OLS-FE] fold {fold} FAILED: {exc}")

        # Combine all CV records into one long DataFrame
        df_cv = pd.concat(
            [
                pd.DataFrame(recs_gam),
                pd.DataFrame(recs_prophet),
                pd.DataFrame(recs_sarima),
                pd.DataFrame(fe_records),
            ],
            ignore_index=True,
        )
        # Multi-level summary (mean ± std per model per metric)
        summary_cv = summarise_cv(df_cv)

    # ── Step 7: Package and return everything ─────────────────────────────────
    if verbose:
        print("[build_fitted_models] Done. Returning artefact dict.")

    return {
        # ── Raw data ──────────────────────────────────────────────────────────
        "ts":             ts,           # aggregate weekly DataFrame
        "panel":          panel,        # panel DataFrame (with fe_pred column added)
        "y_all":          y_all,        # 1-D array of weekly totals
        "t_all":          t_all,        # 1-D array of week indices (0, 1, 2, …)

        # ── Fitted model objects ──────────────────────────────────────────────
        "gam_model":      gam_model,    # GAMSpline instance (call .predict(t))
        "prophet_model":  prophet_model,# ProphetLite instance (call .predict(t))
        "sarima_model":   sarima_model, # SARIMALite instance (call .predict(h=n))
        "fe_model":       fe_model,     # FixedEffectsOLS instance (call .predict(panel))

        # ── In-sample fitted value arrays (same length as ts / y_all) ─────────
        "y_fit_gam":      y_fit_gam,
        "y_fit_prophet":  y_fit_prophet,
        "y_fit_sarima":   y_fit_sarima, # placeholder — SARIMA residuals are differenced
        "y_fit_fe":       y_fit_fe,

        # ── CV results (None unless run_cv=True) ──────────────────────────────
        "df_cv":          df_cv,
        "summary_cv":     summary_cv,
    }
# ══════════════════════════════════════════════════════════════════
# SECTION 9 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  FREIGHT RAIL CARLOAD FORECASTING PIPELINE")
    print("  Models: GAM · ProphetLite · SARIMALite · FixedEffectsOLS")
    print("=" * 65)

    # ── 9.1  Load & prepare ──────────────────────────────────────
    print("\n[1/7] Loading & preparing data…")
    raw   = load_raw_data(DATA_PATH)
    ts    = build_aggregate_timeseries(raw)
    panel = build_panel_data(raw)

    print(f"  Aggregate obs : {len(ts)}")
    print(f"  Panel obs     : {len(panel):,}")
    print(f"  Date range    : {ts['week'].min().date()} → {ts['week'].max().date()}")
    print(f"  Companies     : {raw['company'].nunique()}")
    print(f"  Commodities   : {raw['code'].nunique()}")

    y_all = ts["total"].values.astype(float)
    t_all = ts["week_num"].values.astype(float)

    # ── 9.2  EDA plots ───────────────────────────────────────────
    print("\n[2/7] Generating EDA visualisations…")
    set_plot_style()
    plot_data_overview(ts, raw)
    plot_seasonality_patterns(ts)

    # ── 9.3  CV splits ───────────────────────────────────────────
    print("\n[3/7] Generating CV splits…")
    splits = make_time_splits(n=len(y_all), n_repeats=N_REPEATS, k_folds=K_FOLDS)
    print(f"  Total folds: {len(splits)}")

    # ── 9.4  Cross-validate all models ───────────────────────────
    print("\n[4/7] Cross-validating models…")

    # ── GAM ──
    print("\n  GAM (Penalized Splines)")
    def gam_fit(tr_idx):
        m = GAMSpline(n_knots_trend=15, n_fourier=12, alpha=1.0)
        m.fit(t_all[tr_idx], y_all[tr_idx])
        return m
    def gam_predict(m, te_idx):
        return m.predict(t_all[te_idx])
    recs_gam, fd_gam = run_cv_loop("GAM", splits, gam_fit, gam_predict, y_all)

    # ── ProphetLite ──
    print("\n  Prophet")
    def prophet_fit(tr_idx):
        m = ProphetLite(n_changepoints=20, n_fourier=10, alpha=1.0)
        m.fit(t_all[tr_idx], y_all[tr_idx])
        return m
    def prophet_predict(m, te_idx):
        return m.predict(t_all[te_idx])
    recs_prophet, fd_prophet = run_cv_loop("Prophet", splits, prophet_fit, prophet_predict, y_all)

    # ── SARIMA ──
    print("\n  SARIMA")
    def sarima_fit(tr_idx):
        m = SARIMALite(p=2, q=2, P=1, Q=1, S=SEASONAL_S)
        m.fit(y_all[tr_idx])
        return m
    def sarima_predict(m, te_idx):
        return m.predict(h=len(te_idx))
    recs_sarima, fd_sarima = run_cv_loop("SARIMA", splits, sarima_fit, sarima_predict, y_all)

    # ── Fixed-Effects OLS (panel) ──
    print("\n  OLS Fixed-Effects (panel)")
    fe_records = []; fd_fe = []
    week_sorted = sorted(panel["week"].unique())

    for sp in splits:
        fold = sp["fold"]
        tr_weeks = [week_sorted[i] for i in sp["train_idx"] if i < len(week_sorted)]
        te_weeks = [week_sorted[i] for i in sp["test_idx"]  if i < len(week_sorted)]
        if not tr_weeks or not te_weeks: continue
        tr_panel = panel[panel["week"].isin(tr_weeks)].copy()
        te_panel = panel[panel["week"].isin(te_weeks)].copy()
        try:
            fe_model = FixedEffectsOLS(n_fourier=6, fit_trend=True)
            fe_model.fit(tr_panel)
            te_preds_panel = fe_model.predict(te_panel)
            te_panel = te_panel.copy(); te_panel["pred"] = te_preds_panel
            pred_agg = te_panel.groupby("week")["pred"].sum().reindex(te_weeks).values
            true_agg = panel[panel["week"].isin(te_weeks)].groupby("week")["total"].sum().reindex(te_weeks).values
            pred_agg = np.clip(pred_agg, 0, None)
            m = compute_metrics(true_agg, pred_agg, "OLS-FE"); m["fold"] = fold
            fe_records.append(m); fd_fe.append((sp["test_idx"], true_agg, pred_agg))
            print(f"  [OLS-FE]  fold {fold:2d} | train={len(tr_panel):,} | "
                  f"RMSE={m['RMSE']:>10,.0f} | MAPE={m['MAPE']:.2f}%")
        except Exception as exc:
            print(f"  [OLS-FE] fold {fold} FAILED: {exc}")

    # ── Combine ──
    df_cv = pd.concat([pd.DataFrame(recs_gam), pd.DataFrame(recs_prophet),
                       pd.DataFrame(recs_sarima), pd.DataFrame(fe_records)],
                      ignore_index=True)
#%%
    # ── 9.5  Summarise CV results ─────────────────────────────────
    print("\n[5/7] CV Summary…")
    summary_tbl = summarise_cv(df_cv)
    print("\n" + "─" * 65)
    print("  5×2 K-FOLD CV SUMMARY  (mean ± std)")
    print("─" * 65)
    all_metrics = ["MAE","MSE","RMSE","MAPE","R2"]
    for mdl in ["GAM","Prophet","SARIMA","OLS-FE"]:
        if mdl not in summary_tbl.index: continue
        print(f"\n  {mdl}")
        row = summary_tbl.loc[mdl]
        for met in all_metrics:
            mu  = row[(met,"mean")]; std = row[(met,"std")]
            if met == "MAPE":
                print(f"    {met:<5}: {mu:>8.2f}%  ± {std:>6.2f}%")
            elif met == "R2":
                print(f"    {met:<5}: {mu:>8.4f}   ± {std:>6.4f}")
            else:
                print(f"    {met:<5}: {mu:>12,.0f}  ± {std:>10,.0f}")

    # ── 9.6  Final fits & forecasts ───────────────────────────────
    print("\n[6/7] Fitting final models & generating forecasts…")

    # GAM full fit
    gam_full = GAMSpline(n_knots_trend=15, n_fourier=12, alpha=1.0)
    gam_full.fit(t_all, y_all)
    y_fit_gam = gam_full.predict(t_all)

    # Prophet full fit
    pm_full = ProphetLite(n_changepoints=25, n_fourier=10, alpha=0.5)
    pm_full.fit(t_all, y_all)
    y_fit_prophet = pm_full.predict(t_all)

    # SARIMA full fit
    sm_full = SARIMALite(p=2, q=2, P=1, Q=1, S=SEASONAL_S)
    sm_full.fit(y_all)
    # SARIMA in-sample: approximate fitted = actual (residuals in differenced space)
    y_fit_sarima = y_all.copy()  # placeholder; SARIMA residuals are in differenced space

    # FE OLS full fit
    fe_full = FixedEffectsOLS(n_fourier=6, fit_trend=True)
    fe_full.fit(panel)
    panel["fe_pred"] = fe_full.predict(panel)
    y_fit_fe = panel.groupby("week")["fe_pred"].sum().reindex(ts["week"]).values

    # Forecasts
    fcast_dates, fcast_gam     = forecast_future(ts, gam_full,    FORECAST_H, "prophet")
    _,           fcast_prophet = forecast_future(ts, pm_full,     FORECAST_H, "prophet")
    _,           fcast_sarima  = forecast_future(ts, sm_full,     FORECAST_H, "sarima")

    # FE forecast: extend panel structure
    last_week_panel = panel[panel["week"] == panel["week"].max()].copy()
    last_week_num   = panel["week_num"].max()
    fe_future_preds = []
    for h in range(1, FORECAST_H + 1):
        fp = last_week_panel.copy()
        fp["week_num"] = last_week_num + h
        fp["week_of_year"] = ((fp["week_of_year"] - 1 + h) % 52) + 1
        fp["week"] = fcast_dates[h - 1]
        fp["company"] = fp["company"].astype(panel["company"].dtype)
        fp["code"]    = fp["code"].astype(panel["code"].dtype)
        fe_future_preds.append(fe_full.predict(fp).sum())
    fcast_fe = np.array(fe_future_preds)

    # ── 9.7  All visualisations ───────────────────────────────────
    print("\n[7/7] Rendering all visualisations…")
    set_plot_style()

    fits   = {"GAM": y_fit_gam, "Prophet": y_fit_prophet, "OLS-FE": y_fit_fe}
    fcasts = {"GAM": fcast_gam, "Prophet": fcast_prophet,
              "SARIMA": fcast_sarima, "OLS-FE": fcast_fe}
    fold_data_all = {
        "GAM": fd_gam, "Prophet": fd_prophet,
        "SARIMA": fd_sarima, "OLS-FE": fd_fe
    }
    residuals_dict = {
        "GAM":     y_all - y_fit_gam,
        "Prophet": y_all - y_fit_prophet,
        "OLS-FE":  y_all - y_fit_fe,
    }

    plot_cv_violins(df_cv)
    plot_cv_boxplots(df_cv)
    plot_metric_heatmap(df_cv)
    plot_fold_trajectory(df_cv)
    plot_actual_vs_predicted(fold_data_all)
    plot_fit_and_forecast(ts, fits, fcast_dates, fcasts)
    plot_decomposition_comparison(ts, gam_full, pm_full, t_all)
    plot_residuals_panel(ts, residuals_dict)
    plot_forecast_comparison(ts, fcast_dates, fcasts)
    plot_fe_coefficients(fe_full.coef_df_, top_n=20)
    plot_gam_spline_basis(gam_full, t_all, ts)
    plot_hyperparameter_sensitivity(ts, y_all, t_all, splits)
    plot_scorecard(df_cv)

    # ── Save CV table ─────────────────────────────────────────────
    df_cv.to_csv(f"{OUTPUT_DIR}/cv_results.csv", index=False)
    summary_tbl.to_csv(f"{OUTPUT_DIR}/cv_summary.csv")
    print("\n  ✓ CV tables saved")

    print("\n" + "=" * 65)
    print("  Pipeline complete. Outputs →", OUTPUT_DIR)
    print("=" * 65)

# %%
