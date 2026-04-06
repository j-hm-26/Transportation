"""
forecast_viz.py
===============
Standalone forecast visualisation + export script.
Imports freight_pipeline as a module and uses build_fitted_models()
to obtain fitted model objects, then generates user-controlled plots
and optionally exports predictions to CSV.

USAGE (command-line)
--------------------
# Default: all models, 26-week horizon, show plots interactively
python forecast_viz.py

# Custom horizon and model subset
python forecast_viz.py --horizon 52 --models GAM Prophet --export

# Full options
python forecast_viz.py \
    --data    "Data/Weekly_Cargo_Data_2023_2026_clean_april6.csv" \
    --horizon 26 \
    --models  GAM Prophet SARIMA OLS-FE \
    --history 78 \
    --ci      0.07 \
    --outdir  "Visuals/custom_forecast" \
    --export \
    --no-show

PREREQUISITES
-------------
• freight_pipeline.py must be importable (same directory or on sys.path).
• build_fitted_models() must be present in freight_pipeline.py
  (paste the export block above into that file first).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse          # command-line argument parsing
import os               # directory creation
import sys              # path manipulation

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")    # non-interactive backend; switched later if --show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Local pipeline import ─────────────────────────────────────────────────────
# Ensure the directory containing freight_pipeline.py is on the import path.
# Edit this path if the two files live in different directories.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import freight_pipeline as fp   # gives access to model classes + helpers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CONFIGURATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Default colour palette (mirrors freight_pipeline.py PALETTE)
PALETTE = {
    "GAM":     "#9B59B6",
    "SARIMA":  "#E07B39",
    "Prophet": "#3B82C4",
    "OLS-FE":  "#2DBD6E",
}
BG_COLOR   = "#F8F9FA"   # figure background
GRID_COLOR = "#E2E8F0"   # grid lines
ACCENT     = "#6C63FF"   # generic highlight colour

# All model keys recognised by this script
ALL_MODELS = ["GAM", "Prophet", "SARIMA", "OLS-FE"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    All arguments have sensible defaults so the script runs with zero flags.
    """
    parser = argparse.ArgumentParser(
        description="Freight carload forecast visualisation & export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path to the raw data CSV (forwarded to build_fitted_models)
    parser.add_argument(
        "--data",
        type=str,
        default=fp.DATA_PATH,
        help="Path to the Weekly_Cargo_Data CSV.",
    )

    # Forecast horizon in weeks (controls how far ahead each model predicts)
    parser.add_argument(
        "--horizon",
        type=int,
        default=26,
        help="Forecast horizon in weeks (e.g. 26 = ~6 months, 52 = 1 year).",
    )

    # Subset of models to visualise; must be a subset of ALL_MODELS
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Which models to include in plots and CSV export.",
    )

    # How many historical weeks to show in the zoomed forecast plot
    parser.add_argument(
        "--history",
        type=int,
        default=78,
        help="Number of historical weeks to display in the zoomed forecast chart.",
    )

    # Approximate ± prediction-interval width expressed as a fraction of the point forecast
    parser.add_argument(
        "--ci",
        type=float,
        default=0.07,
        help="Fractional width of the shaded ±CI band (e.g. 0.07 = ±7%%).",
    )

    # Directory where PNG files and the optional CSV are saved
    parser.add_argument(
        "--outdir",
        type=str,
        default="Visuals/forecast_viz",
        help="Output directory for plots and CSV.",
    )

    # Whether to also export forecast values to a CSV file
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="If set, write forecast values to <outdir>/forecasts.csv.",
    )

    # Suppress the interactive plt.show() call (useful in batch / headless runs)
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="Suppress interactive plot window (always save PNGs).",
    )

    # Whether to also run the 5×2 CV (slow; adds CV metric panel to outputs)
    parser.add_argument(
        "--run-cv",
        action="store_true",
        default=False,
        help="Run 5×2 repeated CV and include a metric summary panel.",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – FORECAST GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecasts(
    artefacts: dict,
    models: list[str],
    horizon: int,
) -> tuple[pd.DatetimeIndex, dict]:
    """
    Call each fitted model's predict method for the requested horizon.

    Parameters
    ----------
    artefacts : dict
        Output of fp.build_fitted_models().
    models : list[str]
        Which model keys to forecast (subset of ALL_MODELS).
    horizon : int
        Number of weeks ahead to forecast.

    Returns
    -------
    fcast_dates : pd.DatetimeIndex
        Weekly dates corresponding to the forecast period.
    forecasts   : dict[str, np.ndarray]
        Mapping model_name → 1-D array of forecast carload totals.
    """
    ts    = artefacts["ts"]     # aggregate time-series DataFrame
    panel = artefacts["panel"]  # panel DataFrame (needed for OLS-FE)

    # Build the sequence of future week dates starting the week after the data ends
    last_date   = ts["week"].iloc[-1]
    fcast_dates = pd.date_range(
        start   = last_date + pd.Timedelta(weeks=1),
        periods = horizon,
        freq    = "W-WED",   # same weekday anchor used in freight_pipeline.py
    )

    forecasts = {}   # will hold {model_name: np.ndarray}

    # ── GAM forecast ─────────────────────────────────────────────────────────
    if "GAM" in models:
        # GAMSpline.predict() accepts a 1-D array of week indices
        t_future = np.arange(
            ts["week_num"].iloc[-1] + 1,
            ts["week_num"].iloc[-1] + 1 + horizon,
            dtype=float,
        )
        raw_pred = artefacts["gam_model"].predict(t_future)
        forecasts["GAM"] = np.clip(raw_pred, 0, None)  # carloads cannot be negative

    # ── Prophet forecast ──────────────────────────────────────────────────────
    if "Prophet" in models:
        t_future = np.arange(
            ts["week_num"].iloc[-1] + 1,
            ts["week_num"].iloc[-1] + 1 + horizon,
            dtype=float,
        )
        raw_pred = artefacts["prophet_model"].predict(t_future)
        forecasts["Prophet"] = np.clip(raw_pred, 0, None)

    # ── SARIMA forecast ───────────────────────────────────────────────────────
    if "SARIMA" in models:
        # SARIMALite.predict(h) returns h steps ahead from the end of training
        raw_pred = artefacts["sarima_model"].predict(h=horizon)
        forecasts["SARIMA"] = np.clip(raw_pred, 0, None)

    # ── OLS Fixed-Effects forecast ────────────────────────────────────────────
    if "OLS-FE" in models:
        # Strategy: clone the last observed week's panel rows, then roll forward
        # the week_num and week_of_year fields for each forecast step.
        last_week_panel = panel[panel["week"] == panel["week"].max()].copy()
        last_week_num   = panel["week_num"].max()

        fe_preds = []
        for h in range(1, horizon + 1):
            # Build a one-week panel slice at step h
            fp_slice = last_week_panel.copy()
            fp_slice["week_num"]     = last_week_num + h          # advance time index
            fp_slice["week_of_year"] = ((fp_slice["week_of_year"] - 1 + h) % 52) + 1
            fp_slice["week"]         = fcast_dates[h - 1]         # actual date stamp

            # Preserve category dtypes so the model's dummy builder works correctly
            fp_slice["company"] = fp_slice["company"].astype(panel["company"].dtype)
            fp_slice["code"]    = fp_slice["code"].astype(panel["code"].dtype)

            # Sum panel-level predictions to get the aggregate carload total
            fe_preds.append(artefacts["fe_model"].predict(fp_slice).sum())

        forecasts["OLS-FE"] = np.clip(np.array(fe_preds), 0, None)

    return fcast_dates, forecasts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_forecasts_csv(
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    ts: pd.DataFrame,
    outdir: str,
) -> str:
    """
    Write forecast values to a tidy CSV file with one row per (date, model).

    Columns
    -------
    week            : forecast date (Wednesday of each future week)
    model           : model name ("GAM", "Prophet", "SARIMA", "OLS-FE")
    forecast_carloads : point forecast in raw carload units
    forecast_millions : same value scaled to millions for readability

    The file also includes in-sample summary statistics (last 4 weeks of
    actuals) as a reference block at the top of the long-form table.

    Parameters
    ----------
    fcast_dates : pd.DatetimeIndex
        Dates of the forecast period.
    forecasts : dict
        {model_name: np.ndarray} from generate_forecasts().
    ts : pd.DataFrame
        Aggregate time-series (for the last-actual reference rows).
    outdir : str
        Directory in which to write "forecasts.csv".

    Returns
    -------
    str
        Full path to the saved CSV file.
    """
    rows = []

    # ── Build the long-form forecast table ───────────────────────────────────
    for model_name, values in forecasts.items():
        for date, val in zip(fcast_dates, values):
            rows.append(
                {
                    "week":               date,
                    "model":              model_name,
                    "type":               "forecast",          # distinguishes from actual rows
                    "forecast_carloads":  round(float(val), 2),
                    "forecast_millions":  round(float(val) / 1e6, 6),
                }
            )

    # ── Append last 4 weeks of actuals for context ────────────────────────────
    # These rows make it easy to plot a seamless actual-to-forecast transition
    for _, row in ts.tail(4).iterrows():
        rows.append(
            {
                "week":               row["week"],
                "model":              "Actual",
                "type":               "actual",
                "forecast_carloads":  round(float(row["total"]), 2),
                "forecast_millions":  round(float(row["total"]) / 1e6, 6),
            }
        )

    # ── Assemble DataFrame, sort, and write ───────────────────────────────────
    df_out = (
        pd.DataFrame(rows)
        .sort_values(["week", "model"])  # chronological, then alphabetical by model
        .reset_index(drop=True)
    )

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "forecasts.csv")
    df_out.to_csv(csv_path, index=False)

    print(f"  ✓ Forecast CSV saved → {csv_path}  ({len(df_out)} rows)")
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _set_style():
    """Apply the shared visual theme (mirrors freight_pipeline.py)."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.facecolor":  BG_COLOR,
            "axes.facecolor":    BG_COLOR,
            "grid.color":        GRID_COLOR,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.labelsize":    10,
            "axes.titlesize":    11,
        }
    )


def _save_fig(fig: plt.Figure, outdir: str, fname: str, show: bool):
    """
    Save a figure to <outdir>/<fname> and optionally display it.

    Parameters
    ----------
    fig    : the matplotlib Figure to save
    outdir : output directory (must already exist)
    fname  : filename including extension (e.g. "forecast_ribbon.png")
    show   : if True, call plt.show() before closing
    """
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – PLOT A: ZOOMED FORECAST RIBBON
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_ribbon(
    ts: pd.DataFrame,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list[str],
    horizon: int,
    history_weeks: int,
    ci_pct: float,
    outdir: str,
    show: bool,
):
    """
    Zoomed view showing the last `history_weeks` of actuals plus the full
    forecast period for each selected model, with a ±ci_pct shaded band.

    The vertical dashed line marks the train/forecast boundary.

    Parameters
    ----------
    ts            : aggregate time-series DataFrame
    fcast_dates   : future week dates (DatetimeIndex, length == horizon)
    forecasts     : {model: np.ndarray} point forecasts
    models        : model names to draw (controls which appear in legend)
    horizon       : forecast horizon in weeks (for title)
    history_weeks : how many past weeks to display on the left
    ci_pct        : fractional half-width of the shaded band (e.g. 0.07 = ±7%)
    outdir        : directory to save the PNG
    show          : whether to call plt.show()
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # ── Plot recent actuals ───────────────────────────────────────────────────
    ax.plot(
        ts["week"].iloc[-history_weeks:],          # x: week dates
        ts["total"].iloc[-history_weeks:] / 1e6,   # y: carloads in millions
        color="#333333",
        linewidth=1.8,
        label="Actual",
        zorder=5,                                  # draw actuals on top
    )

    # ── Plot model forecasts ──────────────────────────────────────────────────
    markers = {"GAM": "o", "Prophet": "s", "SARIMA": "^", "OLS-FE": "D"}

    for mdl in models:
        if mdl not in forecasts:
            continue
        fcast = forecasts[mdl]
        color = PALETTE.get(mdl, ACCENT)
        mk    = markers.get(mdl, "o")

        # Shaded confidence band: ± ci_pct of point forecast
        ax.fill_between(
            fcast_dates,
            fcast / 1e6 * (1 - ci_pct),   # lower bound
            fcast / 1e6 * (1 + ci_pct),   # upper bound
            alpha=0.14,
            color=color,
            label=None,                   # don't add band to legend
        )

        # Point forecast line + markers
        ax.plot(
            fcast_dates,
            fcast / 1e6,
            color=color,
            linewidth=2.4,
            marker=mk,
            markersize=5,
            label=f"{mdl} forecast",
        )

    # ── Vertical boundary line ────────────────────────────────────────────────
    boundary = ts["week"].iloc[-1]
    ax.axvline(boundary, color="black", linestyle=":", linewidth=1.3, alpha=0.5)
    ax.text(
        boundary,
        ax.get_ylim()[1] * 0.98,
        "  Forecast →",
        fontsize=9,
        color="grey",
        va="top",
    )

    # ── Labels & legend ───────────────────────────────────────────────────────
    ci_label = f"±{int(ci_pct * 100)}%"
    ax.set_title(
        f"{horizon}-Week Freight Carload Forecast — {', '.join(models)}  ({ci_label} PI)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Carloads (millions)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_A_forecast_ribbon.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PLOT B: IN-SAMPLE FIT + FORECAST (FULL TIME-SERIES)
# ══════════════════════════════════════════════════════════════════════════════

def plot_full_fit_and_forecast(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list[str],
    outdir: str,
    show: bool,
):
    """
    Full time-series view showing:
      • Raw actuals (grey fill + line)
      • In-sample model fits (coloured lines, left panel)
      • Forecast period (right of boundary, same colour)
      • Residual panel below (GAM and Prophet only, if selected)

    Parameters
    ----------
    ts          : aggregate time-series DataFrame (actuals)
    artefacts   : dict from build_fitted_models() (for in-sample fitted arrays)
    fcast_dates : future week dates
    forecasts   : {model: np.ndarray} point forecasts
    models      : model names to draw
    outdir      : output directory
    show        : whether to call plt.show()
    """
    y_all = artefacts["y_all"]

    # ── Collect in-sample fitted value arrays for selected models ─────────────
    # GAM and Prophet have full fitted arrays; SARIMA returns actuals as placeholder
    insample_key_map = {
        "GAM":     "y_fit_gam",
        "Prophet": "y_fit_prophet",
        "SARIMA":  "y_fit_sarima",
        "OLS-FE":  "y_fit_fe",
    }
    fits = {
        mdl: artefacts[insample_key_map[mdl]]
        for mdl in models
        if insample_key_map.get(mdl) in artefacts
    }

    # Decide whether to draw the residual panel (needs at least one non-SARIMA model)
    residual_models = [m for m in models if m in ("GAM", "Prophet", "OLS-FE")]
    n_panels        = 2 if residual_models else 1

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(18, 11 if n_panels == 2 else 7),
        gridspec_kw={"height_ratios": [3, 1.4] if n_panels == 2 else [1]},
    )
    if n_panels == 1:
        axes = [axes]   # always index as axes[0]

    # ── Upper panel: actuals + fits + forecasts ───────────────────────────────
    ax = axes[0]

    # Actual carloads (fill for visual weight, thin line for precision)
    ax.fill_between(ts["week"], y_all / 1e6, alpha=0.15, color="#999999")
    ax.plot(ts["week"], y_all / 1e6, color="#333333", linewidth=0.9,
            label="Actual", alpha=0.8, zorder=1)

    # Model-specific line styles (differentiates overlapping lines)
    styles = {
        "GAM":     ("-",  1.8),
        "Prophet": ("--", 1.8),
        "SARIMA":  ("-.", 1.6),
        "OLS-FE":  (":",  1.6),
    }
    markers = {"GAM": "o", "Prophet": "s", "SARIMA": "^", "OLS-FE": "D"}

    for mdl, fit_vals in fits.items():
        # Skip SARIMA in-sample (it's a copy of actuals — uninformative to plot)
        if mdl == "SARIMA":
            continue
        ls, lw = styles.get(mdl, ("-", 1.5))
        ax.plot(
            ts["week"], fit_vals / 1e6,
            color=PALETTE.get(mdl, ACCENT),
            linewidth=lw, linestyle=ls, alpha=0.82,
            label=f"{mdl} (in-sample fit)",
        )

    # Forecast lines (right of boundary)
    for mdl in models:
        if mdl not in forecasts:
            continue
        mk = markers.get(mdl, "o")
        ax.plot(
            fcast_dates, forecasts[mdl] / 1e6,
            color=PALETTE.get(mdl, ACCENT),
            linewidth=2.3, marker=mk, markersize=4,
            label=f"{mdl} forecast",
        )

    # Boundary line
    boundary = ts["week"].iloc[-1]
    ax.axvline(boundary, color="black", linestyle=":", linewidth=1.3, alpha=0.5)
    ax.text(boundary, ax.get_ylim()[1] * 0.97, "  Forecast →",
            fontsize=9, color="grey")

    ax.set_title(
        "Total Weekly Freight Carloads — In-Sample Fit + Forecast",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Carloads (millions)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    # ── Lower panel: residuals ────────────────────────────────────────────────
    if n_panels == 2:
        ax2 = axes[1]
        for mdl in residual_models:
            resid = (y_all - fits[mdl]) / 1e3   # convert to thousands
            ax2.plot(
                ts["week"], resid,
                color=PALETTE.get(mdl, ACCENT),
                alpha=0.75, linewidth=0.85,
                label=f"{mdl} residual",
            )
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Week")
        ax2.set_ylabel("Residual (thousands)")
        ax2.set_title("In-Sample Residuals")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_B_full_fit_forecast.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – PLOT C: MODEL COMPARISON SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    ts: pd.DataFrame,
    models: list[str],
    history_weeks: int,
    outdir: str,
    show: bool,
):
    """
    One sub-panel per selected model, each showing the last `history_weeks`
    of actuals and that model's forecast with a shaded band.
    Provides an easy side-by-side visual without clutter from overlapping lines.

    Parameters
    ----------
    fcast_dates   : future dates (DatetimeIndex)
    forecasts     : {model: np.ndarray}
    ts            : aggregate time-series
    models        : model names to plot
    history_weeks : historical weeks visible in each sub-panel
    outdir        : output directory
    show          : whether plt.show() is called
    """
    n = len(models)
    if n == 0:
        return

    # Arrange sub-panels in a 2-column grid
    ncols = min(n, 2)
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10 * ncols, 5 * nrows),
        squeeze=False,    # always 2-D axes array
    )
    fig.suptitle(
        "Per-Model Forecast View — Recent History + Projections",
        fontsize=14, fontweight="bold",
    )

    # Flatten axes for easy iteration
    ax_flat = axes.flatten()

    hist_dates = ts["week"].iloc[-history_weeks:]   # slice for x-axis limit
    hist_vals  = ts["total"].iloc[-history_weeks:]  # slice for actual values

    for i, mdl in enumerate(models):
        ax = ax_flat[i]
        color = PALETTE.get(mdl, ACCENT)

        # Actual history
        ax.plot(hist_dates, hist_vals / 1e6, color="#555555",
                linewidth=1.5, label="Actual", alpha=0.85)

        # Forecast (with ±7% shaded band)
        if mdl in forecasts:
            fcast = forecasts[mdl]
            ax.fill_between(
                fcast_dates,
                fcast / 1e6 * 0.93,
                fcast / 1e6 * 1.07,
                alpha=0.18, color=color,
            )
            ax.plot(
                fcast_dates, fcast / 1e6,
                color=color, linewidth=2.4,
                marker="o", markersize=4,
                label=f"{mdl} forecast",
            )

        # Vertical boundary
        boundary = ts["week"].iloc[-1]
        ax.axvline(boundary, color="black", linestyle=":", linewidth=1.2, alpha=0.5)

        ax.set_title(mdl, fontsize=12, fontweight="bold",
                     color=PALETTE.get(mdl, "black"))
        ax.set_xlabel("Week")
        ax.set_ylabel("Carloads (millions)")
        ax.legend(fontsize=8)

    # Hide any unused sub-panels (if n is odd)
    for j in range(n, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_C_model_comparison.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – PLOT D: FORECAST DECOMPOSITION (GAM & PROPHET)
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_decomposition(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    models: list[str],
    horizon: int,
    outdir: str,
    show: bool,
):
    """
    For models that expose `predict_components()` (GAM and ProphetLite),
    decompose the in-sample + forecast signal into Trend and Seasonality
    and plot them on a shared time axis.

    Only drawn if at least one of {GAM, Prophet} is in `models`.

    Parameters
    ----------
    ts          : aggregate time-series DataFrame
    artefacts   : dict from build_fitted_models()
    fcast_dates : future week dates (for x-axis extension)
    models      : selected models (filters which decompositions to draw)
    horizon     : forecast horizon (for t_future construction)
    outdir      : output directory
    show        : whether plt.show() is called
    """
    # Identify which decomposable models are selected
    decompose_map = {
        "GAM":     ("gam_model",     "GAM"),
        "Prophet": ("prophet_model", "Prophet"),
    }
    selected = {k: v for k, v in decompose_map.items() if k in models}
    if not selected:
        return  # nothing to draw

    t_all = artefacts["t_all"]

    # Extend t-axis into the forecast period
    t_future = np.arange(
        t_all[-1] + 1,
        t_all[-1] + 1 + horizon,
        dtype=float,
    )
    t_extended  = np.concatenate([t_all, t_future])         # full time axis
    dates_extended = list(ts["week"]) + list(fcast_dates)   # matching dates

    n_models = len(selected)
    fig, axes = plt.subplots(
        n_models, 2,           # one row per model, columns = Trend / Seasonality
        figsize=(16, 5 * n_models),
        sharex=True,
    )
    if n_models == 1:
        axes = axes.reshape(1, 2)   # ensure 2-D shape

    fig.suptitle(
        "Forecast Decomposition — Trend & Seasonal Components",
        fontsize=13, fontweight="bold",
    )

    for row, (mdl_key, (artefact_key, mdl_label)) in enumerate(selected.items()):
        model  = artefacts[artefact_key]
        color  = PALETTE.get(mdl_label, ACCENT)

        # predict_components() returns trend and seasonality over a t-array
        trend, seas = model.predict_components(t_extended)

        # ── Trend ──────────────────────────────────────────────────────────
        ax_t = axes[row, 0]
        # Shade the forecast region
        ax_t.axvspan(fcast_dates[0], fcast_dates[-1], alpha=0.06, color=color)
        ax_t.axvline(ts["week"].iloc[-1], color="black",
                     linestyle=":", linewidth=1, alpha=0.4)
        ax_t.plot(dates_extended, trend / 1e6, color=color, linewidth=2.2)
        ax_t.set_title(f"{mdl_label} — Trend Component")
        ax_t.set_ylabel("Carloads (millions)")

        # ── Seasonality ────────────────────────────────────────────────────
        ax_s = axes[row, 1]
        ax_s.axvspan(fcast_dates[0], fcast_dates[-1], alpha=0.06, color=color)
        ax_s.axvline(ts["week"].iloc[-1], color="black",
                     linestyle=":", linewidth=1, alpha=0.4)
        ax_s.plot(dates_extended, seas / 1e6, color=color,
                  linewidth=1.2, alpha=0.85)
        ax_s.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax_s.set_title(f"{mdl_label} — Seasonal Component")
        ax_s.set_xlabel("Week")

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_D_decomposition.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – PLOT E (OPTIONAL): CV METRIC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_summary(df_cv: pd.DataFrame, models: list[str], outdir: str, show: bool):
    """
    Simple bar chart of mean RMSE and MAPE from the CV results.
    Only called when --run-cv is set and df_cv is not None.

    Parameters
    ----------
    df_cv   : long-form CV results DataFrame (from build_fitted_models)
    models  : model names to include
    outdir  : output directory
    show    : whether plt.show() is called
    """
    # Filter to selected models only
    df_plot = df_cv[df_cv["model"].isin(models)].copy()
    if df_plot.empty:
        return

    summary = (
        df_plot
        .groupby("model")[["RMSE", "MAPE"]]
        .agg(["mean", "std"])
        .loc[[m for m in models if m in df_plot["model"].unique()]]  # preserve order
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("5×2 CV Performance Summary", fontsize=13, fontweight="bold")

    model_names = summary.index.tolist()
    x = np.arange(len(model_names))
    colors = [PALETTE.get(m, ACCENT) for m in model_names]

    # ── RMSE bar chart ────────────────────────────────────────────────────────
    ax = axes[0]
    means = summary[("RMSE", "mean")] / 1e3    # convert to thousands
    stds  = summary[("RMSE", "std")]  / 1e3
    bars  = ax.bar(x, means, color=colors, alpha=0.85, width=0.55)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)                        # error bars
    ax.bar_label(bars, fmt="%.0f K", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylabel("RMSE (thousands)")
    ax.set_title("Mean CV RMSE (lower = better)")

    # ── MAPE bar chart ────────────────────────────────────────────────────────
    ax = axes[1]
    means = summary[("MAPE", "mean")]
    stds  = summary[("MAPE", "std")]
    bars  = ax.bar(x, means, color=colors, alpha=0.85, width=0.55)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Mean CV MAPE (lower = better)")

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_E_cv_summary.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrates the full workflow:
      1. Parse CLI arguments.
      2. Call build_fitted_models() to fit all models on the full sample.
      3. Generate forecasts for the requested horizon and model subset.
      4. Render all plots and save PNGs.
      5. Optionally export the forecast table to CSV.
    """
    args = parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.horizon < 1:
        raise ValueError("--horizon must be a positive integer.")
    if args.ci < 0 or args.ci > 1:
        raise ValueError("--ci must be between 0 and 1.")
    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {ALL_MODELS}.")

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 65)
    print("  FREIGHT FORECAST VISUALISATION SCRIPT")
    print(f"  Horizon : {args.horizon} weeks")
    print(f"  Models  : {', '.join(args.models)}")
    print(f"  Output  : {args.outdir}")
    print("=" * 65)

    # ── Step 1: Fit models (import from freight_pipeline) ─────────────────────
    print("\n[1/4] Fitting models via freight_pipeline.build_fitted_models() …")
    artefacts = fp.build_fitted_models(
        data_path=args.data,
        run_cv=args.run_cv,   # CV is expensive; only run if explicitly requested
        verbose=True,
    )
    ts    = artefacts["ts"]    # aggregate time-series
    panel = artefacts["panel"] # panel DataFrame

    # ── Step 2: Generate forecasts ────────────────────────────────────────────
    print(f"\n[2/4] Generating {args.horizon}-week forecasts …")
    fcast_dates, forecasts = generate_forecasts(
        artefacts=artefacts,
        models=args.models,
        horizon=args.horizon,
    )

    # Print a quick numeric summary to the console
    print(f"\n  Forecast period: {fcast_dates[0].date()} → {fcast_dates[-1].date()}")
    print(f"  {'Model':<12}  {'Min (M)':>10}  {'Mean (M)':>10}  {'Max (M)':>10}")
    print("  " + "-" * 46)
    for mdl, vals in forecasts.items():
        mn = vals.min() / 1e6; mu = vals.mean() / 1e6; mx = vals.max() / 1e6
        print(f"  {mdl:<12}  {mn:>10.3f}  {mu:>10.3f}  {mx:>10.3f}")

    # ── Step 3: Render plots ──────────────────────────────────────────────────
    print("\n[3/4] Rendering visualisations …")
    _set_style()

    # Plot A – zoomed ribbon view (most useful at a glance)
    plot_forecast_ribbon(
        ts=ts,
        fcast_dates=fcast_dates,
        forecasts=forecasts,
        models=args.models,
        horizon=args.horizon,
        history_weeks=args.history,
        ci_pct=args.ci,
        outdir=args.outdir,
        show=args.show,
    )

    # Plot B – full time-series with in-sample fit + forecast below
    plot_full_fit_and_forecast(
        ts=ts,
        artefacts=artefacts,
        fcast_dates=fcast_dates,
        forecasts=forecasts,
        models=args.models,
        outdir=args.outdir,
        show=args.show,
    )

    # Plot C – one sub-panel per model (easy side-by-side comparison)
    plot_model_comparison(
        fcast_dates=fcast_dates,
        forecasts=forecasts,
        ts=ts,
        models=args.models,
        history_weeks=args.history,
        outdir=args.outdir,
        show=args.show,
    )

    # Plot D – trend / seasonality decomposition (GAM & Prophet only)
    plot_forecast_decomposition(
        ts=ts,
        artefacts=artefacts,
        fcast_dates=fcast_dates,
        models=args.models,
        horizon=args.horizon,
        outdir=args.outdir,
        show=args.show,
    )

    # Plot E – CV metric summary (only if CV was run)
    if args.run_cv and artefacts["df_cv"] is not None:
        plot_cv_summary(
            df_cv=artefacts["df_cv"],
            models=args.models,
            outdir=args.outdir,
            show=args.show,
        )

    # ── Step 4: Export CSV ────────────────────────────────────────────────────
    if args.export:
        print("\n[4/4] Exporting forecast CSV …")
        export_forecasts_csv(
            fcast_dates=fcast_dates,
            forecasts=forecasts,
            ts=ts,
            outdir=args.outdir,
        )
    else:
        print("\n[4/4] Skipping CSV export (pass --export to enable).")

    print("\n" + "=" * 65)
    print(f"  Done. All outputs → {args.outdir}")
    print("=" * 65)


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
