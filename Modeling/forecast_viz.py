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

# Explicit date window
python forecast_viz.py --start 2026-04-09 --end 2026-10-07 --export
forecast_viz.py --start 2025-11-25 --end 2026-25-02 --export --run-cv

# Full options
python forecast_viz.py \
    --data    "Data/Weekly_Cargo_Data_2017_2026.csv" \
    --horizon 26 \
    --models  GAM Prophet SARIMA OLS-FE \
    --history 78 \
    --ci      0.07 \
    --outdir  "Visuals/custom_forecast/2017_onwards" \
    --export \
    --no-show

PREREQUISITES
-------------
• freight_pipeline.py must be importable (same directory or on sys.path).
• build_fitted_models() must be present in freight_pipeline.py
  (paste the export block into that file first).

PLOTS PRODUCED
--------------
  plot_A_forecast_ribbon.png       – zoomed ribbon: recent history + all model forecasts
  plot_B_full_fit_forecast.png     – full time-series: in-sample fits + forecast + residuals
  plot_B2_zoomed_2024_forecast.png – same as B but x-axis clipped to 2024-onward
  plot_C_model_comparison.png      – one sub-panel per model
  plot_D_decomposition.png         – GAM & Prophet trend / seasonal decomposition
  plot_F_gam_splines_vs_actual.png – faceted GAM spline bases overlaid on actuals
  plot_G_variable_importance.png   – bar chart + heatmap of variable importance per model
  plot_E_cv_summary.png            – CV metric summary (only with --run-cv)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse   # command-line argument parsing
import os         # directory creation / path operations
import sys        # sys.path manipulation for local imports

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive Agg backend (PNG output)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

# ── Local pipeline import ─────────────────────────────────────────────────────
# Insert the directory containing freight_pipeline.py at the front of sys.path
# so that `import freight_pipeline` resolves correctly regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import freight_pipeline_v2 as fp   # model classes, helpers, DATA_PATH constant


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CONFIGURATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette — matches freight_pipeline.py so plots share a visual language
PALETTE = {
    "GAM":     "#9B59B6",
    "SARIMA":  "#E07B39",
#    "Prophet": "#3B82C4",
    "OLS-FE":  "#2DBD6E",
}
BG_COLOR   = "#F8F9FA"   # off-white figure background
GRID_COLOR = "#E2E8F0"   # light grey grid lines
ACCENT     = "#6C63FF"   # fallback highlight colour

# All valid model keys accepted by this script
#ALL_MODELS = ["GAM", "Prophet", "SARIMA", "OLS-FE"]
ALL_MODELS = ["GAM", "SARIMA", "OLS-FE"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    All arguments carry sensible defaults so the script runs with zero flags.
    """
    parser = argparse.ArgumentParser(
        description="Freight carload forecast visualisation & export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--data",
        type=str,
        default=fp.DATA_PATH,          # inherit the path constant from the pipeline
        help="Path to the Weekly_Cargo_Data CSV.",
    )

    # ── Forecast horizon (number of weeks) ────────────────────────────────────
    # Used as a fallback when --end is not provided.
    parser.add_argument(
        "--horizon",
        type=int,
        default=26,
        help="Forecast horizon in weeks (e.g. 26 = ~6 months, 52 = 1 year). "
             "Ignored when --end is provided.",
    )

    # ── Optional explicit date bounds (override --horizon) ────────────────────
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Forecast start date (YYYY-MM-DD). "
             "Defaults to the Wednesday one week after the last data point.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Forecast end date (YYYY-MM-DD). Overrides --horizon when provided.",
    )

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Which models to include in every plot and the CSV export.",
    )

    # ── Visual controls ───────────────────────────────────────────────────────
    parser.add_argument(
        "--history",
        type=int,
        default=78,
        help="Number of historical weeks to show in the zoomed forecast chart.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.07,
        help="Fractional half-width of the shaded +/-PI band (e.g. 0.07 = +/-7%%).",
    )

    # ── Output directory ──────────────────────────────────────────────────────
    parser.add_argument(
        "--outdir",
        type=str,
        default="Visuals/forecast_viz/2017_onwards",
        help="Directory where PNG files and the optional CSV are written.",
    )

    # ── CSV export flag ───────────────────────────────────────────────────────
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="Write forecast values to <outdir>/forecasts.csv.",
    )

    # ── Interactive display flag ───────────────────────────────────────────────
    # --no-show suppresses plt.show(); useful in batch / headless environments.
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="Suppress the interactive plot window (PNGs are always saved).",
    )

    # ── Cross-validation flag ─────────────────────────────────────────────────
    parser.add_argument(
        "--run-cv",
        action="store_true",
        default=False,
        help="Run 5x2 repeated CV inside build_fitted_models and add "
             "a metric summary plot (slow).",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATE RANGE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_forecast_dates(
    args: argparse.Namespace,
    last_data_date: pd.Timestamp,
) -> tuple:
    """
    Determine the forecast date range from CLI args.

    Priority:
      1. Both --start and --end given  → use them directly.
      2. Only --end given              → start defaults to last_data_date + 1 week.
      3. Only --start given            → end = start + (horizon - 1) weeks.
      4. Neither given                 → horizon weeks from last_data_date + 1 week.

    Parameters
    ----------
    args           : parsed argparse.Namespace
    last_data_date : pd.Timestamp of the final week in the training data

    Returns
    -------
    fcast_dates : pd.DatetimeIndex  – weekly Wednesday dates for the forecast window
    horizon     : int               – length of fcast_dates
    """
    # Default start: one week after the last observed data point
    if args.start:
        start_date = pd.Timestamp(args.start)
        if start_date <= last_data_date:
            raise ValueError(
                f"--start ({args.start}) must be after the last data date "
                f"({last_data_date.date()})."
            )
    else:
        start_date = last_data_date + pd.Timedelta(weeks=1)

    # End date: either explicit --end or derived from --horizon
    if args.end:
        end_date = pd.Timestamp(args.end)
        if end_date <= start_date:
            raise ValueError(
                f"--end ({args.end}) must be after --start ({start_date.date()})."
            )
        # Build date range between start and end using the W-WED anchor
        fcast_dates = pd.date_range(start=start_date, end=end_date, freq="W-WED")
    else:
        fcast_dates = pd.date_range(
            start=start_date, periods=args.horizon, freq="W-WED"
        )

    horizon = len(fcast_dates)   # actual length after date arithmetic
    return fcast_dates, horizon


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – FORECAST GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecasts(
    artefacts: dict,
    models: list,
    horizon: int,
    fcast_dates: pd.DatetimeIndex,
) -> dict:
    """
    Call each fitted model's predict method for the requested horizon.

    Parameters
    ----------
    artefacts   : dict returned by fp.build_fitted_models()
    models      : list of model keys to forecast (subset of ALL_MODELS)
    horizon     : number of forecast steps (must equal len(fcast_dates))
    fcast_dates : pd.DatetimeIndex of forecast week dates

    Returns
    -------
    forecasts : dict[str, np.ndarray]
        {model_name: array of length horizon} with non-negative carload totals.
    """
    ts    = artefacts["ts"]      # aggregate weekly time-series DataFrame
    panel = artefacts["panel"]   # panel DataFrame (company x commodity x week)

    # Sanity check — horizon and dates must agree
    assert len(fcast_dates) == horizon, (
        f"horizon={horizon} but len(fcast_dates)={len(fcast_dates)}"
    )

    forecasts = {}   # accumulate {model_name: np.ndarray}

    # ── GAM forecast ──────────────────────────────────────────────────────────
    # GAMSpline.predict(t) takes a 1-D array of integer week indices.
    # We extend the week_num sequence from the last observed index.
    if "GAM" in models:
        t_future = np.arange(
            ts["week_num"].iloc[-1] + 1,           # first future step
            ts["week_num"].iloc[-1] + 1 + horizon, # exclusive upper bound
            dtype=float,
        )
        raw = artefacts["gam_model"].predict(t_future)
        forecasts["GAM"] = np.clip(raw, 0, None)   # carloads cannot be negative

    # ── ProphetLite forecast ──────────────────────────────────────────────────
    # Identical interface to GAM: predict on a t-array.
    if "Prophet" in models:
        t_future = np.arange(
            ts["week_num"].iloc[-1] + 1,
            ts["week_num"].iloc[-1] + 1 + horizon,
            dtype=float,
        )
        raw = artefacts["prophet_model"].predict(t_future)
        forecasts["Prophet"] = np.clip(raw, 0, None)

    # ── SARIMALite forecast ───────────────────────────────────────────────────
    # SARIMALite.predict(h) steps forward h periods from the end of training.
    if "SARIMA" in models:
        raw = artefacts["sarima_model"].predict(h=horizon)
        forecasts["SARIMA"] = np.clip(raw, 0, None)

    # ── OLS Fixed-Effects forecast ────────────────────────────────────────────
    # Clone the last observed week's panel rows, then advance week_num and
    # week_of_year for each step h. Sum panel-level predictions to aggregate.
    if "OLS-FE" in models:
        last_week_panel = panel[panel["week"] == panel["week"].max()].copy()
        last_week_num   = float(panel["week_num"].max())

        fe_preds = []
        for h in range(1, horizon + 1):
            fp_slice = last_week_panel.copy()
            fp_slice["week_num"]     = last_week_num + h                        # advance time index
            fp_slice["week_of_year"] = ((fp_slice["week_of_year"] - 1 + h) % 52) + 1
            fp_slice["week"]         = fcast_dates[h - 1]                       # stamp with actual date

            # Restore categorical dtypes so the FE dummy-builder works correctly
            fp_slice["company"] = fp_slice["company"].astype(panel["company"].dtype)
            fp_slice["code"]    = fp_slice["code"].astype(panel["code"].dtype)

            # Sum panel-level predictions to get aggregate weekly total
            fe_preds.append(artefacts["fe_model"].predict(fp_slice).sum())

        forecasts["OLS-FE"] = np.clip(np.array(fe_preds), 0, None)

    return forecasts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_forecasts_csv(
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    ts: pd.DataFrame,
    outdir: str,
) -> str:
    """
    Write forecast values to a tidy long-form CSV file.

    Schema
    ------
    week              : forecast date (Wednesday of each future week)
    model             : model name
    type              : "forecast" | "actual"
    forecast_carloads : point forecast in raw carload units
    forecast_millions : same value scaled to millions

    The last 4 weeks of actuals are appended as reference rows so that
    downstream tools can draw a seamless actual-to-forecast transition.

    Returns
    -------
    str – full path to the saved file.
    """
    rows = []

    # ── Forecast rows (one per date x model) ──────────────────────────────────
    for model_name, values in forecasts.items():
        for date, val in zip(fcast_dates, values):
            rows.append({
                "week":               date,
                "model":              model_name,
                "type":               "forecast",
                "forecast_carloads":  round(float(val), 2),
                "forecast_millions":  round(float(val) / 1e6, 6),
            })

    # ── Actual reference rows (last 4 weeks) ──────────────────────────────────
    for _, row in ts.tail(4).iterrows():
        rows.append({
            "week":               row["week"],
            "model":              "Actual",
            "type":               "actual",
            "forecast_carloads":  round(float(row["total"]), 2),
            "forecast_millions":  round(float(row["total"]) / 1e6, 6),
        })

    # ── Sort (chronological, then alphabetical by model) and write ─────────────
    df_out = (
        pd.DataFrame(rows)
        .sort_values(["week", "model"])
        .reset_index(drop=True)
    )

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "forecasts.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  Forecast CSV saved -> {csv_path}  ({len(df_out)} rows)")
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _set_style():
    """Apply shared visual theme (mirrors freight_pipeline.py set_plot_style)."""
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


def _save_fig(fig: plt.Figure, outdir: str, fname: str, show: bool):
    """
    Save *fig* to <outdir>/<fname> at 150 dpi and optionally call plt.show().

    Parameters
    ----------
    fig    : matplotlib Figure to save
    outdir : output directory (caller is responsible for os.makedirs)
    fname  : filename including extension, e.g. "plot_A_forecast_ribbon.png"
    show   : if True, call plt.show() before closing the figure
    """
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)


def _draw_boundary(ax, boundary_date: pd.Timestamp, y_top_frac: float = 0.97):
    """
    Draw a vertical dotted boundary line + 'Forecast ->' annotation.

    Parameters
    ----------
    ax             : matplotlib Axes
    boundary_date  : date of the last observed data point
    y_top_frac     : fractional position on the y-axis for the text label
    """
    ax.axvline(boundary_date, color="black", linestyle=":", linewidth=1.3, alpha=0.5)
    ylim = ax.get_ylim()
    ax.text(
        boundary_date,
        ylim[0] + (ylim[1] - ylim[0]) * y_top_frac,
        "  Forecast ->",
        fontsize=9, color="grey", va="top",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PLOT A: ZOOMED FORECAST RIBBON
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_ribbon(
    ts: pd.DataFrame,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list,
    horizon: int,
    history_weeks: int,
    ci_pct: float,
    outdir: str,
    show: bool,
):
    """
    Zoomed view: last *history_weeks* of actuals + full forecast window for
    all selected models, with a +/-ci_pct shaded PI band around each forecast.

    Parameters
    ----------
    ts            : aggregate time-series DataFrame
    fcast_dates   : DatetimeIndex of forecast dates
    forecasts     : {model: np.ndarray} from generate_forecasts()
    models        : model names to draw
    horizon       : forecast length in weeks (used in title only)
    history_weeks : how many historical weeks to show on the left
    ci_pct        : fractional half-width of the PI band
    outdir        : output directory
    show          : whether to call plt.show()
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # ── Historical actuals ────────────────────────────────────────────────────
    ax.plot(
        ts["week"].iloc[-history_weeks:],
        ts["total"].iloc[-history_weeks:] / 1e6,
        color="#333333", linewidth=1.8, label="Actual", zorder=5,
    )

    # ── Model forecasts with PI bands ─────────────────────────────────────────
    markers = {"GAM": "o", "Prophet": "s", "SARIMA": "^", "OLS-FE": "D"}
    for mdl in models:
        if mdl not in forecasts:
            continue
        fcast = forecasts[mdl]
        color = PALETTE.get(mdl, ACCENT)
        mk    = markers.get(mdl, "o")
        # Shaded +/- band around the point forecast
        ax.fill_between(
            fcast_dates,
            fcast / 1e6 * (1 - ci_pct),
            fcast / 1e6 * (1 + ci_pct),
            alpha=0.14, color=color,
        )
        # Point-forecast line with markers
        ax.plot(
            fcast_dates, fcast / 1e6,
            color=color, linewidth=2.4, marker=mk, markersize=5,
            label=f"{mdl} forecast",
        )

    # ── Boundary marker and labels ────────────────────────────────────────────
    boundary = ts["week"].iloc[-1]
    _draw_boundary(ax, boundary)

    ci_label = f"+/-{int(ci_pct * 100)}%"
    ax.set_title(
        f"{horizon}-Week Freight Carload Forecast -- {', '.join(models)}  ({ci_label} PI)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Carloads (millions)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_A_forecast_ribbon.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – PLOT B: IN-SAMPLE FIT + FORECAST (FULL & ZOOMED)
# ══════════════════════════════════════════════════════════════════════════════

def _build_fit_forecast_figure(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list,
) -> plt.Figure:
    """
    Internal helper: construct the fit + forecast Figure object.
    The caller may then adjust x-limits before saving, enabling two saved
    versions: the full-range view (Plot B) and a 2024-onward zoom (Plot B2).

    Returns the unsaved Figure.
    """
    y_all = artefacts["y_all"]   # 1-D array of aggregate weekly carload totals

    # Map model keys to their in-sample fitted-value arrays in artefacts dict
    insample_key_map = {
        "GAM":     "y_fit_gam",
        "Prophet": "y_fit_prophet",
        "SARIMA":  "y_fit_sarima",   # placeholder = actuals; skipped in plotting
        "OLS-FE":  "y_fit_fe",
    }
    # Only include models whose artefact key actually exists
    fits = {
        mdl: artefacts[insample_key_map[mdl]]
        for mdl in models
        if insample_key_map.get(mdl) in artefacts
    }

    # Residual panel: only meaningful for models with genuine fitted arrays
    residual_models = [m for m in models if m in ("GAM", "Prophet", "OLS-FE")]
    n_panels = 2 if residual_models else 1

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(18, 11 if n_panels == 2 else 7),
        gridspec_kw={"height_ratios": [3, 1.4] if n_panels == 2 else [1]},
    )
    if n_panels == 1:
        axes = [axes]   # normalise to list so axes[0] always works

    # ── Upper panel: actuals + in-sample fits + forecasts ─────────────────────
    ax = axes[0]

    # Actuals: light fill for visual weight, thin line for precision
    ax.fill_between(ts["week"], y_all / 1e6, alpha=0.12, color="#999999")
    ax.plot(ts["week"], y_all / 1e6,
            color="#333333", linewidth=0.9, label="Actual", alpha=0.8, zorder=1)

    # Line styles per model (avoids colour-only differentiation)
    styles  = {"GAM": ("-", 1.8), "Prophet": ("--", 1.8),
               "SARIMA": ("-.", 1.6), "OLS-FE": (":", 1.6)}
    markers = {"GAM": "o", "Prophet": "s", "SARIMA": "^", "OLS-FE": "D"}

    # In-sample fit lines (SARIMA's "fit" is a copy of actuals — skip it)
    for mdl, fit_vals in fits.items():
        if mdl == "SARIMA":
            continue
        ls, lw = styles.get(mdl, ("-", 1.5))
        ax.plot(
            ts["week"], fit_vals / 1e6,
            color=PALETTE.get(mdl, ACCENT), linewidth=lw, linestyle=ls,
            alpha=0.82, label=f"{mdl} (in-sample fit)",
        )

    # Forecast lines (to the right of the boundary)
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

    boundary = ts["week"].iloc[-1]
    _draw_boundary(ax, boundary)

    ax.set_title(
        "Total Weekly Freight Carloads -- In-Sample Fit + Forecast",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Carloads (millions)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    # ── Lower panel: residuals ─────────────────────────────────────────────────
    if n_panels == 2:
        ax2 = axes[1]
        for mdl in residual_models:
            resid = (y_all - fits[mdl]) / 1e3   # convert to thousands for readability
            ax2.plot(
                ts["week"], resid,
                color=PALETTE.get(mdl, ACCENT),
                alpha=0.75, linewidth=0.85, label=f"{mdl} residual",
            )
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Week")
        ax2.set_ylabel("Residual (thousands)")
        ax2.set_title("In-Sample Residuals")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_full_fit_and_forecast(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list,
    outdir: str,
    show: bool,
):
    """
    Plot B -- complete time-series spanning the full data range.
    Shows in-sample fits, forecast extension, and residual panel.
    """
    fig = _build_fit_forecast_figure(ts, artefacts, fcast_dates, forecasts, models)
    _save_fig(fig, outdir, "plot_B_full_fit_forecast.png", show)


def plot_zoomed_2024_forecast(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    models: list,
    outdir: str,
    show: bool,
):
    """
    Plot B2 -- identical to Plot B but x-axis clipped to 2024-01-01 onward.
    This zoom level makes seasonal fit quality and near-term forecasts
    far easier to read than the full multi-year view.
    """
    fig = _build_fit_forecast_figure(ts, artefacts, fcast_dates, forecasts, models)

    # Apply the 2024 x-limit cutoff to every Axes in the figure
    zoom_start = pd.Timestamp("2024-01-01")
    zoom_end   = fcast_dates[-1] + pd.Timedelta(weeks=2)   # small right-margin

    for ax in fig.axes:
        ax.set_xlim(zoom_start, zoom_end)
        # Reformat x-tick labels to month-year for readability at this zoom
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "Freight Carloads -- In-Sample Fit + Forecast  (2024 to end of horizon)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, outdir, "plot_B2_zoomed_2024_forecast.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – PLOT C: PER-MODEL COMPARISON GRID
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    fcast_dates: pd.DatetimeIndex,
    forecasts: dict,
    ts: pd.DataFrame,
    models: list,
    history_weeks: int,
    outdir: str,
    show: bool,
):
    """
    One sub-panel per selected model, each showing the last *history_weeks*
    of actuals and that model's forecast with a +/-7% PI band.
    A 2-column grid layout is used for space efficiency.
    """
    n = len(models)
    if n == 0:
        return

    ncols  = min(n, 2)
    nrows  = (n + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10 * ncols, 5 * nrows),
        squeeze=False,   # always a 2-D axes array
    )
    fig.suptitle(
        "Per-Model Forecast View -- Recent History + Projections",
        fontsize=14, fontweight="bold",
    )
    ax_flat  = axes.flatten()
    boundary = ts["week"].iloc[-1]

    hist_dates = ts["week"].iloc[-history_weeks:]
    hist_vals  = ts["total"].iloc[-history_weeks:]

    for i, mdl in enumerate(models):
        ax    = ax_flat[i]
        color = PALETTE.get(mdl, ACCENT)

        # Historical actuals for context
        ax.plot(hist_dates, hist_vals / 1e6,
                color="#555555", linewidth=1.5, label="Actual", alpha=0.85)

        # Forecast + PI band
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
                color=color, linewidth=2.4, marker="o", markersize=4,
                label=f"{mdl} forecast",
            )

        _draw_boundary(ax, boundary)
        ax.set_title(mdl, fontsize=12, fontweight="bold",
                     color=PALETTE.get(mdl, "black"))
        ax.set_xlabel("Week")
        ax.set_ylabel("Carloads (millions)")
        ax.legend(fontsize=8)

    # Hide unused sub-panels when n is odd
    for j in range(n, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_C_model_comparison.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – PLOT D: FORECAST DECOMPOSITION (GAM & PROPHET)
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_decomposition(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    models: list,
    horizon: int,
    outdir: str,
    show: bool,
):
    """
    For models that expose predict_components() (GAMSpline and ProphetLite),
    decompose the full in-sample + forecast signal into Trend and Seasonality.

    Only drawn when at least one of {GAM, Prophet} is in *models*.
    """
    # Decomposable models and their artefact keys
    decompose_map = {
        "GAM":     ("gam_model",     "GAM"),
        "Prophet": ("prophet_model", "Prophet"),
    }
    selected = {k: v for k, v in decompose_map.items() if k in models}
    if not selected:
        return   # nothing to draw; skip silently

    t_all = artefacts["t_all"]

    # Build an extended t-array covering in-sample + forecast period
    t_future   = np.arange(t_all[-1] + 1, t_all[-1] + 1 + horizon, dtype=float)
    t_extended = np.concatenate([t_all, t_future])
    dates_ext  = list(ts["week"]) + list(fcast_dates)

    n_models = len(selected)
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(16, 5 * n_models),
        sharex=True,
    )
    if n_models == 1:
        axes = axes.reshape(1, 2)   # ensure 2-D indexing

    fig.suptitle(
        "Forecast Decomposition -- Trend & Seasonal Components",
        fontsize=13, fontweight="bold",
    )

    boundary = ts["week"].iloc[-1]

    for row, (mdl_key, (artefact_key, mdl_label)) in enumerate(selected.items()):
        model = artefacts[artefact_key]
        color = PALETTE.get(mdl_label, ACCENT)

        # predict_components returns (trend_array, seasonal_array) on t_extended
        trend, seas = model.predict_components(t_extended)

        # ── Trend sub-panel ────────────────────────────────────────────────────
        ax_t = axes[row, 0]
        ax_t.axvspan(fcast_dates[0], fcast_dates[-1], alpha=0.06, color=color)
        ax_t.axvline(boundary, color="black", linestyle=":", linewidth=1, alpha=0.4)
        ax_t.plot(dates_ext, trend / 1e6, color=color, linewidth=2.2)
        ax_t.set_title(f"{mdl_label} -- Trend Component")
        ax_t.set_ylabel("Carloads (millions)")

        # ── Seasonal sub-panel ────────────────────────────────────────────────
        ax_s = axes[row, 1]
        ax_s.axvspan(fcast_dates[0], fcast_dates[-1], alpha=0.06, color=color)
        ax_s.axvline(boundary, color="black", linestyle=":", linewidth=1, alpha=0.4)
        ax_s.plot(dates_ext, seas / 1e6, color=color, linewidth=1.2, alpha=0.85)
        ax_s.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax_s.set_title(f"{mdl_label} -- Seasonal Component")
        ax_s.set_xlabel("Week")

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_D_decomposition.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – PLOT E (OPTIONAL): CV METRIC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_summary(df_cv: pd.DataFrame, models: list, outdir: str, show: bool):
    """
    Bar charts of mean +/- std RMSE and MAPE from the cross-validation results.
    Only called when --run-cv is set and df_cv is not None.
    """
    df_plot = df_cv[df_cv["model"].isin(models)].copy()
    if df_plot.empty:
        return

    # Compute mean and std per model, preserving the requested model order
    summary = (
        df_plot
        .groupby("model")[["RMSE", "MAPE"]]
        .agg(["mean", "std"])
        .loc[[m for m in models if m in df_plot["model"].unique()]]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("5x2 CV Performance Summary", fontsize=13, fontweight="bold")

    model_names = summary.index.tolist()
    x      = np.arange(len(model_names))
    colors = [PALETTE.get(m, ACCENT) for m in model_names]

    # ── RMSE panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    means = summary[("RMSE", "mean")] / 1e3   # convert to thousands
    stds  = summary[("RMSE", "std")]  / 1e3
    bars  = ax.bar(x, means, color=colors, alpha=0.85, width=0.55)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.5)
    ax.bar_label(bars, fmt="%.0f K", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylabel("RMSE (thousands)")
    ax.set_title("Mean CV RMSE (lower = better)")

    # ── MAPE panel ────────────────────────────────────────────────────────────
    ax = axes[1]
    means = summary[("MAPE", "mean")]
    stds  = summary[("MAPE", "std")]
    bars  = ax.bar(x, means, color=colors, alpha=0.85, width=0.55)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Mean CV MAPE (lower = better)")

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_E_cv_summary.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – PLOT F: GAM SPLINE BASES vs ACTUAL DATA (FACETED)
# ══════════════════════════════════════════════════════════════════════════════

def plot_gam_splines_vs_actual(
    ts: pd.DataFrame,
    artefacts: dict,
    fcast_dates: pd.DatetimeIndex,
    horizon: int,
    outdir: str,
    show: bool,
    top_n: int = 8,
):
    """
    Faceted plot: top-N weighted GAM spline basis functions, each shown
    individually overlaid against the normalised raw aggregate carload actuals.

    Layout:  top_n sub-panels in a 2-column grid  +  one summary panel
             showing the reconstructed GAM trend vs raw actuals.

    Spline importance is measured by the variance of each basis's weighted
    contribution (basis_scaled * Ridge_coef), so the most influential bases
    appear first.

    Parameters
    ----------
    ts          : aggregate time-series DataFrame
    artefacts   : dict from fp.build_fitted_models()
    fcast_dates : future dates (extend spline into the forecast region)
    horizon     : forecast horizon (for building t_future)
    outdir      : output directory
    top_n       : number of top-variance spline bases to display (default 8)
    show        : whether to call plt.show()
    """
    # Guard: GAM must have been fitted
    if "gam_model" not in artefacts:
        print("  [plot_gam_splines_vs_actual] GAM model not in artefacts; skipping.")
        return

    gam   = artefacts["gam_model"]
    t_all = artefacts["t_all"]
    y_all = artefacts["y_all"]

    # ── Build t-arrays for in-sample and extended (in-sample + forecast) ──────
    t_future   = np.arange(t_all[-1] + 1, t_all[-1] + 1 + horizon, dtype=float)
    t_extended = np.concatenate([t_all, t_future])
    dates_ext  = list(ts["week"]) + list(fcast_dates)

    # ── Extract and weight the spline basis matrix ─────────────────────────────
    # gam._spline_trend is a fitted sklearn SplineTransformer.
    t2d = t_all.reshape(-1, 1)
    B   = gam._spline_trend.transform(t2d)   # shape (n_obs, n_basis_cols)

    # Build a full-width matrix (trend + zero Fourier padding) to use the
    # fitted scaler, then slice out only the trend columns.
    n_fourier_cols = gam._n_fourier_cols
    pad            = np.zeros((len(t_all), n_fourier_cols))
    B_full_scaled  = gam._scaler.transform(np.hstack([B, pad]))
    B_scaled       = B_full_scaled[:, :B.shape[1]]   # trend columns only

    # Weighted contribution: scaled_basis * corresponding Ridge coefficient
    coef     = gam._model.coef_[:B.shape[1]]
    weighted = B_scaled * coef                        # shape (n_obs, n_basis_cols)

    # Rank bases by variance of their weighted contribution (high = influential)
    variances = np.var(weighted, axis=0)
    top_cols  = np.argsort(variances)[::-1][:top_n]

    # ── Extend bases into the forecast region ─────────────────────────────────
    t2d_ext      = t_extended.reshape(-1, 1)
    B_ext        = gam._spline_trend.transform(t2d_ext)
    pad_ext      = np.zeros((len(t_extended), n_fourier_cols))
    B_ext_scaled = gam._scaler.transform(np.hstack([B_ext, pad_ext]))[:, :B_ext.shape[1]]
    weighted_ext = B_ext_scaled * coef               # extended weighted bases

    # Reconstructed GAM trend over the extended period (for summary panel)
    trend_ext, _ = gam.predict_components(t_extended)

    # ── Figure layout: top_n + 1 summary panel, 2-column grid ─────────────────
    n_facets = top_n + 1          # one per basis + one summary
    ncols    = 2
    nrows    = (n_facets + 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"GAM -- Top-{top_n} Weighted Spline Bases vs Actual Carloads",
        fontsize=13, fontweight="bold",
    )
    ax_flat = axes.flatten()

    # Normalise actuals to [0,1] for overlay against zero-centred basis functions.
    # This avoids y-axis scale conflicts while preserving the temporal shape.
    y_min  = y_all.min()
    y_max  = y_all.max()
    y_norm = (y_all - y_min) / (y_max - y_min + 1e-12)

    boundary = ts["week"].iloc[-1]

    for idx, col_idx in enumerate(top_cols):
        ax    = ax_flat[idx]
        color = PALETTE["GAM"]

        # Normalised actuals as greyed background reference
        ax.fill_between(ts["week"], y_norm, alpha=0.12, color="#999999")
        ax.plot(ts["week"], y_norm,
                color="#999999", linewidth=0.8, alpha=0.6, label="Actual (norm.)")

        # Weighted spline basis function extended through forecast period
        ax.plot(
            dates_ext, weighted_ext[:, col_idx],
            color=color, linewidth=1.6, alpha=0.9,
            label=f"Spline basis {col_idx + 1}",
        )

        # Horizontal zero reference and train/forecast boundary
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.axvline(boundary, color="black", linestyle=":", linewidth=1, alpha=0.4)

        # Title shows basis index and variance (indicates relative importance)
        ax.set_title(
            f"Basis {col_idx + 1}  (var = {variances[col_idx]:.4f})",
            fontsize=9,
        )
        ax.set_xlabel("Week")
        ax.legend(fontsize=7, loc="upper left")

    # ── Summary panel: reconstructed trend vs raw actuals ─────────────────────
    ax_sum = ax_flat[top_n]
    ax_sum.fill_between(ts["week"], y_all / 1e6, alpha=0.12, color="#999999")
    ax_sum.plot(ts["week"], y_all / 1e6,
                color="#333333", linewidth=0.9, label="Actual", alpha=0.8)
    ax_sum.plot(
        dates_ext, trend_ext / 1e6,
        color=PALETTE["GAM"], linewidth=2.2, linestyle="--",
        label="GAM Trend (sum of all splines)",
    )
    ax_sum.axvline(boundary, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax_sum.set_title("Summary: Reconstructed GAM Trend vs Actuals", fontsize=9)
    ax_sum.set_xlabel("Week")
    ax_sum.set_ylabel("Carloads (millions)")
    ax_sum.legend(fontsize=7)

    # Hide any extra empty sub-panels (when n_facets is even there are none)
    for j in range(top_n + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_F_gam_splines_vs_actual.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – PLOT G: VARIABLE IMPORTANCE (BAR CHART + HEATMAP)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_importance(artefacts: dict, models: list) -> pd.DataFrame:
    """
    Build a tidy variable-importance DataFrame for all selected models.

    Importance metric
    -----------------
    Ridge / OLS models: mean absolute standardised coefficient within each
    named feature group.  SARIMA: mean absolute parameter value per component.

    Feature groups
    --------------
    GAM         : Trend (spline), Seasonality (Fourier), Calendar effects
    ProphetLite : Piecewise trend, Seasonality (Fourier)
    SARIMA      : AR(p), MA(q), Seasonal AR(P), Seasonal MA(Q)
    OLS-FE      : Company FE, Commodity FE, Fourier Seasonality, Calendar, Trend

    Returns
    -------
    pd.DataFrame with columns: feature, importance, group, model
    """
    rows = []

    # ── GAM ───────────────────────────────────────────────────────────────────
    if "GAM" in models and "gam_model" in artefacts:
        gam  = artefacts["gam_model"]
        coef = np.abs(gam._model.coef_)    # absolute standardised Ridge coefficients

        k1 = gam._n_trend_cols             # end index of trend (spline) block
        k2 = k1 + gam._n_fourier_cols      # end index of Fourier block

        # Mean |coef| per component group captures overall group influence
        rows += [
            {"feature": "Trend (spline)",
             "importance": float(coef[:k1].mean()),
             "group": "Trend", "model": "GAM"},
            {"feature": "Seasonality (Fourier)",
             "importance": float(coef[k1:k2].mean()),
             "group": "Seasonality", "model": "GAM"},
        ]
        # Calendar effects block (present only when a cal_df was passed to fit)
        if len(coef) > k2:
            rows.append({
                "feature":    "Calendar effects",
                "importance": float(coef[k2:].mean()),
                "group":      "Calendar",
                "model":      "GAM",
            })

    # ── ProphetLite ───────────────────────────────────────────────────────────
    if "Prophet" in models and "prophet_model" in artefacts:
        pm   = artefacts["prophet_model"]
        coef = np.abs(pm.model_.coef_)    # .model_ is the Ridge estimator

        k = pm._n_trend_cols              # end of piecewise-trend column block

        rows += [
            {"feature": "Piecewise trend",
             "importance": float(coef[:k].mean()),
             "group": "Trend", "model": "Prophet"},
            {"feature": "Seasonality (Fourier)",
             "importance": float(coef[k:].mean()),
             "group": "Seasonality", "model": "Prophet"},
        ]

    # ── SARIMALite ────────────────────────────────────────────────────────────
    # params_ layout: [phi_1..p, theta_1..q, Phi_1..P, Theta_1..Q, log_sigma2]
    # Use absolute parameter values as a proxy for importance.
    if "SARIMA" in models and "sarima_model" in artefacts:
        sm  = artefacts["sarima_model"]
        par = sm.params_[:-1]             # exclude log-sigma2 at end

        p, q, P, Q = sm.p, sm.q, sm.P, sm.Q

        # Slice each component group from the parameter vector
        ar_coef  = np.abs(par[:p])
        ma_coef  = np.abs(par[p:p + q])
        sar_coef = np.abs(par[p + q:p + q + P])
        sma_coef = np.abs(par[p + q + P:p + q + P + Q])

        for label, arr, grp in [
            (f"AR({p})",  ar_coef,  "AR terms"),
            (f"MA({q})",  ma_coef,  "MA terms"),
            (f"SAR({P})", sar_coef, "Seasonal AR"),
            (f"SMA({Q})", sma_coef, "Seasonal MA"),
        ]:
            if arr.size:
                rows.append({
                    "feature":    label,
                    "importance": float(arr.mean()),
                    "group":      grp,
                    "model":      "SARIMA",
                })

    # ── OLS Fixed-Effects ──────────────────────────────────────────────────────
    # fe_model.coef_df_ has columns: feature, coef, abs_coef.
    # Keep top-20 by |coef|; label by interpretable category.
    if "OLS-FE" in models and "fe_model" in artefacts:
        fe  = artefacts["fe_model"]
        cdf = fe.coef_df_.nlargest(20, "abs_coef").copy()

        def _fe_group(feat: str) -> str:
            """Map OLS-FE feature names to interpretable group labels."""
            if feat.startswith("company_"):                     return "Company FE"
            if feat.startswith("code_"):                        return "Commodity FE"
            if feat.startswith(("sin_", "cos_")):               return "Fourier Seasonality"
            if feat.startswith("q") or feat == "holiday_week":  return "Calendar"
            return "Trend"

        for _, r in cdf.iterrows():
            rows.append({
                "feature":    r["feature"],
                "importance": float(r["abs_coef"]),
                "group":      _fe_group(r["feature"]),
                "model":      "OLS-FE",
            })

    return pd.DataFrame(rows)


def plot_variable_importance(
    artefacts: dict,
    models: list,
    outdir: str,
    show: bool,
):
    """
    Plot G -- two-panel variable importance display for all selected models.

    Left panel  : horizontal bar chart, features ranked by |coefficient|,
                  colour-coded by feature group; model blocks separated by gaps.
    Right panel : heatmap of group-level importance, rows = models,
                  columns = feature groups, cell annotations show raw mean
                  |coef|, colour scale is row-normalised so cross-model
                  magnitudes are visually comparable.

    Parameters
    ----------
    artefacts : dict from fp.build_fitted_models()
    models    : selected model names
    outdir    : output directory
    show      : whether to call plt.show()
    """
    df_imp = _extract_importance(artefacts, models)
    if df_imp.empty:
        print("  [plot_variable_importance] No importance data to plot; skipping.")
        return

    # Colour palette for feature groups (shared across both panels)
    all_groups    = sorted(df_imp["group"].unique().tolist())
    group_colors  = dict(zip(all_groups, sns.color_palette("tab10", len(all_groups))))

    # ── Figure with two sub-plots ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, max(6, len(df_imp) * 0.40 + 2)))
    fig.suptitle(
        "Variable Importance -- All Models  (|standardised coefficient|)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.35)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT SUB-PANEL – horizontal bar chart
    # ══════════════════════════════════════════════════════════════════════════
    ax_bar = fig.add_subplot(gs[0])

    # Sort each model's features by importance descending, then stack models
    df_sorted = (
        df_imp
        .sort_values(["model", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Build y-position vector with a visual gap between model groups
    y_positions = []   # numeric y-coordinate for each bar
    y_labels    = []   # tick labels (model prefix + feature name)
    y_colors    = []   # bar face colours (feature group)
    y_values    = []   # bar lengths (importance)
    current_y   = 0
    gap         = 1.5  # extra vertical space between model blocks

    for mdl in models:
        sub = df_sorted[df_sorted["model"] == mdl]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            y_positions.append(current_y)
            y_labels.append(f"[{mdl}]  {r['feature']}")
            y_colors.append(group_colors.get(r["group"], "#888888"))
            y_values.append(r["importance"])
            current_y += 1
        current_y += gap   # visual separator between model blocks

    y_pos = np.array(y_positions)

    # Draw all bars in one call
    bars = ax_bar.barh(
        y_pos, y_values,
        color=y_colors, alpha=0.85, height=0.75, edgecolor="white",
    )

    # Annotate each bar with its numeric value
    max_val = max(y_values) if y_values else 1.0
    for bar, val in zip(bars, y_values):
        ax_bar.text(
            bar.get_width() + max_val * 0.01,   # small right-offset
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=7.5,
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(y_labels, fontsize=8)
    ax_bar.invert_yaxis()   # highest importance at the top
    ax_bar.set_xlabel("|Standardised Coefficient|")
    ax_bar.set_title("Feature Importance by Model", fontsize=11)

    # Legend mapping group name to colour
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[g], alpha=0.85)
        for g in all_groups
    ]
    ax_bar.legend(
        legend_patches, all_groups,
        title="Feature Group", fontsize=8, title_fontsize=8,
        loc="lower right",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT SUB-PANEL – group-level heatmap
    # ══════════════════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(gs[1])

    # Aggregate to group level: mean importance across features within each group
    df_group = (
        df_imp
        .groupby(["model", "group"])["importance"]
        .mean()
        .reset_index()
    )

    # Pivot to matrix: rows = models, columns = feature groups
    pivot = (
        df_group
        .pivot(index="model", columns="group", values="importance")
        .reindex(index=[m for m in models if m in df_group["model"].unique()])
        .fillna(0.0)
    )

    # Row-normalise so cross-model coefficient magnitudes are comparable visually.
    # Annotations still show the raw (un-normalised) values for interpretability.
    pivot_norm = pivot.div(pivot.max(axis=1) + 1e-12, axis=0)

    sns.heatmap(
        pivot_norm,
        ax=ax_heat,
        annot=pivot.round(3),   # raw values in cell annotations
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Normalised importance (0 = min, 1 = max per model)"},
        annot_kws={"size": 8},
    )
    ax_heat.set_title(
        "Group-Level Importance Heatmap\n(row-normalised, raw values annotated)",
        fontsize=11,
    )
    ax_heat.set_xlabel("Feature Group")
    ax_heat.set_ylabel("Model")
    ax_heat.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    _save_fig(fig, outdir, "plot_G_variable_importance.png", show)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 – MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrates the complete workflow:
      1. Parse and validate CLI arguments.
      2. Resolve the forecast date window (--start / --end / --horizon).
      3. Call fp.build_fitted_models() to fit all four models on the full sample.
      4. Generate point forecasts for every selected model.
      5. Render all plots and save PNGs to --outdir.
      6. Optionally export the forecast table to CSV.
    """
    args = parse_args()

    # ── Input validation ──────────────────────────────────────────────────────
    if args.horizon < 1:
        raise ValueError("--horizon must be a positive integer.")
    if not (0.0 <= args.ci <= 1.0):
        raise ValueError("--ci must be between 0 and 1.")
    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {ALL_MODELS}.")

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 65)
    print("  FREIGHT FORECAST VISUALISATION SCRIPT")
    print(f"  Models  : {', '.join(args.models)}")
    print(f"  Output  : {args.outdir}")
    print("=" * 65)

    # ── Step 1: Fit models via freight_pipeline ────────────────────────────────
    print("\n[1/5] Fitting models via freight_pipeline.build_fitted_models() ...")
    artefacts = fp.build_fitted_models(
        data_path=args.data,
        run_cv=args.run_cv,    # skip CV unless explicitly requested (slow)
        verbose=True,
    )
    ts = artefacts["ts"]   # aggregate weekly time-series DataFrame

    # ── Step 2: Resolve forecast date range ───────────────────────────────────
    # Must happen BEFORE generate_forecasts so horizon and fcast_dates are set.
    print("\n[2/5] Resolving forecast date range ...")
    last_data_date         = ts["week"].iloc[-1]
    fcast_dates, horizon   = resolve_forecast_dates(args, last_data_date)
    print(f"  Window  : {fcast_dates[0].date()} -> {fcast_dates[-1].date()}"
          f"  ({horizon} weeks)")

    # ── Step 3: Generate forecasts ────────────────────────────────────────────
    print(f"\n[3/5] Generating forecasts ...")
    forecasts = generate_forecasts(
        artefacts   = artefacts,
        models      = args.models,
        horizon     = horizon,
        fcast_dates = fcast_dates,
    )

    # Console summary table
    print(f"\n  {'Model':<12}  {'Min (M)':>10}  {'Mean (M)':>10}  {'Max (M)':>10}")
    print("  " + "-" * 46)
    for mdl, vals in forecasts.items():
        print(f"  {mdl:<12}  {vals.min()/1e6:>10.3f}  "
              f"{vals.mean()/1e6:>10.3f}  {vals.max()/1e6:>10.3f}")

    # ── Step 4: Render all plots ───────────────────────────────────────────────
    print("\n[4/5] Rendering visualisations ...")
    _set_style()

    # Plot A – zoomed ribbon: recent history + all model forecasts with PI band
    plot_forecast_ribbon(
        ts=ts, fcast_dates=fcast_dates, forecasts=forecasts,
        models=args.models, horizon=horizon,
        history_weeks=args.history, ci_pct=args.ci,
        outdir=args.outdir, show=args.show,
    )

    # Plot B – full time-series: in-sample fits + forecast + residual panel
    plot_full_fit_and_forecast(
        ts=ts, artefacts=artefacts, fcast_dates=fcast_dates,
        forecasts=forecasts, models=args.models,
        outdir=args.outdir, show=args.show,
    )

    # Plot B2 – same as B but x-axis clipped to 2024-onward for detail
    plot_zoomed_2024_forecast(
        ts=ts, artefacts=artefacts, fcast_dates=fcast_dates,
        forecasts=forecasts, models=args.models,
        outdir=args.outdir, show=args.show,
    )

    # Plot C – per-model comparison grid (one sub-panel per model)
    plot_model_comparison(
        fcast_dates=fcast_dates, forecasts=forecasts, ts=ts,
        models=args.models, history_weeks=args.history,
        outdir=args.outdir, show=args.show,
    )

    # Plot D – GAM & Prophet additive decomposition (trend + seasonality)
    plot_forecast_decomposition(
        ts=ts, artefacts=artefacts, fcast_dates=fcast_dates,
        models=args.models, horizon=horizon,
        outdir=args.outdir, show=args.show,
    )

    # Plot F – faceted GAM spline bases individually vs normalised actuals
    plot_gam_splines_vs_actual(
        ts=ts, artefacts=artefacts, fcast_dates=fcast_dates,
        horizon=horizon, outdir=args.outdir, show=args.show,
    )

    # Plot G – variable importance: ranked bar chart + group-level heatmap
    plot_variable_importance(
        artefacts=artefacts, models=args.models,
        outdir=args.outdir, show=args.show,
    )

    # Plot E – CV metric summary (only when --run-cv was passed)
    if args.run_cv and artefacts.get("df_cv") is not None:
        plot_cv_summary(
            df_cv=artefacts["df_cv"], models=args.models,
            outdir=args.outdir, show=args.show,
        )

    # ── Step 5: Export CSV ────────────────────────────────────────────────────
    if args.export:
        print("\n[5/5] Exporting forecast CSV ...")
        export_forecasts_csv(
            fcast_dates=fcast_dates, forecasts=forecasts,
            ts=ts, outdir=args.outdir,
        )
    else:
        print("\n[5/5] Skipping CSV export (pass --export to enable).")

    print("\n" + "=" * 65)
    print(f"  Done. All outputs -> {args.outdir}")
    print("=" * 65)


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()