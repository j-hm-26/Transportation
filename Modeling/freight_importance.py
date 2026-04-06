"""
freight_importance.py
=====================
Company & Commodity Importance Analysis
========================================

Extends freight_forecasting.py with two analytical lenses that answer
"which companies and commodities matter most, and how?":

SECTION A – GENERALISED ADDITIVE MODEL (GAM)
--------------------------------------------
Built from sklearn's SplineTransformer + Ridge.  Each company and each
commodity gets its own smooth nonlinear trend over time.  The estimated
smooth functions are plotted to reveal how each entity's carload volume
has evolved — going beyond the flat FE coefficient of the OLS model.

Functions
---------
  build_gam_matrices()       Build per-entity spline design matrices
  GAMImportance              Fit one GAM per entity, extract partial effects
  compute_entity_importance()  Rank entities by mean partial effect magnitude
  plot_gam_company_trends()  Smoothed company trajectories over time
  plot_gam_commodity_trends() Smoothed commodity trajectories over time
  plot_importance_bars()     Ranked bar chart of entity importance scores

SECTION B – PRINCIPAL COMPONENT ANALYSIS (PCA)
-----------------------------------------------
Operates on the (weeks × companies) and (weeks × commodities) matrices.
PCA finds the latent axes of co-movement — e.g., "PC1 = system-wide
volume cycle, PC2 = coal vs intermodal divergence".

Functions
---------
  build_weekly_pivot()       Pivot raw panel → weeks × entities matrix
  run_pca()                  Fit PCA, return loadings + explained variance
  plot_pca_scree()           Scree plot: explained variance by component
  plot_pca_loadings()        Loading heatmap: entity contribution per PC
  plot_pca_scores()          PC score time-series (latent factors over time)
  plot_biplot()              2-D biplot: scores + loading arrows together
  plot_company_commodity_corr() Correlation heatmap between entities

SECTION C – COMBINED IMPORTANCE SUMMARY
-----------------------------------------
  plot_importance_summary()  Side-by-side: GAM importance vs PCA loading
                             magnitudes — two views of the same truth.

ENTRY POINT
-----------
Run as __main__ to execute the full analysis and save all figures.
Figures are numbered fig11_* through fig22_* to avoid overwriting the
10 existing figures from freight_forecasting.py.
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════
import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ── Shared constants (mirror freight_forecasting.py) ─────────────
PALETTE = {
    "SARIMA":  "#E07B39",
    "Prophet": "#3B82C4",
    "OLS-FE":  "#2DBD6E",
}
# Colour sequences for multi-entity plots
CO_COLORS  = ["#E07B39","#3B82C4","#2DBD6E","#6C63FF",
               "#F59E0B","#EF4444","#10B981","#8B5CF6"]
CD_COLORS  = (sns.color_palette("tab20", 20)
              + sns.color_palette("Set2", 4))

BG_COLOR   = "#F8F9FA"
GRID_COLOR = "#E2E8F0"
ACCENT     = "#6C63FF"

OUTPUT_DIR = "/home/lord/Desktop/478C_Capstone/CAPPY/Visuals/Importance_Analysis"
DATA_PATH  = "/home/lord/Desktop/478C_Capstone/CAPPY/Data/Weekly_Cargo_Data_2020_2026_all_data.csv"

# Commodity code → readable name
CODE_NAMES = {
    "A": "Chemicals",         "B": "Coal",
    "C": "Coke",              "D": "Crushed Stone/Sand",
    "E": "Farm Products",     "F": "Food & Kindred",
    "G": "Grain Mill",        "H": "Grain",
    "I": "Iron & Steel Scrap","J": "Lumber & Wood",
    "K": "Metallic Ores",     "L": "Metal Products",
    "M": "Motor Vehicles",    "N": "Nonmetallic Minerals",
    "O": "Petroleum Products","P": "Primary Forest",
    "Q": "Pulp & Paper",      "R": "Stone, Clay, Glass",
    "S": "Waste & Scrap",     "T": "All Other",
    "U": "Containers",        "V": "Trailers",
    "YC":"Total Carloads",    "YI":"Total Intermodal",
}

# ══════════════════════════════════════════════════════════════════
# SECTION 0 – DATA HELPERS
# ══════════════════════════════════════════════════════════════════

def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV and return both the raw panel and an enriched
    copy ready for importance analysis.

    Parameters
    ----------
    path : str
        Absolute path to the weekly freight CSV.

    Returns
    -------
    raw : pd.DataFrame
        Minimally cleaned panel (one row per week/company/commodity).
    panel : pd.DataFrame
        raw + derived columns:
          total (originated + received, clipped ≥ 0),
          log_total, week_num (integer), week_of_year, year,
          commodity_name (human-readable label for code).
    """
    raw = pd.read_csv(path, index_col=0)
    raw.columns = (
        raw.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    raw["week"] = pd.to_datetime(raw["week"])
    raw = raw.rename(columns={"commodity_group_code": "code",
                               "commodity_group":      "commodity_group"})
    raw["company"] = raw["company"].astype("category")
    raw["code"]    = raw["code"].astype("category")

    panel = raw.copy()
    panel["originated"] = panel["originated"].fillna(0)
    panel["received"]   = panel["received"].fillna(0)
    panel["total"]      = (panel["originated"] + panel["received"]).clip(lower=0)
    panel["log_total"]  = np.log1p(panel["total"])

    week_map = {w: i for i, w in enumerate(sorted(panel["week"].unique()))}
    panel["week_num"]     = panel["week"].map(week_map).astype(float)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["year"]         = panel["week"].dt.year
    panel["commodity_name"] = panel["code"].map(CODE_NAMES).fillna(panel["code"].astype(str))
    return raw, panel


def build_weekly_pivot(panel: pd.DataFrame,
                       group_col: str,
                       value_col: str = "total",
                       use_name_map: bool = False) -> pd.DataFrame:
    """
    Pivot the panel to a (weeks × entities) matrix.

    Parameters
    ----------
    panel : pd.DataFrame
        Enriched panel from load_and_prepare().
    group_col : str
        Column to use as entities: "company" or "code".
    value_col : str
        Numeric column to aggregate (default "total").
    use_name_map : bool
        If True and group_col == "code", replace codes with full names.

    Returns
    -------
    pd.DataFrame
        Index = week (datetime), columns = entity labels,
        values = summed carloads.  Missing cells filled with 0.
    """
    pivot = (
        panel.groupby(["week", group_col])[value_col]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    if use_name_map and group_col == "code":
        pivot.columns = [CODE_NAMES.get(c, c) for c in pivot.columns]
    return pivot


# ══════════════════════════════════════════════════════════════════
# SECTION A – GENERALISED ADDITIVE MODEL (GAM)
# ══════════════════════════════════════════════════════════════════

def build_spline_features(t: np.ndarray,
                          n_knots: int = 12,
                          degree:  int = 3) -> np.ndarray:
    """
    Build a B-spline basis matrix for the time index t.

    Uses sklearn's SplineTransformer to construct a natural B-spline
    basis with evenly spaced knots.  The resulting columns are smooth
    nonlinear functions of t that the downstream Ridge regression
    combines into a flexible smooth trend.

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        Integer week index (0-based).
    n_knots : int
        Number of interior knots.  More knots → more wiggly fit.
        Recommended range: 6–20 for weekly freight data.
    degree : int
        Polynomial degree of each spline segment (default 3 = cubic).

    Returns
    -------
    np.ndarray, shape (n, n_knots + degree - 1)
        Spline basis matrix, zero-mean centred column-wise.
    """
    st = SplineTransformer(n_knots=n_knots, degree=degree,
                           include_bias=False, knots="uniform")
    B  = st.fit_transform(t.reshape(-1, 1))
    return B - B.mean(axis=0)   # centre so intercept is the grand mean


class GAMImportance:
    """
    Generalised Additive Model (GAM) for entity-level importance.

    For each entity (company or commodity), fits an independent
    penalised spline model:

        log(carloads_e(t)) = f_e(t) + ε_e

    where f_e(t) is a smooth B-spline function estimated by Ridge
    regression.  The fitted smooth f_e captures the nonlinear trend
    of entity e over time, separate from all other entities.

    The "importance" of entity e is defined as the mean absolute
    deviation of f_e from zero — i.e., how much that entity's volume
    deviates from a flat baseline on average.  A high importance score
    means the entity contributes strong, distinctive temporal signal.

    Parameters
    ----------
    n_knots : int
        B-spline knots (controls smoothness).
    degree : int
        Spline polynomial degree (3 = cubic, standard for GAMs).
    alpha : float
        Ridge penalty strength.  Larger values → smoother curves.

    Attributes
    ----------
    entities_       : list of str
        Entity labels (companies or commodity codes).
    smooths_        : dict[str, np.ndarray]
        Fitted smooth f_e(t) values on the training time grid.
    importance_df_  : pd.DataFrame
        Ranked table: entity, mean_abs_effect, std_effect, share_pct.
    """

    def __init__(self, n_knots: int = 12, degree: int = 3,
                 alpha: float = 1.0):
        self.n_knots = n_knots
        self.degree  = degree
        self.alpha   = alpha
        self.entities_:      list | None = None
        self.smooths_:       dict | None = None
        self.importance_df_: pd.DataFrame | None = None

    def fit(self, pivot: pd.DataFrame) -> "GAMImportance":
        """
        Fit one Ridge-spline model per entity in the pivot matrix.

        Parameters
        ----------
        pivot : pd.DataFrame
            Output of build_weekly_pivot() — index is week (datetime),
            columns are entity labels, values are raw carloads.

        Returns
        -------
        self
        """
        t = np.arange(len(pivot), dtype=float)
        B = build_spline_features(t, self.n_knots, self.degree)

        self.entities_ = list(pivot.columns)
        self.smooths_  = {}
        records = []

        for entity in self.entities_:
            y_raw = pivot[entity].values.astype(float)
            y     = np.log1p(np.clip(y_raw, 0, None))  # log scale

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(B, y)
            f_hat = model.predict(B)                    # smooth on log scale

            # Back-transform for interpretable magnitude
            smooth_orig = np.expm1(f_hat)

            self.smooths_[entity] = smooth_orig
            records.append({
                "entity":          entity,
                "mean_abs_effect": float(np.mean(np.abs(smooth_orig - smooth_orig.mean()))),
                "std_effect":      float(np.std(smooth_orig)),
                "mean_level":      float(np.mean(smooth_orig)),
            })

        df = pd.DataFrame(records).sort_values("mean_abs_effect", ascending=False)
        total = df["mean_abs_effect"].sum() + 1e-9
        df["share_pct"] = df["mean_abs_effect"] / total * 100.0
        self.importance_df_ = df.reset_index(drop=True)
        return self

    def get_smooth_df(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """
        Return fitted smooths as a tidy long-form DataFrame.

        Parameters
        ----------
        pivot : pd.DataFrame
            The same pivot used in fit().

        Returns
        -------
        pd.DataFrame with columns: week, entity, smooth_carloads.
        """
        weeks = pivot.index
        rows  = []
        for entity, smooth in self.smooths_.items():
            for w, s in zip(weeks, smooth):
                rows.append({"week": w, "entity": entity, "smooth": s})
        return pd.DataFrame(rows)


def compute_entity_importance(panel:     pd.DataFrame,
                              group_col: str,
                              n_knots:   int = 12,
                              alpha:     float = 1.0) -> GAMImportance:
    """
    Convenience wrapper: build pivot → fit GAM → return fitted model.

    Parameters
    ----------
    panel : pd.DataFrame
        Enriched panel from load_and_prepare().
    group_col : str
        "company" or "code".
    n_knots : int
        Spline knots forwarded to GAMImportance.
    alpha : float
        Ridge penalty forwarded to GAMImportance.

    Returns
    -------
    GAMImportance
        Fitted model with .smooths_, .importance_df_ populated.
    """
    pivot = build_weekly_pivot(panel, group_col)
    gam   = GAMImportance(n_knots=n_knots, alpha=alpha)
    gam.fit(pivot)
    return gam


# ══════════════════════════════════════════════════════════════════
# SECTION A – VISUALISATION (GAM)
# ══════════════════════════════════════════════════════════════════

def plot_gam_company_trends(gam:   GAMImportance,
                            pivot: pd.DataFrame) -> None:
    """
    Plot the GAM-fitted smooth trend for each company on a shared axis.

    Each line shows how that company's estimated carload volume evolves
    over the full dataset window.  The thin raw data are plotted in the
    background at reduced opacity for context.

    Parameters
    ----------
    gam : GAMImportance
        Fitted GAMImportance model (group_col="company").
    pivot : pd.DataFrame
        Weekly company pivot from build_weekly_pivot().
    """
    weeks     = pivot.index
    companies = gam.entities_
    colors    = CO_COLORS[:len(companies)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle("GAM Smooth Trends — Weekly Carloads by Company",
                 fontsize=14, fontweight="bold")

    # Top: raw + smooth for each company
    for company, color in zip(companies, colors):
        raw = pivot[company].values / 1e3
        ax_top.plot(weeks, raw, color=color, linewidth=0.6,
                    alpha=0.25, zorder=1)
        smooth = gam.smooths_[company] / 1e3
        ax_top.plot(weeks, smooth, color=color, linewidth=2.2,
                    label=company, zorder=2)

    ax_top.set_ylabel("Carloads (thousands)")
    ax_top.set_title("Raw data (faint) + GAM Smooth (bold)")
    ax_top.legend(fontsize=8, ncol=4, loc="upper left")

    # Bottom: stacked area of smooths (normalised shares)
    smooth_mat = np.column_stack(
        [gam.smooths_[c] for c in companies]
    )
    smooth_mat = np.clip(smooth_mat, 0, None)
    row_totals = smooth_mat.sum(axis=1, keepdims=True) + 1e-9
    shares     = smooth_mat / row_totals

    ax_bot.stackplot(weeks, shares.T * 100,
                     labels=companies, colors=colors, alpha=0.8)
    ax_bot.set_ylabel("Share of GAM Total (%)")
    ax_bot.set_xlabel("Week")
    ax_bot.set_title("Company Volume Share over Time (GAM-smoothed)")
    ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig11_gam_company_trends.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fig11_gam_company_trends.png")


def plot_gam_commodity_trends(gam:   GAMImportance,
                              pivot: pd.DataFrame,
                              top_n: int = 12) -> None:
    """
    Plot GAM smooths for the top-N commodities by importance score.

    Only the top_n commodities (ranked by mean_abs_effect) are shown
    to keep the chart readable.

    Parameters
    ----------
    gam : GAMImportance
        Fitted GAMImportance model (group_col="code").
    pivot : pd.DataFrame
        Weekly commodity pivot from build_weekly_pivot().
    top_n : int
        How many commodities to include (default 12).
    """
    weeks    = pivot.index
    top_ents = gam.importance_df_["entity"].iloc[:top_n].tolist()
    colors   = (sns.color_palette("tab10", 10)
                + sns.color_palette("Set2", max(0, top_n - 10)))[:top_n]

    n_cols = 3
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 3.5 * n_rows),
                             sharex=True)
    fig.suptitle(f"GAM Smooth Trends — Top {top_n} Commodities by Importance",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, entity, color in zip(axes.flat, top_ents, colors):
        raw    = pivot[entity].values / 1e3
        smooth = gam.smooths_[entity] / 1e3
        ax.fill_between(weeks, raw, alpha=0.12, color=color)
        ax.plot(weeks, raw, color=color, linewidth=0.5, alpha=0.4)
        ax.plot(weeks, smooth, color=color, linewidth=2.0)
        label = CODE_NAMES.get(entity, entity)
        ax.set_title(f"{entity} – {label}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Kcarloads")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)

    # Hide any spare axes
    for ax in axes.flat[top_n:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig12_gam_commodity_trends.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fig12_gam_commodity_trends.png")


def plot_importance_bars(gam_co:  GAMImportance,
                         gam_cd:  GAMImportance) -> None:
    """
    Side-by-side horizontal bar charts of entity importance scores.

    Left panel: company importance.
    Right panel: commodity importance (top 15).

    Importance = mean absolute deviation of the GAM smooth from its
    own mean — higher means the entity's volume fluctuates more
    distinctively and contributes more structural signal to the system.

    Parameters
    ----------
    gam_co : GAMImportance
        Fitted GAM for companies.
    gam_cd : GAMImportance
        Fitted GAM for commodities.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("GAM-Derived Entity Importance (Mean |Partial Effect|)",
                 fontsize=14, fontweight="bold")

    # ── Companies ────────────────────────────────────────────────
    df_co = gam_co.importance_df_.copy()
    sns.barplot(data=df_co, y="entity", x="mean_abs_effect",
                palette=CO_COLORS[:len(df_co)], ax=ax_l,
                orient="h", hue="entity", legend=False)
    for i, row in df_co.iterrows():
        ax_l.text(row["mean_abs_effect"] * 1.01, i,
                  f'{row["share_pct"]:.1f}%', va="center", fontsize=8)
    ax_l.set_title("Company Importance")
    ax_l.set_xlabel("Mean |GAM Effect| (carloads)")
    ax_l.set_ylabel("")

    # ── Commodities (top 15) ─────────────────────────────────────
    df_cd = gam_cd.importance_df_.head(15).copy()
    df_cd["label"] = df_cd["entity"].map(
        lambda c: f'{c} – {CODE_NAMES.get(c, c)}'
    )
    pal_cd = sns.color_palette("tab20", len(df_cd))
    sns.barplot(data=df_cd, y="label", x="mean_abs_effect",
                palette=pal_cd, ax=ax_r,
                orient="h", hue="label", legend=False)
    for i, row in df_cd.iterrows():
        ax_r.text(row["mean_abs_effect"] * 1.01, list(df_cd.index).index(i),
                  f'{row["share_pct"]:.1f}%', va="center", fontsize=8)
    ax_r.set_title("Commodity Importance (Top 15)")
    ax_r.set_xlabel("Mean |GAM Effect| (carloads)")
    ax_r.set_ylabel("")

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig13_gam_importance_bars.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fig13_gam_importance_bars.png")


# ══════════════════════════════════════════════════════════════════
# SECTION B – PCA
# ══════════════════════════════════════════════════════════════════

def run_pca(pivot:      pd.DataFrame,
            n_components: int | None = None,
            scale:        bool = True) -> dict:
    """
    Fit PCA on the weekly entity matrix and return all outputs.

    The matrix is optionally standardised column-wise (recommended when
    entities have very different volume magnitudes — e.g., coal vs coke).
    PCA then finds the orthogonal directions of maximum variance in the
    entity co-movement space.

    Parameters
    ----------
    pivot : pd.DataFrame
        (weeks × entities) matrix from build_weekly_pivot().
    n_components : int, optional
        Number of PCs to retain.  Defaults to min(weeks, entities).
    scale : bool
        Whether to StandardScale columns before PCA (default True).

    Returns
    -------
    dict with keys:
        pca        : fitted sklearn PCA object
        scores     : np.ndarray (weeks × n_components) — PC time-series
        loadings   : pd.DataFrame (entities × n_components) — PC loadings
        explained  : np.ndarray — explained variance ratio per component
        weeks      : pd.DatetimeIndex — time axis for scores
        scaler     : fitted StandardScaler (or None if scale=False)
    """
    X = pivot.values.astype(float)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_max = min(X.shape)
    k     = min(n_components or n_max, n_max)
    pca   = PCA(n_components=k, random_state=42)
    scores = pca.fit_transform(X)

    pc_labels = [f"PC{i+1}" for i in range(k)]
    loadings  = pd.DataFrame(
        pca.components_.T,
        index   = pivot.columns,
        columns = pc_labels,
    )

    return {
        "pca":       pca,
        "scores":    scores,
        "loadings":  loadings,
        "explained": pca.explained_variance_ratio_,
        "weeks":     pivot.index,
        "scaler":    scaler,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION B – VISUALISATION (PCA)
# ══════════════════════════════════════════════════════════════════

def plot_pca_scree(pca_result: dict,
                  title:      str = "PCA Scree Plot") -> None:
    """
    Scree plot with cumulative explained variance overlay.

    The elbow in the individual bar chart shows where additional
    components stop explaining meaningful variance.  The cumulative
    line shows how many PCs are needed to explain e.g. 90% of variance.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca().
    title : str
        Figure title suffix (e.g. "Companies" or "Commodities").
    """
    ev   = pca_result["explained"] * 100
    cumev = np.cumsum(ev)
    k    = len(ev)
    x    = np.arange(1, k + 1)

    fig, ax1 = plt.subplots(figsize=(max(8, k // 2), 5))
    ax2 = ax1.twinx()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.bar(x, ev, color=ACCENT, alpha=0.7, width=0.6)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    ax2.plot(x, cumev, color="#E07B39", marker="o",
             linewidth=2, markersize=5)
    ax2.axhline(90, color="grey", linestyle="--", linewidth=0.8)
    ax2.text(x[-1] * 0.5, 91, "90% threshold", fontsize=8, color="grey")
    ax2.set_ylabel("Cumulative Variance (%)", color="#E07B39")
    ax2.tick_params(axis="y", labelcolor="#E07B39")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    suffix = title.lower().replace(" ", "_")
    fig.savefig(f"{OUTPUT_DIR}/fig14_pca_scree_{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig14_pca_scree_{suffix}.png")


def plot_pca_loadings(pca_result: dict,
                      n_pcs:      int = 5,
                      title:      str = "PCA Loadings",
                      use_names:  bool = False) -> None:
    """
    Heatmap of entity loadings for the first n_pcs components.

    Each cell shows how strongly an entity loads onto a PC.
    Large |loading| → that entity is a major driver of that latent factor.
    Sign indicates direction: positive = moves with the PC, negative = inverse.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca().
    n_pcs : int
        Number of PCs to display (columns in heatmap).
    title : str
        Figure title.
    use_names : bool
        Replace commodity codes with full names in the y-axis.
    """
    load = pca_result["loadings"].iloc[:, :n_pcs].copy()
    if use_names:
        load.index = [f'{c} – {CODE_NAMES.get(c, c)}' for c in load.index]

    fig, ax = plt.subplots(figsize=(max(6, n_pcs * 1.4),
                                    max(5, len(load) * 0.45)))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ev = pca_result["explained"][:n_pcs] * 100
    col_labels = [f"PC{i+1}\n({ev[i]:.1f}%)" for i in range(n_pcs)]
    load.columns = col_labels

    sns.heatmap(load, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Loading"})
    ax.set_ylabel("Entity")
    ax.set_xlabel("Principal Component")

    plt.tight_layout()
    suffix = title.lower().replace(" ", "_")
    fig.savefig(f"{OUTPUT_DIR}/fig15_pca_loadings_{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig15_pca_loadings_{suffix}.png")


def plot_pca_scores(pca_result: dict,
                    n_pcs:      int = 4,
                    title:      str = "PCA Score Time-Series") -> None:
    """
    Time-series plot of the first n_pcs principal component scores.

    Each line is a latent factor that summarises a mode of co-movement
    across entities.  PC1 typically captures the system-wide volume level;
    higher PCs reflect divergences between groups of entities.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca().
    n_pcs : int
        Number of PC score series to plot (stacked panels).
    title : str
        Figure title.
    """
    scores = pca_result["scores"][:, :n_pcs]
    weeks  = pca_result["weeks"]
    ev     = pca_result["explained"][:n_pcs] * 100

    fig, axes = plt.subplots(n_pcs, 1, figsize=(15, 2.8 * n_pcs),
                             sharex=True)
    if n_pcs == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = [ACCENT, "#E07B39", "#3B82C4", "#2DBD6E",
              "#F59E0B", "#EF4444"][:n_pcs]

    for i, (ax, color) in enumerate(zip(axes, colors)):
        ax.fill_between(weeks, scores[:, i], alpha=0.15, color=color)
        ax.plot(weeks, scores[:, i], color=color, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylabel(f"Score")
        ax.set_title(f"PC{i+1}  ({ev[i]:.1f}% of variance)",
                     fontsize=10, loc="left")

    axes[-1].set_xlabel("Week")
    plt.tight_layout()
    suffix = title.lower().replace(" ", "_")
    fig.savefig(f"{OUTPUT_DIR}/fig16_pca_scores_{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig16_pca_scores_{suffix}.png")


def plot_biplot(pca_result:  dict,
                n_top_arrows: int = 8,
                title:        str = "PCA Biplot (PC1 vs PC2)",
                use_names:    bool = False) -> None:
    """
    2-D biplot overlaying PC1 vs PC2 score scatter with loading arrows.

    Each point is a week; each arrow represents one entity.  Arrow direction
    shows how the entity correlates with the two latent factors; arrow length
    shows how strongly it loads.  Weeks that cluster together had similar
    entity profiles.

    Parameters
    ----------
    pca_result : dict
        Output of run_pca().
    n_top_arrows : int
        Only draw arrows for the top-N entities by loading magnitude
        (avoids clutter).
    title : str
        Figure title.
    use_names : bool
        Use full commodity names on arrows.
    """
    scores   = pca_result["scores"]
    loadings = pca_result["loadings"]
    ev       = pca_result["explained"] * 100
    weeks    = pca_result["weeks"]

    # Scale arrows to fit in score space
    sx, sy = scores[:, 0], scores[:, 1]
    scale  = 0.85 * min(np.abs([sx.min(), sx.max(),
                                 sy.min(), sy.max()]))

    # Identify top entities by combined PC1+PC2 loading magnitude
    load12 = np.sqrt(loadings["PC1"]**2 + loadings["PC2"]**2)
    top_idx = load12.nlargest(n_top_arrows).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Colour points by year
    years   = pd.to_datetime(weeks).year
    cmap    = plt.cm.viridis
    yr_min, yr_max = years.min(), years.max()
    yr_norm = (years - yr_min) / max(yr_max - yr_min, 1)

    sc = ax.scatter(sx, sy, c=yr_norm, cmap=cmap, s=18,
                    alpha=0.65, zorder=2)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_ticks(np.linspace(0, 1, yr_max - yr_min + 1))
    cbar.set_ticklabels(range(yr_min, yr_max + 1))
    cbar.set_label("Year")

    # Loading arrows
    lx = loadings.loc[top_idx, "PC1"].values * scale
    ly = loadings.loc[top_idx, "PC2"].values * scale

    for entity, dx, dy in zip(top_idx, lx, ly):
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                   color="#E07B39", lw=1.8))
        lbl = f'{entity} – {CODE_NAMES.get(entity, entity)}' if use_names else entity
        ax.text(dx * 1.06, dy * 1.06, lbl,
                fontsize=8, color="#E07B39", fontweight="bold",
                ha="center", va="center")

    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"PC1  ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2  ({ev[1]:.1f}% variance)")

    plt.tight_layout()
    suffix = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(f"{OUTPUT_DIR}/fig17_pca_biplot_{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig17_pca_biplot_{suffix}.png")


def plot_company_commodity_corr(panel: pd.DataFrame) -> None:
    """
    Two correlation heatmaps:
      Left : (companies × companies) — correlation of weekly volumes.
      Right: (commodities × commodities) — same for commodity groups.

    Strong off-diagonal correlations reveal entities that move together
    — useful for understanding systemic vs. idiosyncratic risk in
    freight volumes.  Hierarchical clustering reorders rows/columns so
    that similar entities are adjacent.

    Parameters
    ----------
    panel : pd.DataFrame
        Enriched panel from load_and_prepare().
    """
    co_pivot = build_weekly_pivot(panel, "company")
    cd_pivot = build_weekly_pivot(panel, "code")

    # Exclude synthetic aggregates YC / YI if present
    cd_pivot = cd_pivot[[c for c in cd_pivot.columns
                          if c not in ("YC", "YI")]]
    cd_pivot.columns = [f'{c}–{CODE_NAMES.get(c, c)}' for c in cd_pivot.columns]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Weekly Carload Volume Correlation (Hierarchically Clustered)",
                 fontsize=14, fontweight="bold")

    for ax, piv, title in [
        (ax_l, co_pivot, "Companies"),
        (ax_r, cd_pivot, "Commodities"),
    ]:
        corr = piv.corr()
        # Hierarchical clustering for better visual order
        dist = pdist(corr.values, metric="euclidean")
        link = linkage(dist, method="ward")
        order = dendrogram(link, no_plot=True)["leaves"]
        corr_ordered = corr.iloc[order, :].iloc[:, order]

        sns.heatmap(corr_ordered, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=(piv.shape[1] <= 10), fmt=".2f",
                    linewidths=0.3, ax=ax,
                    cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0,  labelsize=7)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig18_correlation_heatmaps.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fig18_correlation_heatmaps.png")


# ══════════════════════════════════════════════════════════════════
# SECTION C – COMBINED IMPORTANCE SUMMARY
# ══════════════════════════════════════════════════════════════════

def plot_importance_summary(gam_co:     GAMImportance,
                            gam_cd:     GAMImportance,
                            pca_co:     dict,
                            pca_cd:     dict) -> None:
    """
    Four-panel summary comparing GAM importance and PCA PC1 loadings
    for both companies and commodities.

    The GAM importance captures nonlinear temporal contribution;
    PCA loading magnitude captures linear co-movement contribution.
    Agreement between them strengthens confidence in the ranking.

    Parameters
    ----------
    gam_co : GAMImportance  – company GAM
    gam_cd : GAMImportance  – commodity GAM
    pca_co : dict           – company PCA result (run_pca output)
    pca_cd : dict           – commodity PCA result
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Entity Importance: GAM Temporal Effect vs PCA Loading Magnitude",
                 fontsize=14, fontweight="bold")

    def _gam_bar(ax, gam, colors, title, top_n=None):
        df = gam.importance_df_.copy()
        if top_n:
            df = df.head(top_n)
        labels = [CODE_NAMES.get(e, e) if e in CODE_NAMES else e
                  for e in df["entity"]]
        ax.barh(labels[::-1], df["mean_abs_effect"].values[::-1],
                color=colors[:len(df)], alpha=0.82)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |GAM Effect| (carloads)")

    def _pca_bar(ax, pca_result, title, top_n=None, use_names=False):
        load = pca_result["loadings"]["PC1"].abs().sort_values(ascending=False)
        if top_n:
            load = load.head(top_n)
        if use_names:
            labels = [f'{c}–{CODE_NAMES.get(c, c)}' for c in load.index]
        else:
            labels = list(load.index)
        ax.barh(labels[::-1], load.values[::-1],
                color=ACCENT, alpha=0.75)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ev1 = pca_result["explained"][0] * 100
        ax.set_xlabel(f"|PC1 Loading|  (PC1 explains {ev1:.1f}% var.)")

    _gam_bar(axes[0, 0], gam_co,
             CO_COLORS, "Companies — GAM Importance")
    _pca_bar(axes[0, 1], pca_co,
             "Companies — |PC1 Loading|")

    _gam_bar(axes[1, 0], gam_cd,
             sns.color_palette("tab20", 15),
             "Commodities — GAM Importance (Top 12)", top_n=12)
    _pca_bar(axes[1, 1], pca_cd,
             "Commodities — |PC1 Loading| (Top 12)",
             top_n=12, use_names=True)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig19_importance_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fig19_importance_summary.png")


# ══════════════════════════════════════════════════════════════════
# SECTION D – ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    set_style = lambda: (
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05),
        plt.rcParams.update({
            "figure.facecolor": BG_COLOR,
            "axes.facecolor":   BG_COLOR,
            "grid.color":       GRID_COLOR,
            "axes.spines.top":  False,
            "axes.spines.right": False,
        })
    )
    set_style()

    print("=" * 62)
    print("  FREIGHT IMPORTANCE ANALYSIS  (GAM + PCA)")
    print("=" * 62)

    # ── A.0  Load data ───────────────────────────────────────────
    print("\n[1/5] Loading data…")
    raw, panel = load_and_prepare(DATA_PATH)
    print(f"  Panel rows : {len(panel):,}")
    print(f"  Companies  : {panel['company'].nunique()}")
    print(f"  Commodities: {panel['code'].nunique()}")

    # ── A.1  Build pivots ────────────────────────────────────────
    print("\n[2/5] Building weekly pivot matrices…")
    co_pivot = build_weekly_pivot(panel, "company")
    cd_pivot = build_weekly_pivot(panel, "code")

    # Remove synthetic roll-up codes from commodity pivot
    cd_pivot = cd_pivot[[c for c in cd_pivot.columns
                          if c not in ("YC", "YI")]]
    print(f"  Company pivot  : {co_pivot.shape}")
    print(f"  Commodity pivot: {cd_pivot.shape}")

    # ── A.2  Fit GAMs ────────────────────────────────────────────
    print("\n[3/5] Fitting GAM importance models…")
    gam_co = compute_entity_importance(panel, "company", n_knots=14, alpha=0.5)
    print("\n  Company importance ranking:")
    print(gam_co.importance_df_[["entity","mean_abs_effect","share_pct"]].to_string(index=False))

    # Fit on filtered pivot (no synthetic aggregates YC/YI)
    gam_cd = GAMImportance(n_knots=14, alpha=0.5)
    gam_cd.fit(cd_pivot)
    print("\n  Commodity importance ranking (top 10):")
    print(gam_cd.importance_df_.head(10)[["entity","mean_abs_effect","share_pct"]].to_string(index=False))

    # ── A.3  Run PCA ─────────────────────────────────────────────
    print("\n[4/5] Running PCA…")
    pca_co = run_pca(co_pivot, n_components=8,  scale=True)
    pca_cd = run_pca(cd_pivot, n_components=12, scale=True)

    print(f"\n  Company PCA  — variance explained by first 5 PCs:")
    for i, ev in enumerate(pca_co["explained"][:5]):
        print(f"    PC{i+1}: {ev*100:.1f}%")

    print(f"\n  Commodity PCA — variance explained by first 5 PCs:")
    for i, ev in enumerate(pca_cd["explained"][:5]):
        print(f"    PC{i+1}: {ev*100:.1f}%")

    # ── A.4  Plot all figures ────────────────────────────────────
    print("\n[5/5] Rendering figures…")

    # GAM figures
    plot_gam_company_trends(gam_co, co_pivot)
    plot_gam_commodity_trends(gam_cd, cd_pivot, top_n=12)
    plot_importance_bars(gam_co, gam_cd)

    # PCA figures — companies
    plot_pca_scree(pca_co,    title="PCA Scree — Companies")
    plot_pca_loadings(pca_co, n_pcs=min(5, len(pca_co["explained"])),
                      title="PCA Loadings — Companies")
    plot_pca_scores(pca_co,   n_pcs=4, title="PCA Scores — Companies")
    plot_biplot(pca_co, n_top_arrows=8, title="PCA Biplot — Companies")

    # PCA figures — commodities
    plot_pca_scree(pca_cd,    title="PCA Scree — Commodities")
    plot_pca_loadings(pca_cd, n_pcs=min(5, len(pca_cd["explained"])),
                      title="PCA Loadings — Commodities", use_names=True)
    plot_pca_scores(pca_cd,   n_pcs=4, title="PCA Scores — Commodities")
    plot_biplot(pca_cd, n_top_arrows=10, title="PCA Biplot — Commodities",
                use_names=True)

    # Correlation heatmaps
    plot_company_commodity_corr(panel)

    # Combined summary
    plot_importance_summary(gam_co, gam_cd, pca_co, pca_cd)

    print("\n" + "=" * 62)
    print("  Analysis complete.  Figures saved to:", OUTPUT_DIR)
    print("=" * 62)
