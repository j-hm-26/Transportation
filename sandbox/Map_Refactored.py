# =========================
# IMPORTS
# =========================
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import panel as pn

pn.extension("deckgl")


# =========================
# CONFIG
# =========================
PARQUET_FILE = "Data/rails_processed.parquet"
CORR_FILE = "Data/correlations.csv"

DEFAULT_COLOR = [180, 180, 180]


# =========================
# DATA LOADER
# =========================
class RailDataLoader:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def load(self):
        df = gpd.read_parquet(self.parquet_path)

        # -------------------------
        # Normalize owner
        # -------------------------
        df["owner"] = (
            df["RROWNER1"]
            .fillna("UNKNOWN")
            .str.strip()
            .str.upper()
        )

        # -------------------------
        # Convert geometry → coordinates
        # -------------------------
        def geom_to_coords(geom):
            if geom is None:
                return None

            try:
                if geom.geom_type == "LineString":
                    return [[float(x), float(y)] for x, y in geom.coords]

                elif geom.geom_type == "MultiLineString":
                    coords = []
                    for line in geom.geoms:
                        coords.extend([[float(x), float(y)] for x, y in line.coords])
                    return coords
            except:
                return None

            return None

        df["coordinates"] = df.geometry.apply(geom_to_coords)

        # Drop invalid rows
        df = df.dropna(subset=["coordinates"])

        # Ensure numeric miles
        df["MILES"] = pd.to_numeric(df["MILES"], errors="coerce")

        return df


# =========================
# VISUALIZER
# =========================
class RailVisualizer:
    def __init__(self, df, corr_df):
        self.df = df
        self.corr_df = corr_df

        # -------------------------
        # Slider for animation
        # -------------------------
        self.week_slider = pn.widgets.IntSlider(
            name="Week Block",
            start=int(corr_df["week_block"].min()),
            end=int(corr_df["week_block"].max()),
            value=int(corr_df["week_block"].min())
        )

    # -------------------------
    # Build correlation lookup
    # -------------------------
    def build_corr_lookup(self, week):
        df = self.corr_df[self.corr_df["week_block"] == week]

        corr_map = {}

        for row in df.itertuples():
            a = row.Company_A.upper()
            b = row.Company_B.upper()

            # If no numeric correlation column, simulate strength
            val = getattr(row, "correlation", None)

            # If correlation not provided, assume binary strength
            if val is None:
                val = 1.0

            corr_map.setdefault(a, []).append((b, val))
            corr_map.setdefault(b, []).append((a, val))

        return corr_map

    # -------------------------
    # Apply correlation coloring
    # -------------------------
    def apply_correlation(self, df, week):
        corr_map = self.build_corr_lookup(week)

        colors = []
        hover_pairs = []
        hover_vals = []

        for owner in df["owner"]:
            relations = corr_map.get(owner, [])

            if not relations:
                colors.append(DEFAULT_COLOR)
                hover_pairs.append("None")
                hover_vals.append(0)
                continue

            # Get strongest correlation
            partner, val = max(relations, key=lambda x: abs(x[1]))

            # Color logic
            if val > 0.5:
                color = [0, 0, 255]      # Blue (positive)
            elif val < -0.5:
                color = [255, 0, 0]      # Red (negative)
            else:
                color = DEFAULT_COLOR

            colors.append(color)
            hover_pairs.append(partner)
            hover_vals.append(round(val, 3))

        df = df.copy()
        df["color"] = colors
        df["pair"] = hover_pairs
        df["corr"] = hover_vals

        return df

    # -------------------------
    # Build layer
    # -------------------------
    def build_layer(self, df):
        df_clean = pd.DataFrame({
            "coordinates": df["coordinates"],
            "color": df["color"],
            "owner": df["owner"],
            "pair": df["pair"],
            "corr": df["corr"],
            "MILES": df["MILES"]
        }).dropna()

        return pdk.Layer(
            "PathLayer",
            data=df_clean,
            get_path="coordinates",
            get_color="color",
            width_min_pixels=2,
            pickable=True
        )

    # -------------------------
    # Build deck
    # -------------------------
    def build_deck(self):
        week = self.week_slider.value

        df = self.apply_correlation(self.df, week)

        layer = self.build_layer(df)

        return pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=39,
                longitude=-98,
                zoom=3
            ),
            tooltip={
                "html": """
                <b>Owner:</b> {owner}<br>
                <b>Top Pair:</b> {pair}<br>
                <b>Correlation:</b> {corr}<br>
                <b>Miles:</b> {MILES}
                """
            }
        )

    # -------------------------
    # Panel view (animated)
    # -------------------------
    def view(self):
        return pn.Column(
            "# Rail Correlation Explorer",
            self.week_slider,
            pn.bind(lambda w: pdk.Deck(
                layers=[self.build_layer(self.apply_correlation(self.df, w))],
                initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3),
                tooltip={
                    "html": """
                    <b>Owner:</b> {owner}<br>
                    <b>Top Pair:</b> {pair}<br>
                    <b>Correlation:</b> {corr}
                    """
                }
            ), self.week_slider)
        )


# =========================
# APP
# =========================
class RailApp:
    def __init__(self):
        self.df = RailDataLoader(PARQUET_FILE).load()
        self.corr_df = pd.read_csv(CORR_FILE)

        self.visualizer = RailVisualizer(self.df, self.corr_df)

    def run(self):
        return self.visualizer.view()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app = RailApp()
    pn.serve(app.run(), show=True)