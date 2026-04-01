# Map Refactored
import pandas as pd
import geopandas as gpd
import networkx as nx
import pydeck as pdk
import panel as pn

pn.extension('deckgl')

#####################################
# CONFIG
#####################################

PARQUET_FILE = "Data/rails_processed.parquet"

MAJOR_CARRIERS = ['BNSF', 'CN', 'CP', 'CPKC', 'CSX', 'KCS', 'NS', 'UP']

COLOR_MAP = {
    "UP": [0, 50, 154],
    "CN": [196, 16, 32],
    "BNSF": [0, 64, 134],
    "CP": [196, 16, 32],
    "CPKC": [196, 16, 32],
    "KCS": [237, 28, 36],
    "CSX": [0, 51, 153],
    "NS": [0, 102, 71],
}

#####################################
# DATA LOADER
#####################################

class RailDataLoader:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.df = None

    def load(self):
        self.df = gpd.read_parquet(self.parquet_path)
        self._prepare()
        return self.df

    def _prepare(self):
        # Normalize owner
        self.df["owner"] = (
            self.df["RROWNER1"]
            .fillna("UNKNOWN")
            .str.strip()
            .str.upper()
        )

        # Add color
        self.df["color"] = self.df["owner"].apply(
            lambda x: COLOR_MAP.get(x, [180, 180, 180])
        )

        # Ensure coordinates exist
        self.df = self.df.dropna(subset=["coordinates"])

        # 👇 IMPORTANT: ensure year exists (for slider)
        if "year" not in self.df.columns:
            self.df["year"] = 2020  # fallback default

#####################################
# GRAPH BUILDER
#####################################

class RailGraphBuilder:
    def __init__(self, df):
        self.df = df
        self.G = nx.DiGraph()

    def build(self):
        for row in self.df.itertuples():
            u = row.FRFRANODE
            v = row.TOFRANODE

            self.G.add_edge(
                u,
                v,
                miles=row.MILES,
                owner=row.owner
            )

        return self.G

#####################################
# VISUALIZER
#####################################

class RailVisualizer:
    def __init__(self, df):
        self.df = df

        self.year_slider = pn.widgets.IntSlider(
            name="Year",
            start=int(df["year"].min()),
            end=int(df["year"].max()),
            value=int(df["year"].min())
        )

        self.rail_selector = pn.widgets.CheckBoxGroup(
            name="Railroads",
            options=sorted(df["owner"].unique()),
            value=list(df["owner"].unique())
        )

    def filter_data(self):
        df = self.df[
            (self.df["year"] <= self.year_slider.value) &
            (self.df["owner"].isin(self.rail_selector.value))
        ]
        return df

    def build_layer(self, df):
        return pdk.Layer(
            "PathLayer",
            data=df,
            get_path="coordinates",
            get_color="color",
            width_min_pixels=2,
            pickable=True
        )

    def build_deck(self):
        df = self.filter_data()

        layer = self.build_layer(df)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=39,
                longitude=-98,
                zoom=3
            ),
            tooltip={
                "html": "<b>Owner:</b> {owner}<br><b>Miles:</b> {MILES}"
            }
        )

        return deck

    def view(self):
        return pn.Column(
            "# Rail Network Explorer",
            self.year_slider,
            self.rail_selector,
            pn.bind(self.build_deck)
        )

#####################################
# APP
#####################################

class RailApp:
    def __init__(self):
        self.loader = RailDataLoader(PARQUET_FILE)
        self.df = self.loader.load()

        self.graph = RailGraphBuilder(self.df).build()
        self.visualizer = RailVisualizer(self.df)

    def run(self):
        return self.visualizer.view()

#####################################
# MAIN
#####################################

if __name__ == "__main__":
    app = RailApp()
    pn.serve(app.run(), show=True)