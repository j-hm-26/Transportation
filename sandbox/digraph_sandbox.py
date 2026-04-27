#####################################
# DIGRAPH SANDBOX (CACHED VERSION)
#####################################
#%%
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pydeck as pdk
import numpy as np
import pandas as pd

# -------------------- PATH SETUP -------------------- #
#%%
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

RAW_FILE = os.path.join(DATA_DIR, "NTAD_North_American_Rail_Network_Lines.geojson")
PARQUET_FILE = os.path.join(DATA_DIR, "rails_processed.parquet")
GRAPH_FILE = os.path.join(DATA_DIR, "rail_graph.pkl")

FORCE_REBUILD = True   # <-- Set True to rebuild everything
FILTER_MAJOR_ONLY = False # OPTIONAL FILTER: keep only major carriers, if included will be gray

#%%
def map_rail_code_to_name(code):
    code_to_name = {
        'UP': 'Union Pacific Railroad',
        'BNSF': 'BNSF Railway',
        'NS': 'Norfolk Southern Railway',
        'CSX': 'CSX Transportation',
        'CSXT': 'CSX Transportation',
        'CN': 'Canadian National Railway',
        'CPKC': 'Canadian Pacific Kansas City',
        'KCS': 'Kansas City Southern',  # Now part of CPKC, but included for legacy data
        'CP': 'Canadian Pacific',       # Now part of CPKC, but included for legacy data
    }
    return code_to_name.get(code, 'Unknown or Non-Class I Railroad')
        
# -------------------- LOAD OR BUILD DATAFRAME -------------------- #

if os.path.exists(PARQUET_FILE) and not FORCE_REBUILD:
    print("Loading cached GeoDataFrame from parquet...")
    rails_geojson = gpd.read_parquet(PARQUET_FILE)

else:
    print("Building GeoDataFrame from raw GeoJSON...")
    rails_geojson = gpd.read_file(RAW_FILE)

    # Normalize owner
    rails_geojson["owner"] = (
        rails_geojson["RROWNER1"]
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    # Only the top 9 Major carriers (10 would include XXXX which we cannot ID)
    #major_carriers = rails_geojson["RROWNER1"].value_counts().head(8).index
    # have the major carriers that correspond to our analysis
    major_carriers = ['BNSF', 'CN', 'CP', 'CPKC', 'CSX',
                      'KCS', 'NS', 'UP']

    # Example usage:
    rails_geojson['company_name'] = rails_geojson['RROWNER1'].apply(map_rail_code_to_name)

    if FILTER_MAJOR_ONLY:
        rails_geojson = rails_geojson[
            rails_geojson["RROWNER1"].isin(major_carriers)
        ].copy()

#TODO Change these to have specific colors for each rail
    def owner_color(owner):
        if owner in major_carriers:
            # Major Carriers: 'UP', 'CN', 
            match owner:
                case "UP":
                    color = [0, 50, 154, 256]
                case "CN":
                    color = [196, 16, 32, 256]
                case 'BNSF':
                    color = [0, 64, 134, 256]
                case 'CP':
                    color = [196, 16, 32, 256]
                case 'CPKC':
                    color = [196, 16, 32, 256]
                case 'KCS':
                    color = [237, 28, 36, 256]
                case 'CSX':
                    color = [0, 51, 153, 256]
                # case 'CPRS':
                #     color = [58, 26, 69, 256]
                # case 'CSXT': 
                #     color = [235, 176, 52, 256]
                case 'NS':
                    color = [0, 102, 71, 256]
                # case 'USG':
                #     color = [103, 17, 38, 256]
                # case 'CSAO':
                #     color = [58, 144, 231, 256]
                # case 'PVTX':
                #     color = [53, 151, 88, 256]
            return color
        # otherwise, just make it gray
        return [180, 180, 180, 128]

    unique_owners = rails_geojson["RROWNER1"].unique()
    color_map = {owner: owner_color(owner) for owner in unique_owners}

    rails_geojson["color"] = rails_geojson["RROWNER1"].map(color_map)

    # Simplify geometry
    rails_geojson["geometry"] = rails_geojson["geometry"].simplify(
        tolerance=0.01,
        preserve_topology=True
    )

    #Rename ROWNERS (Private) 
    rails_geojson.loc[rails_geojson["RROWNER1"] == "PVTX", "RROWNER1"] = "Private"

    def linestring_to_coords(geom):
        if geom is None:
            return None
        return [[float(x), float(y)] for x, y in geom.coords]

    rails_geojson["coordinates"] = rails_geojson["geometry"].apply(linestring_to_coords)

    rails_geojson = rails_geojson.dropna(subset=["coordinates"])

    rails_geojson.to_parquet(PARQUET_FILE)
    print("Saved processed GeoDataFrame to parquet.")

#%%

# -------------------- LOAD OR INCREMENTALLY BUILD GRAPH -------------------- #

if os.path.exists(GRAPH_FILE) and not FORCE_REBUILD:
    print("Loading cached graph...")
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
else:
    print("Creating new graph...")
    G = nx.DiGraph()


# Track existing edges to avoid duplicates
existing_edges = set(G.edges())

new_edges = 0

for row in rails_geojson.itertuples(index=False):
    u = row.FRFRANODE
    v = row.TOFRANODE

    if (u, v) in existing_edges:
        continue

    if not G.has_node(u):
        G.add_node(u, pos=row.geometry.coords[0])

    if not G.has_node(v):
        G.add_node(v, pos=row.geometry.coords[-1])

    G.add_edge(
        u,
        v,
        miles=row.MILES,
        owner=row.owner,
        color=row.color
    )

    new_edges += 1

print(f"Added {new_edges} new edges this run.")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")

# Save updated graph
with open(GRAPH_FILE, "wb") as f:
    pickle.dump(G, f)

print("Graph saved.")

#%%
# -------------------- INTERACTIVE PYDECK -------------------- #
deck_df = rails_geojson[[
    "coordinates",
    "color",
    "owner",
    "MILES",
    "STATEAB",
    "TRACKS"
]].copy()

# Convert everything to pure Python types
def clean_row(row):
    color = [int(c) for c in row["color"]]
    if len(color) == 3:
        color.append(255) # Force full opacity
    return {
        "coordinates": [
            # Ensure this is [Longitude, Latitude]
            [float(coord[0]), float(coord[1])] for coord in row["coordinates"]
        ],
        "color": [int(c) for c in row["color"]],
        "owner": str(row["owner"]) if pd.notna(row["owner"]) else "",
        "MILES": float(row["MILES"]) if pd.notna(row["MILES"]) else 0,
        "STATEAB": str(row["STATEAB"]) if pd.notna(row["STATEAB"]) else "",
        "TRACKS": int(row["TRACKS"]) if pd.notna(row["TRACKS"]) else 0
    }

deck_data = [clean_row(row) for _, row in deck_df.iterrows()]
# Convert your clean list back to a DataFrame
clean_df = pd.DataFrame(deck_data)

#%%
layer = pdk.Layer(
    "PathLayer",
    data=clean_df,
    get_path="coordinates",
    get_color="color",
    width_min_pixels=2, # Make it thick so it's visible
    pickable=True
)


tooltip = {
    "html": """
    <b>Owner:</b> {owner} <br/>
    <b>Miles:</b> {MILES} <br/>
    <b>State:</b> {STATEAB} <br/>
    <b>Tracks:</b> {TRACKS}
    """
}

# 3. Use a 'manual' map provider to bypass Mapbox entirely
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=39.0, 
        longitude=-98.0, 
        zoom=3, 
        pitch=0
    ),
    tooltip=tooltip,
    map_style="light"
)

deck.to_html("Rails_Mapping.html")
#%%

# -------------------- TOPOLOGY VISUALIZATION -------------------- #

# pos = {n: G.nodes[n]["pos"] for n in G.nodes}
# deg = dict(G.degree())

# node_colors = [deg[n] for n in G.nodes]

# plt.figure(figsize=(12,12))

# nodes = nx.draw_networkx_nodes(
#     G,
#     pos,
#     node_size=5,
#     node_color=node_colors,
#     cmap="plasma"
# )

# nx.draw_networkx_edges(G, pos, arrows=False, width=0.2)

# plt.colorbar(nodes, label="Node Degree")
# plt.show()
#%%