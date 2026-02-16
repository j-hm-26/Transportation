
# Libraries
import pandas as pd
import pyarrow
#import pyarrow.parquet as pq

# Filepaths

WAYBILL_PATH = "./Data/stb_waybill_data/2023/PublicUseWaybillSample2023/PublicUseWaybillSample2023.txt"
COLS_PATH = "./Data/stb_waybill_data/col_sizes_descriptions.csv"
OUTPUT_PATH = "./Data/cleaned_waybill_data/waybill_data_2023.parquet"

# ================================
# Load column layout
# ================================
colnames_raw = pd.read_csv(COLS_PATH)
colnames = colnames_raw.copy()

# Build FWF colspecs (0-indexed)
colnames["indicies"] = list(zip(colnames["Start Position"] - 1, colnames["End Position"]))

# Clean metadata
colnames["dtype"] = colnames["dtype"].str.strip()
colnames["Data Description"] = colnames["Data Description"].str.strip()

# Extract numeric columns
numeric_cols = colnames.loc[colnames["dtype"] == "float", "Data Description"].tolist()

# ================================
# Load waybill data
# ================================
waybill_data = pd.read_fwf(
    WAYBILL_PATH,
    colspecs=colnames["indicies"].tolist(),
    names=colnames["Data Description"].tolist(),
    dtype=str,   # Load everything as string to preserve leading zeros and avoid parsing errors
    header=None
)

# ================================
# Clean data
# ================================
# Strip whitespace
waybill_data = waybill_data.apply(lambda x: x.str.strip())

# Convert numeric columns safely
waybill_data[numeric_cols] = waybill_data[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Parse date columns
waybill_data["Waybill Date (mm/dd/yy)"] = pd.to_datetime(
    waybill_data["Waybill Date (mm/dd/yy)"], format="%m%d%y", errors="coerce"
)

waybill_data["Accounting Period (mm/yy)"] = pd.to_datetime(
    waybill_data["Accounting Period (mm/yy)"], format="%m%y", errors="coerce"
)

# ================================
# Optional: Validate row length
# ================================
row_lengths = waybill_data.apply(lambda row: row.astype(str).str.len().sum(), axis=1)
if not all(row_lengths == 247):
    print("Warning: Some rows may not match 247-byte length!")

# ================================
# Save cleaned data
# ================================
waybill_data.to_parquet(OUTPUT_PATH, index=False)

print(f"Waybill data successfully cleaned and saved to {OUTPUT_PATH}")
print(waybill_data.head())