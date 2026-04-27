
# Libraries
import pandas as pd
import pyarrow
#import pyarrow.parquet as pq

# Filepaths

WAYBILL_PATHS = ["./Data/stb_waybill_data/2016/PublicUseWaybillSample2016-REV-1-13-23/2016PUWS-REV-1-13-23.txt",
                 "./Data/stb_waybill_data/2017/PublicUseWaybillSample2017-REV-1-13-23/2017PUWS-REV-1-13-23.txt",
                 "./Data/stb_waybill_data/2018/PublicUseWaybillSample2018-REV-1-13-23/2018PUWS-REV-1-13-23.txt",
                 "./Data/stb_waybill_data/2019/PublicUseWaybillSample2019/PublicUseWaybillSample2019.txt",
                 "./Data/stb_waybill_data/2020/PublicUseWaybillSample2020/PublicUseWaybillSample2020.txt",
                 "./Data/stb_waybill_data/2021/PublicUseWaybillSample2021/PublicUseWaybillSample2021.txt",
                 "./Data/stb_waybill_data/2022/PublicUseWaybillSample2022/PublicUseWaybillSample2022.txt",
                 "./Data/stb_waybill_data/2023/PublicUseWaybillSample2023/PublicUseWaybillSample2023.txt"]
COLS_PATH = "./Data/stb_waybill_data/col_sizes_descriptions.csv"
OUTPUT_PATHS = ["./Data/cleaned_waybill_data/waybill_data_2016.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2017.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2018.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2019.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2020.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2021.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2022.parquet",
                "./Data/cleaned_waybill_data/waybill_data_2023.parquet"]

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
for i in range(len(WAYBILL_PATHS)):
    waybill_data = pd.read_fwf(
        WAYBILL_PATHS[i],
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

    waybill_data.drop(columns=["Blank"], inplace=True)
    # ================================
    # Optional: Validate row length
    # ================================
    row_lengths = waybill_data.apply(lambda row: row.astype(str).str.len().sum(), axis=1)
    if not all(row_lengths == 247):
        print("Warning: Some rows may not match 247-byte length!")

    # ================================
    # Save cleaned data
    # ================================
    waybill_data.to_parquet(OUTPUT_PATHS[i], index=False)

    print(f"Waybill data successfully cleaned and saved to {OUTPUT_PATHS[i]}")
    print(waybill_data.head())
