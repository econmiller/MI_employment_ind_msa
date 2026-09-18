import io
import json
import os
import pandas as pd
import requests

# BLS blocks generic User-Agents. Always supply a descriptive identifier.
HEADERS = {
    "User-Agent": "MichiganEmploymentTracker/1.0 (contact@example.com)"
}
BASE_URL = "https://download.bls.gov/pub/time.series/sm/"


def fetch_bls_tsv(filename: str) -> pd.DataFrame:
  """Fetch a TSV flat file from the BLS server and strip whitespace."""
  print(f"Fetching {filename}...")
  url = BASE_URL + filename
  res = requests.get(url, headers=HEADERS)
  res.raise_for_status()

  # BLS files use tabs; read all fields as strings initially
  df = pd.read_csv(io.StringIO(res.text), sep="\t", dtype=str)
  df.columns = [c.strip() for c in df.columns]
  return df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


def calculate_hierarchy_depth(supersector: str, industry: str) -> int:
  """Determine indentation level (depth) based on BLS industry code structure."""
  if industry == "00000000":
    return 0  # Total Nonfarm (Root)
  if industry in ["05000000", "06000000", "07000000", "08000000"]:
    return 1  # Major Domain Aggregates (Total Private, Goods, Services)
  if industry.endswith("000000"):
    return 2  # Broad Supersector (e.g. Manufacturing 30000000)
  if industry.endswith("0000"):
    return 3  # Subsector / 3-digit NAICS
  if industry.endswith("00"):
    return 4  # 4-digit NAICS industry
  return 5  # Detailed 5-to-6 digit NAICS industry


def run_pipeline():
  print("Starting BLS data pipeline...")

  # 1. Fetch reference lookup tables
  areas_df = fetch_bls_tsv("sm.area")
  industries_df = fetch_bls_tsv("sm.industry")
  series_df = fetch_bls_tsv("sm.series")

  area_dict = dict(zip(areas_df["area_code"], areas_df["area_name"]))
  industry_dict = dict(
      zip(industries_df["industry_code"], industries_df["industry_name"])
  )

  # 2. Filter series metadata
  # Michigan state_code = '26', Employment = '01', Non-Seasonally Adjusted = 'U'
  mi_series = series_df[
      (series_df["state_code"] == "26")
      & (series_df["data_type_code"] == "01")
      & (series_df["seasonal"] == "U")
  ].copy()

  print(f"Found {len(mi_series)} relevant employment series for Michigan.")

  # 3. Fetch Michigan split data files (23a and 23b) and concatenate
  df_23a = fetch_bls_tsv("sm.data.23a.Michigan")
  df_23b = fetch_bls_tsv("sm.data.23b.Michigan")
  data_df = pd.concat([df_23a, df_23b], ignore_index=True)

  # Filter out annual averages ('M13')
  data_df = data_df[data_df["period"] != "M13"].copy()
  data_df["value"] = pd.to_numeric(data_df["value"], errors="coerce")

  # Keep only records matching our target series
  data_df = data_df[data_df["series_id"].isin(mi_series["series_id"])].copy()

  # Create sortable date column: YYYY-MM
  data_df["month_num"] = data_df["period"].str.replace("M", "")
  data_df["year_month"] = (
      data_df["year"] + "-" + data_df["month_num"].str.zfill(2)
  )

  # Identify available periods for MoM and YoY calculations
  distinct_periods = sorted(data_df["year_month"].unique())
  if len(distinct_periods) < 13:
    print("Warning: Fewer than 13 periods found; YoY calculations may be null.")

  latest_period = distinct_periods[-1]
  prior_month = (
      distinct_periods[-2] if len(distinct_periods) >= 2 else latest_period
  )
  prior_year = (
      distinct_periods[-13] if len(distinct_periods) >= 13 else latest_period
  )

  print(
      f"Latest period: {latest_period} | Previous month: {prior_month} |"
      f" Year-ago: {prior_year}"
  )

  # Pivot data: series_id vs target periods
  target_periods = [latest_period, prior_month, prior_year]
  filtered_data = data_df[data_df["year_month"].isin(target_periods)]

  pivoted = filtered_data.pivot_table(
      index="series_id", columns="year_month", values="value", aggfunc="first"
  ).reset_index()

  for p in target_periods:
    if p not in pivoted.columns:
      pivoted[p] = None

  pivoted.rename(
      columns={
          latest_period: "current_val",
          prior_month: "prev_month_val",
          prior_year: "prev_year_val",
      },
      inplace=True,
  )

  # Calculate changes
  pivoted["mom_chg"] = (
      pivoted["current_val"] - pivoted["prev_month_val"]
  ).round(2)
  pivoted["yoy_chg"] = (
      pivoted["current_val"] - pivoted["prev_year_val"]
  ).round(2)
  pivoted["yoy_pct"] = (
      (
          (pivoted["current_val"] - pivoted["prev_year_val"])
          / pivoted["prev_year_val"]
      )
      * 100
  ).round(2)

  # 4. Merge metadata
  merged = pd.merge(mi_series, pivoted, on="series_id", how="inner")

  merged["industry_name"] = merged["industry_code"].map(
      lambda c: industry_dict.get(c, f"Industry {c}")
  )
  merged["area_name"] = merged["area_code"].map(
      lambda c: area_dict.get(c, f"Area {c}")
  )
  merged["depth"] = merged.apply(
      lambda r: calculate_hierarchy_depth(
          r["supersector_code"], r["industry_code"]
      ),
      axis=1,
  )

  # 5. Structure payload grouped by Area Code
  output_data = {
      "metadata": {
          "latest_period": latest_period,
          "prior_month": prior_month,
          "prior_year": prior_year,
          "unit": "Thousands of Persons",
      },
      "areas": {},
      "data": {},
  }

  for area_code, group in merged.groupby("area_code"):
    area_title = area_dict.get(area_code, f"Area {area_code}")
    output_data["areas"][area_code] = area_title

    # Sort so aggregates appear before detailed breakdown
    group_sorted = group.sort_values(by=["industry_code"])

    records = []
    for _, row in group_sorted.iterrows():
      records.append({
          "series_id": row["series_id"],
          "industry_code": row["industry_code"],
          "industry_name": row["industry_name"],
          "depth": int(row["depth"]),
          "current_val": (
              float(row["current_val"])
              if pd.notnull(row["current_val"])
              else None
          ),
          "mom_chg": (
              float(row["mom_chg"]) if pd.notnull(row["mom_chg"]) else None
          ),
          "yoy_chg": (
              float(row["yoy_chg"]) if pd.notnull(row["yoy_chg"]) else None
          ),
          "yoy_pct": (
              float(row["yoy_pct"]) if pd.notnull(row["yoy_pct"]) else None
          ),
      })
    output_data["data"][area_code] = records

  # 6. Save JSON locally
  os.makedirs("public/data", exist_ok=True)
  output_path = "michigan_employment.json"
#  output_path = "public/data/michigan_employment.json"
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

  print(f"\nSuccess! File written to: {output_path}")
  print(f"Total Michigan areas processed: {len(output_data['areas'])}")
  print(
      "Sample area entries (Statewide 00000):"
      f" {len(output_data['data'].get('00000', []))} industries"
  )


if __name__ == "__main__":
  run_pipeline()
