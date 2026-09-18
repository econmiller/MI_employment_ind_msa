import io
import json
import os
import shutil
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "MichiganEmploymentTracker/1.0 (contact@example.com)"
}
BASE_URL = "https://download.bls.gov/pub/time.series/sm/"

EXCLUDED_CODES = {
    "05000000",  # Total Private
    "06000000",  # Goods Producing
    "07000000",  # Service-Providing
    "08000000",  # Private Service Providing
}


def fetch_bls_tsv(filename: str) -> pd.DataFrame:
  print(f"Fetching {filename}...")
  url = BASE_URL + filename
  res = requests.get(url, headers=HEADERS)
  res.raise_for_status()

  df = pd.read_csv(io.StringIO(res.text), sep="\t", dtype=str)
  df.columns = [c.strip() for c in df.columns]
  return df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


def assign_clean_depth(ind_code: str) -> int:
  if ind_code == "00000000":
    return 0

  # Supersectors: Level 1
  if ind_code.endswith("0000000") or (
      ind_code.endswith("000000")
      and ind_code
      not in ["31000000", "32000000", "41000000", "42000000", "43000000"]
  ):
    return 1

  # Subsectors: Level 2
  if ind_code in [
      "31000000",
      "32000000",
      "41000000",
      "42000000",
      "43000000",
  ]:
    return 2

  if ind_code.endswith("0000"):
    return 2
  if ind_code.endswith("00"):
    return 3
  return 4


def run_pipeline():
  print("Starting BLS data pipeline...")

  # 1. Fetch metadata
  areas_df = fetch_bls_tsv("sm.area")
  industries_df = fetch_bls_tsv("sm.industry")
  series_df = fetch_bls_tsv("sm.series")

  area_dict = dict(zip(areas_df["area_code"], areas_df["area_name"]))
  industry_dict = dict(
      zip(industries_df["industry_code"], industries_df["industry_name"])
  )

  # Filter Michigan Non-Seasonally Adjusted Employment series
  mi_series = series_df[
      (series_df["state_code"] == "26")
      & (series_df["data_type_code"] == "01")
      & (series_df["seasonal"] == "U")
  ].copy()
  mi_series = mi_series[~mi_series["industry_code"].isin(EXCLUDED_CODES)]

  # 2. Fetch split data
  df_23a = fetch_bls_tsv("sm.data.23a.Michigan")
  df_23b = fetch_bls_tsv("sm.data.23b.Michigan")
  data_df = pd.concat([df_23a, df_23b], ignore_index=True)

  data_df = data_df[data_df["period"] != "M13"].copy()
  data_df["value"] = pd.to_numeric(data_df["value"], errors="coerce")
  data_df = data_df[data_df["series_id"].isin(mi_series["series_id"])].copy()

  data_df["month_num"] = data_df["period"].str.replace("M", "")
  data_df["year_month"] = (
      data_df["year"] + "-" + data_df["month_num"].str.zfill(2)
  )

  distinct_periods = sorted(data_df["year_month"].unique())
  latest_period = distinct_periods[-1]
  prior_month = (
      distinct_periods[-2] if len(distinct_periods) >= 2 else latest_period
  )
  prior_year = (
      distinct_periods[-13] if len(distinct_periods) >= 13 else latest_period
  )

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

  merged = pd.merge(mi_series, pivoted, on="series_id", how="inner")
  merged["industry_name"] = merged["industry_code"].map(
      lambda c: industry_dict.get(c, f"Industry {c}")
  )
  merged["area_name"] = merged["area_code"].map(
      lambda c: area_dict.get(c, f"Area {c}")
  )
  merged["clean_depth"] = merged["industry_code"].apply(assign_clean_depth)

  # OPTIMIZATION 1: Drop all levels deeper than Level 3
  merged = merged[merged["clean_depth"] <= 3].copy()

  # OPTIMIZATION 2: Split output into per-area files
  output_dir = os.path.join("public", "data", "areas")
  os.makedirs(output_dir, exist_ok=True)

  metadata_payload = {
      "metadata": {
          "latest_period": latest_period,
          "prior_month": prior_month,
          "prior_year": prior_year,
          "unit": "Thousands of Persons",
      },
      "areas": {},
  }

  for area_code, group in merged.groupby("area_code"):
    area_title = area_dict.get(area_code, f"Area {area_code}")
    metadata_payload["areas"][area_code] = area_title

    # Preserve natural parent-before-child ordering
    group_sorted = group.sort_values(by=["industry_code"])

    records = []
    for _, row in group_sorted.iterrows():
      records.append({
          "series_id": row["series_id"],
          "industry_code": row["industry_code"],
          "industry_name": row["industry_name"],
          "clean_depth": int(row["clean_depth"]),
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

    # Save per-area JSON file
    area_file_path = os.path.join(output_dir, f"{area_code}.json")
    with open(area_file_path, "w", encoding="utf-8") as f:
      json.dump(records, f, separators=(",", ":"))

  # Save metadata JSON file
  meta_file_path = os.path.join("public", "data", "metadata.json")
  with open(meta_file_path, "w", encoding="utf-8") as f:
    json.dump(metadata_payload, f, separators=(",", ":"))

  print(f"\nPipeline finished.")
  print(f"Metadata written to: {meta_file_path}")
  print(f"Generated {len(metadata_payload['areas'])} area files in {output_dir}")


if __name__ == "__main__":
  run_pipeline()
