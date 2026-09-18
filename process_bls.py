import io
import json
import os
import pandas as pd
import requests

# BLS blocks generic User-Agents. Always provide an identifier.
HEADERS = {
    "User-Agent": "MichiganEmploymentTracker/1.0 (contact@example.com)"
}
BASE_URL = "https://download.bls.gov/pub/time.series/sm/"

# Aggregate domain codes to exclude so all major supersectors attach directly to Total Nonfarm
EXCLUDED_CODES = {
    "05000000",  # Total Private
    "06000000",  # Goods Producing
    "07000000",  # Service-Providing
    "08000000",  # Private Service Providing
}


def fetch_bls_tsv(filename: str) -> pd.DataFrame:
  """Fetch a TSV flat file from BLS, clean whitespace, and return a DataFrame."""
  print(f"Fetching {filename}...")
  url = BASE_URL + filename
  res = requests.get(url, headers=HEADERS)
  res.raise_for_status()

  df = pd.read_csv(io.StringIO(res.text), sep="\t", dtype=str)
  df.columns = [c.strip() for c in df.columns]
  return df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


def assign_clean_depth(ind_code: str) -> int:
  """Assign clean hierarchy levels.

  Level 0: Total Nonfarm
  Level 1: Major Supersectors (Mining, Construction, Manufacturing, Trade,
  Gov)
  Level 2: Subsectors (Durable Goods 31000000 and Non-Durable 32000000 under
  30000000)
  Level 3+: Detailed NAICS industries
  """
  if ind_code == "00000000":
    return 0

  # Supersectors: Ending in 7 zeros, or broad codes except 31000000 & 32000000
  if ind_code.endswith("0000000") or (
      ind_code.endswith("000000") and ind_code not in ["31000000", "32000000", "41000000", "42000000", "43000000"]
  ):
    return 1

  # Durable Goods & Non-Durable Goods are Level 2 under Manufacturing (30000000)
  if ind_code in ["31000000", "32000000", "41000000", "42000000", "43000000"]:
    return 2

  if ind_code.endswith("0000"):
    return 2
  if ind_code.endswith("00"):
    return 3
  return 4


def build_nested_tree(records: list[dict]) -> list[dict]:
  """Transform a flat list of records into a nested _children tree structure."""
  root_nodes = []
  stack = []

  for item in records:
    # Prepare node with empty children array
    node = dict(item)
    node["_children"] = []
    depth = node["clean_depth"]

    if depth == 0:
      root_nodes.append(node)
      stack.clear()
      stack.append(node)
    else:
      # Pop stack until we find a parent node at an earlier depth
      while stack and stack[-1]["clean_depth"] >= depth:
        stack.pop()

      if stack:
        stack[-1]["_children"].append(node)
      else:
        # Fallback: attach directly under root if orphan
        if root_nodes:
          root_nodes[0]["_children"].append(node)
        else:
          root_nodes.append(node)

      stack.append(node)

  # Clean up empty _children lists so Tabulator doesn't render inactive toggles
  def prune_empty_children(nodes):
    for n in nodes:
      if not n["_children"]:
        del n["_children"]
      else:
        prune_empty_children(n["_children"])

  prune_empty_children(root_nodes)
  return root_nodes


def run_pipeline():
  print("Starting BLS data pipeline...")

  # 1. Reference metadata
  areas_df = fetch_bls_tsv("sm.area")
  industries_df = fetch_bls_tsv("sm.industry")
  series_df = fetch_bls_tsv("sm.series")

  area_dict = dict(zip(areas_df["area_code"], areas_df["area_name"]))
  industry_dict = dict(
      zip(industries_df["industry_code"], industries_df["industry_name"])
  )

  # 2. Filter target Michigan non-seasonally adjusted employment series
  mi_series = series_df[
      (series_df["state_code"] == "26")
      & (series_df["data_type_code"] == "01")
      & (series_df["seasonal"] == "U")
  ].copy()

  # Exclude domain codes
  mi_series = mi_series[~mi_series["industry_code"].isin(EXCLUDED_CODES)]
  print(f"Tracking {len(mi_series)} distinct employment series.")

  # 3. Fetch both Michigan data split files
  df_23a = fetch_bls_tsv("sm.data.23a.Michigan")
  df_23b = fetch_bls_tsv("sm.data.23b.Michigan")
  data_df = pd.concat([df_23a, df_23b], ignore_index=True)

  # Clean values and exclude annual averages (M13)
  data_df = data_df[data_df["period"] != "M13"].copy()
  data_df["value"] = pd.to_numeric(data_df["value"], errors="coerce")
  data_df = data_df[data_df["series_id"].isin(mi_series["series_id"])].copy()

  # Construct sortable YYYY-MM
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

  print(
      f"Latest: {latest_period} | Prior: {prior_month} | Year-Ago:"
      f" {prior_year}"
  )

  # Pivot target periods
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

  # Merge metadata
  merged = pd.merge(mi_series, pivoted, on="series_id", how="inner")
  merged["industry_name"] = merged["industry_code"].map(
      lambda c: industry_dict.get(c, f"Industry {c}")
  )
  merged["area_name"] = merged["area_code"].map(
      lambda c: area_dict.get(c, f"Area {c}")
  )
  merged["clean_depth"] = merged["industry_code"].apply(assign_clean_depth)

  # 4. Build output schema with pre-nested trees
  output_data = {
      "metadata": {
          "latest_period": latest_period,
          "prior_month": prior_month,
          "prior_year": prior_year,
          "unit": "Thousands of Persons",
      },
      "areas": {},
      "trees": {},  # Fully nested trees per area code
  }

  for area_code, group in merged.groupby("area_code"):
    area_title = area_dict.get(area_code, f"Area {area_code}")
    output_data["areas"][area_code] = area_title

    # Sort so parent aggregate rows always precede child industries
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

    # Build and store nested hierarchy for this area
    output_data["trees"][area_code] = build_nested_tree(records)

  # 5. Save output (minified for quick transfer)
  output_path = "michigan_employment.json"
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, separators=(",", ":"))

  print(f"\nSuccess! Pre-nested tree saved to: {output_path}")


if __name__ == "__main__":
  run_pipeline()
