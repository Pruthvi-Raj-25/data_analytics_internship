import time
from pathlib import Path
import pandas as pd
import requests

# Base configuration
BASE_URL = "https://api.mfapi.in/mf"
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Scheme definitions from Tasks 4 & 5
SCHEMES = {
    # Task 4
    "hdfc_top_100_direct": {
        "code": 125497,
        "name": "HDFC Top 100 Direct Plan",
    },
    # Task 5
    "sbi_bluechip": {"code": 119551, "name": "SBI Bluechip Fund"},
    "icici_bluechip": {"code": 120503, "name": "ICICI Prudential Bluechip"},
    "nippon_large_cap": {"code": 118632, "name": "Nippon India Large Cap"},
    "axis_bluechip": {"code": 119092, "name": "Axis Bluechip Fund"},
    "kotak_bluechip": {"code": 120841, "name": "Kotak Bluechip Fund"},
}


def fetch_scheme_nav(
    scheme_key: str, scheme_code: int, scheme_name: str
) -> pd.DataFrame | None:
    """Fetches NAV from mfapi.in, parses the JSON response, and returns a DataFrame."""
    url = f"{BASE_URL}/{scheme_code}"
    print(f"Fetching NAV data for [{scheme_name}] (AMFI: {scheme_code})...")

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            print(f"[!] No NAV records found for scheme {scheme_code}")
            return None

        # Extract metadata and daily historical NAV data
        meta = json_data.get("meta", {})
        nav_records = json_data.get("data", [])

        df = pd.DataFrame(nav_records)

        # Attach metadata as columns for easy joining/analysis
        df["scheme_code"] = meta.get("scheme_code", scheme_code)
        df["scheme_name"] = meta.get("scheme_name", scheme_name)
        df["fund_house"] = meta.get("fund_house", None)
        df["scheme_category"] = meta.get("scheme_category", None)

        # Format columns: convert date and numeric NAV
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.sort_values(by="date", ascending=False).reset_index(drop=True)

        # Save individual raw CSV to data/raw/
        output_file = RAW_DATA_DIR / f"{scheme_key}_{scheme_code}_nav.csv"
        df.to_csv(output_file, index=False)
        print(
            f"-> Successfully saved {len(df):,} records to '{output_file}' | Latest NAV: {df['nav'].iloc[0]} ({df['date'].iloc[0].strftime('%Y-%m-%d')})"
        )

        return df

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data for scheme {scheme_code}: {e}")
        return None


def run_nav_fetch():
    print("=" * 80)
    print("DAY 1: FETCHING LIVE & HISTORICAL NAV VIA MFAPI.IN")
    print("=" * 80)

    all_schemes_data = []

    for key, info in SCHEMES.items():
        df = fetch_scheme_nav(key, info["code"], info["name"])
        if df is not None:
            all_schemes_data.append(df)
        time.sleep(0.5)  # Respectful pause between requests

    # Optionally combine all 6 schemes into one unified key schemes CSV
    if all_schemes_data:
        combined_df = pd.concat(all_schemes_data, ignore_index=True)
        combined_file = RAW_DATA_DIR / "key_schemes_nav_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        print("\n" + "=" * 80)
        print(
            f"Combined dataset saved: {combined_file} ({len(combined_df):,} total rows across {len(all_schemes_data)} schemes)"
        )
        print("=" * 80)


if __name__ == "__main__":
    run_nav_fetch()