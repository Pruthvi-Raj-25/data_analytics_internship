from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")


def load_and_inspect_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Task 3: Load all provided CSV datasets and inspect properties & anomalies."""
    print("=" * 80)
    print("TASK 3: LOADING AND INSPECTING DATASETS")
    print("=" * 80)

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    datasets = {}

    for idx, filepath in enumerate(csv_files, start=1):
        name = filepath.stem
        df = pd.read_csv(filepath)
        datasets[name] = df

        print(f"\n[{idx}/{len(csv_files)}] Dataset: {filepath.name}")
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print("\nData Types & Null Counts:")
        info_df = pd.DataFrame(
            {
                "Dtype": df.dtypes.astype(str),
                "Null_Count": df.isnull().sum(),
                "Null_Pct": (df.isnull().sum() / len(df) * 100).round(2),
            }
        )
        print(info_df.to_string())

        print("\nSample Data (First 3 rows):")
        print(df.head(3).to_string())

        # Anomaly checks
        anomalies = []
        if df.duplicated().sum() > 0:
            anomalies.append(f"{df.duplicated().sum()} duplicate rows detected")

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if "nav" in col.lower() or "aum" in col.lower():
                negative_vals = (df[col] < 0).sum()
                if negative_vals > 0:
                    anomalies.append(
                        f"Found {negative_vals} negative values in {col}"
                    )

        if anomalies:
            print(f"-> Anomalies noted: {', '.join(anomalies)}")
        else:
            print("-> No obvious anomalies detected.")
        print("-" * 80)

    return datasets


def explore_fund_master(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Task 6: Explore fund master schema, unique houses, categories, and risk grades."""
    print("\n" + "=" * 80)
    print("TASK 6: EXPLORING FUND MASTER")
    print("=" * 80)

    # Match key matching 'fund_master' (case-insensitive)
    fund_master_key = next(
        (k for k in datasets if "fund_master" in k.lower() or "master" in k.lower()),
        None,
    )

    if not fund_master_key:
        print("[!] fund_master dataset not found among loaded CSVs.")
        return None

    fm = datasets[fund_master_key]

    # Helper function to find case-insensitive column names
    def find_col(candidates):
        for col in fm.columns:
            if col.strip().lower() in candidates:
                return col
        return None

    fund_house_col = find_col(["fund_house", "amc_name", "amc", "fund_family"])
    category_col = find_col(["category", "scheme_category", "broad_category"])
    sub_category_col = find_col(
        ["sub_category", "scheme_sub_category", "sub_cat"]
    )
    risk_col = find_col(["risk_grade", "risk_level", "risk_rating", "risk"])
    code_col = find_col(["scheme_code", "amfi_code", "amfi_scheme_code", "code"])

    print(
        f"Fund Master Total Records: {len(fm):,} across {fm.shape[1]} features"
    )

    if fund_house_col:
        print(f"\nUnique Fund Houses ({fm[fund_house_col].nunique()}):")
        print(fm[fund_house_col].dropna().unique()[:10], "... [Truncated]")

    if category_col:
        print(f"\nUnique Categories ({fm[category_col].nunique()}):")
        print(fm[category_col].dropna().unique())

    if sub_category_col:
        print(f"\nUnique Sub-Categories ({fm[sub_category_col].nunique()}):")
        print(fm[sub_category_col].dropna().unique()[:10], "... [Truncated]")

    if risk_col:
        print(f"\nUnique Risk Grades ({fm[risk_col].nunique()}):")
        print(fm[risk_col].value_counts(dropna=False).to_string())

    if code_col:
        print("\nAMFI Scheme Code Structure:")
        print(
            f"Field: '{code_col}' | Data Type: {fm[code_col].dtype} | Unique: {fm[code_col].nunique()}"
        )
        print("Sample AMFI Codes:", fm[code_col].dropna().head(5).tolist())

    return fm


def validate_amfi_codes(datasets: dict[str, pd.DataFrame]):
    """Task 7: Confirm every code in fund_master exists in nav_history."""
    print("\n" + "=" * 80)
    print("TASK 7: VALIDATING AMFI CODES (FUND_MASTER vs NAV_HISTORY)")
    print("=" * 80)

    fm_key = next((k for k in datasets if "master" in k.lower()), None)
    nav_key = next((k for k in datasets if "nav" in k.lower()), None)

    if not fm_key or not nav_key:
        print(
            f"[!] Cannot perform validation. Found master: {fm_key}, Found nav: {nav_key}"
        )
        return

    fm_df = datasets[fm_key]
    nav_df = datasets[nav_key]

    # Resolve AMFI code column names
    def get_code_col(df):
        for col in df.columns:
            if any(term in col.lower() for term in ["code", "amfi"]):
                return col
        return df.columns[0]

    fm_code_col = get_code_col(fm_df)
    nav_code_col = get_code_col(nav_df)

    fm_codes = set(fm_df[fm_code_col].dropna().astype(str).str.strip())
    nav_codes = set(nav_df[nav_code_col].dropna().astype(str).str.strip())

    missing_in_nav = fm_codes - nav_codes
    missing_in_fm = nav_codes - fm_codes

    match_pct = (
        (len(fm_codes - missing_in_nav) / len(fm_codes) * 100)
        if len(fm_codes) > 0
        else 0
    )

    print(f"Total Unique Scheme Codes in {fm_key}: {len(fm_codes)}")
    print(f"Total Unique Scheme Codes in {nav_key}: {len(nav_codes)}")
    print(f"Codes in {fm_key} matching into {nav_key}: {match_pct:.2f}%")

    if missing_in_nav:
        print(
            f"\n[Warning] {len(missing_in_nav)} scheme code(s) in fund_master are missing from nav_history."
        )
        print("Sample missing codes:", list(missing_in_nav)[:5])
    else:
        print(
            f"\n[Validation Success] All scheme codes in {fm_key} exist in {nav_key}."
        )

    # Short Data Quality Summary
    print("\n" + "=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    print(f"1. Total Datasets Processed: {len(datasets)}")
    print(
        f"2. Referential Integrity: {len(fm_codes - missing_in_nav)}/{len(fm_codes)} master codes found in NAV history."
    )
    print(
        f"3. Orphan NAV Codes: {len(missing_in_fm)} codes present in nav_history but not defined in fund_master."
    )
    print("=" * 80)


if __name__ == "__main__":
    loaded_datasets = load_and_inspect_datasets(RAW_DATA_DIR)
    explore_fund_master(loaded_datasets)
    validate_amfi_codes(loaded_datasets)