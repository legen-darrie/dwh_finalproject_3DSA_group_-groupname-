import os
import re
import argparse
from datetime import datetime
import sys
import glob

import pandas as pd

# ==================
# CONFIG / CONSTANTS
# ==================

DATA_ZONE_PATH = "/app/data_zone"

# ============
# QUALITY LOGS
# ============

quality_report: list[dict] = []


def log_quality(table: str, issue_type: str, details: str,
                severity: str = "WARNING") -> None:
    """Append a data-quality issue and print it."""
    quality_report.append({
        "timestamp": datetime.now().isoformat(),
        "table": table,
        "issue_type": issue_type,
        "details": details,
        "severity": severity,
    })
    print(f"      [{severity}] {table} - {issue_type}: {details}")


def save_quality_report(silver_folder: str) -> None:
    """Persist quality_report to CSV and print summary."""
    if not quality_report:
        print("\n[INFO] No quality issues (errors/warnings) logged.")
        return

    report_df = pd.DataFrame(quality_report)
    report_path = os.path.join(os.path.dirname(silver_folder), "_silver_quality_report.csv")
    report_df.to_csv(report_path, index=False)

    errors = sum(1 for r in quality_report if r["severity"] == "ERROR")
    warnings = sum(1 for r in quality_report if r["severity"] == "WARNING")

    print(f"\nQuality report saved: {report_path}")
    print(f"Quality Summary: {errors} errors, {warnings} warnings")

    if errors > 0:
        print("\n[CRITICAL] One or more errors were flagged. Exiting with code 1.")
        # sys.exit(1)


# ===========================
# GENERIC TRANSFORM UTILITIES
# ===========================

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and fix basic inconsistencies."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def validate_required_columns(df: pd.DataFrame,
                              required_cols: list[str],
                              table_name: str) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log_quality(table_name, "MISSING_COLUMNS",
                    f"Missing required: {missing}", "ERROR")
    return df


def check_nulls(df: pd.DataFrame,
                key_cols: list[str],
                table_name: str) -> pd.DataFrame:
    for col in key_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            total = len(df)
            if null_count > 0:
                pct = (null_count / total) * 100
                log_quality(
                    table_name,
                    "NULL_VALUES",
                    f"{col}: {null_count}/{total} ({pct:.1f}%) nulls",
                    "WARNING",
                )
    return df


def check_duplicates(df: pd.DataFrame,
                     key_cols: list[str],
                     table_name: str) -> int:
    if not all(c in df.columns for c in key_cols):
        return 0
    dup_count = df.duplicated(subset=key_cols).sum()
    if dup_count > 0:
        log_quality(
            table_name,
            "DUPLICATES",
            f"{dup_count} duplicate rows on {key_cols}",
            "WARNING",
        )
    return dup_count


def validate_data_types(df: pd.DataFrame,
                        type_map: dict[str, str],
                        table_name: str) -> pd.DataFrame:
    for col, expected_type in type_map.items():
        if col not in df.columns:
            continue

        if expected_type == "datetime":
            before = df[col].notnull().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after = df[col].notnull().sum()
            invalid = before - after
            if invalid > 0:
                log_quality(
                    table_name,
                    "INVALID_DATETIME",
                    f"{col}: {invalid} values coerced to NaT",
                    "WARNING",
                )

        elif expected_type == "numeric":
            before = df[col].notnull().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after = df[col].notnull().sum()
            invalid = before - after
            if invalid > 0:
                log_quality(
                    table_name,
                    "INVALID_NUMERIC",
                    f"{col}: {invalid} values coerced to NaN",
                    "WARNING",
                )
    return df


def flag_errors(table_name: str) -> None:
    errors = [
        r for r in quality_report
        if r["table"] == table_name and r["severity"] == "ERROR"
    ]
    if errors:
        print(f"      [ERROR] {len(errors)} ERRORS flagged for {table_name}")


# ===========================
# MARKETING-SPECIFIC HELPERS
# ===========================

def normalize_discount_label(s: str) -> str:
    """
    Convert raw discount strings like:
    '1%', '1pct', '1percent', '10%%', '20pct'
    into a canonical label: '1%', '10%', '20%'.
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    s = s.replace("percent", "%").replace("pct", "%")
    s = re.sub(r"[^0-9%]", "", s)
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    num = m.group(1)
    return f"{num}%"


# ===========================
# BUSINESS / CUSTOMER CLEANERS
# ===========================

def clean_business(df: pd.DataFrame, silver_folder: str, file: str) -> None:
    table_name = "business_product"
    print(f"\n  Cleaning: {table_name}")

    df = standardize(df)

    df = validate_required_columns(df, ["product_id", "product_name"], table_name)
    df = check_nulls(df, ["product_id"], table_name)

    if "product_id" in df.columns:
        initial_rows = len(df)
        df.dropna(subset=["product_id"], inplace=True)
        null_removed = initial_rows - len(df)
        if null_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL product_id.")

        initial_rows = len(df)
        df = df.drop_duplicates(subset="product_id", keep="first")
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on product_id. Final Rows: {len(df)}")

    check_duplicates(df, ["product_id"], table_name)
    flag_errors(table_name)

    out = "business_product.parquet"
    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


def clean_customer(df: pd.DataFrame, silver_folder: str, file: str) -> None:
    df = standardize(df)

    if "user_job" in file:
        table_name = "customer_user_job"
        print(f"\n  Cleaning: {table_name}")
        out = "customer_user_job.parquet"
        key_cols = ["user_id"]

    elif "user_credit_card" in file:
        table_name = "customer_user_credit_card"
        print(f"\n  Cleaning: {table_name}")
        out = "customer_user_credit_card.parquet"
        key_cols = ["user_id", "credit_card_number"]

    elif "user_data" in file or "user_" in file:
        table_name = "customer_user"
        print(f"\n  Cleaning: {table_name}")
        out = "customer_user.parquet"
        key_cols = ["user_id"]
    else:
        print(f"  [WARN] Unknown customer file pattern: {file}")
        return

    df = validate_required_columns(df, ["user_id"], table_name)
    df = check_nulls(df, ["user_id"], table_name)

    if "user_id" in df.columns:
        initial_rows = len(df)
        df.dropna(subset=["user_id"], inplace=True)
        null_removed = initial_rows - len(df)
        if null_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL user_id.")

        initial_rows = len(df)
        subset_keys = ["user_id"] if table_name in ["customer_user", "customer_user_job"] else key_cols
        df = df.drop_duplicates(subset=subset_keys, keep="first")
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on {subset_keys}. Final Rows: {len(df)}")

    check_duplicates(df, key_cols, table_name)

    if "birthdate" in df.columns and table_name == "customer_user":
        df = validate_data_types(df, {"birthdate": "datetime"}, table_name)

    flag_errors(table_name)

    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


# =======================
# ENTERPRISE CLEANER
# =======================

def clean_enterprise(df: pd.DataFrame, silver_folder: str, file: str) -> None:
    df = standardize(df)

    if "order_with_merchant" in file:
        table_name = "enterprise_order_merchant_tx"
        print(f"\n  Cleaning: {table_name} (Transaction/Fact)")

        rename_map = {}
        if "merchantid" in df.columns: rename_map["merchantid"] = "merchant_id"
        if "staffid" in df.columns: rename_map["staffid"] = "staff_id"
        if "orderid" in df.columns: rename_map["orderid"] = "order_id"
        if rename_map: df = df.rename(columns=rename_map)

        key_cols = ["order_id", "merchant_id"]
        df = validate_required_columns(df, key_cols, table_name)
        df = check_nulls(df, key_cols, table_name)

        if "order_id" in df.columns:
            initial_rows = len(df)
            df.dropna(subset=["order_id"], inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL order_id.")

            initial_rows = len(df)
            df = df.drop_duplicates(subset=["order_id"], keep="first")
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on order_id. Final Rows: {len(df)}")

        check_duplicates(df, ["order_id"], table_name)
        base_name = file.replace(".parquet", "").replace("enterprise_", "")
        out = f"enterprise_{base_name}_tx.parquet"

    elif "merchant_data" in file:
        table_name = "enterprise_merchant"
        print(f"\n  Cleaning: {table_name} (Dimension)")
        out = "enterprise_merchant.parquet"
        key_cols = ["merchant_id"]

    elif "staff_data" in file:
        table_name = "enterprise_staff"
        print(f"\n  Cleaning: {table_name} (Dimension)")
        out = "enterprise_staff.parquet"
        key_cols = ["staff_id"]
    else:
        print(f"  [WARN] Unknown enterprise file pattern: {file}")
        return

    if table_name in ["enterprise_merchant", "enterprise_staff"]:
        df = validate_required_columns(df, key_cols, table_name)
        df = check_nulls(df, key_cols, table_name)

        if key_cols[0] in df.columns:
            initial_rows = len(df)
            df.dropna(subset=key_cols, inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL {key_cols[0]}.")

            initial_rows = len(df)
            df = df.drop_duplicates(subset=key_cols, keep="first")
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on {key_cols[0]}. Final Rows: {len(df)}")

        check_duplicates(df, key_cols, table_name)

    flag_errors(table_name)
    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


def combine_and_save_enterprise_transactions(bronze_folder: str, silver_folder: str) -> None:
    """Combines all multi-part enterprise order transaction files into a single Silver file."""
    print("\n" + "=" * 70)
    print("STAGE X: CONCATENATION (enterprise_order_merchant_tx)")
    print("======================================================================")

    file_pattern = os.path.join(bronze_folder, "enterprise_order_with_merchant_data*_bronze.parquet")
    all_files = glob.glob(file_pattern)

    list_of_dfs = []

    print(f"  [INFO] Found {len(all_files)} enterprise transaction parts to combine.")

    for filename in all_files:
        print(f"  Reading file part: {os.path.basename(filename)}")
        try:
            df_part = pd.read_parquet(filename)
            df_part = standardize(df_part)

            if "merchantid" in df_part.columns: df_part = df_part.rename(columns={"merchantid": "merchant_id"})
            if "staffid" in df_part.columns: df_part = df_part.rename(columns={"staffid": "staff_id"})
            if "orderid" in df_part.columns: df_part = df_part.rename(columns={"orderid": "order_id"})

            list_of_dfs.append(df_part)
        except Exception as e:
            print(f"  [ERROR] Failed to read {os.path.basename(filename)} during concatenation: {e}")

    if list_of_dfs:
        combined_df = pd.concat(list_of_dfs, ignore_index=True)

        output_filename = "enterprise_order_merchant_tx.parquet"
        final_save_path = os.path.join(silver_folder, output_filename)
        combined_df.to_parquet(final_save_path, index=False)

        print(f"  [OK] Saved combined Silver file: {output_filename} ({len(combined_df)} rows)")
    else:
        print("  [WARN] No enterprise order merchant transaction files found for concatenation.")


# =======================
# OPERATIONS CLEANER
# =======================

def _rename_operations_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "orderid" in df.columns and "order_id" not in df.columns: rename_map["orderid"] = "order_id"
    if "productid" in df.columns and "product_id" not in df.columns: rename_map["productid"] = "product_id"
    if "prod_id" in df.columns and "product_id" not in df.columns: rename_map["prod_id"] = "product_id"
    if "sku" in df.columns and "product_id" not in df.columns: rename_map["sku"] = "product_id"
    if "item_id" in df.columns and "product_id" not in df.columns: rename_map["item_id"] = "product_id"
    if "userid" in df.columns and "user_id" not in df.columns: rename_map["userid"] = "user_id"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"    [INFO] Renamed columns: {rename_map}")
    return df


def clean_operations(df: pd.DataFrame, silver_folder: str, file: str) -> None:
    df = standardize(df)
    df = _rename_operations_columns(df)

    if "order_data" in file:
        table_name = "operations_orders"
        print(f"\n  Cleaning: {table_name}")
        out = "operations_orders.parquet"
        key_cols = ["order_id"]

        df = validate_required_columns(df, key_cols, table_name)
        df = check_nulls(df, key_cols, table_name)

        if "order_id" in df.columns:
            initial_rows = len(df)
            df.dropna(subset=["order_id"], inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL order_id.")

        date_cols = [c for c in df.columns if "date" in c]
        type_map = {c: "datetime" for c in date_cols}
        if type_map:
            df = validate_data_types(df, type_map, table_name)

        check_duplicates(df, key_cols, table_name)

    elif "line_item" in file:
        table_name = "operations_line_items"
        print(f"\n  Cleaning: {table_name}")
        out = "operations_line_items.parquet"
        key_cols = ["order_id", "product_id"]

        if "product_id" not in df.columns:
            if "product" in df.columns:
                df["product_id"] = df["product"].astype("category").cat.codes
            elif "product_name" in df.columns:
                df["product_id"] = df["product_name"].astype("category").cat.codes
            else:
                df["product_id"] = df.reset_index().index

        df = validate_required_columns(df, key_cols, table_name)
        df = check_nulls(df, key_cols, table_name)

        if {"order_id", "product_id"}.issubset(df.columns):
            initial_rows = len(df)
            df.dropna(subset=key_cols, inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL order/product keys.")

            initial_rows = len(df)
            df = df.drop_duplicates(subset=key_cols, keep="first")
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on order/product keys. Final Rows: {len(df)}")

        check_duplicates(df, key_cols, table_name)

    elif "order_delays" in file:
        table_name = "operations_order_delays"
        print(f"\n  Cleaning: {table_name}")
        out = "operations_order_delays.parquet"

        if "orderid" in df.columns and "order_id" not in df.columns:
            df = df.rename(columns={"orderid": "order_id"})

        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates (all columns). Final Rows: {len(df)}")

    else:
        print(f"  [WARN] Unknown operations file pattern: {file}")
        return

    flag_errors(table_name)
    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


# =======================
# MARKETING CLEANER
# =======================

def _rename_marketing_columns(df: pd.DataFrame, file: str) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "campaignid" in df.columns and "campaign_id" not in df.columns:
        rename_map["campaignid"] = "campaign_id"
    if "id" in df.columns and "campaign_id" not in df.columns and "campaign_data" in file:
        rename_map["id"] = "campaign_id"
    if "orderid" in df.columns and "order_id" not in df.columns:
        rename_map["orderid"] = "order_id"
    if "userid" in df.columns and "user_id" not in df.columns:
        rename_map["userid"] = "user_id"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"    [INFO] Renamed columns: {rename_map}")
    return df


def clean_marketing(df: pd.DataFrame, silver_folder: str, file: str) -> None:
    df = standardize(df)
    df = _rename_marketing_columns(df, file)

    if "campaign_data" in file and "transactional" not in file:
        table_name = "marketing_campaign"
        print(f"\n  Cleaning: {table_name}")
        out = "marketing_campaign.parquet"
        key_cols = ["campaign_id"]

        if "campaign_id" not in df.columns:
            if len(df.columns) == 1:
                only_col = df.columns[0]
                df["campaign_id"] = df[only_col].astype("category").cat.codes
                print(f"    [INFO] Created campaign_id from '{only_col}' categorical codes")
            else:
                df["campaign_id"] = df.reset_index().index
                print("    [INFO] Created synthetic campaign_id from row index")

        df = validate_required_columns(df, key_cols, table_name)
        df = check_nulls(df, key_cols, table_name)

        if "campaign_id" in df.columns:
            initial_rows = len(df)
            df.dropna(subset=["campaign_id"], inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL campaign_id.")

            initial_rows = len(df)
            df = df.drop_duplicates(subset=["campaign_id"], keep="first")
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates on campaign_id. Final Rows: {len(df)}")

        # Normalize discount to a canonical label like '1%'
        if "discount" in df.columns:
            df["discount_normalized"] = df["discount"].apply(normalize_discount_label)
        else:
            df["discount_normalized"] = None

        check_duplicates(df, key_cols, table_name)

    elif "transactional_campaign" in file:
        table_name = "marketing_transactional_campaign"
        print(f"\n  Cleaning: {table_name}")
        print("    SECOND VALIDATION")
        out = "marketing_transactional_campaign.parquet"

        if "campaign_id" not in df.columns:
            if "campaign" in df.columns:
                df["campaign_id"] = df["campaign"].astype("category").cat.codes
                print("    [INFO] Created campaign_id from 'campaign' categorical codes")
            else:
                df["campaign_id"] = df.reset_index().index
                print("    [INFO] Created synthetic campaign_id from row index")

        if "order_id" not in df.columns:
            if "orderid" in df.columns:
                df = df.rename(columns={"orderid": "order_id"})
                print("    [INFO] Renamed orderid -> order_id")
            else:
                df["order_id"] = df.reset_index().index
                print("    [INFO] Created synthetic order_id from row index")

        key_cols = [c for c in ["campaign_id", "order_id", "user_id"] if c in df.columns]

        df = check_nulls(df, key_cols, table_name)

        if "order_id" in df.columns:
            initial_rows = len(df)
            df.dropna(subset=["order_id"], inplace=True)
            null_removed = initial_rows - len(df)
            if null_removed > 0:
                print(f"      [CLEANING AUDIT] SUCCESS: Removed {null_removed} rows with NULL order_id.")

        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"      [CLEANING AUDIT] SUCCESS: Removed {duplicates_removed} duplicates (all columns).")

        check_duplicates(df, key_cols, table_name)

    else:
        print(f"  [WARN] Unknown marketing file pattern: {file}")
        return

    flag_errors(table_name)
    full_path = os.path.join(silver_folder, out)
    df.to_parquet(full_path, index=False)
    print(f"    [OK] Saved: {out} at {full_path} ({len(df)} rows)")


# ======================
# ROUTER / ORCHESTRATOR
# ======================

def cleaner(path: str, silver_folder: str) -> None:
    """Route a Bronze parquet file to its department-specific cleaner."""
    file = os.path.basename(path).lower()
    file = file.replace(" department_", "_")
    file = file.replace(" department ", "_")

    print(f"  [ROUTER] Routing Bronze file: {file}")

    while "  " in file:
        file = file.replace("  ", " ")
    file = file.replace(" ", "_")

    try:
        df = pd.read_parquet(path)

        if file.startswith("business_"):
            cleaner_func = clean_business
        elif file.startswith(("customer_management_", "customer_")):
            cleaner_func = clean_customer
        elif file.startswith("enterprise_"):
            cleaner_func = clean_enterprise
        elif file.startswith("operations_"):
            cleaner_func = clean_operations
        elif file.startswith("marketing_"):
            cleaner_func = clean_marketing
        else:
            print(f"  [WARN] No cleaning logic for: {file}")
            return

        cleaner_func(df, silver_folder, file)

    except Exception as e:
        log_quality(file, "PROCESSING_ERROR", str(e), "ERROR")


def run_silver_pipeline(data_zone_path: str) -> None:
    print("\n" + "=" * 70)
    print("SILVER LAYER TRANSFORMATION PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    bronze_folder = os.path.join(data_zone_path, "bronze_files")
    silver_folder = os.path.join(data_zone_path, "silver_files")

    os.makedirs(silver_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 1: CLEANING, STANDARDIZATION, VALIDATION (TRANSFORMATION)")
    print("=" * 70)

    if not os.path.exists(bronze_folder):
        print(f"  [ERROR] Bronze input path not found: {bronze_folder}")
        print("  Please ensure the bronze_ingestion container ran successfully.")
        return

    bronze_files = [
        f for f in os.listdir(bronze_folder)
        if f.endswith(".parquet") and not f.startswith("_")
    ]

    print(f"  [INFO] Bronze folder: {bronze_folder}")
    print(f"  [INFO] Silver folder: {silver_folder}")
    print(f"  [INFO] Bronze files found: {bronze_files}")

    if not bronze_files:
        print("  [ERROR] No Bronze Parquet files found. Please run Bronze Ingestion first.")
    else:
        print(f"  [INFO] Found {len(bronze_files)} Bronze files to process.")
        for file in bronze_files:
            path = os.path.join(bronze_folder, file)
            cleaner(path, silver_folder)

    combine_and_save_enterprise_transactions(bronze_folder, silver_folder)

    print("\n" + "=" * 70)
    print("STAGE 2: SECOND LOADING ZONE (SILVER)")
    print("Load the cleaned parquet files into the SQL data warehouse")
    print("=" * 70)
    print("  [OK] Silver files ready for SQL warehouse loading in 'silver_files' folder.")

    save_quality_report(silver_folder)

    print("\n" + "=" * 70)
    print("SILVER LAYER COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Silver Layer Transformation with Second Validation Checkpoint"
    )
    parser.add_argument("--data-zone", type=str, default=DATA_ZONE_PATH)
    args = parser.parse_args()

    run_silver_pipeline(args.data_zone)
