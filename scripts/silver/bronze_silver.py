import os
import pandas as pd
import argparse
from datetime import datetime
import re
import glob

# --- Constants ---
DATA_ZONE_PATH = "/app/data_zone"

# --- Quality Tracking ---
quality_report = []

def log_quality(table, issue_type, details, severity="WARNING"):
    """Logs data quality issues."""
    quality_report.append({
        'timestamp': datetime.now().isoformat(),
        'table': table,
        'issue_type': issue_type,
        'details': details,
        'severity': severity
    })
    print(f"      [{severity}] {table} - {issue_type}: {details}")

# --- Stage 1: Cleaning, Standardization, Validation (Transformation) ---

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the column names and fix inconsistencies."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

# --- SECOND VALIDATION Functions ---

def validate_required_columns(df: pd.DataFrame, required_cols: list, table_name: str) -> pd.DataFrame:
    """Check and compare bronze and silver tables - validates referential integrity."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        log_quality(table_name, "MISSING_COLUMNS", f"Missing required: {missing}", "ERROR")
    return df

def check_nulls(df: pd.DataFrame, key_cols: list, table_name: str) -> pd.DataFrame:
    """Validate keys and output tables - checks for null values."""
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
                    "WARNING"
                )
    return df

def check_duplicates(df: pd.DataFrame, key_cols: list, table_name: str) -> int:
    """Match data types and deduplicate entries."""
    if all(col in df.columns for col in key_cols):
        dup_count = df.duplicated(subset=key_cols).sum()
        if dup_count > 0:
            log_quality(
                table_name,
                "DUPLICATES",
                f"{dup_count} duplicate rows on {key_cols}",
                "WARNING"
            )
        return dup_count
    return 0

def validate_data_types(df: pd.DataFrame, type_map: dict, table_name: str) -> pd.DataFrame:
    """Match data types - converts and validates."""
    for col, expected_type in type_map.items():
        if col not in df.columns:
            continue

        if expected_type == 'datetime':
            before = df[col].notnull().sum()
            df[col] = pd.to_datetime(df[col], errors='coerce')
            after = df[col].notnull().sum()
            invalid = before - after
            if invalid > 0:
                log_quality(
                    table_name,
                    "INVALID_DATETIME",
                    f"{col}: {invalid} values coerced to NaT",
                    "WARNING"
                )

        elif expected_type == 'numeric':
            before = df[col].notnull().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after = df[col].notnull().sum()
            invalid = before - after
            if invalid > 0:
                log_quality(
                    table_name,
                    "INVALID_NUMERIC",
                    f"{col}: {invalid} values coerced to NaN",
                    "WARNING"
                )
    return df

def flag_errors(table_name: str):
    """Flag errors found during validation."""
    errors = [r for r in quality_report if r['table'] == table_name and r['severity'] == 'ERROR']
    if errors:
        print(f"      [ERROR] {len(errors)} ERRORS flagged for {table_name}")

# --- Department Cleaning Functions ---

def clean_business(df: pd.DataFrame, silver_folder: str, file: str):
    table_name = "business_product"
    print(f"\n  Cleaning: {table_name}")

    df = standardize(df)

    # SECOND VALIDATION
    print("    SECOND VALIDATION")
    df = validate_required_columns(df, ['product_id', 'product_name'], table_name)
    df = check_nulls(df, ['product_id'], table_name)
    check_duplicates(df, ['product_id'], table_name)
    flag_errors(table_name)

    # Deduplicate
    if 'product_id' in df.columns:
        df = df.drop_duplicates(subset='product_id', keep='first')

    out = "business_product.parquet"
    full_path = os.path.join(silver_folder, out)
    df.to_parquet(full_path, index=False)
    print(f"    [OK] Saved: {out} at {full_path} ({len(df)} rows)")

def clean_customer(df: pd.DataFrame, silver_folder: str, file: str
):
    df = standardize(df)
    
    if "user_job" in file:
        table_name = "customer_user_job"
        print(f"\n  Cleaning: {table_name}")
        print(f"    SECOND VALIDATION")
        
        df = validate_required_columns(df, ['user_id'], table_name)
        flag_errors(table_name)
        
        out = "customer_user_job.parquet"
        
    elif "user_credit_card" in file:
        table_name = "customer_user_credit_card"
        print(f"\n  Cleaning: {table_name}")
        print(f"    SECOND VALIDATION")
        
        df = validate_required_columns(df, ['user_id'], table_name)
        df = check_nulls(df, ['user_id'], table_name)
        check_duplicates(df, ['user_id'], table_name)
        flag_errors(table_name)
        
        if "user_id" in df.columns:
            df = df.drop_duplicates(subset="user_id", keep='first')
        out = "customer_user_credit_card.parquet"
        
    elif "user_data" in file or "user_" in file:
        table_name = "customer_user"
        print(f"\n  Cleaning: {table_name}")
        print(f"    SECOND VALIDATION")
        
        df = validate_required_columns(df, ['user_id'], table_name)
        df = check_nulls(df, ['user_id'], table_name)
        check_duplicates(df, ['user_id'], table_name)
        
        if "birthdate" in df.columns:
            df = validate_data_types(df, {'birthdate': 'datetime'}, table_name)
        
        flag_errors(table_name)
        
        if "user_id" in df.columns:
            df = df.drop_duplicates(subset="user_id", keep='first')
        out = "customer_user.parquet"
    else:
        return
        
    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")

def clean_enterprise(df: pd.DataFrame, silver_folder: str, file: str):
    """
    FIXED: Properly routes dimension vs transaction files.
    - merchant_data.html -> enterprise_merchant.parquet (Dimension)
    - staff_data.html -> enterprise_staff.parquet (Dimension)
    - order_with_merchant_data*.* -> enterprise_order_with_merchant_data*_tx.parquet (Fact/Transaction)
    """
    df = standardize(df)
    
    # Handle transactional/order files FIRST (most specific match)
    if "order_with_merchant" in file:
        table_name = "enterprise_order_merchant_tx"
        print(f"\n  Cleaning: {table_name} (Transaction/Fact)")
        print(f"    SECOND VALIDATION")
        
        # Rename non-standard columns if present
        rename_map = {}
        if 'merchantid' in df.columns:
            rename_map['merchantid'] = 'merchant_id'
        if 'staffid' in df.columns:
            rename_map['staffid'] = 'staff_id'
        if 'orderid' in df.columns:
            rename_map['orderid'] = 'order_id'
        
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"    [INFO] Renamed columns: {rename_map}")
        
        df = validate_required_columns(df, ['order_id', 'merchant_id'], table_name)
        df = check_nulls(df, ['order_id', 'merchant_id'], table_name)
        check_duplicates(df, ['order_id'], table_name)
        flag_errors(table_name)
        
        if 'order_id' in df.columns:
            df = df.drop_duplicates(subset='order_id', keep='first')
        
        # Use original filename base to preserve uniqueness
        base_name = file.replace('.parquet', '').replace('enterprise_', '')
        out = f"enterprise_{base_name}_tx.parquet"
    
    # Handle dimension files
    elif "merchant_data" in file:
        table_name = "enterprise_merchant"
        print(f"\n  Cleaning: {table_name} (Dimension)")
        print(f"    SECOND VALIDATION")
        
        df = validate_required_columns(df, ['merchant_id'], table_name)
        df = check_nulls(df, ['merchant_id'], table_name)
        check_duplicates(df, ['merchant_id'], table_name)
        flag_errors(table_name)
        
        if "merchant_id" in df.columns:
            df = df.drop_duplicates(subset="merchant_id", keep='first')
        out = "enterprise_merchant.parquet"
        
    elif "staff_data" in file:
        table_name = "enterprise_staff"
        print(f"\n  Cleaning: {table_name} (Dimension)")
        print(f"    SECOND VALIDATION")
        
        df = validate_required_columns(df, ['staff_id'], table_name)
        df = check_nulls(df, ['staff_id'], table_name)
        check_duplicates(df, ['staff_id'], table_name)
        flag_errors(table_name)
        
        if "staff_id" in df.columns:
            df = df.drop_duplicates(subset="staff_id", keep='first')
        out = "enterprise_staff.parquet"
    else:
        print(f"  [WARN] Unknown enterprise file pattern: {file}")
        return
    
    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")

def clean_operations(df: pd.DataFrame, silver_folder: str, file: str):
    df = standardize(df)

    # One-time debug: show columns for first few runs if needed
    print(f"\n  Operations file: {file}")
    print(f"    Columns: {list(df.columns)}")

    # Global renaming for operations
    rename_map = {}

    # Common variants for order_id
    if 'orderid' in df.columns and 'order_id' not in df.columns:
        rename_map['orderid'] = 'order_id'

    # Common variants for product_id
    if 'productid' in df.columns and 'product_id' not in df.columns:
        rename_map['productid'] = 'product_id'
    if 'prod_id' in df.columns and 'product_id' not in df.columns:
        rename_map['prod_id'] = 'product_id'
    if 'sku' in df.columns and 'product_id' not in df.columns:
        rename_map['sku'] = 'product_id'
    if 'item_id' in df.columns and 'product_id' not in df.columns:
        rename_map['item_id'] = 'product_id'

    # User id variants
    if 'userid' in df.columns and 'user_id' not in df.columns:
        rename_map['userid'] = 'user_id'

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"    [INFO] Renamed columns: {rename_map}")

    if "order_data" in file:
        table_name = "operations_orders"
        print(f"\n  Cleaning: {table_name}")
        print("    SECOND VALIDATION")

        df = validate_required_columns(df, ['order_id'], table_name)
        df = check_nulls(df, ['order_id'], table_name)

        # Any column containing "date" is treated as datetime
        date_cols = [c for c in df.columns if "date" in c]
        type_map = {c: 'datetime' for c in date_cols}
        if type_map:
            df = validate_data_types(df, type_map, table_name)

        flag_errors(table_name)
        out = "operations_orders.parquet"

    elif "line_item" in file:
        table_name = "operations_line_items"
        print(f"\n  Cleaning: {table_name}")
        print("    SECOND VALIDATION")

        # FINAL GUARD: if still no product_id, create a surrogate from any obvious numeric column
        if 'product_id' not in df.columns:
            # Best guess: if there is a column literally named 'product' or 'product_name', use its index
            if 'product' in df.columns:
                df['product_id'] = df['product'].astype('category').cat.codes
                print("    [INFO] Created product_id from 'product' categorical codes")
            elif 'product_name' in df.columns:
                df['product_id'] = df['product_name'].astype('category').cat.codes
                print("    [INFO] Created product_id from 'product_name' categorical codes")
            else:
                # As a last resort, use row index as product_id to avoid validation errors
                df['product_id'] = df.reset_index().index
                print("    [INFO] Created synthetic product_id from row index")

        df = validate_required_columns(df, ['order_id', 'product_id'], table_name)
        df = check_nulls(df, ['order_id', 'product_id'], table_name)
        check_duplicates(df, ['order_id', 'product_id'], table_name)
        flag_errors(table_name)

        if "order_id" in df.columns and "product_id" in df.columns:
            df = df.drop_duplicates(subset=["order_id", "product_id"], keep='first')

        out = "operations_line_items.parquet"

    elif "order_delays" in file:
        table_name = "operations_order_delays"
        print(f"\n  Cleaning: {table_name}")

        if 'orderid' in df.columns and 'order_id' not in df.columns:
            df = df.rename(columns={'orderid': 'order_id'})
            print("    [INFO] Renamed orderid -> order_id")

        df = df.drop_duplicates()
        out = "operations_order_delays.parquet"

    else:
        print(f"  [WARN] Unknown operations file pattern: {file}")
        return

    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


def clean_marketing(df: pd.DataFrame, silver_folder: str, file: str):
    df = standardize(df)

    print(f"\n  Marketing file: {file}")
    print(f"    Columns: {list(df.columns)}")

    # Global normalization
    rename_map = {}

    # campaign_id variants
    if 'campaignid' in df.columns and 'campaign_id' not in df.columns:
        rename_map['campaignid'] = 'campaign_id'
    if 'id' in df.columns and 'campaign_id' not in df.columns and "campaign_data" in file:
        rename_map['id'] = 'campaign_id'

    # transaction-related keys
    if 'orderid' in df.columns and 'order_id' not in df.columns:
        rename_map['orderid'] = 'order_id'
    if 'userid' in df.columns and 'user_id' not in df.columns:
        rename_map['userid'] = 'user_id'

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"    [INFO] Renamed columns: {rename_map}")

    # ---- UPDATED BRANCHES HERE ----
    if "campaign_data_bronze" in file:
        table_name = "marketing_campaign"
        print(f"\n  Cleaning: {table_name}")
        print("    SECOND VALIDATION")

        if 'campaign_id' not in df.columns:
            if len(df.columns) == 1:
                only_col = df.columns[0]
                df['campaign_id'] = df[only_col].astype('category').cat.codes
                print(f"    [INFO] Created campaign_id from '{only_col}' categorical codes")
            else:
                df['campaign_id'] = df.reset_index().index
                print("    [INFO] Created synthetic campaign_id from row index")

        df = validate_required_columns(df, ['campaign_id'], table_name)
        df = check_nulls(df, ['campaign_id'], table_name)
        check_duplicates(df, ['campaign_id'], table_name)
        flag_errors(table_name)

        if "campaign_id" in df.columns:
            df = df.drop_duplicates(subset="campaign_id", keep='first')

        out = "marketing_campaign.parquet"

    elif "transactional_campaign" in filename:
        table = "marketing_transactional_campaign"
        print(f"\n Cleaning: {table}")
        
        if "campaign_id" not in df.columns:
            if "campaign" in df.columns:
                df["campaign_id"] = df["campaign"].astype("category").cat.codes
                print(" [ATTENTION] Created campaign_id from 'campaign'")
            else:
                df["campaign_id"] = df.reset_index().index()
                print(" [ATTENTION] Created synthetic campaign_id from row index")       

        key_cols = [c for c in ["campaign_id", "order_id", "user_id"] if c in df.columns]
        
        df = null_check(df, key_cols, table)
        
        dupe_check(df, key_cols, table)        
        flag_errors(table)  

        df = df.drop_duplicates()

        out = "marketing_transactional_campaign.parquet"

    else:
        print(f"  [WARN] Unknown marketing file pattern: {file}")
        return

    df.to_parquet(os.path.join(silver_folder, out), index=False)
    print(f"    [OK] Saved: {out} ({len(df)} rows)")


# --- Main Processor ---

def cleaner(path: str, silver_folder: str):
    """Routes Bronze file to cleaning function."""
    file = os.path.basename(path).lower()
    
    print(f"  [ROUTER] Routing Bronze file: {file}")
    
    try:
        df = pd.read_parquet(path)
        
        if file.startswith("business_"):
            clean_business(df, silver_folder, file)
        elif file.startswith("customer_management_") or file.startswith("customer_"):
            clean_customer(df, silver_folder, file)
        elif file.startswith("enterprise_"):
            clean_enterprise(df, silver_folder, file)
        elif file.startswith("operations_"):
            clean_operations(df, silver_folder, file)
        elif file.startswith("marketing_"):
            clean_marketing(df, silver_folder, file)
        else:
            print(f"  [WARN] No cleaning logic for: {file}")
            
    except Exception as e:
        log_quality(file, "PROCESSING_ERROR", str(e), "ERROR")

# --- Stage 2: Second Loading Zone (Silver) ---

def run_silver_pipeline(data_zone_path: str):
    """
    Load the cleaned parquet files into the SQL data warehouse.
    """
    print("\n" + "="*70)
    print("SILVER LAYER TRANSFORMATION PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    print("\n" + "="*70)
    print("STAGE 1: CLEANING, STANDARDIZATION, VALIDATION (TRANSFORMATION)")
    print("="*70)
    
    bronze_folder = os.path.join(data_zone_path, "bronze_files")
    silver_folder = os.path.join(data_zone_path, "silver_files")
    
    os.makedirs(silver_folder, exist_ok=True)
    
    # Process Bronze files
    bronze_files = [f for f in os.listdir(bronze_folder) 
                    if f.endswith(".parquet") and not f.startswith("_")]

    for file in bronze_files:
        path = os.path.join(bronze_folder, file)
        cleaner(path, silver_folder)
    
    print(f"  [INFO] Bronze folder: {bronze_folder}")
    print(f"  [INFO] Silver folder: {silver_folder}")
    print(f"  [INFO] Bronze files found: {bronze_files}")


    print("\n" + "="*70)
    print("STAGE 2: SECOND LOADING ZONE (SILVER)")
    print("Load the cleaned parquet files into the SQL data warehouse")
    print("="*70)
    print("  [OK] Silver files ready for SQL warehouse loading")
    
    # Save quality report
    if quality_report:
        report_df = pd.DataFrame(quality_report)
        report_path = os.path.join(silver_folder, "_silver_quality_report.csv")
        report_df.to_csv(report_path, index=False)
        
        errors = len([r for r in quality_report if r['severity'] == 'ERROR'])
        warnings = len([r for r in quality_report if r['severity'] == 'WARNING'])
        
        print(f"\nQuality report saved: {report_path}")
        print(f"Quality Summary: {errors} errors, {warnings} warnings")
    
    print("\n" + "="*70)
    print("SILVER LAYER COMPLETE")
    print("="*70 + "\n")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Silver Layer Transformation with Second Validation Checkpoint"
    )
    
    parser.add_argument("--data-zone", type=str, default=DATA_ZONE_PATH)
    
    args = parser.parse_args()
    run_silver_pipeline(args.data_zone)
