import pandas as pd
from sqlalchemy import create_engine, text
import os
import gc
import re

pd.options.mode.chained_assignment = None

DATA_ZONE_PATH = "/app/data_zone"
SILVER_PATH = os.path.join(DATA_ZONE_PATH, "silver_files")

engine = create_engine(
    "postgresql+psycopg2://shopzada_user:root@postgres_db:5432/shopzada_dwh"
)


def load_silver_data(file_name: str, max_rows=1000) -> pd.DataFrame:
    file_path = os.path.join(SILVER_PATH, f"{file_name}.parquet")
    print(f"Reading Silver file: {file_name}.parquet")
    try:
        df = pd.read_parquet(file_path).head(max_rows)
        print(f"  -> Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"[ERROR] Could not read {file_name}.parquet: {e}")
        return pd.DataFrame()


def cleanup_memory():
    gc.collect()
    print("  [INFO] Memory cleanup complete.")


def create_production_schema():
    print("\n" + "=" * 60)
    print("🚀 STAGE 1: CREATING PRODUCTION GOLD SCHEMA")
    print("=" * 60)

    tables_no_fk = """
    CREATE SCHEMA IF NOT EXISTS gold;

    CREATE TABLE IF NOT EXISTS gold.user_dim (
      user_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      user_id VARCHAR(64) UNIQUE NOT NULL,
      user_name VARCHAR(100) NOT NULL,
      user_job VARCHAR(50),
      user_job_lvl VARCHAR(10),
      creation_date TIMESTAMP,
      street VARCHAR(100),
      state VARCHAR(50),
      city VARCHAR(50),
      country VARCHAR(50),
      birthdate TIMESTAMP,
      gender VARCHAR(10),
      device_address VARCHAR(100),
      user_type VARCHAR(30)
    );

    CREATE TABLE IF NOT EXISTS gold.product_dim (
      product_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      product_id VARCHAR(64) UNIQUE NOT NULL,
      product_name VARCHAR(120) NOT NULL,
      product_type VARCHAR(60),
      product_unit_price NUMERIC(12,2) NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS gold.merchant_dim (
      merchant_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      merchant_id VARCHAR(64) UNIQUE NOT NULL,
      merchant_creation_date TIMESTAMP,
      merchant_name VARCHAR(120) NOT NULL,
      merchant_street VARCHAR(100),
      merchant_state VARCHAR(50),
      merchant_city VARCHAR(50),
      merchant_country VARCHAR(50),
      merchant_contact_no VARCHAR(40)
    );

    CREATE TABLE IF NOT EXISTS gold.staff_dim (
      staff_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      staff_id VARCHAR(64) UNIQUE NOT NULL,
      staff_name VARCHAR(100) NOT NULL,
      staff_job_lvl VARCHAR(10),
      staff_creation_date TIMESTAMP,
      staff_street VARCHAR(100),
      staff_state VARCHAR(50),
      staff_city VARCHAR(50),
      staff_country VARCHAR(50),
      staff_contact_no VARCHAR(40)
    );

    CREATE TABLE IF NOT EXISTS gold.campaign_dim (
      campaign_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      campaign_id VARCHAR(64) UNIQUE NOT NULL,
      campaign_name VARCHAR(120) NOT NULL,
      campaign_description VARCHAR(255),
      campaign_discount VARCHAR(10)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(tables_no_fk))

    print("✅ PRODUCTION SCHEMA CREATED")


def truncate_gold_tables():
    print("\nSTAGE 2: TRUNCATING TABLES")
    tables = ["user_dim", "product_dim", "merchant_dim", "staff_dim", "campaign_dim"]
    with engine.connect() as conn:
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE gold.{table} RESTART IDENTITY;"))
            print(f"  [OK] Truncated gold.{table}")
        conn.commit()


# -------- Dimension loaders --------

def load_user_dim():
    print("  Processing user_dim...")
    user_df = load_silver_data("customer_user", max_rows=500)
    user_job_df = load_silver_data("customer_user_job", max_rows=500)

    if user_df.empty:
        print("  [WARN] customer_user is empty, skipping user_dim")
        return

    if not user_job_df.empty:
        merged = user_df.merge(
            user_job_df[["user_id", "job_title", "job_level"]],
            on="user_id",
            how="left",
        )
    else:
        merged = user_df.copy()
        merged["job_title"] = None
        merged["job_level"] = None

    user_final = merged.rename(
        columns={
            "name": "user_name",
            "job_title": "user_job",
            "job_level": "user_job_lvl",
        }
    )[
        [
            "user_id",
            "user_name",
            "user_job",
            "user_job_lvl",
            "creation_date",
            "street",
            "state",
            "city",
            "country",
            "birthdate",
            "gender",
            "device_address",
            "user_type",
        ]
    ]

    user_final.to_sql(
        "user_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f"  [OK] Loaded {len(user_final)} rows into gold.user_dim")


def load_product_dim():
    print("  Processing product_dim...")
    prod = load_silver_data("business_product", max_rows=500)
    if prod.empty:
        print("  [WARN] business_product empty, skipping product_dim")
        return

    product_final = prod.rename(columns={"price": "product_unit_price"})[
        ["product_id", "product_name", "product_type", "product_unit_price"]
    ]

    product_final.to_sql(
        "product_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f"  [OK] Loaded {len(product_final)} rows into gold.product_dim")


def load_merchant_dim():
    print("  Processing merchant_dim...")
    m = load_silver_data("enterprise_merchant", max_rows=500)
    if m.empty:
        print("  [WARN] enterprise_merchant empty, skipping merchant_dim")
        return

    merchant_final = m.rename(
        columns={
            "creation_date": "merchant_creation_date",
            "name": "merchant_name",
            "street": "merchant_street",
            "state": "merchant_state",
            "city": "merchant_city",
            "country": "merchant_country",
            "contact_number": "merchant_contact_no",
        }
    )[
        [
            "merchant_id",
            "merchant_creation_date",
            "merchant_name",
            "merchant_street",
            "merchant_state",
            "merchant_city",
            "merchant_country",
            "merchant_contact_no",
        ]
    ]

    merchant_final.to_sql(
        "merchant_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f"  [OK] Loaded {len(merchant_final)} rows into gold.merchant_dim")


def load_staff_dim():
    print("  Processing staff_dim...")
    s = load_silver_data("enterprise_staff", max_rows=500)
    if s.empty:
        print("  [WARN] enterprise_staff empty, skipping staff_dim")
        return

    staff_final = s.rename(
        columns={
            "name": "staff_name",
            "job_level": "staff_job_lvl",
            "creation_date": "staff_creation_date",
            "street": "staff_street",
            "state": "staff_state",
            "city": "staff_city",
            "country": "staff_country",
            "contact_number": "staff_contact_no",
        }
    )[
        [
            "staff_id",
            "staff_name",
            "staff_job_lvl",
            "staff_creation_date",
            "staff_street",
            "staff_state",
            "staff_city",
            "staff_country",
            "staff_contact_no",
        ]
    ]

    staff_final.to_sql(
        "staff_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f"  [OK] Loaded {len(staff_final)} rows into gold.staff_dim")


def load_campaign_dim():
    print("  Processing campaign_dim...")
    c = load_silver_data("marketing_campaign", max_rows=500)
    if c.empty:
        print("  [WARN] marketing_campaign empty, skipping campaign_dim")
        return

    # Use Silver normalized discount if available, otherwise fall back
    if "discount_normalized" in c.columns:
        c["campaign_discount"] = c["discount_normalized"]
    elif "discount" in c.columns:
        c["campaign_discount"] = c["discount"]
    else:
        c["campaign_discount"] = None

    campaign_final = c[
        [
            "campaign_id",
            "campaign_name",
            "campaign_description",
            "campaign_discount",
        ]
    ]

    campaign_final.to_sql(
        "campaign_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f"  [OK] Loaded {len(campaign_final)} rows into gold.campaign_dim")


def load_dimensions():
    print("\nSTAGE 3: LOADING DIMENSIONS")

    load_user_dim()
    load_product_dim()
    load_merchant_dim()
    load_staff_dim()
    load_campaign_dim()

    cleanup_memory()


if __name__ == "__main__":
    print("🏆 SHOPZADA GOLD LAYER PIPELINE STARTING...")

    create_production_schema()
    truncate_gold_tables()
    load_dimensions()

    print("\n" + "=" * 80)
    print("🎉 GOLD LAYER COMPLETE - PRODUCTION DATA WAREHOUSE READY!")
    print("📊 pgAdmin: localhost:5050")
    print("=" * 80)
