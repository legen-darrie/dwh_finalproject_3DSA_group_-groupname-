import os
import gc
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

pd.options.mode.chained_assignment = None

DATA_ZONE_PATH = "/app/data_zone"
SILVER_PATH = os.path.join(DATA_ZONE_PATH, "silver_files")

engine = create_engine(
    "postgresql+psycopg2://shopzada_user:root@postgres_db:5432/shopzada_dwh"
)

# =======================
# Generic helpers
# =======================

def load_silver_data(file_name: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Read a Silver parquet file.
    If max_rows is given, return only the first N rows; otherwise return all rows.
    """
    file_path = os.path.join(SILVER_PATH, f"{file_name}.parquet")
    print(f"Reading Silver file: {file_name}.parquet")
    try:
        df = pd.read_parquet(file_path)
        if max_rows is not None:
            df = df.head(max_rows)
        print(f" -> Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"[ERROR] Could not read {file_name}.parquet: {e}")
        return pd.DataFrame()


def cleanup_memory():
    gc.collect()
    print(" [INFO] Memory cleanup complete.")


# =======================
# Schema creation
# =======================

def create_production_schema():
    print("\n" + "=" * 60)
    print("STAGE 1: CREATING PRODUCTION GOLD SCHEMA")
    print("=" * 60)

    ddl = """
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

    CREATE TABLE IF NOT EXISTS gold.credit_card_dim (
        credit_card_key INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        credit_card_number VARCHAR(32) UNIQUE NOT NULL,
        user_id VARCHAR(64) NOT NULL,
        card_type VARCHAR(30),
        bank_name VARCHAR(60),
        expiry_date DATE
    );

    CREATE TABLE IF NOT EXISTS gold.date_dim (
        date_key INT PRIMARY KEY,
        full_date DATE NOT NULL,
        year INT NOT NULL,
        quarter INT NOT NULL,
        month INT NOT NULL,
        month_name VARCHAR(15),
        day INT NOT NULL,
        day_name VARCHAR(15),
        is_weekend BOOLEAN
    );

    CREATE TABLE IF NOT EXISTS gold.order_line_fact (
        order_line_key   INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        order_id         VARCHAR(64),
        order_line_id    VARCHAR(64),
        user_key         INT,
        product_key      INT,
        merchant_key     INT,
        staff_key        INT,
        campaign_key     INT,
        credit_card_key  INT,
        order_date_key   INT,
        quantity         NUMERIC(12,2),
        line_amount      NUMERIC(14,2),
        CONSTRAINT fk_user_dim
            FOREIGN KEY (user_key) REFERENCES gold.user_dim(user_key),
        CONSTRAINT fk_product_dim
            FOREIGN KEY (product_key) REFERENCES gold.product_dim(product_key),
        CONSTRAINT fk_merchant_dim
            FOREIGN KEY (merchant_key) REFERENCES gold.merchant_dim(merchant_key),
        CONSTRAINT fk_staff_dim
            FOREIGN KEY (staff_key) REFERENCES gold.staff_dim(staff_key),
        CONSTRAINT fk_campaign_dim
            FOREIGN KEY (campaign_key) REFERENCES gold.campaign_dim(campaign_key),
        CONSTRAINT fk_credit_card_dim
            FOREIGN KEY (credit_card_key) REFERENCES gold.credit_card_dim(credit_card_key),
        CONSTRAINT fk_date_dim
            FOREIGN KEY (order_date_key) REFERENCES gold.date_dim(date_key)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(ddl))

    print("✅ PRODUCTION SCHEMA CREATED")


def truncate_gold_tables():
    print("\nSTAGE 2: TRUNCATING GOLD TABLES")
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                TRUNCATE TABLE
                    gold.order_line_fact,
                    gold.credit_card_dim,
                    gold.date_dim,
                    gold.user_dim,
                    gold.product_dim,
                    gold.merchant_dim,
                    gold.staff_dim,
                    gold.campaign_dim
                RESTART IDENTITY CASCADE;
                """
            )
        )
        print(" [OK] Truncated all gold tables")
        conn.commit()


# =======================
# Dimension loaders
# =======================

def load_user_dim():
    print(" Processing user_dim...")
    user_df = load_silver_data("customer_user")
    user_job_df = load_silver_data("customer_user_job")

    if user_df.empty:
        print(" [WARN] customer_user is empty, skipping user_dim")
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

    user_final["user_name"] = user_final["user_name"].astype(str).str.slice(0, 100)
    user_final["user_job"] = user_final["user_job"].astype(str).str.slice(0, 50)
    user_final["user_job_lvl"] = user_final["user_job_lvl"].astype(str).str.slice(0, 10)
    user_final["street"] = user_final["street"].astype(str).str.slice(0, 100)
    user_final["state"] = user_final["state"].astype(str).str.slice(0, 50)
    user_final["city"] = user_final["city"].astype(str).str.slice(0, 50)
    user_final["country"] = user_final["country"].astype(str).str.slice(0, 50)
    user_final["gender"] = user_final["gender"].astype(str).str.slice(0, 10)
    user_final["device_address"] = (
        user_final["device_address"].astype(str).str.slice(0, 100)
    )
    user_final["user_type"] = user_final["user_type"].astype(str).str.slice(0, 30)

    user_final = user_final.drop_duplicates(subset=["user_id"])

    user_final.to_sql(
        "user_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(user_final)} rows into gold.user_dim")


def load_product_dim():
    print(" Processing product_dim...")
    prod = load_silver_data("business_product")
    if prod.empty:
        print(" [WARN] business_product empty, skipping product_dim")
        return

    product_final = prod.rename(columns={"price": "product_unit_price"})[
        ["product_id", "product_name", "product_type", "product_unit_price"]
    ]

    product_final["product_name"] = (
        product_final["product_name"].astype(str).str.slice(0, 120)
    )
    product_final["product_type"] = (
        product_final["product_type"].astype(str).str.slice(0, 60)
    )

    product_final.to_sql(
        "product_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(product_final)} rows into gold.product_dim")


def load_merchant_dim():
    print(" Processing merchant_dim...")
    m = load_silver_data("enterprise_merchant")
    if m.empty:
        print(" [WARN] enterprise_merchant empty, skipping merchant_dim")
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

    merchant_final["merchant_name"] = (
        merchant_final["merchant_name"].astype(str).str.slice(0, 120)
    )
    merchant_final["merchant_street"] = (
        merchant_final["merchant_street"].astype(str).str.slice(0, 100)
    )
    merchant_final["merchant_state"] = (
        merchant_final["merchant_state"].astype(str).str.slice(0, 50)
    )
    merchant_final["merchant_city"] = (
        merchant_final["merchant_city"].astype(str).str.slice(0, 50)
    )
    merchant_final["merchant_country"] = (
        merchant_final["merchant_country"].astype(str).str.slice(0, 50)
    )
    merchant_final["merchant_contact_no"] = (
        merchant_final["merchant_contact_no"].astype(str).str.slice(0, 40)
    )

    merchant_final.to_sql(
        "merchant_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(merchant_final)} rows into gold.merchant_dim")


def load_staff_dim():
    print(" Processing staff_dim...")
    s = load_silver_data("enterprise_staff")
    if s.empty:
        print(" [WARN] enterprise_staff empty, skipping staff_dim")
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

    staff_final["staff_name"] = staff_final["staff_name"].astype(str).str.slice(0, 100)
    staff_final["staff_job_lvl"] = (
        staff_final["staff_job_lvl"].astype(str).str.slice(0, 10)
    )
    staff_final["staff_street"] = (
        staff_final["staff_street"].astype(str).str.slice(0, 100)
    )
    staff_final["staff_state"] = (
        staff_final["staff_state"].astype(str).str.slice(0, 50)
    )
    staff_final["staff_city"] = staff_final["staff_city"].astype(str).str.slice(0, 50)
    staff_final["staff_country"] = (
        staff_final["staff_country"].astype(str).str.slice(0, 50)
    )
    staff_final["staff_contact_no"] = (
        staff_final["staff_contact_no"].astype(str).str.slice(0, 40)
    )

    staff_final.to_sql(
        "staff_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(staff_final)} rows into gold.staff_dim")


def load_campaign_dim():
    print(" Processing campaign_dim...")
    c = load_silver_data("marketing_campaign")
    if c.empty:
        print(" [WARN] marketing_campaign empty, skipping campaign_dim")
        return

    if "discount_normalized" in c.columns:
        c["campaign_discount"] = c["discount_normalized"]
    elif "discount" in c.columns:
        c["campaign_discount"] = c["discount"]
    else:
        c["campaign_discount"] = None

    campaign_final = c[
        ["campaign_id", "campaign_name", "campaign_description", "campaign_discount"]
    ]

    campaign_final["campaign_name"] = (
        campaign_final["campaign_name"].astype(str).str.slice(0, 120)
    )
    campaign_final["campaign_description"] = (
        campaign_final["campaign_description"].astype(str).str.slice(0, 255)
    )
    campaign_final["campaign_discount"] = (
        campaign_final["campaign_discount"].astype(str).str.slice(0, 10)
    )

    campaign_final.to_sql(
        "campaign_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(campaign_final)} rows into gold.campaign_dim")


def load_credit_card_dim():
    print(" Processing credit_card_dim...")
    cc = load_silver_data("customer_user_credit_card")
    if cc.empty:
        print(" [WARN] customer_user_credit_card empty, skipping credit_card_dim")
        return

    cc.columns = [c.lower() for c in cc.columns]
    rename_map = {}
    if "credit_card_number" in cc.columns:
        rename_map["credit_card_number"] = "credit_card_number"
    if "user_id" in cc.columns:
        rename_map["user_id"] = "user_id"
    if "card_type" in cc.columns:
        rename_map["card_type"] = "card_type"
    if "bank_name" in cc.columns:
        rename_map["bank_name"] = "bank_name"
    if "expiry_date" in cc.columns:
        rename_map["expiry_date"] = "expiry_date"

    cc = cc.rename(columns=rename_map)
    keep = ["credit_card_number", "user_id", "card_type", "bank_name", "expiry_date"]
    cc = cc[[c for c in keep if c in cc.columns]]

    if "expiry_date" in cc.columns:
        cc["expiry_date"] = pd.to_datetime(
            cc["expiry_date"], errors="coerce"
        ).dt.date

    for col, length in [
        ("credit_card_number", 32),
        ("user_id", 64),
        ("card_type", 30),
        ("bank_name", 60),
    ]:
        if col in cc.columns:
            cc[col] = cc[col].astype(str).str.slice(0, length)

    cc = cc.drop_duplicates(subset=["credit_card_number"])

    cc.to_sql(
        "credit_card_dim", engine, schema="gold", if_exists="append", index=False
    )
    print(f" [OK] Loaded {len(cc)} rows into gold.credit_card_dim")


# =======================
# Date dimension
# =======================

def build_date_range():
    candidates = []
    for fname in ["customer_user", "enterprise_merchant", "enterprise_staff"]:
        df = load_silver_data(fname, max_rows=50_000)
        if df.empty:
            continue
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "birth"]):
                s = pd.to_datetime(df[col], errors="coerce")
                candidates.append(s)

    if not candidates:
        return pd.date_range("2020-01-01", "2020-12-31", freq="D")

    all_dates = pd.concat(candidates).dropna().dt.normalize()
    if all_dates.empty:
        return pd.date_range("2020-01-01", "2020-12-31", freq="D")

    return pd.date_range(all_dates.min(), all_dates.max(), freq="D")


def load_date_dim():
    print(" Processing date_dim...")
    dr = build_date_range()
    df = pd.DataFrame({"full_date": dr})
    df["year"] = df["full_date"].dt.year
    df["quarter"] = df["full_date"].dt.quarter
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()
    df["day"] = df["full_date"].dt.day
    df["day_name"] = df["full_date"].dt.day_name()
    df["is_weekend"] = df["day_name"].isin(["Saturday", "Sunday"])
    df["date_key"] = (
        df["year"] * 10000 + df["month"] * 100 + df["day"]
    ).astype(int)

    df = df[
        [
            "date_key",
            "full_date",
            "year",
            "quarter",
            "month",
            "month_name",
            "day",
            "day_name",
            "is_weekend",
        ]
    ]

    df.to_sql("date_dim", engine, schema="gold", if_exists="append", index=False)
    print(f" [OK] Loaded {len(df)} rows into gold.date_dim")


# =======================
# Fact loader
# =======================

def load_order_line_fact():
    print(" Processing order_line_fact...")

    # 1) Load source data from Silver
    orders = load_silver_data("operations_orders")
    lines = load_silver_data("operations_line_items")
    ord_merch = load_silver_data("enterprise_order_merchant_tx")
    mkt_tx = load_silver_data("marketing_transactional_campaign")

    if orders.empty or lines.empty:
        print(
            f" [WARN] operations_orders empty={orders.empty}, "
            f"operations_line_items empty={lines.empty}"
        )
        print(" [WARN] Skipping order_line_fact load")
        return

    # Standardize column names
    orders.columns = orders.columns.str.lower()
    lines.columns = lines.columns.str.lower()
    ord_merch.columns = ord_merch.columns.str.lower()
    mkt_tx.columns = mkt_tx.columns.str.lower()

    # Ensure key types compatible
    for df in (orders, lines, ord_merch, mkt_tx):
        if "order_id" in df.columns:
            df["order_id"] = df["order_id"].astype(str)
        if "campaign_id" in df.columns:
            df["campaign_id"] = df["campaign_id"].astype(str)

    print(
        f" Orders: {len(orders)} rows, "
        f"Lines: {len(lines)} rows, "
        f"Ord+Merch: {len(ord_merch)} rows, "
        f"MktTx: {len(mkt_tx)} rows"
    )

    # 2) Build base fact at line grain: lines × orders
    fact = lines.merge(orders, on="order_id", how="inner")

    # Join merchant & staff
    fact = fact.merge(
        ord_merch[["order_id", "merchant_id", "staff_id"]],
        on="order_id",
        how="left",
    )

    # Join campaign_id from marketing transactional campaign
    if not mkt_tx.empty and "campaign_id" in mkt_tx.columns:
        fact = fact.merge(
            mkt_tx[["order_id", "campaign_id"]],
            on="order_id",
            how="left",
        )

    print(f" After base joins: {len(fact)} rows")
    print(" FACT COLUMNS:", list(fact.columns))

    # 3) Load dimension key maps
    with engine.connect() as conn:
        user_dim = pd.read_sql("SELECT user_key, user_id FROM gold.user_dim", conn)
        prod_dim = pd.read_sql(
            "SELECT product_key, product_id FROM gold.product_dim",
            conn,
        )
        merch_dim = pd.read_sql(
            "SELECT merchant_key, merchant_id FROM gold.merchant_dim", conn
        )
        staff_dim = pd.read_sql(
            "SELECT staff_key, staff_id FROM gold.staff_dim", conn
        )
        camp_dim = pd.read_sql(
            "SELECT campaign_key, campaign_id FROM gold.campaign_dim", conn
        )
        date_dim = pd.read_sql(
            "SELECT date_key, full_date FROM gold.date_dim", conn
        )

    # 4) Map natural IDs to surrogate keys

    # user_key
    fact = fact.merge(
        user_dim[["user_key", "user_id"]],
        on="user_id",
        how="left",
    )

    # product_key: since product_id values differ, keep this join best-effort
    if "product_id" in fact.columns and "product_id" in prod_dim.columns:
        fact = fact.merge(
            prod_dim[["product_key", "product_id"]],
            on="product_id",
            how="left",
        )
    else:
        fact["product_key"] = None

    # merchant_key
    fact = fact.merge(
        merch_dim[["merchant_key", "merchant_id"]],
        on="merchant_id",
        how="left",
    )

    # staff_key
    fact = fact.merge(
        staff_dim[["staff_key", "staff_id"]],
        on="staff_id",
        how="left",
    )

    # campaign_key
    camp_dim["campaign_id"] = camp_dim["campaign_id"].astype(str)
    if "campaign_id" in fact.columns:
        fact = fact.merge(
            camp_dim[["campaign_key", "campaign_id"]],
            on="campaign_id",
            how="left",
        )
    else:
        fact["campaign_key"] = None

    # 5) Order_date_key using transaction_date
    fact["order_date"] = pd.to_datetime(
        fact["transaction_date"], errors="coerce"
    ).dt.date

    date_dim["full_date"] = pd.to_datetime(
        date_dim["full_date"], errors="coerce"
    ).dt.date

    fact = fact.merge(
        date_dim.rename(columns={"full_date": "order_date"}),
        on="order_date",
        how="left",
    )
    fact.rename(columns={"date_key": "order_date_key"}, inplace=True)

    # 6) Quantity and prices – use cleaned quantity from line items and operations price
    non_null_qty = fact["quantity"].notna().sum() if "quantity" in fact.columns else 0
    print(" Quantity non-null before cast:", non_null_qty, "of", len(fact))

    fact["quantity"] = pd.to_numeric(fact.get("quantity"), errors="coerce")
    fact["unit_price"] = pd.to_numeric(fact.get("price"), errors="coerce")
    fact["line_amount"] = fact["quantity"] * fact["unit_price"]

    overlap = fact["quantity"].notna() & fact["unit_price"].notna()
    print(" rows with both qty and price:", overlap.sum(), "of", len(fact))
    print(" non-null quantity:", fact["quantity"].notna().sum(), "of", len(fact))
    print(" non-null unit_price:", fact["unit_price"].notna().sum(), "of", len(fact))
    print(" non-null line_amount:", fact["line_amount"].notna().sum(), "of", len(fact))

    # 7) Synthetic order_line_id (row number within each order)
    fact["order_line_id"] = (
        fact.groupby("order_id").cumcount() + 1
    ).astype(str)

    # 8) Final projection – include all linked dimension keys
    fact_final = fact[
        [
            "order_id",
            "order_line_id",
            "user_key",
            "product_key",
            "merchant_key",
            "staff_key",
            "campaign_key",
            "order_date_key",
            "quantity",
            "line_amount",
        ]
    ].copy()

    print(
        f" Final fact shape: {len(fact_final)} rows x "
        f"{len(fact_final.columns)} cols"
    )

    fact_final.to_sql(
        "order_line_fact",
        engine,
        schema="gold",
        if_exists="append",
        index=False,
        chunksize=1000,
    )
    print(f" [OK] Loaded {len(fact_final)} rows into gold.order_line_fact")


# =======================
# Orchestration
# =======================

def load_dimensions_and_facts():
    print("\nSTAGE 3: LOADING DIMENSIONS AND FACTS")
    load_user_dim()
    load_product_dim()
    load_merchant_dim()
    load_staff_dim()
    load_campaign_dim()
    load_credit_card_dim()
    load_date_dim()
    load_order_line_fact()
    cleanup_memory()


if __name__ == "__main__":
    print("SHOPZADA GOLD LAYER PIPELINE STARTING...")
    create_production_schema()
    truncate_gold_tables()
    load_dimensions_and_facts()
    print("\n" + "=" * 80)
    print("GOLD LAYER COMPLETE - 8 TABLES SHOULD NOW BE AVAILABLE IN SCHEMA gold.")
    print("=" * 80)
