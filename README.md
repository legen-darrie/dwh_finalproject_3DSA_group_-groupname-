# ShopZada - End-to-End Medallion Data Warehouse Engineering Pipeline

ShopZada is a fictional e-commerce platform that is experiencing data fragmentation across its departments: Business, Customer Management, Enterprise, Marketing, and Operations.

Given this, Group LLN has developed a demo project that implements a full end-to-end Extract, Transform, Load *(ELT)* data warehouse following a Medallion structure (Bronze, Silver, and Gold) with Kimball-style dimensional Modeling principles. It ingests data from the different departments, transforms its contents, and utilizes star-schema models for analytics and Business Intelligence (BI).

---

## Architecture Overview

**Bronze Layer**  
*Raw landing Zone*

- Ingests data sources following the formats: CSV, Parquet, Pickle, Excel, JSON, and HTML.
- Preserves the original structure and adds audit columns (timestamp, department, filename, row UUID)

**Silver Layer**  
*Cleaning and Standardization*

- Uses Pandas to standardize and transform schemas, data types, and keys, followed by data-quality checkpoints.
- Writes cleaned Parquet tables referencing its original data sources.

**Gold Layer**  
*Star-Schema Models*  

- Integrates the setup for Kimball-style fact and dimension tables.
- Allows SQL querying and BI application.


Orchestration is made possible through **Apache Airflow**, and the entire stack is contained within a **Docker Compose** for reproducibility and portability.

---

## GitHub Repository Structure

```
project-root/
├── dashboard/
├── docs/
├── infra/
│   └── docker-compose.yml
├── scripts/
│   └── bronze/
│   └── data/
│   │   └── landing/
│   │   │   └── bronze_files/
│   │   │   └── gold_files/
│   │   │   └── silver_files/
│   │   └── source/
│   │   │   └── business/
│   │   │   └── customer management/
│   │   │   └── enterprise/
│   │   │   └── marketing/
│   │   │   └── operations/
│   └── gold/
│   └── silver/
├── sql/
├── workflows
└── README.md
```

---

## Getting Started

### **Prerequisites**
1. Install the following before starting:
- Docker and Docker Compose
- Python
- pgAdmin4 or PostgreSQL
- Apache Airflow
- Tableau

2. Download the dataset repository.  
3. Setup the environment.

```
docker compose -f infra/docker-compose.yml up -d
```

### **Bronze Layer (Load Data)**
4. Upload the raw files from the dataset to their corresponding folders.
- './source_data/business'
- './source_data/customer'
- './source_data/enterprise'
- './source_data/marketing'
- './source_data/operations'

5. Run the Bronze pipeline

```
docker compose -f infra/docker-compose.yml up bronze_ingestion --build
```

### **Silver Layer (Clean Data)**
6. Run the Silver pipeline

```
docker compose -f infra/docker-compose.yml up silver_transformation --build
```

### **Gold Layer (Star-Schema)**
7. Execute the SQL scripts in 'sql/' to  create and fill the fact and dimension tables from the parquet files coming from the Silver layer.

```
docker compose -f infra/docker-compose.yml up gold_load --build
```

### **Airflow Initialization**
8. Run the Apache airflow orchestrator.

```
docker compose -f infra/docker-compose.yml run --rm airflow airflow db init
```
```
docker compose -f infra/docker-compose.yml up -d airflow
```

### **User Interface Commands**
9. Airflow UI: http://localhost:8080  
pgAdmin UI: http://localhost:5050  
-- admin@shopzada.com  
-- pgadmin_password

postgreSQL:
- Host: postgres_db
- Port: 15432
- Database: shopzada_dwh
- User: shopzada_user
- Password: root

### **Optional: All-in-One**
10. Run the code below to set up the whole pipeline in one command.

```
docker compose -f infra/docker-compose.yml build --no-cache
```
---

## Data Quality and Validation Checkpoints

There are two explicit checkpoints where data validation is executed:  
**Bronze Validation**
- Verifies file existence, size, and populated dataframes.  

**Silver Validation**  
- Inspects and addresses null, duplicates, and data mismatching types, while enforcing columns and keys.

Both validation checkpoints produce consolidated log reports at the end of their respective medallion layer summarizing the code's execution.

---

## Test Cases

Listed below are several demonstration scenarios that align with the data warehouse's capabilities:
  
1. **Incremental Load:** The warehouse is able to ingest new files and assimilate well into the existing data. 
2. **Auditing:** Audit columns are created and presented for traceability and accountability, following data governance.  
3. **Full Pipeline:** New source files will undergo the whole process from ingestion to analytics, automatically.

---

## Tech Stack

- **Processing:** Python, Pandas, Pyarrow, Psycopg2, etc.  
- **Storage:** Bronze/Silver: Parquet files; Gold: Relational warehouse
- **Orchestration:** Apache Airflow (DAGs)
- **Containerization:** Docker/Docker Compose
- **Business Intelligence:** pgAdmin4/PostgreSQL, Tableau

