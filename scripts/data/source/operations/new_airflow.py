from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Infrastructure_Engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'shopzada_medallion_pipeline',
    default_args=default_args,
    description='Orchestrator for Bronze, Silver, Gold',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # Task 1: Check source data
    check_source = BashOperator(
        task_id='check_source_availability',
        bash_command='ls /app/scripts/data/source'
    )

    # Task 2: Bronze (you can later replace echo with docker/compose call)
    signal_bronze = BashOperator(
        task_id='trigger_bronze_ingestion',
        bash_command='echo "Triggering Bronze Container execution..."'
    )

    # Task 3: Silver
    signal_silver = BashOperator(
        task_id='trigger_silver_transformation',
        bash_command='echo "Bronze success. Triggering Silver Container..."'
    )

    # Task 4: Gold load (runs your gold script inside Airflow container)
    run_gold_load = BashOperator(
        task_id='run_gold_load',
        bash_command='python /app/scripts/data/landing/modded_goldload.py'
        # adjust path if your gold script is mounted elsewhere
    )

    check_source >> signal_bronze >> signal_silver >> run_gold_load
