FROM apache/airflow:2.8.3-python3.11
USER airflow
RUN pip install --no-cache-dir pandas pyarrow openpyxl lxml
COPY new_airflow.py /opt/airflow/dags/new_airflow.py
