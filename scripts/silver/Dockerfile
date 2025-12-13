FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas pyarrow openpyxl lxml
COPY scripts/silver/modded_silverval.py . 
CMD ["python", "modded_silverval.py"]