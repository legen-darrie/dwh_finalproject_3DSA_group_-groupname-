FROM python:3.11-slim

WORKDIR /app

# Install pandas and SQLAlchemy dependencies
# psycopg2-binary is the PostgreSQL driver
RUN pip install pandas sqlalchemy psycopg2-binary pyarrow fastparquet

# Copy the Gold script into the container
COPY scripts/gold/modded_goldload.py . 
CMD ["python", "modded_goldload.py"]

CMD ["python", "modded_goldload.py"]