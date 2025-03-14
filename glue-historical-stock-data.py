import sys
import requests
import json
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

# Initialize Spark
spark = SparkSession.builder.appName("StockHistoricalData").getOrCreate()

# AWS S3 Config
S3_BUCKET = "tradeshastra-raw"
API_KEY = "sk-live-A3vG8gN5k0rbWhJFO1KVK1vaK5Cw5Rvma21yS0RS"
BASE_URL = "https://stock.indianapi.in/historical_data"

# Define Schema to Handle Empty Data
schema = StructType([
    StructField("date", DateType(), True),
    StructField("price", DoubleType(), True),
    StructField("stock_name", StringType(), True)
])

def fetch_historical_data(stock_name, period="6m", filter_type="price"):
    """Fetch historical stock data from API and extract necessary fields"""
    url = BASE_URL
    headers = {"x-api-key": API_KEY}
    params = {"stock_name": stock_name, "period": period, "filter": filter_type}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        datasets = response.json().get("datasets", [])
        
        # Extract values from datasets
        for dataset in datasets:
            if dataset["metric"] == "Price":  # Ensure we're getting price data
                return [{"date": record[0], "price": float(record[1]), "stock_name": stock_name} for record in dataset["values"]]
    
    print(f"No data available for {stock_name}")
    return []

# Fetch Data for Multiple Stocks
stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
all_data = []

for stock in stocks:
    data = fetch_historical_data(stock, period="6m")
    all_data.extend(data)

# Handle Empty Dataset: Create an Empty DataFrame if No Data Found
if not all_data:
    print("No data found for any stock. Creating an empty DataFrame.")
    df = spark.createDataFrame([], schema)
else:
    df = spark.createDataFrame(all_data)

# Convert timestamp to Date and Create Partitions
df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))  # Convert string to Date
df = df.withColumn("year", col("date").substr(1, 4))
df = df.withColumn("month", col("date").substr(6, 2))

# Define S3 Partition Path
s3_path = f"s3://{S3_BUCKET}/historical_stock_data/"

# Write Data to S3 (Partitioned by Year & Month)
df.write.mode("overwrite").partitionBy("year", "month").parquet(s3_path)

print(f"Stored historical stock data in {s3_path}")
