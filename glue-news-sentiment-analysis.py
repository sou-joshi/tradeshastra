import sys
import json
import boto3
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# Initialize Spark and Glue Context
spark = SparkSession.builder.appName("MoneyControlNewsProcessing").getOrCreate()
glueContext = GlueContext(spark)
sc = spark.sparkContext
job = Job(glueContext)

# Read arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
job.init(args['JOB_NAME'], args)

# Function to fetch and clean article content
def fetch_article_content(url):
    try:
        response = urlopen(url)
        html_content = response.read().decode("utf-8")

        # Create boto3 client inside UDF to avoid PicklingError
        bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Prompt ensuring only the article content is returned
        prompt = (
            "Extract and return only the main article content from the following HTML. "
            "Ensure no additional information, summaries, or formatting details are returned. "
            "Strictly return only the clean article text:\n\n"
            f"{html_content}"
        )

        model_request = {
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.3
        }

        response = bedrock_runtime.invoke_model(
            modelId="us.meta.llama3-3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(model_request)
        )

        # Parse response
        result = json.loads(response["body"].read().decode("utf-8"))
        content = result.get("content", "").strip()

        return content if content else None

    except (URLError, HTTPError) as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

# Function to extract company tickers using NER
def extract_tickers(content):
    try:
        if not content:
            return None

        # Create boto3 client inside UDF to avoid PicklingError
        bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Prompt ensuring only the company names are returned
        prompt = (
            "Identify and return only the company names mentioned in this article, separated by commas. "
            "Do not return any extra text, explanations, or formatting details. "
            "Only return a comma-separated list of company names:\n\n"
            f"{content}"
        )

        model_request = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.2
        }

        response = bedrock_runtime.invoke_model(
            modelId="us.meta.llama3-3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(model_request)
        )

        # Parse response
        result = json.loads(response["body"].read().decode("utf-8"))
        tickers = result.get("content", "").strip()

        # Ensure no extra text, only company names
        if tickers:
            return tickers.replace("\n", "").replace(" ", "").strip(",")
        else:
            return None

    except Exception as e:
        print(f"Error extracting tickers: {e}")
        return None

# Register UDFs for Spark (using `@udf` instead of passing client)
fetch_article_content_udf = udf(fetch_article_content, StringType())
extract_tickers_udf = udf(extract_tickers, StringType())

# Read input Parquet file from S3
input_s3_path = "s3://tradeshastra-raw/moneycontrol-news/"
df = glueContext.read.format("parquet").load(input_s3_path)

# Apply transformations
df = df.withColumn("content", fetch_article_content_udf(col("link")))
df = df.withColumn("ticker", extract_tickers_udf(col("content")))

# Write output to a new S3 bucket
output_s3_path = "s3://tradeshastra-raw/moneycontrol-news-final/"
df.write.mode("overwrite").parquet(output_s3_path)

# Commit the job
job.commit()
