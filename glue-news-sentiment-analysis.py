import os
import sys
import boto3
import nltk
import yfinance as yf
from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType, DoubleType, StructType, StructField

# Download NLTK resources
nltk.download("vader_lexicon")

# Create Spark Session
spark = SparkSession.builder.appName("NewsSentimentAnalysis").getOrCreate()

# S3 Paths
INPUT_S3_PATH = "s3://tradeshastra-raw/moneycontrol-news/"
OUTPUT_S3_PATH = "s3://tradeshastra-conformed/moneycontrol-news/"

# Load Data from S3
df = spark.read.parquet(INPUT_S3_PATH)

# Load Pre-trained BERT Models from Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to Extract Stock Tickers using BERT NER
def extract_tickers(text):
    if not text or text == "Full content not available":
        return []
    
    entities = ner_pipeline(text)
    tickers = set()

    for entity in entities:
        if entity["entity"].startswith("B-ORG"):  # BERT detects organizations well
            symbol = entity["word"].upper()
            try:
                stock = yf.Ticker(symbol)
                if stock.info.get("regularMarketPrice") is not None:  # Validate real ticker
                    tickers.add(symbol)
            except:
                continue

    return list(tickers)

# Function to Assign Sentiment Score using BERT
def analyze_sentiment(text):
    if not text or text == "Full content not available":
        return {"score": 0, "sentiment": "Neutral"}

    result = sentiment_pipeline(text[:512])[0]  # BERT has a token limit, so truncate if needed
    sentiment_label = result["label"]
    sentiment_score = result["score"]

    if sentiment_label == "POSITIVE":
        sentiment = "Positive"
    elif sentiment_label == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {"score": sentiment_score, "sentiment": sentiment}

# Function to Assign Sentiment Per Ticker
def assign_sentiment_per_ticker(text):
    tickers = extract_tickers(text)
    sentiment = analyze_sentiment(text)
    
    ticker_sentiments = [{"ticker": ticker, "score": sentiment["score"], "sentiment": sentiment["sentiment"]} for ticker in tickers]
    
    return ticker_sentiments

# Define UDFs for Spark Processing
extract_tickers_udf = udf(extract_tickers, ArrayType(StringType()))
analyze_sentiment_udf = udf(lambda text: analyze_sentiment(text)["score"], DoubleType())
classify_sentiment_udf = udf(lambda text: analyze_sentiment(text)["sentiment"], StringType())

# Define StructType for Ticker Sentiments
ticker_sentiment_schema = ArrayType(
    StructType([
        StructField("ticker", StringType(), True),
        StructField("score", DoubleType(), True),
        StructField("sentiment", StringType(), True)
    ])
)
ticker_sentiment_udf = udf(assign_sentiment_per_ticker, ticker_sentiment_schema)

# Apply Transformations
df = df.withColumn("tickers", extract_tickers_udf(col("content"))) \
       .withColumn("overall_sentiment_score", analyze_sentiment_udf(col("content"))) \
       .withColumn("overall_sentiment", classify_sentiment_udf(col("content"))) \
       .withColumn("ticker_sentiments", ticker_sentiment_udf(col("content")))

# Save Processed Data to S3 in Parquet Format
df.write.mode("overwrite").parquet(OUTPUT_S3_PATH)

print("Glue Job Completed Successfully.")
