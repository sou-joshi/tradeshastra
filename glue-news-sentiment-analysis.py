import os
import sys
import boto3
import json
import requests
import logging
import nltk
import yfinance as yf
from bs4 import BeautifulSoup
from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, DoubleType

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Spark Session
spark = SparkSession.builder.appName("NewsSentimentAnalysis").getOrCreate()

# AWS S3 Configuration
INPUT_S3_PATH = "s3://tradeshastra-raw/moneycontrol-news/"
OUTPUT_S3_PATH = "s3://tradeshastra-conformed/moneycontrol-news/"

# Initialize Transformers Pipelines
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Download Required NLTK Resources
nltk.download("vader_lexicon")

# Load Data from S3
df = spark.read.parquet(INPUT_S3_PATH)

def fetch_article_content(url):
    """Fetches the full article content from a URL if the original content is missing."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text() for p in paragraphs if p.get_text()])
            
            return article_text if article_text else "Content not available"
        else:
            logging.warning(f"Failed to fetch content from {url} - HTTP {response.status_code}")
            return "Content not available"
    except requests.RequestException as e:
        logging.error(f"Request failed for {url} - {str(e)}")
        return "Content not available"

def extract_tickers(text):
    """Extracts stock tickers using BERT Named Entity Recognition (NER)."""
    if not text or text == "Content not available":
        return ""
    
    entities = ner_pipeline(text)
    tickers = set()

    for entity in entities:
        if entity["entity"].startswith("B-ORG"):
            symbol = entity["word"].upper()
            try:
                stock = yf.Ticker(symbol)
                if stock.info.get("regularMarketPrice") is not None:
                    tickers.add(symbol)
            except Exception as e:
                logging.debug(f"Failed to validate ticker {symbol}: {str(e)}")

    return ", ".join(tickers)

def analyze_sentiment(text):
    """Analyzes sentiment of the text using BERT Sentiment Analysis Model."""
    if not text or text == "Content not available":
        return {"score": 0, "sentiment": "Neutral"}
    
    result = sentiment_pipeline(text[:512])[0]  # Limit text length for BERT
    sentiment_label = result["label"]
    sentiment_score = result["score"]

    sentiment = "Positive" if sentiment_label == "POSITIVE" else "Negative" if sentiment_label == "NEGATIVE" else "Neutral"

    return {"score": sentiment_score, "sentiment": sentiment}

def assign_sentiment_per_ticker(text):
    """Assigns sentiment scores to extracted tickers."""
    tickers = extract_tickers(text)
    sentiment = analyze_sentiment(text)

    ticker_sentiments = [{"ticker": ticker, "score": sentiment["score"], "sentiment": sentiment["sentiment"]} for ticker in tickers.split(", ")]
    
    return json.dumps(ticker_sentiments)

# Define UDFs for Spark Processing
fetch_content_udf = udf(fetch_article_content, StringType())
extract_tickers_udf = udf(extract_tickers, StringType())
analyze_sentiment_udf = udf(lambda text: analyze_sentiment(text)["score"], DoubleType())
classify_sentiment_udf = udf(lambda text: analyze_sentiment(text)["sentiment"], StringType())
ticker_sentiment_udf = udf(assign_sentiment_per_ticker, StringType())

# Apply Transformations
df = df.withColumn("content", fetch_content_udf(col("link"))) \
       .withColumn("tickers", extract_tickers_udf(col("content"))) \
       .withColumn("overall_sentiment_score", analyze_sentiment_udf(col("content"))) \
       .withColumn("overall_sentiment", classify_sentiment_udf(col("content"))) \
       .withColumn("ticker_sentiments", ticker_sentiment_udf(col("content")))

# Write Data to S3 in Parquet Format
df.write.mode("overwrite").parquet(OUTPUT_S3_PATH)

logging.info(f"Sentiment analysis job completed successfully. Data stored at {OUTPUT_S3_PATH}")
