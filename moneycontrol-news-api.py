import json
import requests
import boto3
import pandas as pd
import hashlib
import logging
from bs4 import BeautifulSoup
from datetime import datetime

# Setup Logging
LOG_GROUP = "/lambda/MoneycontrolNewsLambda"
AWS_REGION = "us-east-1"

# AWS Clients
s3_client = boto3.client("s3")
logs_client = boto3.client("logs", region_name=AWS_REGION)

# AWS S3 Configuration
S3_BUCKET_NAME = "tradeshastra-raw"
NEWS_FOLDER = "moneycontrol-news/"
METADATA_FILE = "metadata/news_ids.txt"

# Moneycontrol News URLs to Scrape
MONEYCONTROL_NEWS_URLS = [
    "https://www.moneycontrol.com/news/business/markets/",
    "https://www.moneycontrol.com/news/business/",
    #"https://www.moneycontrol.com/news/business/stocks/",
    #"https://www.moneycontrol.com/news/business/economy/",
    #"https://www.moneycontrol.com/news/business/ipo/",
]

# Setup Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def setup_cloudwatch_logging():
    """
    Ensures that CloudWatch Log Group and Log Stream exist before logging.
    """
    try:
        # Create Log Group if it doesn’t exist
        logs_client.create_log_group(logGroupName=LOG_GROUP)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass  # Log group already exists

    try:
        # Create Log Stream
        logs_client.create_log_stream(
            logGroupName=LOG_GROUP,
            logStreamName="MoneycontrolNewsScraper"
        )
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass  # Log stream already exists
    except Exception as e:
        print(f"Error creating CloudWatch log stream: {e}")

def log_to_cloudwatch(level, message):
    """
    Logs execution details in AWS CloudWatch.
    """
    try:
        log_event = {
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "message": f"{level}: {message}"
        }

        # Ensure log stream exists before writing logs
        setup_cloudwatch_logging()

        logs_client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName="MoneycontrolNewsScraper",
            logEvents=[log_event]
        )

    except logs_client.exceptions.ResourceNotFoundException:
        print("Log stream does not exist. Creating now...")
        setup_cloudwatch_logging()
    except Exception as e:
        print(f"Error writing to CloudWatch logs: {e}")


def generate_news_id(title, published_date):
    """
    Generates a unique hash ID for deduplication.
    """
    return hashlib.md5(f"{title}{published_date}".encode()).hexdigest()

def fetch_existing_news_ids():
    """
    Retrieves stored news article IDs from S3 to avoid duplicates.
    """
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=METADATA_FILE)
        existing_ids = obj["Body"].read().decode("utf-8").split("\n")
        return set(existing_ids)
    except s3_client.exceptions.NoSuchKey:
        log_to_cloudwatch("INFO", "No metadata file found. Assuming first-time run.")
        return set()
    except Exception as e:
        log_to_cloudwatch("ERROR", f"Error reading metadata file: {e}")
        return set()

def update_metadata_file(new_news_ids):
    """
    Updates metadata file in S3 with newly processed news article IDs.
    """
    if not new_news_ids:
        return

    try:
        existing_ids = fetch_existing_news_ids()
        combined_ids = existing_ids.union(set(new_news_ids))
        metadata_content = "\n".join(combined_ids)

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=METADATA_FILE,
            Body=metadata_content.encode("utf-8"),
            ContentType="text/plain"
        )
        log_to_cloudwatch("INFO", f"Updated metadata file with {len(new_news_ids)} new entries.")
    except Exception as e:
        log_to_cloudwatch("ERROR", f"Error updating metadata: {e}")

def get_full_article_content(url):
    """
    Extracts full article content from Moneycontrol news page.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        article_body = soup.find("div", class_="article__content")

        if article_body:
            return article_body.get_text(separator="\n").strip()
        else:
            return "Full content not available"
    except Exception as e:
        log_to_cloudwatch("ERROR", f"Error fetching article from {url}: {e}")
        return "Error retrieving content"

def fetch_news_articles():
    """
    Scrapes up to 50 latest news articles from Moneycontrol.
    Stops scraping once 50 articles are collected.
    
    Returns:
        list: List of news articles with metadata and full content
    """
    news_data = []
    max_articles = 50  # Stop after collecting 50 articles

    for category_url in MONEYCONTROL_NEWS_URLS:
        try:
            response = requests.get(category_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Extract all news article blocks
            articles = soup.find_all("li", class_="clearfix")
            if not articles:
                log_to_cloudwatch("WARNING", f"No articles found at {category_url}")

            for article in articles:
                if len(news_data) >= max_articles:
                    return news_data  # Stop once 50 articles are collected

                title_element = article.find("h2")
                link_element = article.find("a")

                if not title_element or not link_element:
                    continue  # Skip if title or link is missing

                title = title_element.text.strip()
                link = link_element["href"]

                # Get publication date if available
                date_element = article.find("span", class_="article__date")
                published_date = date_element.text.strip() if date_element else "Unknown Date"

                # Generate a unique news ID
                news_id = generate_news_id(title, published_date)

                # Fetch full article content
                full_content = get_full_article_content(link)

                news_data.append({
                    "news_id": news_id,
                    "title": title,
                    "category": category_url.split("/")[-2],  # Extract last part of URL as category
                    "link": link,
                    "published_date": published_date,
                    "content": full_content,
                    "source": "Moneycontrol",
                    "retrieved_at": datetime.utcnow().isoformat()
                })

        except requests.exceptions.RequestException as e:
            log_to_cloudwatch("ERROR", f"Error fetching news from {category_url}: {e}")

    return news_data

def save_to_s3(data):
    """
    Saves deduplicated news articles to S3 in Parquet format.
    """
    if not data:
        log_to_cloudwatch("INFO", "No new data to save.")
        return

    existing_news_ids = fetch_existing_news_ids()
    new_data = [news for news in data if news["news_id"] not in existing_news_ids]

    if not new_data:
        log_to_cloudwatch("INFO", "No new articles found (all duplicates).")
        return

    df = pd.DataFrame(new_data)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    s3_key = f"{NEWS_FOLDER}moneycontrol_news_{timestamp}.parquet"

    df.to_parquet("/tmp/news.parquet", index=False)
    s3_client.upload_file("/tmp/news.parquet", S3_BUCKET_NAME, s3_key)

    update_metadata_file([news["news_id"] for news in new_data])
    log_to_cloudwatch("INFO", f"Saved {len(new_data)} new articles to S3.")

def lambda_handler(event, context):
    """
    AWS Lambda function entry point. Scrapes Moneycontrol news and stores it in S3.
    """
    log_to_cloudwatch("INFO", "Starting Moneycontrol News Scraper...")
    news_data = fetch_news_articles()
    save_to_s3(news_data)
    log_to_cloudwatch("INFO", "Pipeline execution completed.")

    return {"statusCode": 200, "body": json.dumps("Success")}
