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

# Stock Keyword Mapping
STOCK_KEYWORDS = {
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS",
    "ioc": "IOC.NS", "indian oil corporation": "IOC.NS",
    "bpcl": "BPCL.NS", "bharat petroleum": "BPCL.NS",
    "hpcl": "HPCL.NS", "hindustan petroleum": "HPCL.NS",
    "mrpl": "MRPL.NS", "mangalore refinery": "MRPL.NS",
    "tcs": "TCS.NS", "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS", "hcl": "HCLTECH.NS", "hcl tech": "HCLTECH.NS",
    "hdfc": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS", "icici bank": "ICICIBANK.NS",
    "kotak": "KOTAKBANK.NS", "kotak bank": "KOTAKBANK.NS",
    "axis": "AXISBANK.NS", "axis bank": "AXISBANK.NS",
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "bob": "BANKBARODA.NS", "bank of baroda": "BANKBARODA.NS",
    "canara": "CANBK.NS", "canara bank": "CANBK.NS",
    "pnb": "PNB.NS", "punjab national bank": "PNB.NS",
    "union bank": "UNIONBANK.NS",
    "tatamotors": "TATAMOTORS.NS", "tata motors": "TATAMOTORS.NS",
    "maruti": "MARUTI.NS", "mahindra": "M&M.NS", "m&m": "M&M.NS",
    "bajaj": "BAJAJ-AUTO.NS", "bajaj auto": "BAJAJ-AUTO.NS",
    "itc": "ITC.NS", "nestle": "NESTLEIND.NS",
    "britannia": "BRITANNIA.NS", "dabur": "DABUR.NS",
    "sun pharma": "SUNPHARMA.NS", "cipla": "CIPLA.NS",
    "tata steel": "TATASTEEL.NS", "hindalco": "HINDALCO.NS",
    "jsw steel": "JSWSTEEL.NS", "coal india": "COALINDIA.NS",
    "bharti airtel": "BHARTIARTL.NS", "reliance jio": "RELIANCEJIO.NS",
    "vodafone idea": "IDEA.NS", "tata power": "TATAPOWER.NS",
    "ntpc": "NTPC.NS", "powergrid": "POWERGRID.NS"
}

# News Categories (Latest News Pages)
MONEYCONTROL_NEWS_URLS = [
    "https://www.moneycontrol.com/news/business/markets/",
    "https://www.moneycontrol.com/news/business/",
    "https://www.moneycontrol.com/news/business/stocks/",
    "https://www.moneycontrol.com/news/business/economy/",
    "https://www.moneycontrol.com/news/business/ipo/",
]

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_news_id(title, published_date):
    """Generate a unique news ID based on title and published date."""
    return hashlib.md5(f"{title}{published_date}".encode()).hexdigest()

def analyze_content_for_companies(content):
    """
    Analyze given text to match keywords for companies.
    
    Returns:
        - List of matched stock symbols.
        - List of matched keyword strings.
    """
    matched_symbols = set()
    matched_keywords = set()

    for keyword, symbol in STOCK_KEYWORDS.items():
        if keyword in content:
            matched_symbols.add(symbol)
            matched_keywords.add(keyword)

    return list(matched_symbols), list(matched_keywords)

def fetch_news_articles():
    """
    Fetch news articles from Moneycontrol news pages.
    This version uses the title (in lowercase) for company matching and deduplicates articles
    based on a unique news_id generated from the title and published date.
    """
    # Use a dictionary to store articles by their unique news_id to avoid duplicates.
    news_data = {}

    for category_url in MONEYCONTROL_NEWS_URLS:
        try:
            response = requests.get(category_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            articles = soup.find_all("li", class_="clearfix")

            for article in articles:
                title_element = article.find("h2")
                link_element = article.find("a")
                date_element = article.find("span", class_="article__date")

                if not title_element or not link_element:
                    continue

                title = title_element.text.strip()
                link = link_element["href"]
                published_date = date_element.text.strip() if date_element else "Unknown"
                news_id = generate_news_id(title, published_date)

                # Skip if this news_id already exists (duplicate article)
                if news_id in news_data:
                    continue

                # Use only the title (converted to lowercase) for keyword matching.
                related_symbols, matched_keywords = analyze_content_for_companies(title.lower())

                if related_symbols:
                    news_data[news_id] = {
                        "news_id": news_id,
                        "link": link,
                        "title": title,
                        "related_companies": related_symbols,
                        "company_names": matched_keywords,
                        "published_date": published_date,
                        "retrieved_at": datetime.utcnow().isoformat()
                    }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from {category_url}: {e}")

    # Return only unique articles.
    return list(news_data.values())

def save_to_s3(data):
    """
    Save the news data to an S3 bucket in Parquet format.
    """
    if not data:
        logger.info("No relevant articles found.")
        return

    df = pd.DataFrame(data)
    timestamp = datetime.utcnow()
    year, month, day = timestamp.strftime("%Y"), timestamp.strftime("%m"), timestamp.strftime("%d")
    s3_key = f"{NEWS_FOLDER}year={year}/month={month}/day={day}/filtered_news_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.parquet"

    df.to_parquet("/tmp/news.parquet", index=False)
    s3_client.upload_file("/tmp/news.parquet", S3_BUCKET_NAME, s3_key)
    logger.info(f"Saved {len(data)} articles to S3 at {s3_key}.")

def lambda_handler(event, context):
    """
    Main Lambda handler function.
    """
    try:
        logger.info("Starting Moneycontrol News Scraper (Latest News Only)...")
        relevant_news = fetch_news_articles()
        save_to_s3(relevant_news)
        logger.info("Pipeline execution completed.")
        return {"statusCode": 200, "body": json.dumps("Success")}
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"Failed: {str(e)}")}
