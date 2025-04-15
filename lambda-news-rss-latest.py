import json
import boto3
import requests
import hashlib
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import xml.etree.ElementTree as ET

# AWS Config
AWS_REGION = "us-east-1"
S3_BUCKET = "tradeshastra-raw"
S3_PREFIX = "rss/it-news"
DYNAMODB_TABLE = "moneycontrol_news_data"

# AWS Clients
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# RSS Feeds
RSS_FEEDS = [
    # Economic Times (Stable)
    "https://economictimes.indiatimes.com/tech/rssfeeds/5880659.cms",
    "https://economictimes.indiatimes.com/industry/telecom/rssfeeds/13357270.cms",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/news/economy/rssfeeds/1977021501.cms",

    # Mint / LiveMint (Stable)
    "https://www.livemint.com/rss/technology",
    "https://www.livemint.com/rss/companies",
    "https://www.livemint.com/rss/markets",

    # The Hindu Business Line (Stable)
    "https://www.thehindubusinessline.com/info-tech/feeder/default.rss",
    "https://www.thehindubusinessline.com/companies/feeder/default.rss",
    "https://www.thehindubusinessline.com/markets/feeder/default.rss",

    # Times of India (Stable)
    "https://timesofindia.indiatimes.com/rssfeeds/5880659.cms",
    "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"
]

# IT stock keywords mapped to NSE codes
STOCK_KEYWORDS = {
    # Large Cap IT Companies
    "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hcl technologies": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "techm": "TECHM.NS",
    "ltimindtree": "LTIM.NS",
    "l&t infotech": "LTIM.NS",
    "l&t technology services": "LTTS.NS",
    "lnt": "LTIM.NS",
    "mindtree": "LTIM.NS",
    "mphasis": "MPHASIS.NS",
    "cognizant": "CTSH",  # Listed on NASDAQ but operates in India
    "cts": "CTSH",
    
    # Mid Cap IT Companies
    "persistent systems": "PERSISTENT.NS",
    "persistent": "PERSISTENT.NS",
    "lt ts": "LTTS.NS",
    "l&t ts": "LTTS.NS",
    "cyient": "CYIENT.NS",
    "cyient technologies": "CYIENT.NS",
    "coforge": "COFORGE.NS",
    "niit technologies": "COFORGE.NS",  # Former name of Coforge
    "zensar technologies": "ZENSARTECH.NS",
    "zensar": "ZENSARTECH.NS",
    "mindteck": "MINDTECK.NS",
    "mindteck india": "MINDTECK.NS",
    "sonata software": "SONATSOFTW.NS",
    "sonata": "SONATSOFTW.NS",
    "quick heal technologies": "QUICKHEAL.NS",
    "quickheal": "QUICKHEAL.NS",
    
    # Emerging IT Companies
    "intellect design arena": "INTELLECT.NS",
    "intellect": "INTELLECT.NS",
    "eclerx services": "ECLERX.NS",
    "eclerx": "ECLERX.NS",
    "3i infotech": "3IINFOLTD.NS",
    "3i infoltd": "3IINFOLTD.NS",
    "first source solutions": "FSL.NS",
    "firstsource": "FSL.NS",
    "kelton tech": "KELTECH.NS",
    "kelton": "KELTECH.NS",
    
    # IT Consulting & Services
    "hexaware technologies": None,  # Now private
    "hexaware": None,
    "larsen & toubro infotech": "LTIM.NS",
    "larsen and toubro infotech": "LTIM.NS",
    "lti": "LTIM.NS",
    
    # Product Companies
    "tanla platforms": "TANLA.NS",
    "tanla": "TANLA.NS",
    "newgen software": "NEWGEN.NS",
    "newgen": "NEWGEN.NS",
    "subex": "SUBEXLTD.NS",
    "subex limited": "SUBEXLTD.NS",
    
    # Emerging Technologies
    "nucleus software": "NUCLEUS.NS",
    "nucleus": "NUCLEUS.NS",
    "maveric systems": "MAVERIC.NS",
    "maveric": "MAVERIC.NS",
    
    # IT Education
    "niit": "NIITLTD.NS",
    "niit limited": "NIITLTD.NS",
    "aptech": "APTECHT.NS",
    "aptech limited": "APTECHT.NS",
    
    # Special Cases
    "infosys bpm": "INFY.NS",  # Subsidiary
    "tcs digital": "TCS.NS",   # Subsidiary
    "wipro digital": "WIPRO.NS" # Subsidiary
}

def generate_id(link):
    return hashlib.md5(link.encode()).hexdigest()

def news_exists(news_id):
    try:
        res = table.get_item(Key={"news_id": news_id})
        return "Item" in res
    except Exception as e:
        logger.error(f"Error checking DynamoDB for {news_id}: {e}")
        return False

def save_to_dynamodb(item):
    try:
        table.put_item(Item=item)
    except Exception as e:
        logger.error(f"Failed to insert into DynamoDB: {e}")

def detect_related_companies(title, description):
    content = (title + " " + (description or "")).lower()
    symbols, matched = set(), set()
    for keyword, symbol in STOCK_KEYWORDS.items():
        if keyword in content:
            symbols.add(symbol)
            matched.add(keyword)
    return list(symbols), list(matched)

def parse_rss_feed(url):
    new_articles = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(resp.content)

        for item in root.findall(".//item"):
            title = item.find("title").text or ""
            link = item.find("link").text
            pub_date = item.find("pubDate").text or datetime.utcnow().isoformat()
            desc = item.find("description").text or ""
            news_id = generate_id(link)

            if news_exists(news_id):
                continue

            companies, keywords = detect_related_companies(title, desc)

            record = {
                "news_id": news_id,
                "title": title,
                "link": link,
                "published_date": pub_date,
                "description": desc,
                "retrieved_at": datetime.utcnow().isoformat(),
                "related_companies": companies,
                "matched_keywords": keywords
            }

            save_to_dynamodb(record)
            new_articles.append(record)
    except Exception as e:
        logger.error(f"Error processing feed {url}: {e}")
    return new_articles

def save_to_s3(records):
    if not records:
        logger.info("No new articles found.")
        return

    df = pd.DataFrame(records)
    df["ingestion_ts"] = datetime.utcnow().isoformat()

    table_arrow = pa.Table.from_pandas(df)
    buffer = pa.BufferOutputStream()
    pq.write_table(table_arrow, buffer)

    now = datetime.utcnow()
    s3_key = f"{S3_PREFIX}/year={now.year}/month={now.month}/day={now.day}/news_{now.strftime('%H%M%S')}.parquet"

    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue().to_pybytes())
    logger.info(f"Saved {len(records)} new articles to {s3_key}")

def lambda_handler(event, context):
    all_new_articles = []
    for feed in RSS_FEEDS:
        new_articles = parse_rss_feed(feed)
        all_new_articles.extend(new_articles)

    save_to_s3(all_new_articles)

    return {
        "statusCode": 200,
        "body": json.dumps(f"Fetched {len(all_new_articles)} new articles")
    }
