import json
import requests
import boto3
from datetime import datetime, timedelta
import time
from botocore.exceptions import ClientError

# AWS Clients
ssm = boto3.client("ssm")
dynamodb = boto3.resource("dynamodb")
table_name = "StockPriceHistory"
table = dynamodb.Table(table_name)

# Fetch API Key from AWS Parameter Store
def get_api_key():
    """Retrieve API key from AWS SSM Parameter Store"""
    response = ssm.get_parameter(Name="/indianapi_key", WithDecryption=True)
    return response["Parameter"]["Value"]

API_KEY = get_api_key()  # Securely fetch API key
BASE_URL = "https://stock.indianapi.in/stock"

# Stock symbols
STOCKS = ["RELIANCE", "IOC", "BPCL", "HPCL", "MRPL", "TCS", "INFY", "HCLTECH",
          "WIPRO", "TECHM", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK",
          "INDUSINDBK", "SBIN", "BANKBARODA", "CANBK", "PNB", "UNIONBANK",
          "TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "HINDUNILVR",
          "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "SUNPHARMA", "DRREDDY",
          "CIPLA", "AUROPHARMA", "LUPIN", "TATASTEEL", "HINDALCO", "JSWSTEEL",
          "COALINDIA", "VEDL", "BHARTIARTL", "RELIANCEJIO", "IDEA", "MTNL",
          "TATACOMM", "NTPC", "POWERGRID", "ADANIGREEN", "TATAPOWER", "NHPC"]


def create_dynamodb_table():
    """Creates DynamoDB table if it does not exist"""
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "symbol", "KeyType": "HASH"},  # Partition Key
                {"AttributeName": "timestamp", "KeyType": "RANGE"}  # Sort Key
            ],
            AttributeDefinitions=[
                {"AttributeName": "symbol", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"}
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 10, "WriteCapacityUnits": 10}
        )
        print("Creating table... waiting for completion")
        table.wait_until_exists()
        print(f"Table {table_name} created successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table {table_name} already exists, proceeding...")
        else:
            raise

def fetch_stock_data(symbol, retries=3, delay=2):
    """Fetch stock market data from Indianapi with max retries set to 3"""
    url = f"{BASE_URL}"
    headers = {"x-api-key": API_KEY}
    params = {"name": symbol}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Extracting data based on response format
                current_price = data.get("currentPrice", {})
                bse_price = current_price.get("BSE", "0.0")
                nse_price = current_price.get("NSE", "0.0")

                return {
                    "symbol": symbol,
                    "bse_price": bse_price,
                    "nse_price": nse_price,
                    "timestamp": datetime.utcnow().isoformat(),  # Store history
                    "last_updated": datetime.utcnow().isoformat()
                }
            else:
                print(f"Attempt {attempt+1}: API failed for {symbol} (Status Code: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: Network error for {symbol} - {e}")

        time.sleep(delay * (2 ** attempt))  # Exponential backoff (2s → 4s)
    
    return {"symbol": symbol, "error": "Max retries exceeded"}

def store_in_dynamodb(stock_info):
    """Store stock data in DynamoDB for historical tracking"""
    try:
        table.put_item(Item=stock_info)
    except ClientError as e:
        print(f"DynamoDB Error: {e}")

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    create_dynamodb_table()  # Ensure table exists
    stock_data = [fetch_stock_data(stock) for stock in STOCKS]

    for data in stock_data:
        if "error" not in data:
            store_in_dynamodb(data)

    return {
        "statusCode": 200,
        "body": json.dumps(stock_data, indent=2)
    }
