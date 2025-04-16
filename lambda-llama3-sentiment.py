import os
import json
import boto3
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Bedrock model and inference profile
LLAMA_MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:864899837989:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def fetch_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        selectors = [
            "div.article_content", "div.content_wrapper",
            "div.article-body", "article", "div.story-content"
        ]

        for selector in selectors:
            container = soup.select_one(selector)
            if container:
                for elem in container.find_all(['script', 'style', 'iframe']):
                    elem.decompose()
                return " ".join(p.get_text().strip() for p in container.find_all("p"))

        return soup.title.get_text().strip() if soup.title else ""
    except Exception as e:
        return f"Failed to fetch article: {str(e)}"

def call_llama3_bedrock(prompt):
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        response = bedrock.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        return response_body.get("generation", "No generation returned")
    except Exception as e:
        raise RuntimeError(f"Bedrock invocation failed: {str(e)}")

def lambda_handler(event, context):
    try:
        url = event.get("url")
        company = event.get("company", "a company")

        if not url:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing URL"})}

        article_text = fetch_article_text(url)
        if article_text.startswith("Failed"):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "text_metrics": {
                            "length": len(article_text),
                            "snippet": article_text[:120]
                        }
                    },
                    "errors": [article_text]
                })
            }

        prompt = (
            f"Perform a sentiment analysis for the following news article about {company}.\n\n"
            f"Text:\n{article_text}\n\n"
            f"Return the sentiment as one of: positive, negative, or neutral. "
            f"Also provide a confidence score between 0 and 1 in JSON format."
        )

        llama_response = call_llama3_bedrock(prompt)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "company": company,
                    "text_metrics": {
                        "length": len(article_text),
                        "snippet": article_text[:120]
                    },
                    "llama_output": llama_response
                }
            }, indent=2)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "text_metrics": {},
                },
                "errors": [f"Unexpected error: {str(e)}"]
            }, indent=2)
        }

# For local testing
if __name__ == "__main__":
    test_event = {
        "url": "https://www.moneycontrol.com/news/business/tech-mahindra-share-price-updates-1234567.html",
        "company": "Tech Mahindra"
    }
    print(json.dumps(json.loads(lambda_handler(test_event, None)["body"]), indent=2))
