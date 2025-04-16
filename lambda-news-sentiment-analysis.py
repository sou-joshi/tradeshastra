import json
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.environ.get("HF_API_TOKEN", "your_token_here_if_testing_locally")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
CONFIDENCE_THRESHOLD = 0.7

MODELS = {
    "finbert-tone": "yiyanghkust/finbert-tone",
    "finbert-prosus": "ProsusAI/finbert",
    "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment"
}

MODEL_ALIAS = {
    "yiyanghkust/finbert-tone": "finbert-tone",
    "ProsusAI/finbert": "finbert-prosus",
    "cardiffnlp/twitter-roberta-base-sentiment": "twitter-roberta"
}


# Standardized label mapping
LABEL_MAPPING = {
    "finbert-tone": {
        "neutral": "neutral",
        "positive": "positive",
        "negative": "negative"
    },
    "finbert-prosus": {
        "neutral": "neutral",
        "positive": "positive",
        "negative": "negative"
    },
    "twitter-roberta": {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
}

def standardize_label(model_name, original_label):
    """Convert model-specific labels to standard labels"""
    model_key = MODEL_ALIAS.get(model_name, model_name)  # Normalize to short key

    normalized_label = original_label.upper() if model_key == "twitter-roberta" else original_label.lower()
    mapped = LABEL_MAPPING.get(model_key, {}).get(normalized_label)

    if mapped:
        return mapped
    else:
        logger.warning(f"Unmapped label '{original_label}' for model '{model_name}'")
        return normalized_label.replace("label_", "")



def interpret_confidence(score):
    """Human-readable confidence interpretation"""
    if score > 0.9: return "very high confidence"
    elif score > 0.7: return "high confidence"
    elif score > 0.5: return "moderate confidence"
    else: return "low confidence"

def fetch_article_text(url):
    """Extract main content from news article"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "lxml")
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

def analyze_sentiment(text, model_name):
    """Get sentiment analysis from Hugging Face API"""
    endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
    try:
        response = requests.post(
            endpoint,
            headers=HEADERS,
            json={"inputs": text[:512]},  # Truncate to avoid token limits
            timeout=15
        )
        response.raise_for_status()
        results = response.json()

        if isinstance(results, list):
            predictions = results[0] if (results and isinstance(results[0], list)) else results
            if predictions:
                top_prediction = max(predictions, key=lambda x: x["score"])
                original_label = top_prediction["label"]
                standardized_label = standardize_label(model_name, original_label)

                return {
                    "label": standardized_label,
                    "score": round(top_prediction["score"], 4),
                    "confidence": interpret_confidence(top_prediction["score"]),
                    "model": model_name,
                    "status": "success"
                }

        elif isinstance(results, dict) and "label" in results:
            original_label = results["label"]
            standardized_label = standardize_label(model_name, original_label)
            return {
                "label": standardized_label,
                "score": round(results["score"], 4),
                "confidence": interpret_confidence(results["score"]),
                "model": model_name,
                "status": "success"
            }

        return {
            "error": "Unexpected response structure",
            "raw_response": results,
            "model": model_name,
            "status": "failed"
        }

    except Exception as e:
        return {
            "error": str(e),
            "model": model_name,
            "status": "failed"
        }

def lambda_handler(event, context):
    """Main Lambda entry point"""
    try:
        url = event.get("url")
        company = event.get("company", "a company")

        if not url:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "URL parameter is required",
                    "timestamp": datetime.utcnow().isoformat()
                })
            }

        logger.info(f"Fetching article from URL: {url}")
        article_text = fetch_article_text(url)
        if article_text.startswith("Failed"):
            logger.error(article_text)
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": article_text,
                    "url": url,
                    "timestamp": datetime.utcnow().isoformat()
                })
            }

        analysis_text = f"News about {company}: {article_text}"
        results = {}

        for name, model in MODELS.items():
            logger.info(f"Analyzing with model: {name}")
            results[name] = analyze_sentiment(analysis_text, model)
            logger.info(f"Result for {name}: {json.dumps(results[name], indent=2)}")

        response_data = {
            "url": url,
            "company": company,
            "timestamp": datetime.utcnow().isoformat(),
            "text_length": len(article_text),
            "analysis": results,
            "text_snippet": article_text[:200] + "..." if len(article_text) > 200 else article_text,
            "confidence_interpretation": {
                "very high confidence": ">90% certainty",
                "high confidence": "70-90% certainty",
                "moderate confidence": "50-70% certainty",
                "low confidence": "<50% certainty"
            }
        }

        logger.info("==== FINAL CLEAN OUTPUT ====")
        logger.info(json.dumps(response_data, indent=2))
        print(json.dumps(response_data, indent=2))

        return {
            "statusCode": 200,
            "body": json.dumps(response_data, indent=2)
        }

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Internal server error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        }
