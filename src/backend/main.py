
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Literal
from datetime import datetime
import uuid
import requests
import re

from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI(
    title="SentimentIQ API",
    description="AI-Powered Sentiment Intelligence Platform",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU; use 0 for GPU if available
    )
except Exception as e:
    print(f"Warning: Transformer model load failed: {e}")
    sentiment_analyzer = None

lemmatizer = WordNetLemmatizer()


results_store = {}


class AnalyzeRequest(BaseModel):
    url: HttpUrl

class WordFrequency(BaseModel):
    word: str
    count: int

class TopPhrase(BaseModel):
    phrase: str
    sentiment: str
    score: float

# class SentimentResult(BaseModel):
#     id: str
#     url: str
#     sentiment: Literal["positive", "negative", "neutral"]
#     confidence: float
#     distribution: dict
#     wordFrequencies: List[WordFrequency]
#     topPhrases: List[TopPhrase]
#     summary: str
#     analyzedAt: str
#     sourceType: Literal["website", "youtube", "twitter", "unknown"]
#     totalTextsAnalyzed: int
class SentimentScore(BaseModel):
    text: str
    label: Literal["positive", "negative", "neutral"]
    score: float

class SentimentResult(BaseModel):
    id: str
    url: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    distribution: dict
    wordFrequencies: List[WordFrequency]
    topPhrases: List[TopPhrase]
    # scores: List[dict]
    scores: List[SentimentScore]
    summary: str
    analyzedAt: str
    sourceType: Literal["website", "youtube", "twitter", "unknown"]
    totalTextsAnalyzed: int



def detect_source_type(url: str) -> str:
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    else:
        return "website"

# def fetch_with_jina(url: str) -> str:
#     """Fetch clean Markdown content using free Jina Reader API."""
#     jina_url = f"https://r.jina.ai/{url}"
#     headers = {
#         "Accept": "application/json",
#         "X-Return-Format": "markdown",
#         "X-Retain-Images": "none"
#     }
#     try:
#         response = requests.get(jina_url, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         content = data.get("data", {}).get("content") or data.get("content", "")
#         content = data.get("data", {}).get("content") or data.get("content", "")
#         content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # extra safety
#         return content
#     except Exception as e:
#         print(f"Jina Reader error: {e}")
#         return ""

# ... (keep all your imports and setup same)

def fetch_with_jina(url: str) -> str:
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "application/json",
        "X-Return-Format": "markdown",
        "X-Retain-Images": "none",
        "X-Timeout": "30000"  # Give more time for YouTube
    }
    try:
        response = requests.get(jina_url, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data.get("data", {}).get("content") or data.get("content", "")
        # Extra cleanup
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # Remove leftover links
        return content.strip()
    except Exception as e:
        print(f"Jina error: {e}")
        return ""

def extract_texts(markdown: str, source_type: str) -> List[str]:
    if not markdown:
        return []

    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    texts = []
    junk_keywords = [
        "view", "views", "ago", "streamed", "playback", "sign in", "bot", "error",
        "share", "playlist", "transcript", "subscribe", "java:", "bedrock:", "discord",
        "follow along", "hashtag", "channel", "live", "more"
    ]

    if source_type == "youtube":
        # Focus on lines that look like real comments
        for line in lines:
            cleaned = line.strip()
            if len(cleaned) < 20 or len(cleaned) > 600:
                continue
            if any(junk in cleaned.lower() for junk in junk_keywords):
                continue
            if cleaned.startswith(("[]", "[", "Back", "Search", "If playback", "An error", "Sign in", "Follow along")):
                continue
            # Prefer lines with emojis (common in comments) or @username
            if "😂" in cleaned or "❤" in cleaned or "@" in cleaned or "legend" in cleaned.lower() or "rip" in cleaned.lower():
                texts.append(cleaned)
            elif len(texts) < 20:  # Fallback if too few
                texts.append(cleaned)

    # Dedupe
    seen = set()
    final_texts = []
    for t in texts:
        low = t.lower()
        if low not in seen:
            final_texts.append(t)
            seen.add(low)

    return final_texts[:100]  # Limit to avoid noise

def extract_texts(markdown: str, source_type: str) -> List[str]:
    """Parse Markdown into individual text segments (comments/tweets/etc.)"""
    if not markdown:
        return []

    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    texts = []

    if source_type == "youtube":
       
        for line in lines:
            if len(line) > 30 and len(line) < 1000 and not line.startswith(("#", "[", ">", "- ", "* ")):
               
                match = re.match(r"^@?(\w[\w.-]*\w)\s*[:\-–—]\s*(.+)", line)
                if match:
                    text = match.group(2).strip()
                else:
                    text = line
                if len(text) > 20:
                    texts.append(text)

    elif source_type == "twitter":
        for line in lines:
            cleaned = line.lstrip("#*-_> ")
            if 20 < len(cleaned) < 500:
                texts.append(cleaned)

    else:  
        for line in lines:
            cleaned = line.lstrip("#*-_> ")
            if 30 < len(cleaned) < 1000:
                texts.append(cleaned)

    
    if len(texts) < 10:
        paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
        for para in paragraphs:
            cleaned = re.sub(r"^#+", "", para).strip()
            if 50 < len(cleaned) < 2000:
                texts.append(cleaned)

    return texts[:400] 

def preprocess_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    except:
        tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

def analyze_sentiment(texts: List[str]) -> dict:
    if not texts:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "distribution": {"positive": 33, "negative": 33, "neutral": 34},
            "scores": []
        }

    positive_count = negative_count = neutral_count = 0
    all_scores = []

    for text in texts[:100]:  
        if len(text) < 10:
            continue

        if sentiment_analyzer:
            try:
                result = sentiment_analyzer(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                if label == 'positive':
                    positive_count += 1
                else:
                    negative_count += 1
                all_scores.append({"text": text[:100], "label": label, "score": score})
            except:
                neutral_count += 1
        else:
           
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointing']
            text_lower = text.lower()
            pos = sum(w in text_lower for w in positive_words)
            neg = sum(w in text_lower for w in negative_words)
            if pos > neg:
                positive_count += 1
            elif neg > pos:
                negative_count += 1
            else:
                neutral_count += 1

    total = positive_count + negative_count + neutral_count or 1
    distribution = {
        "positive": round(positive_count / total * 100),
        "negative": round(negative_count / total * 100),
        "neutral": round(neutral_count / total * 100)
    }

    if positive_count >= negative_count and positive_count >= neutral_count:
        sentiment = "positive"
        confidence = positive_count / total * 100
    elif negative_count >= positive_count and negative_count >= neutral_count:
        sentiment = "negative"
        confidence = negative_count / total * 100
    else:
        sentiment = "neutral"
        confidence = neutral_count / total * 100

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "distribution": distribution,
        "scores": all_scores
    }

def extract_word_frequencies(texts: List[str], top_n: int = 15) -> List[WordFrequency]:
    all_words = []
    for text in texts:
        processed = preprocess_text(text)
        all_words.extend(processed.split())
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    return [WordFrequency(word=word, count=count) for word, count in top_words]

def extract_top_phrases(sentiment_scores: List[dict]) -> List[TopPhrase]:
    phrases = []
    for score_data in sentiment_scores[:10]:  # More top phrases
        phrases.append(TopPhrase(
            phrase=score_data.get("text", "")[:80] + "...",
            sentiment=score_data.get("label", "neutral"),
            score=round(score_data.get("score", 0.5), 3)
        ))
    return phrases

def generate_summary(url: str, sentiment: str, texts_count: int) -> str:
    return f"Analysis of {url} shows predominantly {sentiment} sentiment based on {texts_count} extracted text segments (comments, posts, or content). Key themes and user opinions are highlighted through word frequencies and top sentiment-bearing phrases."

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/analyze", response_model=SentimentResult)
async def analyze_url(request: AnalyzeRequest):
    url = str(request.url)
    source_type = detect_source_type(url)

    markdown = fetch_with_jina(url)
    texts = extract_texts(markdown, source_type)

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable content from the URL (try checking if the page is public and has comments/text)."
        )

    sentiment_data = analyze_sentiment(texts)
    word_frequencies = extract_word_frequencies(texts)
    top_phrases = extract_top_phrases(sentiment_data.get("scores", []))

    summary = generate_summary(url, sentiment_data["sentiment"], len(texts))

    result_id = str(uuid.uuid4())
    # result = SentimentResult(
    #     id=result_id,
    #     url=url,
    #     sentiment=sentiment_data["sentiment"],
    #     confidence=sentiment_data["confidence"],
    #     distribution=sentiment_data["distribution"],
    #     wordFrequencies=word_frequencies,
    #     topPhrases=top_phrases,
    #     summary=summary,
    #     analyzedAt=datetime.utcnow().isoformat(),
    #     sourceType=source_type,
    #     totalTextsAnalyzed=len(texts)
    # )
    result = SentimentResult(
    id=result_id,
    url=url,
    sentiment=sentiment_data["sentiment"],
    confidence=sentiment_data["confidence"],
    distribution=sentiment_data["distribution"],
    wordFrequencies=word_frequencies,
    topPhrases=top_phrases,
    scores=sentiment_data.get("scores", []),  # <-- ADD THIS
    summary=summary,
    analyzedAt=datetime.utcnow().isoformat(),
    sourceType=source_type,
    totalTextsAnalyzed=len(texts)
    )


    results_store[result_id] = result
    return result

@app.get("/results/{result_id}", response_model=SentimentResult)
async def get_result(result_id: str):
    if result_id not in results_store:
        raise HTTPException(status_code=404, detail="Result not found")
    return results_store[result_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)