"""
Sentiment Analysis Dashboard - Backend API
Real-time sentiment analysis from social media, reviews, and text
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List
from datetime import datetime
import uvicorn

# Sentiment Analysis Libraries
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis Dashboard API",
    version="1.0.0",
    description="Real-time sentiment analysis for social media, reviews, and text"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class TextAnalysisRequest(BaseModel):
    text: str
    source: str = "general"  # e.g., "twitter", "review", "feedback"

class SentimentResponse(BaseModel):
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    polarity_score: float
    intensity: dict
    source: str
    timestamp: str

class BulkAnalysisRequest(BaseModel):
    texts: List[dict]  # Each dict should have "text" and optional "source"

class DashboardStats(BaseModel):
    total_analyzed: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    average_sentiment_score: float
    trending_words: List[dict]
    sentiment_by_source: dict

# In-memory storage (for demo)
analysis_history = []
sentiment_scores = []

# Initialize sentiment analyzers
textblob_analyzer = TextBlob
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment using multiple methods
    Returns: dict with sentiment, confidence, and scores
    """
    
    # Clean text
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty")
    
    # TextBlob Analysis
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity  # -1 to 1
    textblob_subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    # VADER Analysis (better for social media)
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']  # -1 to 1
    
    # Average the scores
    average_polarity = (textblob_polarity + vader_compound) / 2
    
    # Determine sentiment label
    if average_polarity > 0.1:
        sentiment = "positive"
    elif average_polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Calculate confidence (how sure we are)
    confidence = abs(average_polarity)
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "polarity_score": round(average_polarity, 3),
        "textblob_polarity": round(textblob_polarity, 3),
        "vader_compound": round(vader_compound, 3),
        "intensity": {
            "positive": round(vader_scores['pos'], 3),
            "negative": round(vader_scores['neg'], 3),
            "neutral": round(vader_scores['neu'], 3)
        }
    }

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract top keywords from text"""
    # Simple keyword extraction (remove stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'be', 'have', 'has',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
    }
    
    words = text.lower().split()
    keywords = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
    
    # Get most common
    word_freq = Counter(keywords)
    return [word for word, _ in word_freq.most_common(top_n)]

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze sentiment of provided text"""
    try:
        # Analyze sentiment
        analysis = analyze_sentiment(request.text)
        
        # Create response
        response = SentimentResponse(
            text=request.text,
            sentiment=analysis["sentiment"],
            confidence=analysis["confidence"],
            polarity_score=analysis["polarity_score"],
            intensity=analysis["intensity"],
            source=request.source,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in history
        analysis_history.append({
            "text": request.text,
            "sentiment": analysis["sentiment"],
            "polarity_score": analysis["polarity_score"],
            "source": request.source,
            "timestamp": response.timestamp
        })
        sentiment_scores.append(analysis["polarity_score"])
        
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-bulk")
async def analyze_bulk(request: BulkAnalysisRequest):
    """Analyze multiple texts at once"""
    try:
        results = []
        for item in request.texts:
            text = item.get("text")
            source = item.get("source", "general")
            
            if not text:
                continue
            
            analysis = analyze_sentiment(text)
            
            result = {
                "text": text,
                "sentiment": analysis["sentiment"],
                "confidence": analysis["confidence"],
                "polarity_score": analysis["polarity_score"],
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Store in history
            analysis_history.append(result)
            sentiment_scores.append(analysis["polarity_score"])
        
        return {
            "status": "success",
            "analyzed_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get overall sentiment statistics"""
    try:
        if not analysis_history:
            return DashboardStats(
                total_analyzed=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                positive_percentage=0.0,
                negative_percentage=0.0,
                neutral_percentage=0.0,
                average_sentiment_score=0.0,
                trending_words=[],
                sentiment_by_source={}
            )
        
        # Count sentiments
        total = len(analysis_history)
        positive_count = sum(1 for item in analysis_history if item["sentiment"] == "positive")
        negative_count = sum(1 for item in analysis_history if item["sentiment"] == "negative")
        neutral_count = sum(1 for item in analysis_history if item["sentiment"] == "neutral")
        
        # Calculate percentages
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        
        # Average sentiment score
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Extract trending words
        all_texts = " ".join([item["text"] for item in analysis_history])
        trending = extract_keywords(all_texts, top_n=10)
        trending_words = [{"word": word, "count": all_texts.lower().count(word)} for word in trending]
        
        # Sentiment by source
        sentiment_by_source = {}
        for item in analysis_history:
            source = item["source"]
            if source not in sentiment_by_source:
                sentiment_by_source[source] = {"positive": 0, "negative": 0, "neutral": 0}
            sentiment_by_source[source][item["sentiment"]] += 1
        
        return DashboardStats(
            total_analyzed=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            positive_percentage=round(positive_pct, 2),
            negative_percentage=round(negative_pct, 2),
            neutral_percentage=round(neutral_pct, 2),
            average_sentiment_score=round(avg_score, 3),
            trending_words=trending_words,
            sentiment_by_source=sentiment_by_source
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
async def get_analysis_history(limit: int = 100):
    """Get recent analysis history"""
    return {
        "total": len(analysis_history),
        "recent": analysis_history[-limit:]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sentiment Analysis Dashboard API",
        "total_analyzed": len(analysis_history)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
