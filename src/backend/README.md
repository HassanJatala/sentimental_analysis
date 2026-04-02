# Python Backend - SentimentIQ

This directory contains the Python FastAPI backend code for the SentimentIQ platform.

## ⚠️ Important Note

This Python backend must be deployed **externally** (Railway, Render, AWS, etc.) as Lovable only supports TypeScript/Deno edge functions.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Set Frontend Environment Variable

Create a `.env` file in the frontend root:

```
VITE_API_URL=http://localhost:8000
```

For production, set this to your deployed backend URL.

## Deployment Options

### Railway
```bash
# Create railway.json with this content:
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
  }
}
```

### Render
Create a `render.yaml`:
```yaml
services:
  - type: web
    name: sentimentiq-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### AWS Lambda / Elastic Beanstalk
See AWS documentation for Python deployment guides.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze a URL |
| `/results/{id}` | GET | Get analysis by ID |

## Project Structure

```
backend/
├── main.py           # FastAPI app entry point
├── requirements.txt  # Python dependencies
├── scraper/
│   ├── __init__.py
│   ├── web.py        # Website scraper
│   ├── youtube.py    # YouTube comments
│   └── twitter.py    # Twitter/X scraper
├── nlp/
│   ├── __init__.py
│   ├── preprocessor.py
│   └── sentiment.py  # Sentiment analysis
└── models/
    └── schemas.py    # Pydantic models
```
