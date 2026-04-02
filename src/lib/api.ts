// API configuration for connecting to your Python FastAPI backend
// Update this URL to point to your deployed Python backend

import { ClassDictionary } from "clsx";

// Replace this URL with your deployed Python backend URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
// Example: const API_BASE_URL = 'https://your-app.railway.app';
export interface SentimentScore {
  text: string;
  label: 'positive' | 'negative' | 'neutral';
  score: number;
}
export interface SentimentResult {
  id: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  wordFrequencies: { word: string; count: number }[];
  topPhrases: { phrase: string; sentiment: string; score: number }[];
  scores: SentimentScore[];
  summary: string;
  analyzedAt: string;
  sourceType: 'website' | 'youtube' | 'twitter' | 'unknown';
  totalTextsAnalyzed: number;
  
}

export interface AnalyzeRequest {
  url: string;
}

export interface ApiError {
  message: string;
  status: number;
}

class SentimentAPI {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async analyze(url: string): Promise<SentimentResult> {
    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const error: ApiError = {
        message: 'Failed to analyze URL',
        status: response.status,
      };
      throw error;
    }

    return response.json();
  }

  async getResults(id: string): Promise<SentimentResult> {
    const response = await fetch(`${this.baseUrl}/results/${id}`);

    if (!response.ok) {
      const error: ApiError = {
        message: 'Failed to fetch results',
        status: response.status,
      };
      throw error;
    }

    return response.json();
  }
  

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

export const sentimentAPI = new SentimentAPI();
export interface AnalysisHistoryItem {
  id: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  analyzedAt: string;
}

// Demo data for when the backend is not connected
export const generateDemoResult = (url: string): SentimentResult => {
  const sentiments = ['positive', 'negative', 'neutral'] as const;
  const primarySentiment = sentiments[Math.floor(Math.random() * 3)];
  
  const distribution = {
    positive: Math.random() * 60 + 10,
    negative: Math.random() * 40 + 5,
    neutral: Math.random() * 30 + 10,
  };
  
  const total = distribution.positive + distribution.negative + distribution.neutral;
  distribution.positive = Math.round((distribution.positive / total) * 100);
  distribution.negative = Math.round((distribution.negative / total) * 100);
  distribution.neutral = 100 - distribution.positive - distribution.negative;

  const wordFrequencies = [
    { word: 'excellent', count: Math.floor(Math.random() * 50) + 20 },
    { word: 'product', count: Math.floor(Math.random() * 40) + 15 },
    { word: 'quality', count: Math.floor(Math.random() * 35) + 10 },
    { word: 'service', count: Math.floor(Math.random() * 30) + 10 },
    { word: 'recommend', count: Math.floor(Math.random() * 25) + 8 },
    { word: 'experience', count: Math.floor(Math.random() * 20) + 5 },
    { word: 'amazing', count: Math.floor(Math.random() * 18) + 5 },
    { word: 'helpful', count: Math.floor(Math.random() * 15) + 3 },
    { word: 'fast', count: Math.floor(Math.random() * 12) + 3 },
    { word: 'delivery', count: Math.floor(Math.random() * 10) + 2 },
  ].sort((a, b) => b.count - a.count);

  const topPhrases = [
    { phrase: 'great customer service', sentiment: 'positive', score: 0.92 },
    { phrase: 'fast shipping', sentiment: 'positive', score: 0.88 },
    { phrase: 'high quality product', sentiment: 'positive', score: 0.85 },
    { phrase: 'would recommend', sentiment: 'positive', score: 0.82 },
    { phrase: 'needs improvement', sentiment: 'negative', score: 0.65 },
  ];
  const scores: SentimentScore[] = [
  {
    text: 'This product is excellent and works perfectly.',
    label: 'positive',
    score: 0.97,
  },
  {
    text: 'Customer service was helpful and responsive.',
    label: 'positive',
    score: 0.91,
  },
  {
    text: 'Delivery took longer than expected.',
    label: 'negative',
    score: 0.74,
  },
  {
    text: 'The packaging was okay.',
    label: 'neutral',
    score: 0.55,
  },
];


  return {
    id: crypto.randomUUID(),
    url,
    sentiment: primarySentiment,
    confidence: Math.random() * 20 + 75,
    distribution,
    wordFrequencies,
    topPhrases,
    summary: `Analysis of ${url} reveals predominantly ${primarySentiment} sentiment. The content shows strong indicators of customer satisfaction with particular emphasis on product quality and service delivery. Key themes include customer experience, product reliability, and brand perception.`,
    scores,
    analyzedAt: new Date().toISOString(),
    sourceType: url.includes('youtube') ? 'youtube' : url.includes('twitter') || url.includes('x.com') ? 'twitter' : 'website',
    totalTextsAnalyzed: Math.floor(Math.random() * 500) + 50,
  };
};
