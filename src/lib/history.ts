import { SentimentResult } from '@/lib/api';

const HISTORY_KEY = 'sentiment_history';

export interface AnalysisHistoryItem {
  id: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  analyzedAt: string;
}

export const getHistory = (): AnalysisHistoryItem[] => {
  const raw = localStorage.getItem(HISTORY_KEY);
  return raw ? JSON.parse(raw) : [];
};

export const saveToHistory = (result: SentimentResult) => {
  const history = getHistory();

  const newItem: AnalysisHistoryItem = {
    id: result.id,
    url: result.url,
    sentiment: result.sentiment,
    confidence: result.confidence,
    analyzedAt: result.analyzedAt,
  };

  const filtered = history.filter((h) => h.url !== result.url);

  localStorage.setItem(
    HISTORY_KEY,
    JSON.stringify([newItem, ...filtered].slice(0, 10))
  );
};
