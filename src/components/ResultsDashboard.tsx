
import { SentimentResult } from '@/lib/api';
import { SentimentCard } from './SentimentCard';
import { SentimentChart } from './SentimentChart';
import { WordCloud } from './WordCloud';
import { AnalysisSummary } from './AnalysisSummary';
import { TopPhrases } from './TopPhrases';
import { ExternalLink, RotateCcw } from 'lucide-react';
import { Button } from './ui/button';
import { useState } from 'react';

interface ResultsDashboardProps {
  result: SentimentResult;
  onReset: () => void;
}

export const ResultsDashboard = ({ result, onReset }: ResultsDashboardProps) => {
  const [filter, setFilter] = useState<'all' | 'positive' | 'negative' | 'neutral'>('all');

  // Filter content based on sentiment
  const filteredContent = result.scores?.filter((item) => {
    if (filter === 'all') return true;
    return item.label === filter;
  }) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-display font-bold text-foreground">
            Analysis Results
          </h2>
          <a
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors mt-1"
          >
            <span className="truncate max-w-md">{result.url}</span>
            <ExternalLink className="w-4 h-4 flex-shrink-0" />
          </a>
        </div>
        <Button variant="outline" onClick={onReset} className="gap-2">
          <RotateCcw className="w-4 h-4" />
          New Analysis
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SentimentCard result={result} />
        <SentimentChart result={result} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <WordCloud result={result} />
        <TopPhrases result={result} />
      </div>

      {/* ===== Content Details Section ===== */}
      <div className=" glass-card rounded-2xl shadow-lg p-6">
        <div className="flex  items-center justify-between mb-4">
          <h3 className="text-xl text-white font-bold ">Content Details</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg transition ${
                filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('positive')}
              className={`px-4 py-2 rounded-lg transition ${
                filter === 'positive' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Positive
            </button>
            <button
              onClick={() => setFilter('negative')}
              className={`px-4 py-2 rounded-lg transition ${
                filter === 'negative' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Negative
            </button>
            <button
              onClick={() => setFilter('neutral')}
              className={`px-4 py-2 rounded-lg transition ${
                filter === 'neutral' ? 'bg-gray-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Neutral
            </button>
          </div>
        </div>

        <div className="space-y-4 max-h-96 hover:cursor-pointer px-10 overflow-y-auto  overflow-x-hidden ">
          {filteredContent.slice(0, 50).map((item, index) => (
            <div
              key={index}
              className="group border  border-gray-200 rounded-lg p-4 bg-gray-300 transition transform transition-transform duration-300 ease-in-out hover:scale-105"
            >
              <div className="flex items-start justify-between mb-2">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    item.label === 'positive'
                      ? 'bg-green-100 text-green-700'
                      : item.label === 'negative'
                      ? 'bg-red-100 text-red-700'
                      : 'bg-gray-100 text-gray-700'
                  }`}
                >
                  {item.label}
                </span>
                {item.text && item.text.slice(0, 20) && (
                  <span className="text-sm text-gray-500 truncate">
                    {item.text.slice(0, 20)}...
                  </span>
                )}
              </div>
              <p className="text-gray-700 ">{item.text}</p>
            </div>
          ))}
        </div>
      </div>
      {/* ===== End Content Details Section ===== */}

      <AnalysisSummary result={result} />
    </div>
  );
};
