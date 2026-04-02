import { useEffect, useState } from 'react';
import { getHistory, AnalysisHistoryItem } from '@/lib/history';
interface Props {
  onSelect: (id: string) => void;
  refreshFlag?: number; // optional trigger to refresh history
}

export const AnalysisHistory = ({ onSelect, refreshFlag }: Props) => {
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);

  useEffect(() => {
    setHistory(getHistory());
    console.log('Loaded history:', getHistory());
  }, [refreshFlag]); // refresh when refreshFlag changes
  
  if (history.length === 0) return null;

  return (
    <div className="bg-white glass-card dark:bg-zinc-900 rounded-2xl shadow-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Recent Analyses</h3>
      <div className="space-y-3">
        {history.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className="w-full text-left p-3 rounded-xl border bg-gray-50 dark:hover:bg-zinc-800 transition"
          >
            <div className="flex justify-between items-center">
              <span className="truncate text-sm font-medium">{item.url}</span>
              <span
                className={`text-xs px-2 py-1 rounded-full ${
                  item.sentiment === 'positive'
                    ? 'bg-green-100 text-green-700'
                    : item.sentiment === 'negative'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {item.sentiment}
              </span>
            </div>
             {/* Full URL in smaller text below */}
      <div className="text-xs pb-2 text-black font-bold mt-1 break-all">
        Full URL: {item.url}
      </div>

            <div className="text-xs pb-5 font-bold text-gray-500 mt-1">
              Confidence: {item.confidence.toFixed(1)}%
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
