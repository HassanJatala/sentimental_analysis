import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { SentimentResult } from '@/lib/api';

interface TopWordsChartProps {
  result: SentimentResult;
  topN?: number;
}

export const TopPhrases = ({ result, topN = 10 }: TopWordsChartProps) => {
  // Prepare chart data: only valid words
  const chartData = result.wordFrequencies
    .filter((wf) => wf.word && wf.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, topN)
    .map((wf) => ({
      word: wf.word,
      count: wf.count,
    }));

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 animate-fade-in" style={{ animationDelay: '0.4s' }}>
      <h3 className="text-xl font-bold text-gray-800 mb-4">Top Contributors</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 20, left: 20, bottom: 50 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="word"
            tick={{ fontSize: 14, angle: -30, textAnchor: 'end' }}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number, name: string, props) => {
              const word = (props?.payload as any)?.word;
              return [`Count: ${value}`, `Word: ${word}`];
            }}
          />
          <Bar dataKey="count" fill="#3b82f6" barSize={30} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
