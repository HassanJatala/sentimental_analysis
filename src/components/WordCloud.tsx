import { SentimentResult } from '@/lib/api';

interface WordCloudProps {
  result: SentimentResult;
}

export const WordCloud = ({ result }: WordCloudProps) => {
  const maxCount = Math.max(...result.wordFrequencies.map((w) => w.count));

  const getSize = (count: number) => {
    const ratio = count / maxCount;
    if (ratio > 0.8) return 'text-3xl font-bold';
    if (ratio > 0.6) return 'text-2xl font-semibold';
    if (ratio > 0.4) return 'text-xl font-medium';
    if (ratio > 0.2) return 'text-lg';
    return 'text-base';
  };

  const getColor = (index: number) => {
    const colors = [
      'text-primary',
      'text-positive',
      'text-chart-accent',
      'text-foreground',
      'text-muted-foreground',
    ];
    return colors[index % colors.length];
  };

  return (
    <div className="glass-card rounded-2xl p-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
      <h3 className="text-lg font-display font-semibold text-foreground mb-4">
        Key Words
      </h3>
      <div className="flex flex-wrap items-center justify-center gap-4 min-h-[200px]">
        {result.wordFrequencies.slice(0, 15).map((word, index) => (
          <span
            key={word.word}
            className={`${getSize(word.count)} ${getColor(index)} transition-all duration-300 hover:scale-110 cursor-default`}
            title={`${word.word}: ${word.count} occurrences`}
          >
            {word.word}
          </span>
        ))}
      </div>
    </div>
  );
};
