import { ThumbsUp, ThumbsDown, Minus, TrendingUp } from 'lucide-react';
import { SentimentResult } from '@/lib/api';
import { cn } from '@/lib/utils';

interface SentimentCardProps {
  result: SentimentResult;
}

export const SentimentCard = ({ result }: SentimentCardProps) => {
  const sentimentConfig = {
    positive: {
      icon: ThumbsUp,
      label: 'Positive',
      color: 'text-positive',
      bg: 'bg-positive/10',
      border: 'border-positive/30',
    },
    negative: {
      icon: ThumbsDown,
      label: 'Negative',
      color: 'text-negative',
      bg: 'bg-negative/10',
      border: 'border-negative/30',
    },
    neutral: {
      icon: Minus,
      label: 'Neutral',
      color: 'text-neutral',
      bg: 'bg-neutral/10',
      border: 'border-neutral/30',
    },
  };

  const config = sentimentConfig[result.sentiment];
  const Icon = config.icon;

  return (
    <div className="glass-card rounded-2xl p-6 animate-fade-in">
      <h3 className="text-lg font-display font-semibold text-foreground mb-4">
        Overall Sentiment
      </h3>
      <div className="flex items-center gap-4">
        <div className={cn('p-4 rounded-xl', config.bg, 'border', config.border)}>
          <Icon className={cn('w-8 h-8', config.color)} />
        </div>
        <div>
          <p className={cn('text-2xl font-display font-bold', config.color)}>
            {config.label}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground text-sm">
              {result.confidence.toFixed(1)}% confidence
            </span>
          </div>
        </div>
      </div>
      <div className="mt-6 grid grid-cols-3 gap-3">
        {Object.entries(result.distribution).map(([key, value]) => (
          <div
            key={key}
            className={cn(
              'p-3 rounded-lg text-center',
              key === 'positive' && 'bg-positive/10',
              key === 'negative' && 'bg-negative/10',
              key === 'neutral' && 'bg-neutral/10'
            )}
          >
            <p className={cn(
              'text-xl font-bold',
              key === 'positive' && 'text-positive',
              key === 'negative' && 'text-negative',
              key === 'neutral' && 'text-neutral'
            )}>
              {value}%
            </p>
            <p className="text-xs text-muted-foreground capitalize">{key}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
