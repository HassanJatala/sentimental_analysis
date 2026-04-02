import { FileText, Clock, Link, BarChart3 } from 'lucide-react';
import { SentimentResult } from '@/lib/api';
import { format } from 'date-fns';

interface AnalysisSummaryProps {
  result: SentimentResult;
}

export const AnalysisSummary = ({ result }: AnalysisSummaryProps) => {
  const sourceTypeLabels = {
    website: 'Website',
    youtube: 'YouTube',
    twitter: 'Twitter/X',
    unknown: 'Unknown Source',
  };

  return (
    <div className="glass-card rounded-2xl p-6 animate-fade-in" style={{ animationDelay: '0.3s' }}>
      <h3 className="text-lg font-display font-semibold text-foreground mb-4">
        Analysis Summary
      </h3>
      <div className="space-y-4">
        <p className="text-muted-foreground leading-relaxed">
          {result.summary}
        </p>
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-secondary">
              <Link className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Source Type</p>
              <p className="text-sm font-medium text-foreground">
                {sourceTypeLabels[result.sourceType]}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-secondary">
              <BarChart3 className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Texts Analyzed</p>
              <p className="text-sm font-medium text-foreground">
                {result.totalTextsAnalyzed.toLocaleString()}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-secondary">
              <Clock className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Analyzed At</p>
              <p className="text-sm font-medium text-foreground">
                {format(new Date(result.analyzedAt), 'MMM d, yyyy HH:mm')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-secondary">
              <FileText className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Analysis ID</p>
              <p className="text-sm font-medium text-foreground font-mono">
                {result.id.slice(0, 8)}...
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
