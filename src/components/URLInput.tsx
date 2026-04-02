import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search, Sparkles, Globe, Youtube, Twitter } from 'lucide-react';

interface URLInputProps {
  onAnalyze: (url: string) => void;
  isLoading: boolean;
}

export const URLInput = ({ onAnalyze, isLoading }: URLInputProps) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onAnalyze(url.trim());
    }
  };

  const getSourceIcon = () => {
    if (url.includes('youtube.com') || url.includes('youtu.be')) {
      return <Youtube className="w-5 h-5 text-negative" />;
    }
    if (url.includes('twitter.com') || url.includes('x.com')) {
      return <Twitter className="w-5 h-5 text-primary" />;
    }
    return <Globe className="w-5 h-5 text-muted-foreground" />;
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="relative group">
        <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 via-chart-accent/20 to-primary/20 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="relative flex items-center gap-3 p-2 bg-card border border-border rounded-xl">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-secondary/50">
            {getSourceIcon()}
          </div>
          <Input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste any URL to analyze sentiment..."
            className="flex-1 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 h-12 text-lg"
            required
          />
          <Button
            type="submit"
            size="lg"
            variant="glow"
            disabled={isLoading || !url.trim()}
            className="px-6"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                <span>Analyze</span>
              </>
            )}
          </Button>
        </div>
      </div>
      <div className="flex items-center justify-center gap-6 mt-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <Globe className="w-4 h-4" /> Websites
        </span>
        <span className="flex items-center gap-2">
          <Youtube className="w-4 h-4" /> YouTube
        </span>
        <span className="flex items-center gap-2">
          <Twitter className="w-4 h-4" /> Twitter/X
        </span>
      </div>
    </form>
  );
};
