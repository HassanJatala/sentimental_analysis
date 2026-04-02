import { Brain, Github, ExternalLink } from 'lucide-react';
import { Button } from './ui/button';

export const Header = () => {
  return (
    <header className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="font-display font-bold text-lg text-foreground">
              SentimentIQ
            </h1>
            <p className="text-xs text-muted-foreground hidden sm:block">
              AI-Powered Sentiment Analysis
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          
          <Button variant="outline" size="sm" className="gap-2" onClick={() => window.open('https://github.com/HassanJatala', '_blank')}>
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </Button>
        </div>
      </div>
    </header>
  );
};
