import { Brain, Sparkles, Search, BarChart3 } from 'lucide-react';
import { useEffect, useState } from 'react';

const loadingSteps = [
  { icon: Search, text: 'Fetching content from URL...' },
  { icon: Brain, text: 'Processing with NLP models...' },
  { icon: Sparkles, text: 'Analyzing sentiment patterns...' },
  { icon: BarChart3, text: 'Generating insights...' },
];

export const LoadingState = () => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % loadingSteps.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const CurrentIcon = loadingSteps[currentStep].icon;

  return (
    <div className="flex flex-col items-center justify-center py-20 animate-fade-in">
      <div className="relative">
        <div className="absolute inset-0 bg-primary/20 rounded-full blur-2xl animate-pulse-slow" />
        <div className="relative p-6 bg-card border border-border rounded-2xl">
          <CurrentIcon className="w-12 h-12 text-primary animate-float" />
        </div>
      </div>
      <h3 className="mt-8 text-xl font-display font-semibold text-foreground">
        Analyzing Content
      </h3>
      <p className="mt-2 text-muted-foreground animate-pulse">
        {loadingSteps[currentStep].text}
      </p>
      <div className="flex gap-2 mt-6">
        {loadingSteps.map((_, index) => (
          <div
            key={index}
            className={cn(
              'w-2 h-2 rounded-full transition-all duration-300',
              index === currentStep
                ? 'bg-primary w-6'
                : 'bg-muted-foreground/30'
            )}
          />
        ))}
      </div>
    </div>
  );
};

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}
