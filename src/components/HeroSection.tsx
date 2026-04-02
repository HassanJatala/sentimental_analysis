import { Sparkles, Zap, Shield, BarChart3 } from 'lucide-react';

export const HeroSection = () => {
  const features = [
    { icon: Zap, text: 'Real-time Analysis' },
    { icon: Shield, text: 'NLP-Powered' },
    { icon: BarChart3, text: 'Visual Insights' },
  ];

  return (
    <div className="text-center py-12 animate-fade-in">
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-6">
        <Sparkles className="w-4 h-4" />
        AI-Powered Sentiment Intelligence
      </div>
      <h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-bold text-foreground mb-4">
        Understand{' '}
        <span className="gradient-text">Public Opinion</span>
        <br />
        in Seconds
      </h1>
      <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
        Extract meaningful insights from any public URL. Analyze websites, YouTube comments,
        and social media with advanced NLP and machine learning.
      </p>
      <div className="flex flex-wrap items-center justify-center gap-6">
        {features.map((feature) => (
          <div
            key={feature.text}
            className="flex items-center gap-2 text-muted-foreground"
          >
            <feature.icon className="w-5 h-5 text-primary" />
            <span>{feature.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
