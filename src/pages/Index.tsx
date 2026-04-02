import { useState, useEffect } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';
import { Header } from '@/components/Header';
import { HeroSection } from '@/components/HeroSection';
import { URLInput } from '@/components/URLInput';
import { LoadingState } from '@/components/LoadingState';
import { ResultsDashboard } from '@/components/ResultsDashboard';
import { DemoModeNotice } from '@/components/DemoModeNotice';
import { sentimentAPI, generateDemoResult, SentimentResult } from '@/lib/api';
import { saveToHistory } from '@/lib/history';
import { AnalysisHistory } from '@/components/AnalysisHistory';


const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);


  useEffect(() => {
    const checkBackend = async () => {
      const connected = await sentimentAPI.healthCheck();
      setBackendConnected(connected);
      if (!connected) {
        setIsDemoMode(true);
      }
    };
    checkBackend();
  }, []);

  const handleAnalyze = async (url: string) => {
  setIsLoading(true);
  setResult(null);

  try {
    let data: SentimentResult;

    if (backendConnected) {
      data = await sentimentAPI.analyze(url);
    } else {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      data = generateDemoResult(url);
    }

    setResult(data);
    saveToHistory(data);

    // trigger history refresh
    setHistoryRefresh((prev) => prev + 1);

    setIsDemoMode(!backendConnected);
    toast.success('Analysis complete!');
  } catch (error) {
    console.error(error);
    toast.error('Analysis failed. Using demo mode.');
    const demoResult = generateDemoResult(url);
    setResult(demoResult);
    saveToHistory(demoResult);
    setHistoryRefresh((prev) => prev + 1);
    setIsDemoMode(true);
  } finally {
    setIsLoading(false);
  }
};

const handleSelectHistory = async (id: string) => {
  try {
    const data = await sentimentAPI.getResults(id);
    setResult(data);
  } catch {
    toast.error('Could not load previous result');
  }
};

  const handleReset = () => {
    setResult(null);
    setIsDemoMode(false);
  };


  return (
    <div className="min-h-screen bg-background">
      <Toaster position="top-right" />
      
      {/* Background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-chart-accent/5 rounded-full blur-3xl" />
      </div>

      <div className="relative">
        <Header />
        
        <main className="container mx-auto px-4 py-8">
          {!result && !isLoading && (
            <>
              <HeroSection />
              <div className="mt-8">
                <URLInput onAnalyze={handleAnalyze} isLoading={isLoading} />
              </div>
              {backendConnected === false && (
                <div className="mt-8 max-w-2xl mx-auto">
                  <DemoModeNotice isDemo={true} />
                </div>
              )}
            </>
          )}

          {isLoading && <LoadingState />}

          <AnalysisHistory onSelect={handleSelectHistory} refreshFlag={historyRefresh} />
          
          {result && !isLoading && (
            <div className="max-w-6xl mx-auto">
              <DemoModeNotice isDemo={isDemoMode} />
              <div className="mt-6">
                {result && !isLoading && (
  <div className="max-w-6xl mx-auto space-y-6">
    


    <ResultsDashboard result={result} onReset={handleReset} />
  </div>
)}

              </div>
            </div>
          )}
        </main>

        <footer className="border-t border-border/50 py-8 mt-16">
          <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
            <p>Built with React + Python FastAPI + AI-Powered NLP Sentiment Analysis</p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;
