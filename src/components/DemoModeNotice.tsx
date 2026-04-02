import { AlertCircle, ExternalLink } from 'lucide-react';

interface DemoModeNoticeProps {
  isDemo: boolean;
}

export const DemoModeNotice = ({ isDemo }: DemoModeNoticeProps) => {
  if (!isDemo) return null;

  return (
    <div className="bg-neutral/10 border border-neutral/30 rounded-xl p-4 flex items-start gap-3 animate-fade-in">
      <AlertCircle className="w-5 h-5 text-neutral flex-shrink-0 mt-0.5" />
      <div>
        <p className="text-sm font-medium text-neutral">Demo Mode Active</p>
        <p className="text-sm text-muted-foreground mt-1">
          The Python backend is not connected. Showing simulated results for demonstration.
          Deploy your FastAPI backend and set the <code className="text-xs bg-secondary px-1.5 py-0.5 rounded">VITE_API_URL</code> environment variable to connect.
        </p>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline mt-2"
        >
          View Backend Setup Guide
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
};
