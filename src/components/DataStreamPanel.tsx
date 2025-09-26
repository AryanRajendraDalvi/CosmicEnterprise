import { Activity, Satellite, AlertTriangle, TrendingUp } from "lucide-react";

interface DataStreamPanelProps {
  title: string;
  children: React.ReactNode;
  status?: "normal" | "warning" | "critical";
  compact?: boolean;
}

export function DataStreamPanel({ title, children, status = "normal", compact = false }: DataStreamPanelProps) {
  const getStatusColor = () => {
    switch (status) {
      case "warning": return "border-yellow-500/40 shadow-yellow-500/20";
      case "critical": return "border-red-500/40 shadow-red-500/20";
      default: return "border-purple-500/30 shadow-purple-500/10";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "warning": return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case "critical": return <AlertTriangle className="w-4 h-4 text-red-400 animate-pulse" />;
      default: return <Activity className="w-4 h-4 text-green-400" />;
    }
  };

  return (
    <div className={`mission-control-panel rounded-lg ${getStatusColor()} hologram-effect ${compact ? 'p-4' : 'p-6'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-mono text-slate-300 tracking-wider uppercase">
          {title}
        </h3>
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
        </div>
      </div>
      
      {children}
      
      {/* Data Flow Animation */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-400/50 to-transparent data-stream"></div>
    </div>
  );
}