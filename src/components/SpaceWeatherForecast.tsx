import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";
import { AlertTriangle, TrendingUp, Loader2 } from "lucide-react";
import { getForecast, ForecastData } from "../services/api";

interface ForecastPoint {
  time: string;
  kpIndex: number;
  stormProb: number;
  severity: string;
}

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "Severe": return "destructive";
    case "Moderate": return "default";
    case "Minor": return "secondary";
    default: return "secondary";
  }
};

const getSeverityIcon = (severity: string) => {
  if (severity === "Severe") return <AlertTriangle className="w-4 h-4" />;
  if (severity === "Moderate") return <TrendingUp className="w-4 h-4" />;
  return null;
};

const getSeverityLevel = (kpIndex: number): string => {
  if (kpIndex >= 7) return "Severe";
  if (kpIndex >= 5) return "Moderate";
  return "Minor";
};

export function SpaceWeatherForecast() {
  const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch forecast data on component mount
  useEffect(() => {
    const fetchForecast = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getForecast([24, 48, 72]);
        
        // Transform API data to chart format
        const points: ForecastPoint[] = [];
        const now = new Date();
        
        // Add current point
        points.push({
          time: "00:00",
          kpIndex: 2.3,
          stormProb: 5,
          severity: "Minor"
        });
        
        // Add forecast points
        Object.entries(data).forEach(([key, forecast]: [string, ForecastData]) => {
          const hours = parseInt(key.replace('h', ''));
          const timeLabel = `${hours}:00`;
          
          points.push({
            time: timeLabel,
            kpIndex: forecast.kp_predicted,
            stormProb: Math.round(forecast.storm_probabilities.severe_storm_kp7 * 100),
            severity: getSeverityLevel(forecast.kp_predicted)
          });
        });
        
        setForecastData(points);
      } catch (err) {
        setError("Failed to load forecast data");
        console.error("Error fetching forecast:", err);
        
        // Fallback to mock data
        const mockData: ForecastPoint[] = [
          { time: "00:00", kpIndex: 2.3, stormProb: 5, severity: "Minor" },
          { time: "06:00", kpIndex: 3.1, stormProb: 12, severity: "Minor" },
          { time: "12:00", kpIndex: 4.2, stormProb: 23, severity: "Moderate" },
          { time: "18:00", kpIndex: 5.8, stormProb: 45, severity: "Moderate" },
          { time: "24:00", kpIndex: 6.9, stormProb: 67, severity: "Severe" },
          { time: "30:00", kpIndex: 7.2, stormProb: 73, severity: "Severe" },
          { time: "36:00", kpIndex: 6.1, stormProb: 52, severity: "Moderate" },
          { time: "42:00", kpIndex: 4.8, stormProb: 34, severity: "Moderate" },
          { time: "48:00", kpIndex: 3.5, stormProb: 18, severity: "Minor" },
          { time: "54:00", kpIndex: 2.8, stormProb: 9, severity: "Minor" },
          { time: "60:00", kpIndex: 2.1, stormProb: 4, severity: "Minor" },
          { time: "66:00", kpIndex: 1.9, stormProb: 3, severity: "Minor" },
          { time: "72:00", kpIndex: 2.4, stormProb: 7, severity: "Minor" },
        ];
        
        setForecastData(mockData);
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-8">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500 mb-4" />
        <p className="text-slate-400 text-lg">Loading space weather forecast...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-400 text-lg mb-2">Error loading forecast data</p>
        <p className="text-slate-400">{error}</p>
      </div>
    );
  }

  const currentStorm = forecastData.find(d => d.time === "24:00") || forecastData[4];
  const maxStormProb = Math.max(...forecastData.map(d => d.stormProb));
  const maxKpIndex = Math.max(...forecastData.map(d => d.kpIndex));

  return (
    <div className="space-y-4">
      {/* Current Alert - More compact for mission control */}
      <div className="control-surface p-4 rounded-lg border-red-500/40">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 animate-pulse" />
            <span className="text-base font-mono text-slate-300">STORM ALERT</span>
          </div>
          <div className="text-xs font-mono text-red-400 bg-red-500/10 px-2 py-1 rounded">
            {currentStorm?.severity} RISK
          </div>
        </div>
        <div className="text-base text-slate-400">
          Probability: <span className="text-red-400 font-mono">{currentStorm?.stormProb}%</span> | 
          Kp Index: <span className="text-cyan-400 font-mono">{currentStorm?.kpIndex}</span>
        </div>
      </div>

      {/* 72-Hour Forecast - Streamlined */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-base font-mono text-slate-300">72-HOUR FORECAST</span>
          <div className="text-sm font-mono text-slate-400">
            MAX: <span className="text-red-400">{maxStormProb}%</span> | 
            PEAK Kp: <span className="text-cyan-400">{maxKpIndex}</span>
          </div>
        </div>
        <div className="h-48">
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Hours from now', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  yAxisId="left"
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Storm Probability (%)', angle: -90, position: 'insideLeft' }}
                />
                <YAxis 
                  yAxisId="right" 
                  orientation="right"
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Kp Index', angle: 90, position: 'insideRight' }}
                />
                <Tooltip 
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="control-surface p-3 rounded shadow-lg">
                          <p className="text-sm font-mono text-slate-300">T+{label}h</p>
                          <p className="text-sm text-red-400">PROB: {data.stormProb}%</p>
                          <p className="text-sm text-cyan-400">Kp: {data.kpIndex}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="stormProb"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="kpIndex"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Key Metrics - Compact Mission Control Style */}
      <div className="grid grid-cols-3 gap-3">
        <div className="control-surface p-3 rounded text-center">
          <div className="text-xl font-mono text-red-400">{currentStorm?.stormProb}%</div>
          <div className="text-sm text-slate-500">STORM RISK</div>
        </div>
        <div className="control-surface p-3 rounded text-center">
          <div className="text-xl font-mono text-cyan-400">{currentStorm?.kpIndex}</div>
          <div className="text-sm text-slate-500">KP INDEX</div>
        </div>
        <div className="control-surface p-3 rounded text-center">
          <div className="text-xl font-mono text-purple-400">12h</div>
          <div className="text-sm text-slate-500">PEAK ETA</div>
        </div>
      </div>
    </div>
  );
}