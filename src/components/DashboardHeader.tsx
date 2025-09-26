import { useState } from "react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Bell, Settings, Satellite, Activity } from "lucide-react";
import { ImageWithFallback } from "./figma/ImageWithFallback";

export function DashboardHeader() {
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [realTimeMode, setRealTimeMode] = useState(false);

  return (
    <div className="border-b glass-morphism relative overflow-hidden">
      <div className="absolute inset-0 aurora-effect opacity-30"></div>
      <div className="relative z-10 flex items-center justify-between p-6">
        <div className="flex items-center gap-4">
          <div className="relative floating">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-500 p-0.5 cosmic-glow">
              <div className="w-full h-full rounded-xl bg-slate-900 flex items-center justify-center">
                <Satellite className="w-6 h-6 text-purple-400" />
              </div>
            </div>
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-green-400 to-cyan-400 rounded-full border-2 border-slate-900 animate-pulse cosmic-glow"></div>
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-cyan-400 to-green-400 bg-clip-text text-transparent">
              Cosmic Weather Insurance
            </h1>
            <p className="text-slate-300">Space Weather Risk Assessment & Insurance Pricing Platform</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-green-400 animate-pulse" />
            <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/30 cosmic-glow">
              Live Data Feed
            </Badge>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="real-time"
              checked={realTimeMode}
              onCheckedChange={setRealTimeMode}
            />
            <Label htmlFor="real-time" className="text-sm text-slate-300">Real-time</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="alerts"
              checked={alertsEnabled}
              onCheckedChange={setAlertsEnabled}
            />
            <Label htmlFor="alerts" className="text-sm text-slate-300">Alerts</Label>
            <Bell className={`w-4 h-4 ${alertsEnabled ? 'text-cyan-400 cosmic-glow' : 'text-slate-500'}`} />
          </div>

          <Button variant="outline" size="sm" className="glass-morphism border-purple-500/30 text-purple-300 hover:bg-purple-500/10">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-slate-900/50 px-6 py-3 flex items-center justify-between text-sm border-t border-purple-500/20">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse cosmic-glow"></div>
            <span className="text-slate-300">NOAA SWPC Connected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse cosmic-glow"></div>
            <span className="text-slate-300">NASA OMNIWeb Active</span>
          </div>
          <div className="flex items-center gap-2">
            <Satellite className="w-4 h-4 text-purple-400" />
            <span className="text-slate-300">5,247 satellites tracked</span>
          </div>
        </div>
        <div className="text-slate-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}