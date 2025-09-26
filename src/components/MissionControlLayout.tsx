import { ReactNode } from "react";
import { EarthScene } from "./EarthScene";

interface MissionControlLayoutProps {
  children: ReactNode;
}

export function MissionControlLayout({ children }: MissionControlLayoutProps) {
  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
      {/* Background Grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(139, 92, 246, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139, 92, 246, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}></div>
      </div>
      
      {/* Main Command Center Layout */}
      <div className="relative z-10 min-h-screen flex">
        {/* Left Panel - Earth Scene & Primary Controls */}
        <div className="shrink-0 border-r border-purple-500/20 mission-control-panel" style={{ width: "560px" }}>
          <div className="p-8 space-y-8">
            {/* Earth Scene */}
            <div className="text-center">
              <h2 className="text-xl font-medium mb-6 text-purple-300 tracking-wider">
                ORBITAL SURVEILLANCE
              </h2>
              <EarthScene />
              
              {/* Status Indicators below the globe */}
              <div className="mt-8 space-y-4">
                <div className="grid grid-cols-4 gap-3">
                  <div className="control-surface p-3 rounded text-center">
                    <div className="text-sm text-slate-400">ASTEROIDS</div>
                    <div className="text-lg font-mono text-orange-400">12%</div>
                  </div>
                  <div className="control-surface p-3 rounded text-center">
                    <div className="text-sm text-slate-400">SOLAR FLARES</div>
                    <div className="text-lg font-mono text-pink-400">28%</div>
                  </div>
                  <div className="control-surface p-3 rounded text-center">
                    <div className="text-sm text-slate-400">SOLAR WINDS</div>
                    <div className="text-lg font-mono text-green-400">34%</div>
                  </div>
                  <div className="control-surface p-3 rounded text-center">
                    <div className="text-sm text-slate-400">GEOMAGNETIC</div>
                    <div className="text-lg font-mono text-cyan-400">19%</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="control-surface p-4 rounded-lg">
                    <div className="text-sm text-slate-400 mb-1">ACTIVE SATELLITES</div>
                    <div className="text-xl font-mono text-cyan-400">5,247</div>
                  </div>
                  <div className="control-surface p-4 rounded-lg">
                    <div className="text-sm text-slate-400 mb-1">STORM RISK</div>
                    <div className="text-xl font-mono text-red-400">HIGH</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Right Panel - Data Streams & Controls */}
        <div className="flex-1 flex flex-col">
          {children}
        </div>
      </div>
      
      {/* Ambient particles removed for a cleaner, more professional presentation */}
    </div>
  );
}