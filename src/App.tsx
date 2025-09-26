import { useState } from "react";
import { MissionControlLayout } from "./components/MissionControlLayout";
import { DataStreamPanel } from "./components/DataStreamPanel";
import { SpaceWeatherForecast } from "./components/SpaceWeatherForecast";
import { SatelliteSelector } from "./components/SatelliteSelector";
import { RiskAssessment } from "./components/RiskAssessment";
import { InsurancePricing } from "./components/InsurancePricing";
import { DataVisualizationDashboard } from "./components/DataVisualizationDashboard";
import { Activity, Satellite, Shield, Calculator, BarChart } from "lucide-react";

interface SatelliteData {
  name: string;
  operator: string;
  orbitType: string;
  altitude: number;
  inclination: number;
  mass: number;
  purpose: string;
  launchYear: number;
  shielding: string;
  value: number;
}

export default function App() {
  const [selectedSatellite, setSelectedSatellite] = useState<SatelliteData | undefined>();
  const [activeView, setActiveView] = useState<"dashboard" | "visualization">("dashboard");
  
  // Mock current storm probability - in real app this would come from the forecast
  const currentStormProbability = 67;

  // Calculate expected loss based on satellite and storm data
  const calculateExpectedLoss = (satellite: SatelliteData, stormProb: number) => {
    if (!satellite) return 0;
    
    const getOrbitVulnerability = (orbitType: string, altitude: number) => {
      switch (orbitType) {
        case "LEO": return altitude < 600 ? 0.3 : 0.4;
        case "MEO": return 0.7;
        case "GEO": return 0.9;
        case "HEO": return 0.8;
        default: return 0.5;
      }
    };

    const getShieldingFactor = (shielding: string) => {
      switch (shielding) {
        case "Basic": return 1.0;
        case "Standard": return 0.8;
        case "Enhanced": return 0.6;
        case "Hardened": return 0.4;
        default: return 0.8;
      }
    };

    const getAgeFactor = (launchYear: number) => {
      const age = 2025 - launchYear;
      return Math.min(1.0, 0.7 + (age * 0.05));
    };

    const orbitVulnerability = getOrbitVulnerability(satellite.orbitType, satellite.altitude);
    const shieldingFactor = getShieldingFactor(satellite.shielding);
    const ageFactor = getAgeFactor(satellite.launchYear);
    
    const baseVulnerability = orbitVulnerability * shieldingFactor * ageFactor;
    const stormImpact = baseVulnerability * (stormProb / 100);
    
    return satellite.value * stormImpact * 0.15; // 15% average loss rate
  };

  const expectedLoss = selectedSatellite ? calculateExpectedLoss(selectedSatellite, currentStormProbability) : 0;

  return (
    <MissionControlLayout>
      {/* Command Header */}
      <div className="border-b border-purple-500/20 mission-control-panel p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
              <Satellite className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-mono text-slate-200 tracking-wider">
                COSMIC WEATHER INSURANCE
              </h1>
              <p className="text-sm text-slate-400 font-mono">
                MISSION CONTROL • ORBITAL RISK ASSESSMENT SYSTEM
              </p>
            </div>
          </div>
          
          {/* Navigation */}
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveView("dashboard")}
              className={`px-4 py-2 rounded-lg font-mono text-sm flex items-center gap-2 transition-all ${
                activeView === "dashboard"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <Activity className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={() => setActiveView("visualization")}
              className={`px-4 py-2 rounded-lg font-mono text-sm flex items-center gap-2 transition-all ${
                activeView === "visualization"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <BarChart className="w-4 h-4" />
              Data Visualization
            </button>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-mono text-slate-400">LIVE FEED</span>
            </div>
            <div className="text-sm font-mono text-slate-400">
              {new Date().toISOString().slice(0, 19)}Z
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6 overflow-y-auto">
        {activeView === "dashboard" ? (
          <>
            {/* Space Weather Stream */}
            <DataStreamPanel title="Space Weather Intelligence" status="critical">
              <SpaceWeatherForecast />
            </DataStreamPanel>

            {/* Dual Panel Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Satellite Operations */}
              <div className="space-y-6">
                <DataStreamPanel title="Satellite Database Query" status="normal">
                  <SatelliteSelector 
                    onSatelliteSelect={setSelectedSatellite}
                    selectedSatellite={selectedSatellite}
                  />
                </DataStreamPanel>
                
                <DataStreamPanel title="Risk Assessment Matrix" status={selectedSatellite ? "warning" : "normal"}>
                  <RiskAssessment 
                    satellite={selectedSatellite}
                    stormProbability={currentStormProbability}
                  />
                </DataStreamPanel>
              </div>

              {/* Insurance Processing */}
              <DataStreamPanel title="Insurance Pricing Engine" status={expectedLoss > 0 ? "warning" : "normal"}>
                <InsurancePricing 
                  satellite={selectedSatellite}
                  expectedLoss={expectedLoss}
                  stormProbability={currentStormProbability}
                />
              </DataStreamPanel>
            </div>

            {/* System Status Footer */}
            <div className="grid grid-cols-3 gap-6">
              <DataStreamPanel title="Data Sources" compact>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">NASA OMNIWeb</span>
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">NOAA SWPC</span>
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">UCS Database</span>
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  </div>
                </div>
              </DataStreamPanel>

              <DataStreamPanel title="AI Models" compact>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">LSTM Storm Prediction</span>
                    <span className="text-cyan-400">95.2%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Monte Carlo Sim</span>
                    <span className="text-green-400">Active</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Risk Mapping</span>
                    <span className="text-purple-400">Real-time</span>
                  </div>
                </div>
              </DataStreamPanel>

              <DataStreamPanel title="System Health" compact>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Processing Load</span>
                    <span className="text-green-400">23%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">API Latency</span>
                    <span className="text-cyan-400">24ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Uptime</span>
                    <span className="text-green-400">99.97%</span>
                  </div>
                </div>
              </DataStreamPanel>
            </div>
          </>
        ) : (
          <DataVisualizationDashboard />
        )}
      </div>
    </MissionControlLayout>
  );
}