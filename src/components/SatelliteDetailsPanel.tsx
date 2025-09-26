import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Satellite, Rocket, Globe, Calendar, Zap, Shield, DollarSign, Activity } from "lucide-react";

interface SatelliteData {
  name: string;
  operator: string;
  satellite_type: string;
  orbit_type: string;
  perigee_km: number;
  apogee_km: number;
  inclination_deg: number;
  mass_kg: number;
  value_usd: number;
  shielding_factor: number;
  launch_year: number;
  expected_lifetime_years: number;
  country: string;
  purpose: string;
  power_watts: number;
  status: string;
}

interface SatelliteDetailsPanelProps {
  satellite: SatelliteData;
}

export function SatelliteDetailsPanel({ satellite }: SatelliteDetailsPanelProps) {
  if (!satellite) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <Satellite className="w-5 h-5 text-purple-400" />
            Satellite Details
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-slate-400 text-center py-8">
            Select a satellite to view detailed information
          </p>
        </CardContent>
      </Card>
    );
  }

  // Calculate derived metrics
  const age = 2025 - satellite.launch_year;
  const isHighValue = satellite.value_usd > 100000000; // > $100M
  const isHeavy = satellite.mass_kg > 1000; // > 1000kg
  const orbitAltitude = (satellite.perigee_km + satellite.apogee_km) / 2;

  // Get shielding level description
  const getShieldingLevel = (factor: number) => {
    if (factor > 0.8) return "Basic";
    if (factor > 0.6) return "Standard";
    if (factor > 0.4) return "Enhanced";
    return "Hardened";
  };

  // Get orbit description
  const getOrbitDescription = (orbitType: string) => {
    switch (orbitType) {
      case "LEO": return "Low Earth Orbit";
      case "MEO": return "Medium Earth Orbit";
      case "GEO": return "Geostationary Orbit";
      case "HEO": return "Highly Elliptical Orbit";
      case "SSO": return "Sun-Synchronous Orbit";
      default: return orbitType;
    }
  };

  return (
    <Card className="space-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2 text-slate-200">
          <div className="flex items-center gap-2">
            <Satellite className="w-5 h-5 text-purple-400" />
            Satellite Details
          </div>
          <Badge 
            variant={isHighValue ? "destructive" : "secondary"}
            className="text-xs"
          >
            {isHighValue ? "High Value Asset" : "Standard Asset"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Satellite Name and Operator */}
        <div className="border-b border-purple-500/20 pb-4">
          <h3 className="text-xl font-bold text-slate-200 mb-1">{satellite.name}</h3>
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Operator:</span>
            <span className="text-cyan-400 font-medium">{satellite.operator}</span>
            <span className="text-slate-600">•</span>
            <span className="text-slate-400">Country:</span>
            <span className="text-green-400">{satellite.country}</span>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="control-surface p-3 rounded-lg text-center">
            <DollarSign className="w-5 h-5 text-green-400 mx-auto mb-1" />
            <p className="text-lg font-bold text-green-400">
              ${(satellite.value_usd / 1000000).toFixed(1)}M
            </p>
            <p className="text-xs text-slate-400">Value</p>
          </div>
          
          <div className="control-surface p-3 rounded-lg text-center">
            <Zap className="w-5 h-5 text-yellow-400 mx-auto mb-1" />
            <p className="text-lg font-bold text-yellow-400">
              {satellite.mass_kg.toLocaleString()} kg
            </p>
            <p className="text-xs text-slate-400">Mass</p>
          </div>
          
          <div className="control-surface p-3 rounded-lg text-center">
            <Globe className="w-5 h-5 text-blue-400 mx-auto mb-1" />
            <p className="text-lg font-bold text-blue-400">
              {orbitAltitude.toLocaleString()} km
            </p>
            <p className="text-xs text-slate-400">Avg. Altitude</p>
          </div>
          
          <div className="control-surface p-3 rounded-lg text-center">
            <Calendar className="w-5 h-5 text-purple-400 mx-auto mb-1" />
            <p className="text-lg font-bold text-purple-400">{age} years</p>
            <p className="text-xs text-slate-400">Age</p>
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="border border-purple-500/20 rounded-lg p-4">
          <h4 className="font-medium text-slate-200 mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Technical Specifications
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex justify-between">
              <span className="text-slate-400">Type:</span>
              <span className="text-slate-200">{satellite.satellite_type}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Orbit:</span>
              <span className="text-slate-200">{getOrbitDescription(satellite.orbit_type)}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Perigee:</span>
              <span className="text-slate-200">{satellite.perigee_km.toLocaleString()} km</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Apogee:</span>
              <span className="text-slate-200">{satellite.apogee_km.toLocaleString()} km</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Inclination:</span>
              <span className="text-slate-200">{satellite.inclination_deg}°</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Power:</span>
              <span className="text-slate-200">{satellite.power_watts.toLocaleString()} W</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Shielding:</span>
              <span className="text-slate-200">{getShieldingLevel(satellite.shielding_factor)}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Lifetime:</span>
              <span className="text-slate-200">{satellite.expected_lifetime_years} years</span>
            </div>
          </div>
        </div>

        {/* Mission Information */}
        <div className="border border-purple-500/20 rounded-lg p-4">
          <h4 className="font-medium text-slate-200 mb-3 flex items-center gap-2">
            <Rocket className="w-4 h-4 text-orange-400" />
            Mission Information
          </h4>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-400">Purpose:</span>
              <span className="text-slate-200">{satellite.purpose}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Launch Year:</span>
              <span className="text-slate-200">{satellite.launch_year}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-slate-400">Status:</span>
              <Badge 
                variant={satellite.status === "Operational" ? "default" : "secondary"}
                className="text-xs"
              >
                {satellite.status}
              </Badge>
            </div>
          </div>
        </div>

        {/* Risk Factors */}
        <div className="border border-red-500/20 rounded-lg p-4 bg-red-500/5">
          <h4 className="font-medium text-slate-200 mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4 text-red-400" />
            Risk Assessment Factors
          </h4>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Orbital Vulnerability</span>
                <span className="text-slate-200">
                  {satellite.orbit_type} Orbit
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div 
                  className="bg-red-500 h-2 rounded-full" 
                  style={{ 
                    width: `${satellite.orbit_type === "LEO" ? 40 : 
                            satellite.orbit_type === "MEO" ? 70 : 
                            satellite.orbit_type === "GEO" ? 90 : 
                            satellite.orbit_type === "HEO" ? 80 : 50}%` 
                  }}
                ></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Shielding Effectiveness</span>
                <span className="text-slate-200">
                  {getShieldingLevel(satellite.shielding_factor)}
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div 
                  className="bg-cyan-500 h-2 rounded-full" 
                  style={{ 
                    width: `${(1 - satellite.shielding_factor) * 100}%` 
                  }}
                ></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Age Factor</span>
                <span className="text-slate-200">
                  {age} years
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full" 
                  style={{ 
                    width: `${Math.min(100, age * 5)}%` 
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}