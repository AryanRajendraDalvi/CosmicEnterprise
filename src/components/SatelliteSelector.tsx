import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Search, Satellite, Plus, Info } from "lucide-react";
import { getSatellites, Satellite as SatelliteType } from "../services/api";
import { SatelliteDetailsPanel } from "./SatelliteDetailsPanel";

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

interface SatelliteDisplay {
  id: string;
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

interface SatelliteSelectorProps {
  onSatelliteSelect: (satellite: SatelliteData) => void;
  selectedSatellite?: SatelliteData;
}

export function SatelliteSelector({ onSatelliteSelect, selectedSatellite }: SatelliteSelectorProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [satellites, setSatellites] = useState<SatelliteDisplay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [customSatellite, setCustomSatellite] = useState<Partial<SatelliteData>>({
    name: "",
    operator: "",
    orbitType: "LEO",
    altitude: 550,
    inclination: 53,
    mass: 1000,
    purpose: "Communications",
    launchYear: 2023,
    shielding: "Standard",
    value: 50000000
  });

  // Fetch satellites from API
  useEffect(() => {
    const fetchSatellites = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getSatellites({ query: searchTerm });
        
        // Convert API satellite data to display format
        const displaySatellites = data.satellites.slice(0, 20).map((sat: SatelliteType) => ({
          id: `${sat.name}-${sat.operator}`,
          name: sat.name,
          operator: sat.operator,
          orbitType: sat.orbit_type,
          altitude: sat.perigee_km, // Using perigee as altitude for display
          inclination: sat.inclination_deg,
          mass: sat.mass_kg,
          purpose: sat.purpose,
          launchYear: sat.launch_year,
          shielding: sat.shielding_factor > 0.8 ? "Basic" : 
                    sat.shielding_factor > 0.6 ? "Standard" : 
                    sat.shielding_factor > 0.4 ? "Enhanced" : "Hardened",
          value: sat.value_usd
        }));
        
        setSatellites(displaySatellites);
      } catch (err) {
        setError("Failed to load satellites");
        console.error("Error fetching satellites:", err);
        
        // Fallback to mock data if API fails
        const mockSatellites: SatelliteDisplay[] = [
          {
            id: "starlink-1234",
            name: "Starlink 1234",
            operator: "SpaceX",
            orbitType: "LEO",
            altitude: 550,
            inclination: 53.0,
            mass: 260,
            purpose: "Communications",
            launchYear: 2022,
            shielding: "Standard",
            value: 500000
          },
          {
            id: "gps-iif-12",
            name: "GPS IIF-12",
            operator: "US Air Force",
            orbitType: "MEO",
            altitude: 20200,
            inclination: 55.0,
            mass: 1630,
            purpose: "Navigation",
            launchYear: 2016,
            shielding: "Hardened",
            value: 450000000
          },
          {
            id: "goes-18",
            name: "GOES-18",
            operator: "NOAA",
            orbitType: "GEO",
            altitude: 35786,
            inclination: 0.1,
            mass: 5192,
            purpose: "Weather",
            launchYear: 2022,
            shielding: "Enhanced",
            value: 1200000000
          },
          {
            id: "hubble",
            name: "Hubble Space Telescope",
            operator: "NASA",
            orbitType: "LEO",
            altitude: 535,
            inclination: 28.5,
            mass: 11110,
            purpose: "Observatory",
            launchYear: 1990,
            shielding: "Enhanced",
            value: 10000000000
          },
          {
            id: "sentinel-2a",
            name: "Sentinel-2A",
            operator: "ESA",
            orbitType: "LEO",
            altitude: 786,
            inclination: 98.6,
            mass: 1140,
            purpose: "Earth Observation",
            launchYear: 2015,
            shielding: "Standard",
            value: 280000000
          }
        ];
        setSatellites(mockSatellites);
      } finally {
        setLoading(false);
      }
    };

    fetchSatellites();
  }, [searchTerm]);

  const filteredSatellites = satellites.filter(sat =>
    sat.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    sat.operator.toLowerCase().includes(searchTerm.toLowerCase()) ||
    sat.purpose.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getOrbitBadgeColor = (orbit: string) => {
    switch (orbit) {
      case "LEO": return "default";
      case "MEO": return "secondary";
      case "GEO": return "outline";
      default: return "outline";
    }
  };

  const handleCustomSubmit = () => {
    if (customSatellite.name && customSatellite.operator) {
      onSatelliteSelect(customSatellite as SatelliteData);
    }
  };

  return (
    <Tabs defaultValue="search" className="w-full">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="search" className="flex items-center gap-2">
          <Search className="w-4 h-4" />
          Search
        </TabsTrigger>
        <TabsTrigger value="details" className="flex items-center gap-2">
          <Info className="w-4 h-4" />
          Details
        </TabsTrigger>
      </TabsList>
      
      <TabsContent value="search" className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
          <Input
            placeholder="SEARCH SATELLITE DATABASE..."
            className="pl-10 bg-slate-800/50 border-purple-500/30 text-slate-300 placeholder:text-slate-500 font-mono text-base"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {loading && <div className="p-3 text-center text-slate-500">Loading satellites...</div>}
          {error && <div className="p-3 text-center text-red-500">{error}</div>}
          {!loading && filteredSatellites.map((satellite) => (
            <div
              key={satellite.id}
              className={`p-3 border rounded cursor-pointer transition-all duration-200 text-sm ${
                selectedSatellite?.name === satellite.name 
                  ? 'border-purple-500/50 bg-purple-500/10 hologram-effect' 
                  : 'border-purple-500/20 hover:border-purple-500/40 hover:bg-purple-500/5'
              }`}
              onClick={() => onSatelliteSelect(satellite)}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-slate-200">{satellite.name}</span>
                <span className="text-sm px-2 py-1 rounded bg-cyan-500/20 text-cyan-400 font-mono">
                  {satellite.orbitType}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm text-slate-500 font-mono">
                <span>{satellite.operator}</span>
                <span className="text-cyan-400">{satellite.altitude.toLocaleString()}km</span>
                <span>{satellite.purpose}</span>
                <span className="text-green-400">${(satellite.value / 1000000).toFixed(1)}M</span>
              </div>
            </div>
          ))}
        </div>
            
        
        {selectedSatellite && (
          <div className="control-surface p-3 rounded border border-purple-500/30">
            <div className="text-sm font-mono text-purple-300 mb-2">SELECTED TARGET</div>
            <div className="grid grid-cols-2 gap-2 text-sm font-mono">
              <div><span className="text-slate-500">NAME:</span> <span className="text-slate-300">{selectedSatellite.name}</span></div>
              <div><span className="text-slate-500">OPR:</span> <span className="text-slate-300">{selectedSatellite.operator}</span></div>
              <div><span className="text-slate-500">ORBIT:</span> <span className="text-cyan-400">{selectedSatellite.orbitType}</span></div>
              <div><span className="text-slate-500">ALT:</span> <span className="text-cyan-400">{selectedSatellite.altitude.toLocaleString()}km</span></div>
              <div><span className="text-slate-500">SHIELD:</span> <span className="text-green-400">{selectedSatellite.shielding}</span></div>
              <div><span className="text-slate-500">VALUE:</span> <span className="text-green-400">${(selectedSatellite.value / 1000000).toFixed(1)}M</span></div>
            </div>
          </div>
        )}
      </TabsContent>
      
      <TabsContent value="details">
        {selectedSatellite ? (
          <SatelliteDetailsPanel 
            satellite={{
              name: selectedSatellite.name,
              operator: selectedSatellite.operator,
              satellite_type: selectedSatellite.orbitType,
              orbit_type: selectedSatellite.orbitType,
              perigee_km: selectedSatellite.altitude,
              apogee_km: selectedSatellite.altitude,
              inclination_deg: selectedSatellite.inclination,
              mass_kg: selectedSatellite.mass,
              value_usd: selectedSatellite.value,
              shielding_factor: 
                selectedSatellite.shielding === "Basic" ? 0.9 :
                selectedSatellite.shielding === "Standard" ? 0.7 :
                selectedSatellite.shielding === "Enhanced" ? 0.5 : 0.3,
              launch_year: selectedSatellite.launchYear,
              expected_lifetime_years: 15,
              country: "Unknown",
              purpose: selectedSatellite.purpose,
              power_watts: 1000,
              status: "Operational"
            }} 
          />
        ) : (
          <div className="text-center py-8 text-slate-500">
            <Info className="w-12 h-12 mx-auto mb-4 text-slate-600" />
            <p>Select a satellite to view detailed information</p>
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}