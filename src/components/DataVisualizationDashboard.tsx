import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from "recharts";
import { 
  Satellite, Shield, TrendingUp, DollarSign, Activity, 
  Globe, Rocket, Calendar, Zap, Loader2, AlertTriangle
} from "lucide-react";
import { getDatabaseStats, getSatellites, getOperators, getForecast } from "../services/api";

// Color palette for charts
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', '#00c49f'];

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

export function DataVisualizationDashboard() {
  const [databaseStats, setDatabaseStats] = useState<any>(null);
  const [satellites, setSatellites] = useState<SatelliteData[]>([]);
  const [operators, setOperators] = useState<string[]>([]);
  const [forecastData, setForecastData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch all data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch database stats
        const stats = await getDatabaseStats();
        setDatabaseStats(stats);
        
        // Fetch operators
        const operatorList = await getOperators();
        setOperators(operatorList);
        
        // Fetch a sample of satellites
        const satelliteData = await getSatellites({ query: "" });
        setSatellites(satelliteData.satellites.slice(0, 50)); // Limit to 50 for performance
        
        // Fetch forecast data
        const forecast = await getForecast([24, 48, 72]);
        const forecastArray = Object.entries(forecast).map(([key, value]) => ({
          time: key,
          kpIndex: value.kp_predicted,
          stormProbability: Math.round(value.storm_probabilities.severe_storm_kp7 * 100),
          confidenceLower: value.confidence_interval.lower,
          confidenceUpper: value.confidence_interval.upper
        }));
        setForecastData(forecastArray);
      } catch (err) {
        setError("Failed to load data");
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4" />
        <p className="text-slate-400 text-xl">Loading cosmic data visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            Data Visualization Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-red-400 text-lg mb-2">Failed to load visualization data</p>
            <p className="text-slate-400">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for visualizations
  const orbitDistribution = databaseStats?.orbit_distribution 
    ? Object.entries(databaseStats.orbit_distribution).map(([name, count]) => ({
        name,
        value: count as number
      }))
    : [];

  const satelliteTypeDistribution = databaseStats?.type_distribution 
    ? Object.entries(databaseStats.type_distribution).map(([name, count]) => ({
        name,
        value: count as number
      }))
    : [];

  const satelliteValueData = satellites
    .sort((a, b) => b.value_usd - a.value_usd)
    .slice(0, 10)
    .map(sat => ({
      name: sat.name.length > 15 ? `${sat.name.substring(0, 15)}...` : sat.name,
      value: sat.value_usd / 1000000 // Convert to millions
    }));

  const satelliteMassData = satellites
    .sort((a, b) => b.mass_kg - a.mass_kg)
    .slice(0, 10)
    .map(sat => ({
      name: sat.name.length > 15 ? `${sat.name.substring(0, 15)}...` : sat.name,
      mass: sat.mass_kg
    }));

  const operatorSatelliteCount = operators.slice(0, 10).map(op => {
    const count = satellites.filter(s => s.operator === op).length;
    return { name: op, count };
  }).filter(op => op.count > 0);

  const launchYearData = satellites.reduce((acc, sat) => {
    const year = sat.launch_year;
    if (!acc[year]) {
      acc[year] = 0;
    }
    acc[year] += 1;
    return acc;
  }, {} as Record<number, number>);

  const launchYearChartData = Object.entries(launchYearData)
    .map(([year, count]) => ({ year: parseInt(year), count }))
    .sort((a, b) => a.year - b.year);

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-200 flex items-center gap-2">
          <Activity className="w-6 h-6 text-purple-400" />
          Data Visualization Dashboard
        </h2>
        <Badge variant="outline" className="text-sm">
          Real-time Data
        </Badge>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="space-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Total Satellites</p>
                <p className="text-2xl font-bold text-slate-200">
                  {databaseStats?.total_satellites?.toLocaleString() || '0'}
                </p>
              </div>
              <Satellite className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="space-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Total Fleet Value</p>
                <p className="text-2xl font-bold text-slate-200">
                  ${(databaseStats?.total_value ? databaseStats.total_value / 1e9 : 0).toFixed(1)}B
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="space-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Operators</p>
                <p className="text-2xl font-bold text-slate-200">
                  {databaseStats?.operator_count || '0'}
                </p>
              </div>
              <Globe className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="space-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Avg. Satellite Value</p>
                <p className="text-2xl font-bold text-slate-200">
                  ${(databaseStats?.average_value ? databaseStats.average_value / 1e6 : 0).toFixed(1)}M
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Orbit Distribution */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Globe className="w-5 h-5 text-blue-400" />
              Satellite Orbit Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={orbitDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {orbitDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value, "Satellites"]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Satellite Type Distribution */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Rocket className="w-5 h-5 text-orange-400" />
              Satellite Type Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={satelliteTypeDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" name="Count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Valuable Satellites */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <DollarSign className="w-5 h-5 text-green-400" />
              Top 10 Most Valuable Satellites (Millions $)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={satelliteValueData}
                  layout="vertical"
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip formatter={(value) => [`$${value}M`, "Value"]} />
                  <Legend />
                  <Bar dataKey="value" name="Value (Millions $)" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Satellite Mass Distribution */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Zap className="w-5 h-5 text-yellow-400" />
              Top 10 Heaviest Satellites (kg)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={satelliteMassData}
                  layout="vertical"
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip formatter={(value) => [`${value} kg`, "Mass"]} />
                  <Legend />
                  <Bar dataKey="mass" name="Mass (kg)" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Launch Year Distribution */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Calendar className="w-5 h-5 text-purple-400" />
              Satellite Launch Timeline
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={launchYearChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    name="Satellites Launched" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.3} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Operator Distribution */}
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Shield className="w-5 h-5 text-cyan-400" />
              Top Operators by Satellite Count
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={operatorSatelliteCount}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" name="Satellite Count" fill="#ff8042" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Space Weather Forecast Visualization */}
      {forecastData.length > 0 && (
        <Card className="space-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-200">
              <Activity className="w-5 h-5 text-red-400" />
              Space Weather Forecast
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="kpIndex" 
                    name="Kp Index" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="stormProbability" 
                    name="Storm Probability (%)" 
                    stroke="#ff8042" 
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}