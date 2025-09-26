import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { AlertTriangle, Shield, Clock, DollarSign, Loader2, BarChart3 } from "lucide-react";
import { getRiskAssessment, RiskAssessmentResponse } from "../services/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

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

interface RiskAssessmentProps {
  satellite?: SatelliteData;
  stormProbability: number;
}

const COLORS = ['#0088fe', '#00c49f', '#ffc658', '#ff8042', '#8884d8'];

export function RiskAssessment({ satellite, stormProbability }: RiskAssessmentProps) {
  const [riskData, setRiskData] = useState<RiskAssessmentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch risk assessment when satellite changes
  useEffect(() => {
    if (!satellite) {
      setRiskData(null);
      return;
    }

    const fetchRiskAssessment = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getRiskAssessment(satellite.name);
        setRiskData(data);
      } catch (err) {
        setError("Failed to load risk assessment");
        console.error("Error fetching risk assessment:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchRiskAssessment();
  }, [satellite]);

  if (!satellite) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            Risk Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-slate-400 text-center py-8 text-lg">
            Select a satellite to view risk assessment
          </p>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            Risk Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-purple-500 mb-4" />
            <p className="text-slate-400 text-lg">Analyzing space weather risks...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            Risk Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-red-400 text-lg mb-2">Error loading risk assessment</p>
            <p className="text-slate-400">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Use API data if available, otherwise fallback to mock calculation
  const impactData = riskData?.impact_assessment['24h'] || null;
  const financialData = impactData?.financial_impact || null;

  // Calculate vulnerability factors (fallback if no API data)
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
  const stormImpact = impactData ? impactData.anomaly_probability : baseVulnerability * (stormProbability / 100);
  
  // Calculate potential losses
  const expectedLoss = financialData ? financialData.expected_loss : satellite.value * stormImpact * 0.15;
  const confidenceInterval = financialData 
    ? { low: financialData.quantiles['50th'], high: financialData.quantiles['95th'] }
    : { low: expectedLoss * 0.7, high: expectedLoss * 1.4 };

  const getRiskLevel = (impact: number) => {
    if (impact > 0.5) return { level: "Critical", color: "destructive" };
    if (impact > 0.3) return { level: "High", color: "destructive" };
    if (impact > 0.15) return { level: "Moderate", color: "default" };
    return { level: "Low", color: "secondary" };
  };

  const riskLevel = getRiskLevel(stormImpact);

  // Prepare data for risk factor visualization
  const riskFactors = [
    { name: "Orbital Vulnerability", value: orbitVulnerability * 100 },
    { name: "Shielding Effectiveness", value: (1 - shieldingFactor) * 100 },
    { name: "Age Factor", value: ageFactor * 100 },
    { name: "Storm Probability", value: stormProbability },
  ];

  // Prepare data for financial impact visualization
  const financialImpactData = [
    { name: "Expected Loss", value: expectedLoss / 1000000 },
    { name: "50th Percentile", value: financialData ? financialData.quantiles['50th'] / 1000000 : 0 },
    { name: "95th Percentile", value: financialData ? financialData.quantiles['95th'] / 1000000 : 0 },
    { name: "99th Percentile", value: financialData ? financialData.quantiles['99th'] / 1000000 : 0 },
  ];

  return (
    <Card className="space-card floating">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-slate-200">
          <AlertTriangle className="w-5 h-5 text-orange-400" />
          Risk Assessment
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Risk Score */}
        <div className="text-center aurora-effect p-6 rounded-xl border border-red-500/30">
          <div className="text-3xl font-bold mb-2 text-red-400 cosmic-glow">
            {(stormImpact * 100).toFixed(1)}%
          </div>
          <Badge variant={riskLevel.color as any} className="mb-2 cosmic-glow text-lg">
            {riskLevel.level} Risk
          </Badge>
          <p className="text-base text-slate-400">
            Expected impact probability from current storm forecast
          </p>
        </div>

        {/* Risk Factors Visualization */}
        <div className="border border-purple-500/20 rounded-lg p-4">
          <h4 className="font-medium mb-4 flex items-center gap-2 text-slate-200 text-lg">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Risk Factor Analysis
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskFactors}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, "Factor"]} />
                  <Bar dataKey="value" name="Risk Factor">
                    {riskFactors.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskFactors}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {riskFactors.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value}%`, "Factor"]} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Risk Factors Details */}
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-base flex items-center gap-2 text-slate-300">
                <Shield className="w-4 h-4 text-purple-400" />
                Orbital Vulnerability
              </span>
              <span className="text-base font-medium text-red-400">
                {(orbitVulnerability * 100).toFixed(0)}%
              </span>
            </div>
            <Progress value={orbitVulnerability * 100} className="h-2" />
            <p className="text-sm text-slate-400 mt-1">
              <span className="text-cyan-400">{satellite.orbitType}</span> orbit at <span className="text-cyan-400">{satellite.altitude.toLocaleString()} km</span> altitude
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-base flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Shielding Effectiveness
              </span>
              <span className="text-base font-medium">
                {((1 - shieldingFactor) * 100).toFixed(0)}%
              </span>
            </div>
            <Progress value={(1 - shieldingFactor) * 100} className="h-2" />
            <p className="text-sm text-muted-foreground mt-1">
              {satellite.shielding} radiation shielding
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-base flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Age Factor
              </span>
              <span className="text-base font-medium">
                {(ageFactor * 100).toFixed(0)}%
              </span>
            </div>
            <Progress value={ageFactor * 100} className="h-2" />
            <p className="text-sm text-muted-foreground mt-1">
              {2025 - satellite.launchYear} years in operation
            </p>
          </div>
        </div>

        {/* Financial Impact Visualization */}
        <div className="border border-green-500/20 rounded-lg p-4">
          <h4 className="font-medium mb-4 flex items-center gap-2 text-slate-200 text-lg">
            <DollarSign className="w-5 h-5 text-green-400" />
            Financial Impact Analysis (Millions $)
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={financialImpactData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => [`$${value}M`, "Value"]} />
                <Bar dataKey="value" name="Financial Impact (Millions $)" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Financial Impact */}
        <div className="border-t border-purple-500/20 pt-4">
          <h4 className="font-medium mb-3 flex items-center gap-2 text-slate-200 text-lg">
            <DollarSign className="w-4 h-4 text-green-400" />
            Expected Financial Impact
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center glass-morphism p-3 rounded-lg cosmic-glow">
              <p className="text-xl font-bold text-red-400">
                ${(expectedLoss / 1000000).toFixed(1)}M
              </p>
              <p className="text-sm text-slate-400">Expected Loss</p>
            </div>
            <div className="text-center glass-morphism p-3 rounded-lg">
              <p className="text-base text-slate-200">
                ${(confidenceInterval.low / 1000000).toFixed(1)}M - ${(confidenceInterval.high / 1000000).toFixed(1)}M
              </p>
              <p className="text-sm text-slate-400">95% Confidence Interval</p>
            </div>
          </div>
        </div>

        {/* Risk Breakdown */}
        <div className="bg-muted/50 p-4 rounded-lg">
          <h5 className="font-medium mb-2 text-lg">Risk Components</h5>
          <div className="space-y-2 text-base">
            <div className="flex justify-between">
              <span>Base Vulnerability:</span>
              <span>{(baseVulnerability * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Storm Probability:</span>
              <span>{stormProbability}%</span>
            </div>
            <div className="flex justify-between">
              <span>Combined Impact:</span>
              <span className="font-medium">{(stormImpact * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}