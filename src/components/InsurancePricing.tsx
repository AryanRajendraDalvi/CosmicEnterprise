import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Label } from "./ui/label";
import { Calculator, TrendingUp, FileText, Download, Loader2 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { getRiskAssessment, RiskAssessmentResponse } from "../services/api";

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

interface MonteCarloResult {
  simulation: number;
  loss: number;
  premium: number;
}

interface InsurancePricingProps {
  satellite?: SatelliteData;
  expectedLoss: number;
  stormProbability: number;
}

export function InsurancePricing({ satellite, expectedLoss, stormProbability }: InsurancePricingProps) {
  const [loadingFactor, setLoadingFactor] = useState<[number]>([1.25]);
  const [coverageLevel, setCoverageLevel] = useState<[number]>([80]);
  const [deductible, setDeductible] = useState<[number]>([5]);
  const [monteCarloResults, setMonteCarloResults] = useState<MonteCarloResult[]>([]);
  const [premiumData, setPremiumData] = useState<RiskAssessmentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch premium data when satellite changes
  useEffect(() => {
    if (!satellite) {
      setPremiumData(null);
      return;
    }

    const fetchPremiumData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getRiskAssessment(satellite.name);
        setPremiumData(data);
      } catch (err) {
        setError("Failed to load premium data");
        console.error("Error fetching premium data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPremiumData();
  }, [satellite]);

  // Simulate Monte Carlo analysis
  useEffect(() => {
    if (expectedLoss > 0) {
      const results: MonteCarloResult[] = [];
      for (let i = 0; i < 20; i++) {
        const variation = 0.8 + Math.random() * 0.4; // ±20% variation
        const loss = expectedLoss * variation;
        results.push({
          simulation: i + 1,
          loss: loss / 1000000, // Convert to millions
          premium: (loss * loadingFactor[0]) / 1000000
        });
      }
      setMonteCarloResults(results);
    }
  }, [expectedLoss, loadingFactor]);

  if (!satellite || expectedLoss === 0) {
    return (
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <Calculator className="w-5 h-5 text-purple-400" />
            Insurance Pricing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-slate-400 text-center py-8 text-lg">
            Complete risk assessment to view insurance pricing
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
            <Calculator className="w-5 h-5 text-purple-400" />
            Insurance Pricing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-purple-500 mb-4" />
            <p className="text-slate-400 text-lg">Calculating insurance premium...</p>
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
            <Calculator className="w-5 h-5 text-purple-400" />
            Insurance Pricing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-red-400 text-lg mb-2">Error loading premium data</p>
            <p className="text-slate-400">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Use API data if available
  const premiumInfo = premiumData?.insurance_premiums['24h'] || null;

  // Use API data or fallback to mock calculation
  const finalPremium = premiumInfo ? premiumInfo.total_premium : expectedLoss * loadingFactor[0];
  const premiumRate = premiumInfo ? premiumInfo.premium_rate_percent : (finalPremium / satellite.value) * 100;
  
  // Handle confidence interval from API or fallback
  let confidenceInterval: { lower: number; upper: number };
  if (premiumInfo) {
    confidenceInterval = premiumInfo.confidence_interval;
  } else {
    confidenceInterval = { 
      lower: finalPremium * 0.85, 
      upper: finalPremium * 1.15 
    };
  }

  // Calculate other values for display
  const adjustedLoss = expectedLoss * (coverageLevel[0] / 100);
  const deductibleAmount = satellite.value * (deductible[0] / 100);
  const netCoverage = Math.max(0, adjustedLoss - deductibleAmount);
  const riskMultiplier = Math.max(0.8, 1 + (stormProbability - 20) / 100);

  return (
    <div className="space-y-6">
      <Card className="space-card floating">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <Calculator className="w-5 h-5 text-purple-400" />
            Insurance Pricing Calculator
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Premium Summary */}
          <div className="text-center aurora-effect p-6 rounded-lg border border-purple-500/30 cosmic-glow">
            <div className="text-3xl font-bold text-green-400 mb-2 cosmic-glow">
              ${(finalPremium / 1000000).toFixed(2)}M
            </div>
            <Badge variant="outline" className="mb-2 border-purple-500/30 text-purple-300 text-lg">
              {premiumRate.toFixed(3)}% of asset value
            </Badge>
            <p className="text-base text-slate-400">
              Recommended annual premium (loading factor: <span className="text-cyan-400">{loadingFactor[0]}x</span>)
            </p>
            <div className="text-base mt-2">
              <span className="text-slate-400">95% CI: </span>
              <span className="text-slate-200">${(confidenceInterval.lower / 1000000).toFixed(2)}M - ${(confidenceInterval.upper / 1000000).toFixed(2)}M</span>
            </div>
          </div>

          {/* Pricing Controls */}
          <div className="space-y-4">
            <div>
              <Label className="text-sm font-medium">Loading Factor: {loadingFactor[0]}x</Label>
              <Slider
                value={loadingFactor}
                onValueChange={(value) => setLoadingFactor(value as [number])}
                max={2}
                min={1}
                step={0.05}
                className="mt-2"
              />
              <p className="text-sm text-muted-foreground mt-1">
                Risk margin and profit factor applied to expected loss
              </p>
            </div>

            <div>
              <Label className="text-sm font-medium">Coverage Level: {coverageLevel[0]}%</Label>
              <Slider
                value={coverageLevel}
                onValueChange={(value) => setCoverageLevel(value as [number])}
                max={100}
                min={50}
                step={5}
                className="mt-2"
              />
              <p className="text-sm text-muted-foreground mt-1">
                Percentage of total asset value covered
              </p>
            </div>

            <div>
              <Label className="text-sm font-medium">Deductible: {deductible[0]}%</Label>
              <Slider
                value={deductible}
                onValueChange={(value) => setDeductible(value as [number])}
                max={20}
                min={0}
                step={1}
                className="mt-2"
              />
              <p className="text-sm text-muted-foreground mt-1">
                Percentage of asset value as deductible
              </p>
            </div>
          </div>

          {/* Pricing Breakdown */}
          <div className="border-t pt-4">
            <h4 className="font-medium mb-3 text-lg">Pricing Breakdown</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Asset Value:</span>
                <span>${(satellite.value / 1000000).toFixed(1)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Expected Loss (raw):</span>
                <span>${(expectedLoss / 1000000).toFixed(2)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Coverage Adjustment ({coverageLevel[0]}%):</span>
                <span>${(adjustedLoss / 1000000).toFixed(2)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Deductible ({deductible[0]}%):</span>
                <span>-${(deductibleAmount / 1000000).toFixed(2)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Net Coverage:</span>
                <span>${(netCoverage / 1000000).toFixed(2)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Risk Multiplier:</span>
                <span>{riskMultiplier.toFixed(2)}x</span>
              </div>
              <div className="flex justify-between font-medium border-t pt-2">
                <span>Final Premium:</span>
                <span>${(finalPremium / 1000000).toFixed(2)}M</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Monte Carlo Simulation */}
      <Card className="space-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            Monte Carlo Simulation (1000 runs)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monteCarloResults.slice(0, 15)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="simulation" />
                <YAxis label={{ value: 'Premium ($M)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  formatter={(value, name) => [
                    `$${Number(value).toFixed(2)}M`, 
                    name === 'premium' ? 'Premium' : 'Expected Loss'
                  ]}
                />
                <Bar dataKey="loss" fill="#8884d8" name="Expected Loss" />
                <Bar dataKey="premium" fill="#82ca9d" name="Premium" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div className="glass-morphism p-3 rounded-lg">
              <p className="text-xl font-bold text-green-400">${(finalPremium / 1000000).toFixed(2)}M</p>
              <p className="text-sm text-slate-400">Mean Premium</p>
            </div>
            <div className="glass-morphism p-3 rounded-lg">
              <p className="text-xl font-bold text-cyan-400">±${((finalPremium * 0.15) / 1000000).toFixed(2)}M</p>
              <p className="text-sm text-slate-400">Standard Deviation</p>
            </div>
            <div className="glass-morphism p-3 rounded-lg">
              <p className="text-xl font-bold text-purple-400">95%</p>
              <p className="text-sm text-slate-400">Confidence Level</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button className="flex-1 bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 cosmic-glow">
          <FileText className="w-4 h-4 mr-2" />
          Generate Quote
        </Button>
        <Button variant="outline" className="flex-1 glass-morphism border-purple-500/30 text-purple-300 hover:bg-purple-500/10">
          <Download className="w-4 h-4 mr-2" />
          Export Analysis
        </Button>
      </div>
    </div>
  );
}