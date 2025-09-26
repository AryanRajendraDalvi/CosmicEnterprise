// API service for the Cosmic Weather Insurance System

const API_BASE_URL = 'http://localhost:5000/api';

// Satellite data types
export interface Satellite {
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

// Forecast data types
export interface ForecastData {
  kp_predicted: number;
  kp_std: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  storm_probabilities: {
    minor_storm_kp5: number;
    major_storm_kp6: number;
    severe_storm_kp7: number;
    extreme_storm_kp8: number;
  };
  forecast_time: string;
  confidence_level: number;
}

// Risk assessment types
export interface RiskAssessment {
  forecast_kp: number;
  anomaly_probability: number;
  anomaly_factors: {
    base_rate: number;
    storm_factor: number;
    orbit_factor: number;
    protection_factor: number;
    age_factor: number;
    mass_factor: number;
  };
  expected_downtime_hours: number;
  financial_impact: {
    expected_loss: number;
    loss_std: number;
    quantiles: {
      '50th': number;
      '75th': number;
      '90th': number;
      '95th': number;
      '99th': number;
    };
    var_95: number;
    var_99: number;
    max_possible_loss: number;
    probability_of_loss: number;
    expected_downtime_hours: number;
  };
  risk_level: string;
  confidence: number;
}

// Insurance premium types
export interface InsurancePremium {
  base_premium: number;
  loaded_premium: number;
  capital_charge: number;
  total_premium: number;
  premium_rate_percent: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  coverage_days: number;
  deductible_recommended: number;
  policy_limits_recommended: number;
  pricing_components: {
    expected_loss: number;
    loading_factor: number;
    capital_charge_rate: number;
    risk_free_rate: number;
  };
  assumptions: string[];
}

// Full risk assessment response
export interface RiskAssessmentResponse {
  satellite_info: Satellite;
  forecast: Record<string, ForecastData>;
  impact_assessment: Record<string, RiskAssessment>;
  insurance_premiums: Record<string, InsurancePremium>;
  assessment_timestamp: string;
  system_confidence: string;
}

// API functions
export async function healthCheck(): Promise<{ status: string; message: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}

export async function getSatellites(params?: {
  query?: string;
  operator?: string;
  orbitType?: string;
  satelliteType?: string;
}): Promise<{ satellites: Satellite[]; count: number }> {
  try {
    const queryParams = new URLSearchParams();
    if (params?.query) queryParams.append('query', params.query);
    if (params?.operator) queryParams.append('operator', params.operator);
    if (params?.orbitType) queryParams.append('orbitType', params.orbitType);
    if (params?.satelliteType) queryParams.append('satelliteType', params.satelliteType);

    const response = await fetch(`${API_BASE_URL}/satellites?${queryParams.toString()}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch satellites:', error);
    throw error;
  }
}

export async function getOperators(): Promise<string[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/satellites/operators`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.operators;
  } catch (error) {
    console.error('Failed to fetch operators:', error);
    throw error;
  }
}

export async function getSatellite(satelliteName: string): Promise<Satellite> {
  try {
    const response = await fetch(`${API_BASE_URL}/satellites/${encodeURIComponent(satelliteName)}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch satellite ${satelliteName}:`, error);
    throw error;
  }
}

export async function getForecast(forecastHours?: number[]): Promise<Record<string, ForecastData>> {
  try {
    const response = await fetch(`${API_BASE_URL}/forecast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        forecast_hours: forecastHours || [24, 48, 72],
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch forecast:', error);
    throw error;
  }
}

export async function getRiskAssessment(
  satelliteName: string,
  forecastHours?: number[]
): Promise<RiskAssessmentResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/risk-assessment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        satellite_name: satelliteName,
        forecast_hours: forecastHours || [24, 48, 72],
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch risk assessment for ${satelliteName}:`, error);
    throw error;
  }
}

export async function getDatabaseStats(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/database/stats`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch database stats:', error);
    throw error;
  }
}