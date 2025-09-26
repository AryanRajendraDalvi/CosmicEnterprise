#!/usr/bin/env python3
"""
Flask API for the Cosmic Weather Insurance System
Exposes the Python backend functionality as REST endpoints for the React frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import sys
import os

# Add the current directory to Python path to import Cosmicpls
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main system
from Cosmicpls import SpaceWeatherInsuranceSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the system (this will take some time as it loads data and trains models)
print("🚀 Initializing Space Weather Insurance System...")
system = SpaceWeatherInsuranceSystem()
system.initialize_system()
print("✅ System ready!")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Cosmic Weather Insurance API is running"
    })

@app.route('/api/satellites', methods=['GET'])
def get_satellites():
    """Get list of satellites with optional search"""
    try:
        query = request.args.get('query', '')
        operator = request.args.get('operator', '')
        orbit_type = request.args.get('orbitType', '')
        satellite_type = request.args.get('satelliteType', '')
        
        # Search satellites
        results = system.satellite_db.search_satellites(
            query=query,
            operator=operator,
            orbit_type=orbit_type,
            satellite_type=satellite_type
        )
        
        # Convert to JSON-serializable format
        satellites = []
        for _, sat in results.iterrows():
            satellites.append({
                'name': sat['satellite_name'],
                'operator': sat['operator'],
                'satellite_type': sat['satellite_type'],
                'orbit_type': sat['orbit_type'],
                'perigee_km': float(sat['perigee_km']),
                'apogee_km': float(sat['apogee_km']),
                'inclination_deg': float(sat['inclination_deg']),
                'mass_kg': float(sat['mass_kg']),
                'value_usd': int(sat['value_usd']),
                'shielding_factor': float(sat['shielding_factor']),
                'launch_year': int(sat['launch_year']),
                'expected_lifetime_years': float(sat['expected_lifetime_years']),
                'country': sat['country'],
                'purpose': sat['purpose'],
                'power_watts': int(sat['power_watts']),
                'status': sat['status']
            })
        
        return jsonify({
            "satellites": satellites,
            "count": len(satellites)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/satellites/operators', methods=['GET'])
def get_operators():
    """Get list of unique satellite operators"""
    try:
        operators = system.satellite_db.list_unique_operators()
        return jsonify({"operators": operators})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/satellites/<satellite_name>', methods=['GET'])
def get_satellite(satellite_name):
    """Get detailed information about a specific satellite"""
    try:
        satellite_info = system.satellite_db.get_satellite_by_name(satellite_name)
        if not satellite_info:
            return jsonify({"error": f"Satellite '{satellite_name}' not found"}), 404
        
        # Convert to JSON-serializable format
        satellite_data = {
            'name': satellite_info['satellite_name'],
            'operator': satellite_info['operator'],
            'satellite_type': satellite_info['satellite_type'],
            'orbit_type': satellite_info['orbit_type'],
            'perigee_km': float(satellite_info['perigee_km']),
            'apogee_km': float(satellite_info['apogee_km']),
            'inclination_deg': float(satellite_info['inclination_deg']),
            'mass_kg': float(satellite_info['mass_kg']),
            'value_usd': int(satellite_info['value_usd']),
            'shielding_factor': float(satellite_info['shielding_factor']),
            'launch_year': int(satellite_info['launch_year']),
            'expected_lifetime_years': float(satellite_info['expected_lifetime_years']),
            'country': satellite_info['country'],
            'purpose': satellite_info['purpose'],
            'power_watts': int(satellite_info['power_watts']),
            'status': satellite_info['status']
        }
        
        return jsonify(satellite_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """Get space weather forecast"""
    try:
        # Get forecast hours from request or use defaults
        data = request.get_json() or {}
        forecast_hours = data.get('forecast_hours', [24, 48, 72])
        
        # Generate forecast using the latest data
        forecast = system.forecaster.forecast_storm(system.space_weather_data, forecast_hours)
        
        # Convert to JSON-serializable format
        forecast_data = {}
        for key, value in forecast.items():
            forecast_data[key] = {
                'kp_predicted': float(value['kp_predicted']),
                'kp_std': float(value['kp_std']),
                'confidence_interval': {
                    'lower': float(value['confidence_interval']['lower']),
                    'upper': float(value['confidence_interval']['upper'])
                },
                'storm_probabilities': {
                    'minor_storm_kp5': float(value['storm_probabilities']['minor_storm_kp5']),
                    'major_storm_kp6': float(value['storm_probabilities']['major_storm_kp6']),
                    'severe_storm_kp7': float(value['storm_probabilities']['severe_storm_kp7']),
                    'extreme_storm_kp8': float(value['storm_probabilities']['extreme_storm_kp8'])
                },
                'forecast_time': value['forecast_time'].isoformat() if hasattr(value['forecast_time'], 'isoformat') else str(value['forecast_time']),
                'confidence_level': value['confidence_level']
            }
        
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/risk-assessment', methods=['POST'])
def get_risk_assessment():
    """Get risk assessment for a satellite"""
    try:
        data = request.get_json()
        if not data or 'satellite_name' not in data:
            return jsonify({"error": "satellite_name is required"}), 400
        
        satellite_name = data['satellite_name']
        forecast_hours = data.get('forecast_hours', [24, 48, 72])
        
        # Generate comprehensive risk assessment
        assessment = system.generate_risk_assessment(satellite_name, forecast_hours)
        
        # Convert to JSON-serializable format
        response = {
            'satellite_info': {
                'name': assessment['satellite_info']['satellite_name'],
                'operator': assessment['satellite_info']['operator'],
                'satellite_type': assessment['satellite_info']['satellite_type'],
                'orbit_type': assessment['satellite_info']['orbit_type'],
                'perigee_km': float(assessment['satellite_info']['perigee_km']),
                'apogee_km': float(assessment['satellite_info']['apogee_km']),
                'inclination_deg': float(assessment['satellite_info']['inclination_deg']),
                'mass_kg': float(assessment['satellite_info']['mass_kg']),
                'value_usd': int(assessment['satellite_info']['value_usd']),
                'shielding_factor': float(assessment['satellite_info']['shielding_factor']),
                'launch_year': int(assessment['satellite_info']['launch_year']),
                'expected_lifetime_years': float(assessment['satellite_info']['expected_lifetime_years']),
                'country': assessment['satellite_info']['country'],
                'purpose': assessment['satellite_info']['purpose'],
                'power_watts': int(assessment['satellite_info']['power_watts']),
                'status': assessment['satellite_info']['status']
            },
            'forecast': {},
            'impact_assessment': {},
            'insurance_premiums': {},
            'assessment_timestamp': assessment['assessment_timestamp'],
            'system_confidence': assessment['system_confidence']
        }
        
        # Process forecast data
        for key, value in assessment['forecast'].items():
            response['forecast'][key] = {
                'kp_predicted': float(value['kp_predicted']),
                'kp_std': float(value['kp_std']),
                'confidence_interval': {
                    'lower': float(value['confidence_interval']['lower']),
                    'upper': float(value['confidence_interval']['upper'])
                },
                'storm_probabilities': {
                    'minor_storm_kp5': float(value['storm_probabilities']['minor_storm_kp5']),
                    'major_storm_kp6': float(value['storm_probabilities']['major_storm_kp6']),
                    'severe_storm_kp7': float(value['storm_probabilities']['severe_storm_kp7']),
                    'extreme_storm_kp8': float(value['storm_probabilities']['extreme_storm_kp8'])
                },
                'forecast_time': value['forecast_time'].isoformat() if hasattr(value['forecast_time'], 'isoformat') else str(value['forecast_time']),
                'confidence_level': value['confidence_level']
            }
        
        # Process impact assessment data
        for key, value in assessment['impact_assessment'].items():
            response['impact_assessment'][key] = {
                'forecast_kp': float(value['forecast_kp']),
                'anomaly_probability': float(value['anomaly_probability']),
                'anomaly_factors': {
                    'base_rate': float(value['anomaly_factors']['base_rate']),
                    'storm_factor': float(value['anomaly_factors']['storm_factor']),
                    'orbit_factor': float(value['anomaly_factors']['orbit_factor']),
                    'protection_factor': float(value['anomaly_factors']['protection_factor']),
                    'age_factor': float(value['anomaly_factors']['age_factor']),
                    'mass_factor': float(value['anomaly_factors']['mass_factor'])
                },
                'expected_downtime_hours': float(value['expected_downtime_hours']),
                'financial_impact': {
                    'expected_loss': float(value['financial_impact']['expected_loss']),
                    'loss_std': float(value['financial_impact']['loss_std']),
                    'quantiles': {
                        '50th': float(value['financial_impact']['quantiles']['50th']),
                        '75th': float(value['financial_impact']['quantiles']['75th']),
                        '90th': float(value['financial_impact']['quantiles']['90th']),
                        '95th': float(value['financial_impact']['quantiles']['95th']),
                        '99th': float(value['financial_impact']['quantiles']['99th'])
                    },
                    'var_95': float(value['financial_impact']['var_95']),
                    'var_99': float(value['financial_impact']['var_99']),
                    'max_possible_loss': float(value['financial_impact']['max_possible_loss']),
                    'probability_of_loss': float(value['financial_impact']['probability_of_loss']),
                    'expected_downtime_hours': float(value['financial_impact']['expected_downtime_hours'])
                },
                'risk_level': value['risk_level'],
                'confidence': value['confidence']
            }
        
        # Process insurance premiums data
        for key, value in assessment['insurance_premiums'].items():
            response['insurance_premiums'][key] = {
                'base_premium': float(value['base_premium']),
                'loaded_premium': float(value['loaded_premium']),
                'capital_charge': float(value['capital_charge']),
                'total_premium': float(value['total_premium']),
                'premium_rate_percent': float(value['premium_rate_percent']),
                'confidence_interval': {
                    'lower': float(value['confidence_interval']['lower']),
                    'upper': float(value['confidence_interval']['upper'])
                },
                'coverage_days': value['coverage_days'],
                'deductible_recommended': float(value['deductible_recommended']),
                'policy_limits_recommended': float(value['policy_limits_recommended']),
                'pricing_components': {
                    'expected_loss': float(value['pricing_components']['expected_loss']),
                    'loading_factor': float(value['pricing_components']['loading_factor']),
                    'capital_charge_rate': float(value['pricing_components']['capital_charge_rate']),
                    'risk_free_rate': float(value['pricing_components']['risk_free_rate'])
                },
                'assumptions': value['assumptions']
            }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/database/stats', methods=['GET'])
def get_database_stats():
    """Get satellite database statistics"""
    try:
        stats = system.satellite_db.get_database_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)