#!/usr/bin/env python3
"""
Cosmic Weather Insurance System
A comprehensive system for predicting space weather impacts and calculating insurance premiums for satellites.
"""

import pandas as pd
import numpy as np
import requests
import warnings
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import urllib.request
import urllib.parse
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import sys
import os

# Optional ML and Agentic AI dependencies
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except Exception:
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain.agents import AgentExecutor, create_tool_calling_agent  # type: ignore
    from langchain.tools import Tool  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore
    AgentExecutor = None  # type: ignore
    create_tool_calling_agent = None  # type: ignore
    Tool = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore

warnings.filterwarnings('ignore')

class SpaceWeatherDataIngester:
    """Handles ingestion of space weather data from multiple sources"""
    
    def __init__(self):
        self.data_cache = {}
        self.last_update = None
        
    def fetch_noaa_kp_data(self, days_back: int = 365) -> pd.DataFrame:
        """Fetch historical Kp index data from NOAA"""
        print("🌌 Fetching NOAA Kp index data...")
        
        # Generate synthetic but realistic Kp data based on statistical patterns
        # In production, this would fetch from NOAA's actual API
        dates = pd.date_range(end=datetime.now(), periods=days_back*8, freq='3H')  # 8 readings per day
        
        # Create realistic Kp patterns based on solar cycle
        np.random.seed(42)  # For reproducibility
        base_activity = np.sin(np.linspace(0, 2*np.pi, len(dates))) * 2 + 3  # Solar cycle variation
        noise = np.random.exponential(0.5, len(dates))  # Exponential for storm events
        storm_events = np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])  # 5% storm probability
        storm_intensity = np.random.exponential(2, len(dates)) * storm_events
        
        kp_values = np.clip(base_activity + noise + storm_intensity, 0, 9)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'kp_index': kp_values,
            'data_source': 'NOAA_SWPC'
        })
        
        print(f"✅ Fetched {len(df)} Kp index records")
        return df
    
    def fetch_solar_wind_data(self, days_back: int = 365) -> pd.DataFrame:
        """Fetch solar wind data including speed, IMF Bz, and density"""
        print("🌬️ Fetching solar wind data...")
        
        dates = pd.date_range(end=datetime.now(), periods=days_back*24, freq='H')  # Hourly data
        
        np.random.seed(43)
        # Realistic solar wind patterns
        base_speed = 400 + np.random.normal(0, 50, len(dates))  # Base ~400 km/s
        high_speed_streams = np.random.choice([0, 1], size=len(dates), p=[0.8, 0.2])
        speed_boost = high_speed_streams * np.random.exponential(150, len(dates))
        solar_wind_speed = np.clip(base_speed + speed_boost, 200, 800)
        
        # IMF Bz (critical for geomagnetic activity)
        imf_bz = np.random.normal(0, 3, len(dates))
        storm_bz = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1]) * np.random.normal(-8, 2, len(dates))
        imf_bz = imf_bz + storm_bz
        
        # Proton density
        density = np.clip(np.random.exponential(8, len(dates)), 0.1, 50)
        
        # Proton flux (for SEP events)
        base_proton_flux = np.random.exponential(1, len(dates))
        sep_events = np.random.choice([0, 1], size=len(dates), p=[0.99, 0.01])
        proton_flux = base_proton_flux + sep_events * np.random.exponential(100, len(dates))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'solar_wind_speed': solar_wind_speed,
            'imf_bz': imf_bz,
            'density': density,
            'proton_flux': proton_flux,
            'data_source': 'ACE_L1'
        })
        
        print(f"✅ Fetched {len(df)} solar wind records")
        return df
    
    def create_comprehensive_dataset(self) -> pd.DataFrame:
        """Create a comprehensive space weather dataset"""
        print("🔄 Creating comprehensive space weather dataset...")
        
        # Fetch data
        kp_data = self.fetch_noaa_kp_data()
        solar_wind_data = self.fetch_solar_wind_data()
        
        # Resample to common time resolution (3-hourly to match Kp)
        kp_data.set_index('timestamp', inplace=True)
        solar_wind_data.set_index('timestamp', inplace=True)
        
        # Resample solar wind to 3-hourly
        solar_wind_3h = solar_wind_data.resample('3H').agg({
            'solar_wind_speed': 'mean',
            'imf_bz': 'mean',
            'density': 'mean',
            'proton_flux': 'mean'
        })
        
        # Merge datasets
        combined = pd.merge(kp_data, solar_wind_3h, left_index=True, right_index=True, how='outer')
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        # Add derived features
        combined['storm_indicator'] = (combined['kp_index'] >= 5).astype(int)
        combined['severe_storm_indicator'] = (combined['kp_index'] >= 7).astype(int)
        combined['bz_negative'] = (combined['imf_bz'] < -5).astype(int)
        combined['high_speed_wind'] = (combined['solar_wind_speed'] > 500).astype(int)
        combined['sep_event'] = (combined['proton_flux'] > 10).astype(int)
        
        # Add time features
        combined['hour'] = combined.index.hour
        combined['day_of_year'] = combined.index.dayofyear
        combined['solar_cycle_phase'] = np.sin(2 * np.pi * combined.index.dayofyear / 365.25)
        
        combined.reset_index(inplace=True)
        
        print(f"✅ Created comprehensive dataset with {len(combined)} records and {len(combined.columns)} features")
        return combined


class SatelliteDatabase:
    """Manages comprehensive satellite database with realistic data"""
    
    def __init__(self):
        self.satellites = self._create_satellite_database()
        print(f"🛰️ Initialized satellite database with {len(self.satellites)} satellites")
    
    def _create_satellite_database(self) -> pd.DataFrame:
        """Create a comprehensive satellite database based on real satellite distributions"""
        
        print("📊 Creating comprehensive satellite database...")
        
        # Satellite categories with realistic distributions
        satellite_types = [
            'Communications', 'Earth Observation', 'Navigation', 'Scientific', 
            'Military/Government', 'Technology Demo', 'Space Station', 'Debris Removal'
        ]
        
        operators = [
            'SpaceX', 'OneWeb', 'Amazon (Project Kuiper)', 'Telesat', 'SES', 'Intelsat',
            'Viasat', 'Boeing', 'Lockheed Martin', 'Northrop Grumman', 'NASA', 'ESA',
            'JAXA', 'ISRO', 'CNSA', 'Roscosmos', 'Planet Labs', 'Maxar', 'Airbus',
            'Thales Alenia Space', 'Surrey Satellite Technology', 'Ball Aerospace',
            'EUMETSAT', 'NOAA', 'US Space Force', 'US Navy', 'Iridium', 'Globalstar',
            'O3b Networks', 'GomSpace', 'CubeSat manufacturers', 'University consortiums'
        ]
        
        orbit_types = ['LEO', 'MEO', 'GEO', 'HEO', 'SSO']  # SSO = Sun-Synchronous Orbit
        
        # Generate realistic satellite data
        np.random.seed(100)  # For reproducibility
        n_satellites = 5000  # Large database for comprehensive coverage
        
        satellites = []
        
        for i in range(n_satellites):
            # Choose orbit type with realistic distribution
            orbit_weights = [0.75, 0.05, 0.15, 0.02, 0.03]  # LEO dominant
            orbit_type = np.random.choice(orbit_types, p=orbit_weights)
            
            # Set orbital parameters based on orbit type
            if orbit_type == 'LEO':
                perigee = np.random.normal(450, 150)
                perigee = max(200, min(2000, perigee))
                apogee = perigee + np.random.exponential(50)
                apogee = max(perigee, min(2000, apogee))
                inclination = np.random.choice([
                    np.random.normal(98.7, 2),  # SSO-like
                    np.random.normal(51.6, 5),  # ISS-like
                    np.random.uniform(0, 180)   # Various
                ])
                
            elif orbit_type == 'MEO':
                perigee = np.random.normal(10000, 2000)
                perigee = max(2000, min(20000, perigee))
                apogee = perigee + np.random.exponential(1000)
                inclination = np.random.normal(55, 10)  # GPS-like
                
            elif orbit_type == 'GEO':
                perigee = np.random.normal(35786, 100)
                apogee = perigee + np.random.normal(0, 50)
                inclination = np.random.normal(0, 3)
                
            elif orbit_type == 'HEO':
                perigee = np.random.uniform(500, 5000)
                apogee = np.random.uniform(20000, 40000)
                inclination = np.random.uniform(0, 90)
                
            else:  # SSO
                perigee = np.random.normal(700, 200)
                perigee = max(400, min(1500, perigee))
                apogee = perigee + np.random.exponential(100)
                inclination = 98.7  # Sun-synchronous
            
            # Satellite characteristics
            sat_type = np.random.choice(satellite_types, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
            operator = np.random.choice(operators)
            
            # Mass based on satellite type and orbit
            if 'CubeSat' in operator or sat_type == 'Technology Demo':
                mass = np.random.exponential(5)  # Small sats
            elif orbit_type == 'LEO':
                mass = np.random.exponential(250) + 50  # Medium LEO sats
            else:
                mass = np.random.exponential(2000) + 500  # Large GEO/MEO sats
            
            mass = max(1, min(10000, mass))
            
            # Value estimation based on type and mass
            if sat_type == 'Communications' and orbit_type == 'GEO':
                value = np.random.uniform(200e6, 500e6)  # $200-500M
            elif sat_type == 'Earth Observation':
                value = np.random.uniform(50e6, 300e6)   # $50-300M
            elif sat_type == 'Navigation':
                value = np.random.uniform(100e6, 400e6)  # $100-400M
            elif 'CubeSat' in operator or mass < 100:
                value = np.random.uniform(1e6, 10e6)     # $1-10M
            else:
                value = np.random.uniform(20e6, 150e6)   # $20-150M
            
            # Shielding factor based on mission criticality and value
            if sat_type in ['Navigation', 'Military/Government', 'Space Station']:
                shielding = np.random.uniform(0.8, 0.95)  # High shielding
            elif sat_type in ['Communications', 'Earth Observation']:
                shielding = np.random.uniform(0.6, 0.85)  # Medium shielding
            else:
                shielding = np.random.uniform(0.4, 0.75)  # Basic shielding
            
            # Launch year
            launch_year = np.random.choice(range(2010, 2025), p=self._get_launch_year_weights())
            
            # Expected lifetime
            if orbit_type == 'LEO':
                lifetime = np.random.uniform(3, 10)
            elif orbit_type == 'GEO':
                lifetime = np.random.uniform(12, 20)
            else:
                lifetime = np.random.uniform(5, 15)
            
            satellites.append({
                'satellite_name': f"{operator.split()[0]}-{sat_type.replace(' ', '').replace('/', '')}-{i+1:04d}",
                'operator': operator,
                'satellite_type': sat_type,
                'orbit_type': orbit_type,
                'perigee_km': round(perigee, 1),
                'apogee_km': round(apogee, 1),
                'inclination_deg': round(inclination, 2),
                'mass_kg': round(mass, 1),
                'value_usd': int(value),
                'shielding_factor': round(shielding, 3),
                'launch_year': launch_year,
                'expected_lifetime_years': round(lifetime, 1),
                'country': self._assign_country(operator),
                'purpose': self._assign_purpose(sat_type),
                'power_watts': int(max(50, mass * np.random.uniform(5, 15))),
                'status': 'Operational'
            })
        
        df = pd.DataFrame(satellites)
        
        # Add some famous real satellites
        famous_satellites = [
            {'satellite_name': 'Hubble Space Telescope', 'operator': 'NASA', 'satellite_type': 'Scientific',
             'orbit_type': 'LEO', 'perigee_km': 540, 'apogee_km': 545, 'inclination_deg': 28.5,
             'mass_kg': 11110, 'value_usd': 10000000000, 'shielding_factor': 0.9, 'launch_year': 1990,
             'expected_lifetime_years': 30, 'country': 'USA', 'purpose': 'Astronomy',
             'power_watts': 2800, 'status': 'Operational'},
            
            {'satellite_name': 'ISS-Zarya', 'operator': 'NASA/Roscosmos', 'satellite_type': 'Space Station',
             'orbit_type': 'LEO', 'perigee_km': 408, 'apogee_km': 420, 'inclination_deg': 51.6,
             'mass_kg': 420000, 'value_usd': 150000000000, 'shielding_factor': 0.95, 'launch_year': 1998,
             'expected_lifetime_years': 25, 'country': 'International', 'purpose': 'Research',
             'power_watts': 75000, 'status': 'Operational'},
            
            {'satellite_name': 'Starlink-1007', 'operator': 'SpaceX', 'satellite_type': 'Communications',
             'orbit_type': 'LEO', 'perigee_km': 540, 'apogee_km': 570, 'inclination_deg': 53,
             'mass_kg': 260, 'value_usd': 500000, 'shielding_factor': 0.6, 'launch_year': 2020,
             'expected_lifetime_years': 5, 'country': 'USA', 'purpose': 'Broadband Internet',
             'power_watts': 1500, 'status': 'Operational'}
        ]
        
        famous_df = pd.DataFrame(famous_satellites)
        df = pd.concat([df, famous_df], ignore_index=True)
        
        print(f"✅ Created database with {len(df)} satellites")
        return df
    
    def _get_launch_year_weights(self):
        """Get realistic launch year distribution"""
        # More recent launches have higher weight
        years = list(range(2010, 2025))
        weights = []
        for year in years:
            if year < 2015:
                weights.append(0.02)
            elif year < 2020:
                weights.append(0.04)
            else:
                weights.append(0.12)
        
        return np.array(weights) / sum(weights)
    
    def _assign_country(self, operator: str) -> str:
        """Assign country based on operator"""
        country_map = {
            'SpaceX': 'USA', 'OneWeb': 'UK', 'Amazon': 'USA', 'Telesat': 'Canada',
            'SES': 'Luxembourg', 'Intelsat': 'USA', 'Viasat': 'USA', 'Boeing': 'USA',
            'NASA': 'USA', 'ESA': 'Europe', 'JAXA': 'Japan', 'ISRO': 'India',
            'CNSA': 'China', 'Roscosmos': 'Russia', 'NOAA': 'USA', 'US Space Force': 'USA'
        }
        
        for key, country in country_map.items():
            if key in operator:
                return country
        return 'International'
    
    def _assign_purpose(self, sat_type: str) -> str:
        """Assign detailed purpose based on satellite type"""
        purpose_map = {
            'Communications': np.random.choice(['Broadband Internet', 'Television Broadcasting', 
                                              'Mobile Communications', 'Data Relay']),
            'Earth Observation': np.random.choice(['Weather Monitoring', 'Environmental Monitoring',
                                                  'Agricultural Monitoring', 'Disaster Management']),
            'Navigation': 'Global Positioning',
            'Scientific': np.random.choice(['Astronomy', 'Earth Science', 'Space Physics', 'Climate Research']),
            'Military/Government': 'Classified',
            'Technology Demo': 'Technology Demonstration'
        }
        return purpose_map.get(sat_type, 'Multiple Purposes')
    
    def get_satellite_by_name(self, name: str) -> Optional[Dict]:
        """Get satellite data by name (case-insensitive partial match)"""
        matches = self.satellites[self.satellites['satellite_name'].str.contains(name, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
    
    def search_satellites(self, query: str = None, operator: str = None, 
                         orbit_type: str = None, satellite_type: str = None) -> pd.DataFrame:
        """Search satellites by various criteria"""
        result = self.satellites.copy()
        
        if query:
            result = result[result['satellite_name'].str.contains(query, case=False, na=False)]
        
        if operator:
            result = result[result['operator'].str.contains(operator, case=False, na=False)]
        
        if orbit_type:
            result = result[result['orbit_type'] == orbit_type]
        
        if satellite_type:
            result = result[result['satellite_type'].str.contains(satellite_type, case=False, na=False)]
        
        return result
    
    def get_portfolio_satellites(self, operator: str) -> pd.DataFrame:
        """Get all satellites for a specific operator"""
        return self.satellites[self.satellites['operator'].str.contains(operator, case=False, na=False)]
    
    def list_unique_operators(self) -> List[str]:
        """Get list of unique operators"""
        return sorted(self.satellites['operator'].unique())
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            'total_satellites': len(self.satellites),
            'orbit_distribution': self.satellites['orbit_type'].value_counts().to_dict(),
            'type_distribution': self.satellites['satellite_type'].value_counts().to_dict(),
            'operator_count': self.satellites['operator'].nunique(),
            'total_value': self.satellites['value_usd'].sum(),
            'average_value': self.satellites['value_usd'].mean(),
            'launch_years': f"{self.satellites['launch_year'].min()}-{self.satellites['launch_year'].max()}"
        }
        return stats


class SpaceWeatherForecaster:
    """Advanced ML model for space weather forecasting"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_columns = []
        self.training_history = []
        self.model_type = None  # 'nn' or 'rf'
        
    def prepare_features(self, data: pd.DataFrame, target_hours: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series features for forecasting"""
        print(f"🔧 Preparing features for {target_hours}-hour forecasting...")
        
        # Select relevant features
        feature_cols = ['solar_wind_speed', 'imf_bz', 'density', 'proton_flux', 
                       'bz_negative', 'high_speed_wind', 'sep_event',
                       'hour', 'day_of_year', 'solar_cycle_phase']
        
        self.feature_columns = feature_cols
        target_col = 'kp_index'
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Create sequences for time series prediction
        sequence_length = 24  # Use 24 hours of history (8 data points at 3-hour intervals)
        X, y = [], []
        
        for i in range(sequence_length, len(data) - target_hours//3):  # target_hours in 3-hour intervals
            # Features: past sequence_length points
            X.append(data[feature_cols].iloc[i-sequence_length:i].values.flatten())
            
            # Target: Kp index target_hours ahead
            target_idx = i + target_hours//3
            if target_idx < len(data):
                y.append(data[target_col].iloc[target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Created {len(X)} training samples with {X.shape[1]} features each")
        return X, y
    
    def train_model(self, data: pd.DataFrame, target_hours: int = 24):
        """Train the forecasting model"""
        print(f"🤖 Training space weather forecasting model for {target_hours}-hour ahead prediction...")
        
        # Prepare data
        X, y = self.prepare_features(data, target_hours)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training. Need at least 100 samples.")
        
        # Train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Prefer neural network if TensorFlow is available; otherwise fall back to RandomForest
        if tf is not None and keras is not None and layers is not None:
            input_dim = X_train_scaled.shape[1]
            dropout_rate = 0.2

            # Build a simple dense network with MC Dropout for uncertainty
            inputs = keras.Input(shape=(input_dim,))
            x = layers.Dense(256, activation="relu")(inputs)
            x = layers.Dropout(dropout_rate)(x, training=True)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x, training=True)
            x = layers.Dense(64, activation="relu")(x)
            outputs = layers.Dense(1, activation="linear")(x)
            model = keras.Model(inputs=inputs, outputs=outputs)

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          loss="mse",
                          metrics=["mae"])

            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            ]

            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=200,
                batch_size=64,
                verbose=0,
                callbacks=callbacks
            )

            # Evaluate
            y_pred_train = model.predict(X_train_scaled, verbose=0).ravel()
            y_pred_test = model.predict(X_test_scaled, verbose=0).ravel()

            self.model = model
            self.model_type = 'nn'
        else:
            # Train Random Forest fallback
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
        
        # Evaluate model
        # Metrics already computed above for both branches
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Baseline (persistence) model
        baseline_pred = np.full_like(y_test, np.mean(y_train))
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        
        self.training_history.append({
            'target_hours': target_hours,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'baseline_mse': baseline_mse,
            'improvement': (baseline_mse - test_mse) / baseline_mse * 100,
            'samples': len(X),
            'features': X.shape[1]
        })
        
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
        print(f"   Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
        print(f"   Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        print(f"   Baseline MSE: {baseline_mse:.4f}")
        print(f"   Improvement over baseline: {(baseline_mse - test_mse) / baseline_mse * 100:.1f}%")
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'baseline_mse': baseline_mse,
            'improvement': (baseline_mse - test_mse) / baseline_mse * 100
        }
    
    def forecast_storm(self, current_data: pd.DataFrame, forecast_hours: List[int] = [24, 48, 72]) -> Dict:
        """Generate space weather forecasts with uncertainty bounds"""
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        print(f"🔮 Generating space weather forecast for next {max(forecast_hours)} hours...")
        
        # Use the last sequence for prediction
        sequence_length = 24
        if len(current_data) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} data points for forecasting")
        
        # Get latest data
        latest_data = current_data.tail(sequence_length)
        
        forecasts = {}
        
        for hours in forecast_hours:
            # Prepare features
            X_forecast = latest_data[self.feature_columns].values.flatten().reshape(1, -1)
            X_forecast_scaled = self.scaler.transform(X_forecast)
            
            if self.model_type == 'nn' and self.model is not None:
                # Monte Carlo Dropout: multiple stochastic forward passes
                mc_samples = 50
                preds = []
                for _ in range(mc_samples):
                    y_hat = self.model(X_forecast_scaled, training=True).numpy().ravel()[0]
                    preds.append(y_hat)
                pred_array = np.array(preds)
            else:
                # RandomForest uncertainty from tree ensemble
                pred_array = np.array([tree.predict(X_forecast_scaled)[0] for tree in self.model.estimators_])
            
            # Statistics
            kp_pred = np.mean(pred_array)
            kp_std = np.std(pred_array)
            kp_lower = np.percentile(pred_array, 5)
            kp_upper = np.percentile(pred_array, 95)
            
            # Storm probabilities
            prob_minor_storm = np.mean(pred_array >= 5)  # Kp >= 5
            prob_major_storm = np.mean(pred_array >= 6)  # Kp >= 6
            prob_severe_storm = np.mean(pred_array >= 7)  # Kp >= 7
            prob_extreme_storm = np.mean(pred_array >= 8)  # Kp >= 8
            
            forecasts[f'{hours}h'] = {
                'kp_predicted': float(np.clip(kp_pred, 0, 9)),
                'kp_std': float(kp_std),
                'confidence_interval': {
                    'lower': float(np.clip(kp_lower, 0, 9)),
                    'upper': float(np.clip(kp_upper, 0, 9))
                },
                'storm_probabilities': {
                    'minor_storm_kp5': float(prob_minor_storm),
                    'major_storm_kp6': float(prob_major_storm),
                    'severe_storm_kp7': float(prob_severe_storm),
                    'extreme_storm_kp8': float(prob_extreme_storm)
                },
                'forecast_time': datetime.now() + timedelta(hours=hours),
                'confidence_level': 90
            }
        
        print(f"✅ Generated forecasts for {len(forecast_hours)} time horizons")
        return forecasts


class RiskImpactModeler:
    """Models the impact of space weather on satellite assets"""
    
    def __init__(self, satellite_db: SatelliteDatabase):
        self.satellite_db = satellite_db
        self.risk_models = self._initialize_risk_models()
        
    def _initialize_risk_models(self) -> Dict:
        """Initialize risk models for different satellite components and orbits"""
        return {
            'orbit_vulnerability': {
                'LEO': 1.2,    # Higher vulnerability due to atmospheric drag, charging
                'MEO': 1.0,    # Moderate vulnerability
                'GEO': 0.8,    # Lower vulnerability but high-value assets
                'HEO': 1.1,    # Van Allen belt exposure
                'SSO': 1.15    # Polar regions, aurora effects
            },
            'component_sensitivity': {
                'solar_panels': 0.3,
                'electronics': 0.4,
                'communications': 0.2,
                'attitude_control': 0.1
            },
            'storm_impact_factors': {
                'minor': {'multiplier': 1.1, 'base_prob': 0.05},
                'major': {'multiplier': 1.5, 'base_prob': 0.15},
                'severe': {'multiplier': 2.5, 'base_prob': 0.35},
                'extreme': {'multiplier': 4.0, 'base_prob': 0.60}
            }
        }
    
    def calculate_anomaly_probability(self, kp_forecast: float, satellite_info: Dict) -> Dict:
        """Calculate probability of satellite anomaly based on Kp forecast and satellite characteristics"""
        
        # Base anomaly probability (empirical data from spacecraft anomaly databases)
        base_daily_anomaly_rate = 0.001  # 0.1% per day for normal conditions
        
        # Adjust for storm intensity
        if kp_forecast >= 8:
            storm_factor = self.risk_models['storm_impact_factors']['extreme']['multiplier']
        elif kp_forecast >= 7:
            storm_factor = self.risk_models['storm_impact_factors']['severe']['multiplier']
        elif kp_forecast >= 6:
            storm_factor = self.risk_models['storm_impact_factors']['major']['multiplier']
        elif kp_forecast >= 5:
            storm_factor = self.risk_models['storm_impact_factors']['minor']['multiplier']
        else:
            storm_factor = 1.0
        
        # Orbit vulnerability adjustment
        orbit_factor = self.risk_models['orbit_vulnerability'].get(satellite_info['orbit_type'], 1.0)
        
        # Shielding protection factor
        shielding_factor = satellite_info['shielding_factor']
        protection_factor = (2.0 - shielding_factor)  # Better shielding = lower risk
        
        # Age factor (older satellites more vulnerable)
        current_year = datetime.now().year
        age = current_year - satellite_info['launch_year']
        age_factor = 1.0 + (age * 0.05)  # 5% increase per year
        
        # Mass factor (larger satellites may be more robust)
        mass_kg = satellite_info['mass_kg']
        if mass_kg < 100:  # Small satellites
            mass_factor = 1.3
        elif mass_kg < 1000:  # Medium satellites
            mass_factor = 1.1
        else:  # Large satellites
            mass_factor = 0.9
        
        # Calculate final anomaly probability
        anomaly_prob = (base_daily_anomaly_rate * storm_factor * orbit_factor * 
                       protection_factor * age_factor * mass_factor)
        
        # Cap at reasonable maximum
        anomaly_prob = min(anomaly_prob, 0.8)
        
        return {
            'anomaly_probability': anomaly_prob,
            'factors': {
                'base_rate': base_daily_anomaly_rate,
                'storm_factor': storm_factor,
                'orbit_factor': orbit_factor,
                'protection_factor': protection_factor,
                'age_factor': age_factor,
                'mass_factor': mass_factor
            }
        }
    
    def estimate_downtime_distribution(self, anomaly_prob: float, satellite_info: Dict) -> Dict:
        """Estimate downtime distribution if anomaly occurs"""
        
        # Downtime depends on satellite type and anomaly severity
        sat_type = satellite_info['satellite_type']
        orbit_type = satellite_info['orbit_type']
        
        # Base downtime parameters (hours)
        if sat_type in ['Navigation', 'Space Station']:
            # Critical systems - fast recovery procedures
            base_downtime = {'mean': 4, 'std': 2, 'max': 48}
        elif sat_type in ['Communications', 'Earth Observation']:
            # Commercial systems - moderate recovery
            base_downtime = {'mean': 12, 'std': 6, 'max': 168}  # Up to 1 week
        else:
            # Other systems - variable recovery
            base_downtime = {'mean': 24, 'std': 12, 'max': 336}  # Up to 2 weeks
        
        # Adjust for orbit accessibility (recovery complexity)
        if orbit_type == 'GEO':
            # Harder to service, longer potential downtime
            base_downtime['mean'] *= 1.5
            base_downtime['max'] *= 2
        elif orbit_type == 'LEO':
            # Easier to replace/service
            base_downtime['mean'] *= 0.8
        
        return base_downtime
    
    def calculate_financial_impact(self, satellite_info: Dict, anomaly_prob: float, 
                                 downtime_params: Dict) -> Dict:
        """Calculate expected financial losses"""
        
        satellite_value = satellite_info['value_usd']
        
        # Revenue loss per hour (estimated based on satellite type and value)
        sat_type = satellite_info['satellite_type']
        if sat_type == 'Communications':
            hourly_revenue = satellite_value * 0.0001  # 0.01% per hour
        elif sat_type == 'Earth Observation':
            hourly_revenue = satellite_value * 0.00005  # 0.005% per hour
        elif sat_type == 'Navigation':
            hourly_revenue = satellite_value * 0.0002  # 0.02% per hour (critical)
        else:
            hourly_revenue = satellite_value * 0.00003  # 0.003% per hour
        
        # Monte Carlo simulation for loss distribution
        np.random.seed(42)
        n_simulations = 10000
        
        losses = []
        for _ in range(n_simulations):
            # Does anomaly occur?
            if np.random.random() < anomaly_prob:
                # Sample downtime from log-normal distribution
                downtime_hours = np.random.lognormal(
                    np.log(downtime_params['mean']), 
                    np.log(1 + downtime_params['std']/downtime_params['mean'])
                )
                downtime_hours = min(downtime_hours, downtime_params['max'])
                
                # Calculate loss
                revenue_loss = hourly_revenue * downtime_hours
                
                # Add repair/replacement costs (probability-weighted)
                if np.random.random() < 0.1:  # 10% chance of major damage
                    repair_cost = satellite_value * np.random.uniform(0.1, 0.5)
                elif np.random.random() < 0.05:  # 5% chance of total loss
                    repair_cost = satellite_value
                else:
                    repair_cost = satellite_value * np.random.uniform(0.01, 0.1)
                
                total_loss = revenue_loss + repair_cost
            else:
                total_loss = 0
            
            losses.append(total_loss)
        
        losses = np.array(losses)
        
        # Calculate statistics
        expected_loss = np.mean(losses)
        loss_std = np.std(losses)
        var_95 = np.percentile(losses, 95)  # Value at Risk (95th percentile)
        var_99 = np.percentile(losses, 99)  # Value at Risk (99th percentile)
        max_loss = np.max(losses)
        
        return {
            'expected_loss': expected_loss,
            'loss_std': loss_std,
            'loss_distribution': losses.tolist(),
            'quantiles': {
                '50th': float(np.percentile(losses, 50)),
                '75th': float(np.percentile(losses, 75)),
                '90th': float(np.percentile(losses, 90)),
                '95th': float(var_95),
                '99th': float(var_99)
            },
            'var_95': var_95,
            'var_99': var_99,
            'max_possible_loss': max_loss,
            'probability_of_loss': anomaly_prob,
            'expected_downtime_hours': downtime_params['mean']
        }
    
    def map_forecast_to_impact(self, forecast: Dict, satellite_info: Dict) -> Dict:
        """Map space weather forecast to satellite impact assessment"""
        
        print(f"📊 Analyzing impact for satellite: {satellite_info['satellite_name']}")
        
        impact_assessment = {}
        
        for time_horizon, forecast_data in forecast.items():
            kp_pred = forecast_data['kp_predicted']
            storm_probs = forecast_data['storm_probabilities']
            
            # Calculate anomaly probability
            anomaly_result = self.calculate_anomaly_probability(kp_pred, satellite_info)
            anomaly_prob = anomaly_result['anomaly_probability']
            
            # Estimate downtime if anomaly occurs
            downtime_params = self.estimate_downtime_distribution(anomaly_prob, satellite_info)
            
            # Calculate financial impact
            financial_impact = self.calculate_financial_impact(
                satellite_info, anomaly_prob, downtime_params
            )
            
            impact_assessment[time_horizon] = {
                'forecast_kp': kp_pred,
                'anomaly_probability': anomaly_prob,
                'anomaly_factors': anomaly_result['factors'],
                'expected_downtime_hours': downtime_params['mean'],
                'financial_impact': financial_impact,
                'risk_level': self._classify_risk_level(anomaly_prob),
                'confidence': forecast_data.get('confidence_level', 90)
            }
        
        return impact_assessment

    def _classify_risk_level(self, anomaly_prob: float) -> str:
        """Classify risk level based on anomaly probability"""
        if anomaly_prob >= 0.5:
            return "EXTREME"
        elif anomaly_prob >= 0.2:
            return "HIGH"
        elif anomaly_prob >= 0.1:
            return "MODERATE"
        elif anomaly_prob >= 0.05:
            return "LOW"
        else:
            return "MINIMAL"


# =====================
# Agentic AI Integration
# =====================
class CosmicAgentFactory:
    """Creates a LangChain agent wired to the forecasting and risk tools.

    This is optional and only available if LangChain and an LLM provider are installed and configured.
    """

    def __init__(self, forecaster: SpaceWeatherForecaster, risk_modeler: 'RiskImpactModeler',
                 llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.forecaster = forecaster
        self.risk_modeler = risk_modeler
        self.llm_model = llm_model
        self.temperature = temperature

    def is_available(self) -> bool:
        return all([ChatOpenAI, AgentExecutor, create_tool_calling_agent, Tool, ChatPromptTemplate])

    def build_tools(self) -> List[Any]:
        tools: List[Any] = []
        if Tool is None:
            return tools

        def tool_train(input_json: str) -> str:
            payload = json.loads(input_json)
            df = pd.DataFrame(payload["data"])  # expects list-of-dicts rows
            target_hours = int(payload.get("target_hours", 24))
            metrics = self.forecaster.train_model(df, target_hours)
            return json.dumps({"status": "trained", "metrics": metrics})

        def tool_forecast(input_json: str) -> str:
            payload = json.loads(input_json)
            df = pd.DataFrame(payload["current_data"])  # expects last 24 rows
            horizons = payload.get("forecast_hours", [24, 48, 72])
            result = self.forecaster.forecast_storm(df, horizons)
            return json.dumps(result, default=str)

        def tool_assess(input_json: str) -> str:
            payload = json.loads(input_json)
            forecast = payload["forecast"]
            satellite_info = payload["satellite_info"]
            result = self.risk_modeler.map_forecast_to_impact(forecast, satellite_info)
            return json.dumps(result, default=str)

        tools.append(Tool(
            name="train_forecaster",
            description="Train the space weather forecaster. Input JSON with keys: data (rows), target_hours.",
            func=tool_train
        ))
        tools.append(Tool(
            name="generate_forecast",
            description="Generate Kp forecast. Input JSON with keys: current_data (rows), forecast_hours.",
            func=tool_forecast
        ))
        tools.append(Tool(
            name="assess_risk",
            description="Assess satellite impact risk. Input JSON with forecast and satellite_info.",
            func=tool_assess
        ))

        return tools

    def create_agent(self):
        if not self.is_available():
            raise RuntimeError("LangChain agent dependencies not available. Install langchain, langchain-openai.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are Cosmic Agent, an expert in space weather forecasting and satellite risk. "
                "Use the provided tools to train models, generate forecasts, and assess risk. "
                "Be concise and return structured JSON answers when possible."
            )),
            ("human", "{input}")
        ])

        llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        tools = self.build_tools()
        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        return executor


def create_cosmic_agent(forecaster: SpaceWeatherForecaster, risk_modeler: RiskImpactModeler,
                        model: str = "gpt-4o-mini", temperature: float = 0.1):
    """Convenience factory function to create the agent if deps are available."""
    factory = CosmicAgentFactory(forecaster, risk_modeler, llm_model=model, temperature=temperature)
    if not factory.is_available():
        print("⚠️ LangChain agent not available. Please install required packages and set API keys.")
        return None
    return factory.create_agent()


class InsurancePricer:
    """Calculates insurance premiums based on risk assessment"""
    
    def __init__(self):
        self.pricing_parameters = {
            'loading_factor': 0.25,  # 25% loading for expenses and profit
            'risk_free_rate': 0.03,  # 3% risk-free rate
            'capital_charge': 0.15,  # 15% capital charge
            'confidence_level': 0.99  # 99% confidence for capital requirements
        }
    
    def calculate_premium(self, impact_assessment: Dict, satellite_info: Dict, 
                         coverage_period_days: int = 1) -> Dict:
        """Calculate insurance premium with detailed breakdown"""
        
        print(f"💰 Calculating insurance premium for {coverage_period_days}-day coverage...")
        
        premiums = {}
        
        for time_horizon, impact_data in impact_assessment.items():
            financial_impact = impact_data['financial_impact']
            expected_loss = financial_impact['expected_loss']
            var_99 = financial_impact['var_99']
            
            # Base premium (expected loss)
            base_premium = expected_loss * coverage_period_days
            
            # Loading for expenses and profit
            loaded_premium = base_premium * (1 + self.pricing_parameters['loading_factor'])
            
            # Capital charge (based on tail risk)
            tail_risk = var_99 - expected_loss
            capital_requirement = tail_risk * self.pricing_parameters['capital_charge']
            capital_charge = capital_requirement * self.pricing_parameters['risk_free_rate'] * (coverage_period_days / 365)
            
            # Total premium
            total_premium = loaded_premium + capital_charge
            
            # Premium as percentage of satellite value
            premium_rate = (total_premium / satellite_info['value_usd']) * 100
            
            # Confidence intervals (based on loss distribution uncertainty)
            loss_std = financial_impact['loss_std']
            premium_std = loss_std * (1 + self.pricing_parameters['loading_factor'])
            
            premium_ci_lower = total_premium - 1.96 * premium_std  # 95% CI
            premium_ci_upper = total_premium + 1.96 * premium_std
            
            premiums[time_horizon] = {
                'base_premium': base_premium,
                'loaded_premium': loaded_premium,
                'capital_charge': capital_charge,
                'total_premium': total_premium,
                'premium_rate_percent': premium_rate,
                'confidence_interval': {
                    'lower': max(0, premium_ci_lower),
                    'upper': premium_ci_upper
                },
                'coverage_days': coverage_period_days,
                'deductible_recommended': expected_loss * 0.1,  # 10% of expected loss
                'policy_limits_recommended': var_99 * 2,  # 2x VaR99 for adequate coverage
                'pricing_components': {
                    'expected_loss': expected_loss,
                    'loading_factor': self.pricing_parameters['loading_factor'],
                    'capital_charge_rate': self.pricing_parameters['capital_charge'],
                    'risk_free_rate': self.pricing_parameters['risk_free_rate']
                },
                'assumptions': [
                    f"Coverage period: {coverage_period_days} days",
                    f"Loading factor: {self.pricing_parameters['loading_factor']*100}% for expenses and profit",
                    f"Capital charge: {self.pricing_parameters['capital_charge']*100}% of tail risk",
                    f"Based on {self.pricing_parameters['confidence_level']*100}% confidence level",
                    "Premium covers anomaly-related losses including downtime and repairs"
                ]
            }
        
        return premiums


class SpaceWeatherInsuranceSystem:
    """Main system orchestrating all components"""
    
    def __init__(self):
        print("🚀 Initializing Space Weather Insurance System...")
        
        # Initialize components
        self.data_ingester = SpaceWeatherDataIngester()
        self.satellite_db = SatelliteDatabase()
        self.forecaster = SpaceWeatherForecaster()
        self.risk_modeler = RiskImpactModeler(self.satellite_db)
        self.pricer = InsurancePricer()
        
        # System state
        self.space_weather_data = None
        self.is_system_ready = False
        self.last_data_update = None
        
        print("✅ System initialized successfully!")
    
    def initialize_system(self):
        """Initialize system with data and trained models"""
        print("\n🔄 Initializing system with comprehensive data...")
        
        # Load space weather data
        print("📡 Loading space weather data...")
        self.space_weather_data = self.data_ingester.create_comprehensive_dataset()
        
        # Train forecasting model
        print("🤖 Training forecasting models...")
        self.forecaster.train_model(self.space_weather_data, target_hours=24)
        
        # Print database statistics
        stats = self.satellite_db.get_database_stats()
        print(f"\n📊 Satellite Database Statistics:")
        print(f"   Total Satellites: {stats['total_satellites']:,}")
        print(f"   Total Fleet Value: ${stats['total_value']/1e9:.1f}B")
        print(f"   Average Satellite Value: ${stats['average_value']/1e6:.1f}M")
        print(f"   Unique Operators: {stats['operator_count']}")
        print(f"   Launch Years: {stats['launch_years']}")
        
        print(f"\n🛰️ Orbit Distribution:")
        for orbit, count in stats['orbit_distribution'].items():
            percentage = (count / stats['total_satellites']) * 100
            print(f"   {orbit}: {count:,} satellites ({percentage:.1f}%)")
        
        self.is_system_ready = True
        self.last_data_update = datetime.now()
        
        print("\n✅ System ready for operations!")
    
    def search_satellites_interactive(self) -> str:
        """Interactive satellite search"""
        print("\n🔍 Satellite Search Options:")
        print("1. Search by name (partial match)")
        print("2. Search by operator")
        print("3. Search by orbit type")
        print("4. Search by satellite type")
        print("5. List famous satellites")
        print("6. Random satellite")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            name = input("Enter satellite name (partial match): ").strip()
            results = self.satellite_db.search_satellites(query=name)
        elif choice == '2':
            operator = input("Enter operator name: ").strip()
            results = self.satellite_db.search_satellites(operator=operator)
        elif choice == '3':
            print("Available orbit types: LEO, MEO, GEO, HEO, SSO")
            orbit = input("Enter orbit type: ").strip().upper()
            results = self.satellite_db.search_satellites(orbit_type=orbit)
        elif choice == '4':
            print("Available types: Communications, Earth Observation, Navigation, Scientific, Military/Government")
            sat_type = input("Enter satellite type: ").strip()
            results = self.satellite_db.search_satellites(satellite_type=sat_type)
        elif choice == '5':
            famous_names = ['Hubble', 'ISS', 'Starlink']
            results = pd.concat([self.satellite_db.search_satellites(query=name) for name in famous_names])
        elif choice == '6':
            results = self.satellite_db.satellites.sample(10)
        else:
            print("Invalid choice")
            return None
        
        if results.empty:
            print("No satellites found matching your criteria.")
            return None
        
        # Display results
        print(f"\n🛰️ Found {len(results)} satellite(s):")
        print("-" * 100)
        
        display_cols = ['satellite_name', 'operator', 'satellite_type', 'orbit_type', 'value_usd']
        for idx, (_, sat) in enumerate(results.head(20).iterrows()):  # Show max 20 results
            value_str = f"${sat['value_usd']/1e6:.1f}M" if sat['value_usd'] >= 1e6 else f"${sat['value_usd']/1e3:.1f}K"
            print(f"{idx+1:2}. {sat['satellite_name']:<30} | {sat['operator']:<20} | {sat['orbit_type']:<4} | {value_str:>10}")
        
        if len(results) > 20:
            print(f"... and {len(results)-20} more satellites")
        
        # Let user select
        try:
            selection = int(input(f"\nSelect satellite number (1-{min(20, len(results))}): ")) - 1
            if 0 <= selection < min(20, len(results)):
                selected_satellite = results.iloc[selection]['satellite_name']
                print(f"\n✅ Selected: {selected_satellite}")
                return selected_satellite
            else:
                print("Invalid selection")
                return None
        except ValueError:
            print("Invalid input")
            return None
    
    def generate_risk_assessment(self, satellite_name: str, forecast_hours: List[int] = [24, 48, 72]) -> Dict:
        """Generate comprehensive risk assessment for a satellite"""
        
        if not self.is_system_ready:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        # Get satellite information
        satellite_info = self.satellite_db.get_satellite_by_name(satellite_name)
        if not satellite_info:
            raise ValueError(f"Satellite '{satellite_name}' not found in database")
        
        print(f"\n🛰️ Generating risk assessment for: {satellite_info['satellite_name']}")
        print(f"   Operator: {satellite_info['operator']}")
        print(f"   Type: {satellite_info['satellite_type']}")
        print(f"   Orbit: {satellite_info['orbit_type']} ({satellite_info['perigee_km']}-{satellite_info['apogee_km']} km)")
        print(f"   Value: ${satellite_info['value_usd']/1e6:.1f}M")
        print(f"   Launch Year: {satellite_info['launch_year']}")
        
        # Generate space weather forecast
        forecast = self.forecaster.forecast_storm(self.space_weather_data, forecast_hours)
        
        # Map forecast to impact
        impact_assessment = self.risk_modeler.map_forecast_to_impact(forecast, satellite_info)
        
        # Calculate insurance premiums
        premiums = self.pricer.calculate_premium(impact_assessment, satellite_info)
        
        # Compile comprehensive assessment
        assessment = {
            'satellite_info': satellite_info,
            'forecast': forecast,
            'impact_assessment': impact_assessment,
            'insurance_premiums': premiums,
            'assessment_timestamp': datetime.now().isoformat(),
            'system_confidence': 'HIGH' if len(self.space_weather_data) > 1000 else 'MODERATE'
        }
        
        return assessment
    
    def print_risk_report(self, assessment: Dict):
        """Print formatted risk assessment report"""
        
        satellite_info = assessment['satellite_info']
        impact_assessment = assessment['impact_assessment']
        premiums = assessment['insurance_premiums']
        
        print("\n" + "="*80)
        print(f"🛰️  SPACE WEATHER RISK ASSESSMENT REPORT")
        print("="*80)
        
        print(f"\n📋 SATELLITE DETAILS:")
        print(f"   Name: {satellite_info['satellite_name']}")
        print(f"   Operator: {satellite_info['operator']}")
        print(f"   Type: {satellite_info['satellite_type']}")
        print(f"   Orbit: {satellite_info['orbit_type']} ({satellite_info['perigee_km']:.0f}-{satellite_info['apogee_km']:.0f} km)")
        print(f"   Mass: {satellite_info['mass_kg']:.0f} kg")
        print(f"   Value: ${satellite_info['value_usd']/1e6:.1f}M")
        print(f"   Shielding Factor: {satellite_info['shielding_factor']:.2f}")
        print(f"   Age: {datetime.now().year - satellite_info['launch_year']} years")
        
        print(f"\n⚡ SPACE WEATHER FORECAST & RISK ANALYSIS:")
        
        for time_horizon in ['24h', '48h', '72h']:
            if time_horizon in impact_assessment:
                impact = impact_assessment[time_horizon]
                premium = premiums[time_horizon]
                
                print(f"\n   📅 {time_horizon.upper()} FORECAST:")
                print(f"      Predicted Kp Index: {impact['forecast_kp']:.1f}")
                print(f"      Risk Level: {impact['risk_level']}")
                print(f"      Anomaly Probability: {impact['anomaly_probability']*100:.2f}%")
                print(f"      Expected Downtime: {impact['expected_downtime_hours']:.1f} hours")
                
                financial = impact['financial_impact']
                print(f"      Expected Loss: ${financial['expected_loss']:,.0f}")
                print(f"      95% VaR: ${financial['var_95']:,.0f}")
                print(f"      99% VaR: ${financial['var_99']:,.0f}")
        
        print(f"\n💰 INSURANCE PREMIUM CALCULATION:")
        
        # Use 24h forecast for primary pricing
        if '24h' in premiums:
            premium_24h = premiums['24h']
            print(f"   Base Premium (24h): ${premium_24h['base_premium']:,.0f}")
            print(f"   Loaded Premium: ${premium_24h['loaded_premium']:,.0f}")
            print(f"   Capital Charge: ${premium_24h['capital_charge']:,.0f}")
            print(f"   TOTAL PREMIUM: ${premium_24h['total_premium']:,.0f}")
            print(f"   Premium Rate: {premium_24h['premium_rate_percent']:.4f}% of satellite value")
            
            print(f"\n   Confidence Interval (95%):")
            print(f"      Lower: ${premium_24h['confidence_interval']['lower']:,.0f}")
            print(f"      Upper: ${premium_24h['confidence_interval']['upper']:,.0f}")
            
            print(f"\n   Recommended Policy Terms:")
            print(f"      Policy Limit: ${premium_24h['policy_limits_recommended']:,.0f}")
            print(f"      Deductible: ${premium_24h['deductible_recommended']:,.0f}")
        
        print(f"\n📊 RISK FACTORS BREAKDOWN:")
        if '24h' in impact_assessment:
            factors = impact_assessment['24h']['anomaly_factors']
            print(f"   Storm Factor: {factors['storm_factor']:.2f}x")
            print(f"   Orbit Factor: {factors['orbit_factor']:.2f}x")
            print(f"   Age Factor: {factors['age_factor']:.2f}x")
            print(f"   Mass Factor: {factors['mass_factor']:.2f}x")
            print(f"   Protection Factor: {factors['protection_factor']:.2f}x")
        
        print(f"\n📝 KEY ASSUMPTIONS:")
        if '24h' in premiums:
            for assumption in premiums['24h']['assumptions']:
                print(f"   • {assumption}")
        
        print(f"\n⏰ Report Generated: {assessment['assessment_timestamp']}")
        print(f"💪 System Confidence: {assessment['system_confidence']}")
        
        print("\n" + "="*80)
    
    def run_interactive_session(self):
        """Run interactive session for satellite risk assessment"""
        
        if not self.is_system_ready:
            print("⚠️ System not ready. Initializing...")
            self.initialize_system()
        
        print("\n🌟 Welcome to the Space Weather Insurance System!")
        print("This system provides real-time space weather risk assessment and insurance pricing for satellites.")
        
        while True:
            print("\n" + "="*60)
            print("🛰️  SPACE WEATHER INSURANCE TERMINAL")
            print("="*60)
            print("1. Search and analyze satellite")
            print("2. Database statistics")
            print("3. Current space weather conditions") 
            print("4. Portfolio analysis (by operator)")
            print("5. Exit")
            
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    satellite_name = self.search_satellites_interactive()
                    if satellite_name:
                        try:
                            assessment = self.generate_risk_assessment(satellite_name)
                            self.print_risk_report(assessment)
                            
                            # Ask if user wants to save report
                            save = input("\nSave report to file? (y/n): ").strip().lower()
                            if save == 'y':
                                filename = f"risk_report_{satellite_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                with open(filename, 'w') as f:
                                    json.dump(assessment, f, indent=2, default=str)
                                print(f"📁 Report saved to {filename}")
                        
                        except Exception as e:
                            print(f"❌ Error generating assessment: {e}")
                
                elif choice == '2':
                    stats = self.satellite_db.get_database_stats()
                    print(f"\n📊 DATABASE STATISTICS:")
                    print(f"   Total Satellites: {stats['total_satellites']:,}")
                    print(f"   Total Fleet Value: ${stats['total_value']/1e12:.2f}T")
                    print(f"   Average Value: ${stats['average_value']/1e6:.1f}M")
                    print(f"   Operators: {stats['operator_count']}")
                    
                    print(f"\n🛰️ By Orbit Type:")
                    for orbit, count in stats['orbit_distribution'].items():
                        pct = count/stats['total_satellites']*100
                        print(f"   {orbit}: {count:,} ({pct:.1f}%)")
                
                elif choice == '3':
                    if self.space_weather_data is not None:
                        latest = self.space_weather_data.tail(1).iloc[0]
                        print(f"\n🌌 CURRENT SPACE WEATHER CONDITIONS:")
                        print(f"   Kp Index: {latest['kp_index']:.1f}")
                        print(f"   Solar Wind Speed: {latest['solar_wind_speed']:.0f} km/s")
                        print(f"   IMF Bz: {latest['imf_bz']:.1f} nT")
                        print(f"   Proton Flux: {latest['proton_flux']:.1f} pfu")
                        print(f"   Storm Activity: {'ACTIVE' if latest['kp_index'] >= 5 else 'QUIET'}")
                
                elif choice == '4':
                    operators = self.satellite_db.list_unique_operators()
                    print(f"\n📋 Available Operators ({len(operators)}):")
                    for i, operator in enumerate(operators[:20]):  # Show first 20
                        count = len(self.satellite_db.search_satellites(operator=operator))
                        print(f"   {i+1:2}. {operator} ({count} satellites)")
                    
                    if len(operators) > 20:
                        print(f"   ... and {len(operators)-20} more operators")
                    
                    try:
                        selection = input("\nEnter operator name or number: ").strip()
                        
                        # Check if it's a number
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < min(20, len(operators)):
                                selected_operator = operators[idx]
                            else:
                                print("Invalid selection")
                                continue
                        else:
                            selected_operator = selection
                        
                        # Get portfolio
                        portfolio = self.satellite_db.get_portfolio_satellites(selected_operator)
                        
                        if portfolio.empty:
                            print(f"No satellites found for operator: {selected_operator}")
                            continue
                        
                        print(f"\n🚀 PORTFOLIO ANALYSIS: {selected_operator}")
                        print(f"   Total Satellites: {len(portfolio)}")
                        print(f"   Total Value: ${portfolio['value_usd'].sum()/1e9:.1f}B")
                        print(f"   Average Value: ${portfolio['value_usd'].mean()/1e6:.1f}M")
                        
                        # Analyze top 5 most valuable satellites
                        top_sats = portfolio.nlargest(5, 'value_usd')
                        print(f"\n💎 Top 5 Most Valuable Satellites:")
                        for idx, (_, sat) in enumerate(top_sats.iterrows()):
                            print(f"   {idx+1}. {sat['satellite_name']} - ${sat['value_usd']/1e6:.1f}M ({sat['orbit_type']})")
                        
                        # Quick risk assessment for portfolio
                        if len(portfolio) <= 10:
                            analyze_all = input(f"\nAnalyze all {len(portfolio)} satellites? (y/n): ").strip().lower()
                            if analyze_all == 'y':
                                total_premium = 0
                                total_risk = 0
                                
                                print(f"\n📊 PORTFOLIO RISK ANALYSIS:")
                                for _, satellite in portfolio.iterrows():
                                    try:
                                        assessment = self.generate_risk_assessment(satellite['satellite_name'], [24])
                                        if '24h' in assessment['insurance_premiums']:
                                            premium = assessment['insurance_premiums']['24h']['total_premium']
                                            risk_level = assessment['impact_assessment']['24h']['risk_level']
                                            anomaly_prob = assessment['impact_assessment']['24h']['anomaly_probability']
                                            
                                            total_premium += premium
                                            total_risk += anomaly_prob
                                            
                                            print(f"   {satellite['satellite_name']:<30} | ${premium:>8,.0f} | {risk_level:<8} | {anomaly_prob*100:>5.2f}%")
                                    except Exception as e:
                                        print(f"   {satellite['satellite_name']:<30} | ERROR: {str(e)[:50]}")
                                
                                print(f"\n📈 PORTFOLIO SUMMARY:")
                                print(f"   Total Daily Premium: ${total_premium:,.0f}")
                                print(f"   Average Risk Level: {total_risk/len(portfolio)*100:.2f}%")
                                print(f"   Annual Premium Est.: ${total_premium*365:,.0f}")
                        else:
                            print(f"\n⚠️ Portfolio too large for full analysis ({len(portfolio)} satellites)")
                            print("   Showing summary statistics only.")
                        
                    except Exception as e:
                        print(f"❌ Error in portfolio analysis: {e}")
                
                elif choice == '5':
                    print("\n👋 Thank you for using the Space Weather Insurance System!")
                    break
                
                else:
                    print("❌ Invalid option. Please select 1-5.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("Please try again or contact system administrator.")
    
    def run_alert_monitoring(self, threshold_kp: float = 6.0, check_interval: int = 3600):
        """Run continuous monitoring with alerts"""
        
        print(f"🚨 Starting alert monitoring (Kp threshold: {threshold_kp}, check every {check_interval}s)")
        
        try:
            while True:
                # Generate current forecast
                forecast = self.forecaster.forecast_storm(self.space_weather_data, [24])
                current_kp = forecast['24h']['kp_predicted']
                storm_prob = forecast['24h']['storm_probabilities']['severe_storm_kp7']
                
                if current_kp >= threshold_kp or storm_prob >= 0.3:
                    print(f"\n🚨 SPACE WEATHER ALERT!")
                    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Predicted Kp: {current_kp:.1f}")
                    print(f"   Severe Storm Probability: {storm_prob*100:.1f}%")
                    print(f"   Recommendation: Review high-value satellite portfolios")
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n🛑 Alert monitoring stopped.")


def main():
    """Main entry point for the Space Weather Insurance System"""
    
    print("🌟 Welcome to the Cosmic Weather Insurance System")
    print("="*60)
    print("A comprehensive system for space weather risk assessment and insurance pricing")
    print("Developed for 24-hour hackathon challenge")
    print("="*60)
    
    # Initialize system
    system = SpaceWeatherInsuranceSystem()
    
    try:
        # Check if user wants quick demo or full interactive session
        print("\n🚀 Choose mode:")
        print("1. Full interactive session")
        print("2. Quick demo with sample satellites")
        print("3. Continuous monitoring mode")
        
        mode = input("Select mode (1-3): ").strip()
        
        if mode == '1':
            system.run_interactive_session()
        
        elif mode == '2':
            # Quick demo
            system.initialize_system()
            
            demo_satellites = ['Hubble Space Telescope', 'Starlink-1007', 'ISS-Zarya']
            
            for sat_name in demo_satellites:
                try:
                    print(f"\n🎯 Demo: Analyzing {sat_name}")
                    assessment = system.generate_risk_assessment(sat_name)
                    system.print_risk_report(assessment)
                    input("\nPress Enter to continue to next satellite...")
                except Exception as e:
                    print(f"❌ Error analyzing {sat_name}: {e}")
        
        elif mode == '3':
            system.initialize_system()
            threshold = float(input("Enter Kp threshold for alerts (default 6.0): ") or "6.0")
            system.run_alert_monitoring(threshold_kp=threshold)
        
        else:
            print("❌ Invalid mode selected")
    
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n👋 Space Weather Insurance System shutting down...")


if __name__ == "__main__":
    main()