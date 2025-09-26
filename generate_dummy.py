import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

print("Starting dummy CSV generation...")

# --- 1. OMNIWeb Data ---
# Solar wind speed, density, IMF Bz, plasma temp.
dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=1000, freq='H'))
omni_data = {
    'Year': dates.year,
    'Day': dates.dayofyear,
    'Hour': dates.hour,
    'BZ_GSM': np.random.uniform(-10, 10, 1000).round(2),
    'Speed': np.random.uniform(300, 800, 1000).round(1),
    'Proton_Density': np.random.uniform(1, 15, 1000).round(2),
    'Plasma_Temp': np.random.uniform(50000, 200000, 1000).round(1)
}
omni_df = pd.DataFrame(omni_data)
# Add some common placeholder values
omni_df.loc[omni_df.sample(frac=0.05).index, 'Speed'] = 999.9
omni_df.to_csv('omni_data.csv', index=False, sep=' ')
print("1. omni_data.csv created.")

# --- 2 & 3. GOES Proton & X-Ray Flux ---
dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=1000, freq='5min'))
proton_flux_data = {
    'time_tag': dates,
    'proton_flux_gt10MeV': np.random.lognormal(mean=0.5, sigma=1.5, size=1000).round(4)
}
xray_flux_data = {
    'time_tag': dates,
    'xray_flux_long': np.random.lognormal(mean=-15, sigma=2, size=1000)
}
pd.DataFrame(proton_flux_data).to_csv('goes_proton_flux.csv', index=False)
pd.DataFrame(xray_flux_data).to_csv('goes_xray_flux.csv', index=False)
print("2. goes_proton_flux.csv created.")
print("3. goes_xray_flux.csv created.")

# --- 4. NOAA Indices (Kp, Dst) ---
dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=1000, freq='3H'))
noaa_indices_data = {
    'datetime': dates,
    'Kp': np.random.randint(0, 9, 1000),
    'Dst': np.random.randint(-150, 20, 1000)
}
pd.DataFrame(noaa_indices_data).to_csv('noaa_indices.csv', index=False)
print("4. noaa_indices.csv created.")

# --- 5. SuperMAG Data ---
dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=1000, freq='T'))
supermag_data = {
    'Date': dates.strftime('%Y-%m-%d'),
    'Time': dates.strftime('%H:%M:%S'),
    'SML': np.random.randint(-1000, -50, 1000),
    'SMU': np.random.randint(50, 500, 1000)
}
pd.DataFrame(supermag_data).to_csv('supermag_data.csv', index=False, sep=' ')
print("5. supermag_data.csv created.")

# --- 6. ESA Data (Similar to NOAA) ---
dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=1000, freq='H'))
esa_data = {
    'datetime': dates,
    'geomagnetic_index': np.random.uniform(0, 8, 1000).round(2)
}
pd.DataFrame(esa_data).to_csv('esa_data.csv', index=False)
print("6. esa_data.csv created.")

# --- 7. Geomagnetic Storm Alerts ---
# Event-based data is different.
alert_times = sorted([datetime.now() - timedelta(hours=random.randint(0, 1000)) for _ in range(50)])
alert_data = {
    'issue_time': alert_times,
    'alert_type': [random.choice(['WATCH', 'WARNING', 'ALERT']) for _ in range(50)],
    'g_level': [f"G{random.randint(1, 5)}" for _ in range(50)]
}
pd.DataFrame(alert_data).to_csv('geomagnetic_storm_alerts.csv', index=False)
print("7. geomagnetic_storm_alerts.csv created.")

# --- 8. UCS Satellite Database ---
# Static data about satellites.
satellite_data = {
    'Name of Satellite': [f'SAT-{i}' for i in range(1000)],
    'Country of Operator/Owner': [random.choice(['USA', 'Russia', 'China', 'ESA', 'Luxembourg']) for _ in range(1000)],
    'Purpose': [random.choice(['Communications', 'Earth Observation', 'Navigation', 'Science']) for _ in range(1000)],
    'Class of Orbit': [random.choice(['LEO', 'MEO', 'GEO', 'Elliptical']) for _ in range(1000)],
    'Date of Launch': [datetime(random.randint(1990, 2024), random.randint(1, 12), random.randint(1, 28)) for _ in range(1000)]
}
pd.DataFrame(satellite_data).to_csv('ucs_satellite_database.csv', index=False)
print("8. ucs_satellite_database.csv created.")

print("\nAll dummy CSV files have been generated successfully!")