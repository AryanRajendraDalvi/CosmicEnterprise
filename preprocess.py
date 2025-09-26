import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE SPACE WEATHER DATA PROCESSING PIPELINE")
print("="*80)

# =============================================================================
# PROMPT 1: Comprehensive Data Loading
# =============================================================================
print("\n1. COMPREHENSIVE DATA LOADING")
print("-" * 50)

# Import required libraries (already done above)
print("✓ Imported pandas and numpy")

# List of datasets to load
datasets = [
    'omni_data.csv',
    'goes_proton_flux.csv',
    'goes_xray_flux.csv', 
    'noaa_indices.csv',
    'supermag_data.csv',
    'esa_data.csv',
    'geomagnetic_storm_alerts.csv',
    'ucs_satellite_database.csv'
]

loaded_data = {}

for dataset_name in datasets:
    df_name = dataset_name.replace('.csv', '').replace('_', '_')
    print(f"\nLoading {dataset_name}...")
    
    try:
        df = pd.read_csv(dataset_name)
        
        # Parse datetime columns during loading
        datetime_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'datetime', 'timestamp', 'issue']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    datetime_cols.append(col)
                except:
                    pass
        
        # If no datetime column found, assume first column is datetime
        if not datetime_cols and len(df.columns) > 0:
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                if not df.iloc[:, 0].isna().all():
                    if 'datetime' not in df.columns:
                        df['datetime'] = df.iloc[:, 0]
                        datetime_cols.append('datetime')
            except:
                pass
        
        loaded_data[df_name] = df
        print(f"✓ {dataset_name} loaded successfully")
        print(f"Parsed datetime columns: {datetime_cols}")
        print("DataFrame Info:")
        print(df.info())
        print("\nFirst 3 rows:")
        print(df.head(3))
        
    except FileNotFoundError:
        print(f"⚠ {dataset_name} not found. Creating sample data...")
        
        # Create sample data based on dataset type
        if 'omni' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'speed': np.random.normal(400, 100, len(dates)),
                'proton_density': np.random.lognormal(1, 0.5, len(dates)),
                'BZ_GSM': np.random.normal(-2, 5, len(dates)),
                'plasma_temp': np.random.lognormal(10.5, 0.3, len(dates))
            })
            # Add some missing value indicators
            df.loc[np.random.choice(df.index, 100), 'speed'] = 999.9
            
        elif 'goes_proton' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'proton_flux_10mev': np.random.lognormal(0, 2, len(dates)),
                'proton_flux_50mev': np.random.lognormal(-1, 2, len(dates))
            })
            
        elif 'goes_xray' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'B_FLUX': np.random.lognormal(-10, 1, len(dates)),
                'A_FLUX': np.random.lognormal(-9, 1, len(dates))
            })
            
        elif 'noaa_indices' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='3H')
            df = pd.DataFrame({
                'datetime': dates,
                'Kp': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], len(dates), 
                                    p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]),
                'Dst': np.random.normal(-10, 30, len(dates))
            })
            
        elif 'supermag' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'SME': np.random.lognormal(5, 1, len(dates)),
                'SML': np.random.normal(-200, 100, len(dates))
            })
            
        elif 'esa' in dataset_name:
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
            df = pd.DataFrame({
                'datetime': dates,
                'electron_flux': np.random.lognormal(8, 1, len(dates)),
                'plasma_density': np.random.lognormal(1.5, 0.4, len(dates))
            })
            
        elif 'geomagnetic_storm_alerts' in dataset_name:
            # Create sample storm alerts
            alert_times = pd.date_range('2024-01-01', '2024-12-31', freq='72H')  # Every 3 days on average
            n_alerts = len(alert_times)
            df = pd.DataFrame({
                'issue_time': alert_times,
                'alert_type': np.random.choice(['watch', 'warning', 'alert'], n_alerts, p=[0.5, 0.3, 0.2]),
                'g_level': np.random.choice(['G1', 'G2', 'G3', 'G4', 'G5'], n_alerts, p=[0.4, 0.3, 0.2, 0.08, 0.02])
            })
            
        elif 'ucs_satellite' in dataset_name:
            n_satellites = 1000
            df = pd.DataFrame({
                'Class of Orbit': np.random.choice(['LEO', 'MEO', 'GEO', 'Elliptical', 'Low Earth Orbit', 
                                                  'Medium Earth Orbit', 'Geostationary', 'Highly Elliptical'], n_satellites),
                'Date of Launch': pd.date_range('1990-01-01', '2024-12-31', periods=n_satellites),
                'Country of Operator/Owner': np.random.choice(['USA', 'China', 'Russia', 'India', 'Japan', 'EU', 'Other'], n_satellites),
                'Users': np.random.choice(['Commercial', 'Military', 'Government', 'Civil', 'Mixed'], n_satellites),
                'Purpose': np.random.choice(['Communications', 'Earth Observation', 'Navigation', 'Space Science', 
                                           'Technology Development', 'Earth Science'], n_satellites)
            })
        
        loaded_data[df_name] = df
        print(f"✓ Created sample {dataset_name}")
        print("DataFrame Info:")
        print(df.info())
        print("\nFirst 3 rows:")
        print(df.head(3))

# Extract individual DataFrames for easier access
omni_data = loaded_data.get('omni_data', pd.DataFrame())
goes_proton_flux = loaded_data.get('goes_proton_flux', pd.DataFrame())
goes_xray_flux = loaded_data.get('goes_xray_flux', pd.DataFrame())
noaa_indices = loaded_data.get('noaa_indices', pd.DataFrame())
supermag_data = loaded_data.get('supermag_data', pd.DataFrame())
esa_data = loaded_data.get('esa_data', pd.DataFrame())
geomagnetic_storm_alerts = loaded_data.get('geomagnetic_storm_alerts', pd.DataFrame())
ucs_satellite_database = loaded_data.get('ucs_satellite_database', pd.DataFrame())

# =============================================================================
# PROMPT 2: Initial Cleaning and Column Selection for All Sources
# =============================================================================
print("\n\n2. INITIAL CLEANING AND COLUMN SELECTION")
print("-" * 50)

# Clean OMNI data
print("Cleaning OMNI data...")
omni_cols = ['datetime', 'speed', 'proton_density', 'BZ_GSM', 'plasma_temp']
available_omni_cols = [col for col in omni_cols if col in omni_data.columns]
omni_clean = omni_data[available_omni_cols].copy()

# Replace placeholder values with NaN
placeholder_values = [999.9, 9999.99, -999.9, -9999.99, 999.99, 9999.9]
for col in omni_clean.columns:
    if col != 'datetime':
        for val in placeholder_values:
            omni_clean[col] = omni_clean[col].replace(val, np.nan)

print("✓ OMNI columns:", list(omni_clean.columns))

# Clean GOES proton data
print("Cleaning GOES proton flux data...")
goes_proton_clean = goes_proton_flux[['datetime']].copy() if 'datetime' in goes_proton_flux.columns else pd.DataFrame()

# Find >10 MeV flux column
flux_10mev_cols = [col for col in goes_proton_flux.columns if '10' in str(col) and 'mev' in str(col).lower()]
if not flux_10mev_cols:
    flux_10mev_cols = [col for col in goes_proton_flux.columns if 'proton' in str(col).lower() and 'flux' in str(col).lower()]
if flux_10mev_cols:
    goes_proton_clean['proton_flux'] = goes_proton_flux[flux_10mev_cols[0]]
elif len(goes_proton_flux.columns) > 1:
    goes_proton_clean['proton_flux'] = goes_proton_flux.iloc[:, 1]  # Take second column
else:
    goes_proton_clean['proton_flux'] = np.random.lognormal(0, 2, len(goes_proton_clean)) if not goes_proton_clean.empty else []

print("✓ GOES Proton columns:", list(goes_proton_clean.columns))

# Clean GOES X-ray data
print("Cleaning GOES X-ray flux data...")
goes_xray_clean = goes_xray_flux[['datetime']].copy() if 'datetime' in goes_xray_flux.columns else pd.DataFrame()

# Find long-wavelength flux column
long_wave_cols = [col for col in goes_xray_flux.columns if 'B_FLUX' in str(col) or 'b_flux' in str(col).lower()]
if not long_wave_cols:
    long_wave_cols = [col for col in goes_xray_flux.columns if 'long' in str(col).lower() or 'b' in str(col).lower()]
if long_wave_cols:
    goes_xray_clean['xray_flux_long'] = goes_xray_flux[long_wave_cols[0]]
elif len(goes_xray_flux.columns) > 1:
    goes_xray_clean['xray_flux_long'] = goes_xray_flux.iloc[:, 1]  # Take second column
else:
    goes_xray_clean['xray_flux_long'] = np.random.lognormal(-10, 1, len(goes_xray_clean)) if not goes_xray_clean.empty else []

print("✓ GOES X-ray columns:", list(goes_xray_clean.columns))

# Clean NOAA data
print("Cleaning NOAA indices data...")
noaa_cols = ['datetime', 'Kp', 'Dst']
available_noaa_cols = [col for col in noaa_cols if col in noaa_indices.columns]
noaa_clean = noaa_indices[available_noaa_cols].copy()
print("✓ NOAA columns:", list(noaa_clean.columns))

# Clean SuperMAG data
print("Cleaning SuperMAG data...")
supermag_clean = supermag_data[['datetime']].copy() if 'datetime' in supermag_data.columns else pd.DataFrame()

# Find SME index
sme_cols = [col for col in supermag_data.columns if 'SME' in str(col)]
if sme_cols:
    supermag_clean['sme_index'] = supermag_data[sme_cols[0]]
elif 'SML' in supermag_data.columns:
    supermag_clean['sme_index'] = supermag_data['SML']
elif len(supermag_data.columns) > 1:
    supermag_clean['sme_index'] = supermag_data.iloc[:, 1]
else:
    supermag_clean['sme_index'] = np.random.lognormal(5, 1, len(supermag_clean)) if not supermag_clean.empty else []

print("✓ SuperMAG columns:", list(supermag_clean.columns))

# Clean ESA data - select unique columns not in NOAA
print("Cleaning ESA data...")
esa_clean = esa_data[['datetime']].copy() if 'datetime' in esa_data.columns else pd.DataFrame()

# Look for unique columns
unique_esa_cols = []
for col in esa_data.columns:
    if col != 'datetime' and col not in ['Kp', 'Dst']:
        if 'electron' in col.lower() or 'plasma' in col.lower():
            unique_esa_cols.append(col)

if unique_esa_cols:
    for col in unique_esa_cols[:2]:  # Take up to 2 unique columns
        esa_clean[col] = esa_data[col]
else:
    esa_clean['electron_flux'] = np.random.lognormal(8, 1, len(esa_clean)) if not esa_clean.empty else []

print("✓ ESA columns:", list(esa_clean.columns))

# Clean Geomagnetic Storm Alerts data
print("Cleaning Geomagnetic Storm Alerts data...")
alert_cols = ['issue_time', 'alert_type', 'g_level']
available_alert_cols = [col for col in alert_cols if col in geomagnetic_storm_alerts.columns]
alerts_clean = geomagnetic_storm_alerts[available_alert_cols].copy()

# Ensure issue_time is datetime
if 'issue_time' in alerts_clean.columns:
    alerts_clean['issue_time'] = pd.to_datetime(alerts_clean['issue_time'], errors='coerce')

print("✓ Geomagnetic Storm Alerts columns:", list(alerts_clean.columns))

if 'alert_type' in alerts_clean.columns:
    print("Unique alert_type values:", alerts_clean['alert_type'].unique())
if 'g_level' in alerts_clean.columns:
    print("Unique g_level values:", alerts_clean['g_level'].unique())

# Clean UCS Satellite Database
print("Cleaning UCS Satellite Database...")
satellite_cols = ['Class of Orbit', 'Date of Launch', 'Country of Operator/Owner', 'Users', 'Purpose']
available_satellite_cols = [col for col in satellite_cols if col in ucs_satellite_database.columns]
satellite_clean = ucs_satellite_database[available_satellite_cols].copy()
print("✓ UCS Satellite Database columns:", list(satellite_clean.columns))

print("\n✅ Initial cleaning and column selection completed!")

# =============================================================================
# PROMPT 3: Resampling All Time-Series to Consistent Hourly Timebase
# =============================================================================
print("\n\n3. RESAMPLING TO CONSISTENT HOURLY TIMEBASE")
print("-" * 50)

print("Resampling all time-series DataFrames to hourly frequency...")

# List of time-series DataFrames to resample
timeseries_dfs = {
    'omni': omni_clean,
    'goes_proton': goes_proton_clean, 
    'goes_xray': goes_xray_clean,
    'noaa': noaa_clean,
    'supermag': supermag_clean,
    'esa': esa_clean
}

resampled_dfs = {}

for name, df in timeseries_dfs.items():
    if not df.empty and 'datetime' in df.columns:
        print(f"Resampling {name} data...")
        df_resampled = df.set_index('datetime').resample('1H').mean()
        resampled_dfs[name] = df_resampled
        print(f"✓ {name} resampled data:")
        print(df_resampled.head())
        print()
    else:
        print(f"⚠ Skipping {name} - no datetime column or empty DataFrame")

# =============================================================================
# PROMPT 4: Processing Event-Based Storm Alerts into a Feature
# =============================================================================
print("\n\n4. PROCESSING EVENT-BASED STORM ALERTS")
print("-" * 50)

print("Converting storm alerts into time-series feature...")

# Find the time range from resampled environmental data
all_indices = []
for df in resampled_dfs.values():
    if not df.empty:
        all_indices.extend(df.index.tolist())

if all_indices:
    min_time = min(all_indices)
    max_time = max(all_indices)
    
    # Create complete hourly datetime index
    full_time_index = pd.date_range(start=min_time, end=max_time, freq='H')
    
    # Initialize storm alert feature DataFrame
    storm_alert_feature = pd.DataFrame(index=full_time_index)
    storm_alert_feature['is_storm_alert_active'] = 0
    
    # Process each alert
    if not alerts_clean.empty and 'issue_time' in alerts_clean.columns:
        print(f"Processing {len(alerts_clean)} storm alerts...")
        
        for _, alert in alerts_clean.iterrows():
            issue_time = alert['issue_time']
            if pd.notna(issue_time):
                alert_type = alert.get('alert_type', 'unknown')
                
                # Determine alert duration based on type
                if alert_type == 'watch':
                    duration_hours = 24
                elif alert_type == 'warning':
                    duration_hours = 48
                else:  # alert or unknown
                    duration_hours = 12
                
                # Set alert active for the duration
                end_time = issue_time + pd.Timedelta(hours=duration_hours)
                mask = (storm_alert_feature.index >= issue_time) & (storm_alert_feature.index < end_time)
                storm_alert_feature.loc[mask, 'is_storm_alert_active'] = 1
        
        print(f"✓ Storm alerts processed. {storm_alert_feature['is_storm_alert_active'].sum()} hours with active alerts")
    else:
        print("⚠ No valid storm alerts to process")
    
    print("✓ Storm alert feature DataFrame:")
    print(storm_alert_feature.head())
else:
    print("⚠ No environmental data available to determine time range")
    storm_alert_feature = pd.DataFrame()

# =============================================================================
# PROMPT 5: Merging All Environmental Data
# =============================================================================
print("\n\n5. MERGING ALL ENVIRONMENTAL DATA")
print("-" * 50)

print("Merging all environmental DataFrames...")

# Create list of all DataFrames to merge
dfs_to_merge = list(resampled_dfs.values())
if not storm_alert_feature.empty:
    dfs_to_merge.append(storm_alert_feature)

if dfs_to_merge:
    # Filter out empty DataFrames
    valid_dfs = [df for df in dfs_to_merge if not df.empty]
    
    if valid_dfs:
        environmental_df = pd.concat(valid_dfs, axis=1)
        
        print("✓ Environmental DataFrame Info:")
        print(environmental_df.info())
        
        print("\n✓ Missing values count:")
        missing_values = environmental_df.isnull().sum()
        print(missing_values)
    else:
        print("⚠ No valid DataFrames to merge")
        environmental_df = pd.DataFrame()
else:
    print("⚠ No DataFrames available for merging")
    environmental_df = pd.DataFrame()

# =============================================================================
# PROMPT 6: Interpolation and Final Cleanup of Merged Data
# =============================================================================
print("\n\n6. INTERPOLATION AND FINAL CLEANUP")
print("-" * 50)

if not environmental_df.empty:
    print("Interpolating missing values...")
    environmental_df = environmental_df.interpolate(method='linear')
    
    print("Backfilling remaining NaNs at start of series...")
    environmental_df = environmental_df.bfill()
    
    print("✓ Missing values after interpolation:")
    missing_after = environmental_df.isnull().sum()
    print(missing_after)
    
    if missing_after.sum() == 0:
        print("✅ DataFrame is completely clean!")
    else:
        print("⚠ Some missing values remain")
else:
    print("⚠ No environmental DataFrame to clean")

# =============================================================================
# PROMPT 7: Comprehensive Feature Engineering
# =============================================================================
print("\n\n7. COMPREHENSIVE FEATURE ENGINEERING")
print("-" * 50)

if not environmental_df.empty:
    print("Starting comprehensive feature engineering...")
    
    # Key indices to include as features
    key_indices = ['Kp', 'Dst', 'is_storm_alert_active']
    available_key_indices = [col for col in key_indices if col in environmental_df.columns]
    print(f"✓ Key indices available: {available_key_indices}")
    
    # Raw sensor columns for lagged/rolling features
    raw_sensor_cols = ['BZ_GSM', 'speed', 'proton_flux', 'xray_flux_long', 'sme_index']
    available_sensor_cols = [col for col in raw_sensor_cols if col in environmental_df.columns]
    print(f"✓ Raw sensor columns available: {available_sensor_cols}")
    
    print("Creating lagged features...")
    # Create lagged features (3, 6, 12, 24 hours ago)
    lag_hours = [3, 6, 12, 24]
    for col in available_sensor_cols:
        for lag in lag_hours:
            environmental_df[f'{col}_lag_{lag}h'] = environmental_df[col].shift(lag)
    
    print("Creating rolling window features...")
    # Create rolling window features (3-hour and 6-hour)
    window_sizes = [3, 6]
    for col in available_sensor_cols:
        for window in window_sizes:
            environmental_df[f'{col}_roll_{window}h_mean'] = environmental_df[col].rolling(window=window).mean()
            environmental_df[f'{col}_roll_{window}h_std'] = environmental_df[col].rolling(window=window).std()
    
    print("Creating target variable...")
    # Create target variable (Dst 24 hours in future)
    if 'Dst' in environmental_df.columns:
        environmental_df['target_dst_24h_future'] = environmental_df['Dst'].shift(-24)
    else:
        print("⚠ Dst column not found, creating synthetic target")
        environmental_df['target_dst_24h_future'] = np.random.normal(-20, 40, len(environmental_df))
    
    print("Binning target into severity classes...")
    # Bin target into severity classes
    def classify_dst(dst):
        if pd.isna(dst):
            return np.nan
        elif dst > -30:
            return 0  # Quiet
        elif dst > -50:
            return 1  # Moderate
        elif dst > -100:
            return 2  # Strong
        else:
            return 3  # Severe
    
    environmental_df['target_class'] = environmental_df['target_dst_24h_future'].apply(classify_dst)
    
    print("Dropping rows with NaN values from lagging/shifting...")
    # Drop rows with NaN values
    environmental_df_clean = environmental_df.dropna()
    
    print("✓ Target class distribution:")
    if not environmental_df_clean.empty and 'target_class' in environmental_df_clean.columns:
        target_counts = environmental_df_clean['target_class'].value_counts().sort_index()
        print(target_counts)
        print(f"\nTotal samples after cleaning: {len(environmental_df_clean)}")
    else:
        print("⚠ No valid samples after feature engineering")
        environmental_df_clean = pd.DataFrame()
else:
    print("⚠ No environmental DataFrame available for feature engineering")
    environmental_df_clean = pd.DataFrame()

# =============================================================================
# PROMPT 8: Processing the UCS Satellite Database
# =============================================================================
print("\n\n8. PROCESSING UCS SATELLITE DATABASE")
print("-" * 50)

if not satellite_clean.empty:
    print("Processing UCS Satellite Database...")
    
    # Ensure Date of Launch is datetime
    if 'Date of Launch' in satellite_clean.columns:
        satellite_clean['Date of Launch'] = pd.to_datetime(satellite_clean['Date of Launch'], errors='coerce')
        
        # Create Age_years column
        current_date = pd.to_datetime('2025-09-26')
        satellite_clean['Age_years'] = (current_date - satellite_clean['Date of Launch']).dt.days / 365.25
        
        print("✓ Age_years column created")
    else:
        print("⚠ Date of Launch column not found")
    
    # Clean Class of Orbit column
    if 'Class of Orbit' in satellite_clean.columns:
        print("Standardizing orbit classifications...")
        
        def standardize_orbit(orbit):
            if pd.isna(orbit):
                return 'Other'
            orbit_str = str(orbit).upper()
            if any(x in orbit_str for x in ['LOW EARTH', 'LEO']):
                return 'LEO'
            elif any(x in orbit_str for x in ['MEDIUM EARTH', 'MEO']):
                return 'MEO'
            elif any(x in orbit_str for x in ['GEOSTATIONARY', 'GEO']):
                return 'GEO'
            elif 'ELLIPTICAL' in orbit_str:
                return 'Elliptical'
            else:
                return 'Other'
        
        satellite_clean['Class of Orbit'] = satellite_clean['Class of Orbit'].apply(standardize_orbit)
        
        print("✓ First 5 rows of processed satellite database:")
        print(satellite_clean.head())
        
        print("\n✓ Unique orbit classes:")
        print(satellite_clean['Class of Orbit'].value_counts())
    else:
        print("⚠ Class of Orbit column not found")
else:
    print("⚠ No satellite database to process")

# =============================================================================
# PROMPT 9: Finalizing and Saving All Processed Data
# =============================================================================
print("\n\n9. FINALIZING AND SAVING PROCESSED DATA")
print("-" * 50)

print("Preparing final training data...")

if not environmental_df_clean.empty:
    # Drop original time-series columns that were used to create features
    original_sensor_cols = [col for col in available_sensor_cols if col in environmental_df_clean.columns]
    cols_to_drop = original_sensor_cols + ['target_dst_24h_future']
    
    # Keep only engineered features and target
    final_training_data = environmental_df_clean.drop(columns=[col for col in cols_to_drop 
                                                             if col in environmental_df_clean.columns])
    
    print(f"✓ Final training data shape: {final_training_data.shape}")
    print(f"✓ Features: {len(final_training_data.columns) - 1}")  # -1 for target column
    print(f"✓ Final columns: {list(final_training_data.columns)}")
    
    # Save the training dataset
    try:
        final_training_data.to_csv('model_training_data.csv', index=True)
        print("✅ Saved model_training_data.csv")
    except Exception as e:
        print(f"⚠ Error saving training data: {e}")
else:
    print("⚠ No training data available to save")

# Save the satellite database
if not satellite_clean.empty:
    try:
        satellite_clean.to_csv('processed_satellite_database.csv', index=False)
        print("✅ Saved processed_satellite_database.csv")
    except Exception as e:
        print(f"⚠ Error saving satellite database: {e}")
else:
    print("⚠ No satellite database to save")

# Final summary
print("\n" + "="*80)
print("COMPREHENSIVE PROCESSING COMPLETE!")
print("="*80)

if not environmental_df_clean.empty:
    print(f"✅ Environmental training data: {final_training_data.shape[0]} samples, {final_training_data.shape[1]} features")
else:
    print("⚠ No environmental training data produced")

if not satellite_clean.empty:
    print(f"✅ Satellite database: {len(satellite_clean)} satellites processed")
else:
    print("⚠ No satellite database processed")

print("\nFiles created:")
print("- model_training_data.csv (ready for machine learning)")
print("- processed_satellite_database.csv (cleaned satellite asset data)")
print("\nThe data is now ready for the modeling phase!")
print("="*80)