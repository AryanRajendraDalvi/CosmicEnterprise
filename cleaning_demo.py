import pandas as pd
import json
import os

def load_and_clean_satellite_data():
    """
    Load the UCS satellite database and prepare it for cleaning.
    """
    # Correct file name
    file_path = 'UCS-Satellite-Database 5-1-2023.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Satellite database file not found: {file_path}")
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Select relevant columns for cleaning
    columns_map = {
        'Name of Satellite, Alternate Names': 'SatelliteName',
        'Class of Orbit': 'OrbitType',
        'Perigee (km)': 'Altitude_km',
        'Inclination (degrees)': 'Inclination_deg',
        'Launch Mass (kg.)': 'Mass_kg'
    }
    
    # Filter to only the columns we need
    available_columns = [col for col in columns_map.keys() if col in df.columns]
    df_subset = df[available_columns].copy()
    
    # Rename columns to match our standard
    rename_map = {k: v for k, v in columns_map.items() if k in available_columns}
    df_subset.rename(columns=rename_map, inplace=True)
    
    return df_subset

def normalize_orbit_type(orbit_type):
    """Normalize orbit type to standard values (LEO/MEO/GEO/HEO)."""
    if pd.isna(orbit_type):
        return None
        
    orbit_str = str(orbit_type).upper().strip()
    if 'LEO' in orbit_str:
        return 'LEO'
    elif 'MEO' in orbit_str:
        return 'MEO'
    elif 'GEO' in orbit_str:
        return 'GEO'
    elif 'HEO' in orbit_str:
        return 'HEO'
    else:
        return orbit_str

def estimate_shielding(orbit_type):
    """Estimate shielding based on orbit type."""
    if pd.isna(orbit_type):
        return 1.0  # Default shielding
    
    # Default shielding values per orbit type
    shielding_defaults = {
        'LEO': 3.0,  # Higher shielding needed for LEO due to higher debris density
        'MEO': 2.0,
        'GEO': 1.5,
        'HEO': 2.5
    }
    
    return shielding_defaults.get(str(orbit_type).upper(), 1.0)

def clean_satellite_row(row):
    """
    Clean a single row of satellite data.
    """
    # Create a copy of the row to avoid modifying the original
    cleaned_row = row.copy()
    
    # Normalize orbit type
    if 'OrbitType' in cleaned_row:
        cleaned_row['OrbitType'] = normalize_orbit_type(cleaned_row['OrbitType'])
    
    # Estimate shielding based on orbit type
    cleaned_row['Shielding'] = estimate_shielding(cleaned_row.get('OrbitType'))
    
    # Ensure numeric fields are properly formatted
    for field in ['Altitude_km', 'Inclination_deg', 'Mass_kg', 'Shielding']:
        if field in cleaned_row and not pd.isna(cleaned_row[field]):
            try:
                cleaned_row[field] = float(cleaned_row[field])
            except (ValueError, TypeError):
                cleaned_row[field] = None
    
    return cleaned_row

def clean_satellite_dataset():
    """
    Main function to clean the satellite dataset.
    """
    print("Loading satellite data...")
    # Load the data
    df = load_and_clean_satellite_data()
    
    # Remove duplicates based on satellite name
    df = df.drop_duplicates(subset=['SatelliteName'], keep='first')
    
    print(f"Loaded {len(df)} satellites. Cleaning first 5 rows as demonstration...")
    
    # Clean the data row by row
    cleaned_data = []
    for idx, row in df.head(5).iterrows():  # Process first 5 rows for demo
        try:
            # Clean the row
            cleaned_row = clean_satellite_row(row)
            cleaned_data.append(cleaned_row)
            print(f"Cleaned row {idx}: {cleaned_row['SatelliteName']} - Orbit: {cleaned_row['OrbitType']}, Shielding: {cleaned_row['Shielding']}")
        except Exception as e:
            print(f"Error cleaning row {idx}: {e}")
    
    return cleaned_data

def save_cleaned_data(cleaned_data, filename='cleaned_satellites.json'):
    """
    Save the cleaned data to a JSON file.
    """
    # Convert to serializable format
    serializable_data = []
    for row in cleaned_data:
        serializable_row = {}
        for key, value in row.items():
            # Handle NaN values
            if pd.isna(value):
                serializable_row[key] = None
            else:
                serializable_row[key] = value
        serializable_data.append(serializable_row)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Cleaned data saved to {filename}")

if __name__ == "__main__":
    # Run the cleaning process
    try:
        cleaned_data = clean_satellite_dataset()
        print("\nDataset cleaned successfully.")
        print("Sample of cleaned data:")
        if cleaned_data:
            print(json.dumps(cleaned_data[0], indent=2, default=str))
        
        # Save the cleaned data
        save_cleaned_data(cleaned_data)
        
    except Exception as e:
        print(f"Error: {e}")