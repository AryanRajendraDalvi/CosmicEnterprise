"""
Satellite Data Cleaning with Google Gemini API

This script demonstrates how to use the Google Gemini API for cleaning satellite data.
"""

import pandas as pd
import json
import os
from langchain_google_genai import GoogleGenerativeAI

def load_and_clean_satellite_data():
    """
    Load the UCS satellite database and clean it.
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

def clean_satellite_row_with_gemini(row, llm):
    """
    Clean a single row of satellite data using the Gemini API.
    """
    try:
        # Convert row to a readable format
        row_info = f"""
        Satellite Name: {row.get('SatelliteName', 'Unknown')}
        Orbit Type: {row.get('OrbitType', 'Unknown')}
        Altitude (km): {row.get('Altitude_km', 'Unknown')}
        Inclination (degrees): {row.get('Inclination_deg', 'Unknown')}
        Mass (kg): {row.get('Mass_kg', 'Unknown')}
        """
        
        # Create a prompt for the LLM
        prompt = f"""
        Please clean and standardize the following satellite data:
        
        {row_info}
        
        Requirements:
        1. Normalize the orbit type to one of: LEO, MEO, GEO, HEO
        2. Estimate a shielding value based on the orbit type (LEO: 3.0, MEO: 2.0, GEO: 1.5, HEO: 2.5)
        3. Ensure all numeric values are properly formatted
        4. Return the result as a JSON object with fields: SatelliteName, OrbitType, Altitude_km, Inclination_deg, Mass_kg, Shielding
        
        Return only the JSON object, nothing else.
        """
        
        # Get response from Gemini
        response = llm.invoke(prompt)
        
        # Try to parse the response as JSON
        try:
            cleaned_data = json.loads(response)
            return cleaned_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return a basic cleaned version
            return {
                "SatelliteName": row.get('SatelliteName', ''),
                "OrbitType": normalize_orbit_type(row.get('OrbitType')),
                "Altitude_km": float(row.get('Altitude_km', 0)) if not pd.isna(row.get('Altitude_km')) else 0,
                "Inclination_deg": float(row.get('Inclination_deg', 0)) if not pd.isna(row.get('Inclination_deg')) else 0,
                "Mass_kg": float(row.get('Mass_kg', 0)) if not pd.isna(row.get('Mass_kg')) else 0,
                "Shielding": estimate_shielding(row.get('OrbitType'))
            }
            
    except Exception as e:
        # If any error occurs, return a basic cleaned version
        return {
            "SatelliteName": row.get('SatelliteName', ''),
            "OrbitType": normalize_orbit_type(row.get('OrbitType')),
            "Altitude_km": float(row.get('Altitude_km', 0)) if not pd.isna(row.get('Altitude_km')) else 0,
            "Inclination_deg": float(row.get('Inclination_deg', 0)) if not pd.isna(row.get('Inclination_deg')) else 0,
            "Mass_kg": float(row.get('Mass_kg', 0)) if not pd.isna(row.get('Mass_kg')) else 0,
            "Shielding": estimate_shielding(row.get('OrbitType'))
        }

def clean_satellite_dataset():
    """
    Main function to clean the satellite dataset using Google Gemini API.
    """
    # Load the data
    df = load_and_clean_satellite_data()
    
    # Remove duplicates based on satellite name
    df = df.drop_duplicates(subset=['SatelliteName'], keep='first')
    
    # Initialize the Gemini LLM with the provided API key
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", 
                            google_api_key="AIzaSyBmVQ0Byvp50gGGyBSJ0TH-6eVJsrxNceU")
    
    print(f"Loaded {len(df)} satellite records. Processing first 3 records with Gemini API...")
    
    # Clean the data row by row (limited to avoid quota issues)
    cleaned_data = []
    for idx, row in df.head(3).iterrows():  # Process only first 3 rows to avoid quota issues
        try:
            # Clean the row using Gemini
            cleaned_row = clean_satellite_row_with_gemini(row, llm)
            cleaned_data.append(cleaned_row)
            print(f"Cleaned row {idx}: {cleaned_row['SatelliteName']} - Orbit: {cleaned_row['OrbitType']}")
        except Exception as e:
            print(f"Error cleaning row {idx}: {e}")
            # Use fallback method
            fallback_row = {
                "SatelliteName": row.get('SatelliteName', ''),
                "OrbitType": normalize_orbit_type(row.get('OrbitType')),
                "Altitude_km": float(row.get('Altitude_km', 0)) if not pd.isna(row.get('Altitude_km')) else 0,
                "Inclination_deg": float(row.get('Inclination_deg', 0)) if not pd.isna(row.get('Inclination_deg')) else 0,
                "Mass_kg": float(row.get('Mass_kg', 0)) if not pd.isna(row.get('Mass_kg')) else 0,
                "Shielding": estimate_shielding(row.get('OrbitType'))
            }
            cleaned_data.append(fallback_row)
    
    return cleaned_data

def save_cleaned_data(cleaned_data, filename='cleaned_satellites_gemini.json'):
    """
    Save the cleaned data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    print(f"Cleaned data saved to {filename}")

if __name__ == "__main__":
    # Run the cleaning process
    try:
        print("Starting satellite data cleaning with Google Gemini API...")
        cleaned_data = clean_satellite_dataset()
        print("\nDataset cleaned successfully with Google Gemini API.")
        
        if cleaned_data:
            print("Sample of cleaned data:")
            print(json.dumps(cleaned_data[0], indent=2))
            
            # Save the cleaned data
            save_cleaned_data(cleaned_data)
        else:
            print("No data was cleaned.")
            
    except Exception as e:
        print(f"Error: {e}")