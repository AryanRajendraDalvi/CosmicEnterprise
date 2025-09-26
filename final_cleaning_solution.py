"""
Final Satellite Data Cleaning Solution

This script provides a complete solution for cleaning satellite data from the UCS database.
It includes both a direct implementation and a LangChain agent demonstration.
"""

import pandas as pd
import json
import os

def load_satellite_data():
    """
    Load the UCS satellite database.
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
    if any(leo_term in orbit_str for leo_term in ['LEO', 'LOW EARTH']):
        return 'LEO'
    elif any(meo_term in orbit_str for meo_term in ['MEO', 'MEDIUM EARTH']):
        return 'MEO'
    elif any(geo_term in orbit_str for geo_term in ['GEO', 'GEOSTATIONARY', 'GEOSYNCHRONOUS']):
        return 'GEO'
    elif any(heo_term in orbit_str for heo_term in ['HEO', 'HIGH EARTH']):
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

def clean_satellite_data(df):
    """
    Clean the satellite data using direct processing.
    """
    print(f"Cleaning {len(df)} satellite records...")
    
    # Remove duplicates based on satellite name
    df_cleaned = df.drop_duplicates(subset=['SatelliteName'], keep='first').copy()
    print(f"Removed duplicates. Remaining records: {len(df_cleaned)}")
    
    # Normalize orbit types
    df_cleaned['OrbitType'] = df_cleaned['OrbitType'].apply(normalize_orbit_type)
    
    # Estimate shielding values
    df_cleaned['Shielding'] = df_cleaned['OrbitType'].apply(estimate_shielding)
    
    # Ensure numeric fields are properly formatted
    numeric_fields = ['Altitude_km', 'Inclination_deg', 'Mass_kg', 'Shielding']
    for field in numeric_fields:
        if field in df_cleaned.columns:
            df_cleaned[field] = pd.to_numeric(df_cleaned[field], errors='coerce')
    
    print("Data cleaning completed.")
    return df_cleaned

def save_cleaned_data(df, filename='cleaned_satellites.json'):
    """
    Save the cleaned data to a JSON file.
    """
    # Convert DataFrame to list of dictionaries
    cleaned_data = df.to_dict('records')
    
    # Handle any remaining NaN values
    for record in cleaned_data:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned data saved to {filename}")

def demonstrate_langchain_approach():
    """
    Demonstrate how LangChain agents could be used for this task.
    """
    print("\n" + "="*60)
    print("LANGCHAIN AGENT APPROACH (DEMONSTRATION)")
    print("="*60)
    print("""
In a full implementation with an OpenAI API key, we would:

1. Create LangChain tools for:
   - Normalizing orbit types
   - Estimating shielding values
   - Cleaning numeric fields
   - Removing duplicates

2. Initialize an agent with these tools:
   from langchain.agents import initialize_agent, Tool
   from langchain_openai import OpenAI
   
   agent = initialize_agent(
       tools=cleaning_tools,
       llm=OpenAI(temperature=0),
       agent="zero-shot-react-description",
       verbose=True
   )

3. Process each satellite record through the agent:
   for row in satellite_data.itertuples():
       result = agent.run(f"Clean this satellite data: {row}")
       cleaned_data.append(result)

Benefits of the LangChain approach:
- More flexible handling of inconsistent data formats
- Ability to adapt to new data patterns through natural language instructions
- Integration with other AI capabilities for complex decision making
- Audit trail of cleaning decisions through agent reasoning

However, for this specific structured data task, the direct approach is:
- More efficient
- More predictable
- Doesn't require external API calls
- Easier to debug and maintain
""")

def main():
    """
    Main function to run the satellite data cleaning process.
    """
    print("UCS Satellite Database Cleaning Tool")
    print("=" * 40)
    
    try:
        # Load the data
        print("Loading satellite data...")
        df = load_satellite_data()
        print(f"Loaded {len(df)} satellite records.")
        
        # Clean the data
        df_cleaned = clean_satellite_data(df)
        
        # Show sample of cleaned data
        print("\nSample of cleaned data:")
        print(df_cleaned.head(3).to_string())
        
        # Save the cleaned data
        save_cleaned_data(df_cleaned)
        
        # Demonstrate LangChain approach
        demonstrate_langchain_approach()
        
        print("\nProcess completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the UCS satellite database file is in the correct location.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()