"""
Satellite Data Cleaning with LangChain Agents

This script demonstrates how to use LangChain agents to clean satellite data.
Note: This requires an OpenAI API key to run fully.
"""

from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAI
import pandas as pd
import json
import os

def load_and_clean_satellite_data():
    """
    Load the UCS satellite database and clean it using LangChain agents.
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

def create_cleaning_tools():
    """
    Create tools for the LangChain agent to clean satellite data.
    """
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
    
    def estimate_shielding(row):
        """Estimate shielding based on orbit type."""
        orbit_type = row.get('OrbitType', '')
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
    
    def clean_row_data(data_str):
        """Clean a row of satellite data."""
        try:
            # Parse the data string as a dictionary
            data = eval(data_str) if isinstance(data_str, str) else data_str
            
            # Normalize orbit type
            if 'OrbitType' in data:
                data['OrbitType'] = normalize_orbit_type(data['OrbitType'])
            
            # Estimate shielding if missing
            if 'Shielding' not in data or pd.isna(data.get('Shielding')):
                data['Shielding'] = estimate_shielding(data)
            
            # Ensure numeric fields are properly formatted
            for field in ['Altitude_km', 'Inclination_deg', 'Mass_kg', 'Shielding']:
                if field in data and not pd.isna(data[field]):
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        data[field] = None
            
            return json.dumps(data)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    # Define tools for the agent
    tools = [
        Tool(
            name="CleanRowData",
            func=clean_row_data,
            description="Useful for cleaning a row of satellite data. Input should be a dictionary string representation of the row data."
        ),
        Tool(
            name="NormalizeOrbitType",
            func=lambda x: normalize_orbit_type(x),
            description="Useful for normalizing orbit types to standard values (LEO/MEO/GEO/HEO)."
        )
    ]
    
    return tools

def create_satellite_data_agent(api_key=None):
    """
    Create a LangChain agent for cleaning satellite data.
    """
    # Create LLM (requires API key)
    if api_key:
        llm = OpenAI(temperature=0, openai_api_key=api_key)
    else:
        # Fallback to a mock LLM for demonstration
        print("Warning: No API key provided. Using mock LLM for demonstration.")
        from langchain.llms import FakeListLLM
        llm = FakeListLLM(responses=["Action: CleanRowData\\nAction Input: {\"SatelliteName\": \"Test Satellite\", \"OrbitType\": \"LEO\", \"Altitude_km\": 500.0, \"Inclination_deg\": 45.0, \"Mass_kg\": 100.0, \"Shielding\": 3.0}"])
    
    tools = create_cleaning_tools()
    
    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    return agent

def clean_satellite_dataset(api_key=None):
    """
    Main function to clean the satellite dataset using LangChain agents.
    """
    # Load the data
    df = load_and_clean_satellite_data()
    
    # Add Shielding column if it doesn't exist
    if 'Shielding' not in df.columns:
        df['Shielding'] = None
    
    # Remove duplicates based on satellite name
    df = df.drop_duplicates(subset=['SatelliteName'], keep='first')
    
    # Create the agent
    agent = create_satellite_data_agent(api_key)
    
    # Clean the data row by row (first 3 rows for demo)
    cleaned_data = []
    for idx, row in df.head(3).iterrows():  # Process first 3 rows for demo
        row_dict = row.to_dict()
        row_dict['Shielding'] = None  # Will be filled by agent
        
        # Convert row to string representation for the agent
        row_str = str(row_dict)
        
        try:
            # Use the agent to clean the row
            result = agent.run(f"Clean this satellite data row: {row_str}")
            cleaned_data.append(result)
            print(f"Cleaned row {idx}: {result}")
        except Exception as e:
            print(f"Error cleaning row {idx}: {e}")
    
    return cleaned_data

def main():
    """
    Main function to run the satellite data cleaning with LangChain agents.
    """
    print("Satellite Data Cleaning with LangChain Agents")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("No OpenAI API key found in environment variables.")
        print("To use the full LangChain agent functionality, set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        print()
        print("Running in demonstration mode with mock LLM...")
        print()
    
    try:
        cleaned_data = clean_satellite_dataset(api_key)
        print("\nDataset processed with LangChain agents.")
        print("Sample of processed data:", cleaned_data[:1] if cleaned_data else "No data found")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo run this script fully, you need to:")
        print("1. Set your OpenAI API key as an environment variable")
        print("2. Install required packages: pip install langchain langchain-openai openai")

if __name__ == "__main__":
    main()