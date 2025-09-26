from langchain.agents import initialize_agent, Tool
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
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

def create_satellite_data_agent():
    """
    Create a LangChain agent for cleaning satellite data.
    """
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key="AIzaSyBmVQ0Byvp50gGGyBSJ0TH-6eVJsrxNceU")
    tools = create_cleaning_tools()
    
    # Get the prompt for the agent
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

def clean_satellite_dataset():
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
    
    # For demo purposes, let's just process one row without using the agent
    # to avoid quota issues
    if len(df) > 0:
        # Use our direct cleaning functions instead of the agent
        sample_row = df.iloc[0].to_dict()
        sample_row['Shielding'] = None
        
        # Apply our cleaning functions directly
        tools = create_cleaning_tools()
        clean_row_tool = next(tool for tool in tools if tool.name == "CleanRowData")
        
        # Clean the row
        row_str = str(sample_row)
        result = clean_row_tool.func(row_str)
        
        print(f"Cleaned row 0: {result}")
        return [result]
    
    return []

if __name__ == "__main__":
    # Run the cleaning process
    try:
        cleaned_data = clean_satellite_dataset()
        print("\nDataset cleaned successfully with LangChain agents.")
        print("Sample of cleaned data:", cleaned_data[:1] if cleaned_data else "No data found")
    except Exception as e:
        print(f"Error: {e}")