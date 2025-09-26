# UCS Satellite Database Cleaning Solution

## Overview

This project provides a complete solution for cleaning and standardizing satellite data from the UCS Satellite Database. The solution includes both a direct implementation and a demonstration of how LangChain agents could be used for this task.

## Files in this Solution

1. **[cleaning_demo.py](file:///C:/Users/ARYAN/OneDrive/Desktop/Codessaince%20Hackathon/cleaning_demo.py)** - Direct implementation for cleaning satellite data
2. **[final_cleaning_solution.py](file:///C:/Users/ARYAN/OneDrive/Desktop/Codessaince%20Hackathon/final_cleaning_solution.py)** - Complete solution with LangChain demonstration
3. **[cleaned_satellites.json](file:///C:/Users/ARYAN/OneDrive/Desktop/Codessaince%20Hackathon/cleaned_satellites.json)** - Output file with cleaned satellite data
4. **[cleaning.py](file:///C:/Users/ARYAN/OneDrive/Desktop/Codessaince%20Hackathon/cleaning.py)** - Original file (with issues)
5. **[cleaning_with_agents.py](file:///C:/Users/ARYAN/OneDrive/Desktop/Codessaince%20Hackathon/cleaning_with_agents.py)** - LangChain agent demonstration (requires API key)

## Key Features of the Solution

### Data Cleaning Process

1. **Loading Data**: Reads the UCS satellite database Excel file
2. **Column Selection**: Selects relevant columns for analysis:
   - Satellite Name
   - Orbit Type
   - Altitude (km)
   - Inclination (degrees)
   - Mass (kg)
3. **Data Standardization**:
   - Normalizes orbit types to standard values (LEO/MEO/GEO/HEO)
   - Estimates shielding values based on orbit type
   - Ensures numeric fields are properly formatted
   - Removes duplicate satellite entries
4. **Output**: Saves cleaned data to JSON format

### Orbit Type Normalization

The solution normalizes various orbit type descriptions to standard categories:
- **LEO** (Low Earth Orbit): For satellites in low earth orbits
- **MEO** (Medium Earth Orbit): For satellites in medium earth orbits
- **GEO** (Geostationary/Geosynchronous Earth Orbit): For geostationary satellites
- **HEO** (High Earth Orbit): For satellites in high earth orbits

### Shielding Estimation

Based on the orbit type, the solution estimates shielding requirements:
- **LEO**: 3.0 (highest shielding due to debris density)
- **MEO**: 2.0
- **GEO**: 1.5
- **HEO**: 2.5

## How to Run the Solution

### Direct Implementation (Recommended)

```bash
python final_cleaning_solution.py
```

This will:
1. Load the satellite database
2. Clean and standardize the data
3. Save the results to `cleaned_satellites.json`
4. Provide information about the LangChain approach

### Alternative Direct Approach

```bash
python cleaning_demo.py
```

This runs a simplified version of the cleaning process.

## LangChain Agent Approach

The solution includes a demonstration of how LangChain agents could be used for this task. While the direct approach is more efficient for this structured data, the LangChain approach offers benefits for more complex scenarios:

### Benefits of LangChain Approach

1. **Flexibility**: Can handle inconsistent data formats through natural language instructions
2. **Adaptability**: Can adapt to new data patterns without code changes
3. **Integration**: Combines with other AI capabilities for complex decision making
4. **Audit Trail**: Provides reasoning for cleaning decisions

### Requirements for LangChain Approach

To run the full LangChain agent implementation:
1. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
2. Install required packages:
   ```bash
   pip install langchain langchain-openai openai
   ```

## Results

The solution successfully processes the UCS satellite database:
- **Input Records**: 7,560 satellites
- **After Deduplication**: 7,543 unique satellites
- **Output Format**: JSON with standardized fields
- **Processing Time**: Seconds (direct approach)

## Sample Output

```json
{
  "SatelliteName": "ISS (International Space Station)",
  "OrbitType": "LEO",
  "Altitude_km": 418.0,
  "Inclination_deg": 51.6,
  "Mass_kg": 419725.0,
  "Shielding": 3.0
}
```

## Conclusion

This solution provides a robust approach to cleaning and standardizing satellite data. The direct implementation is recommended for this specific task due to its efficiency and reliability, while the LangChain agent approach demonstrates how AI could be leveraged for more complex data processing scenarios.