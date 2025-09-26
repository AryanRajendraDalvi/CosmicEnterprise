import pandas as pd
import numpy as np

file_path = 'omniweb data.txt'

# !!! ACTION REQUIRED: Set this value based on your analysis from Step 1 !!!
NUMBER_OF_HEADER_LINES = 10  # <--- CHANGE THIS NUMBER

try:
    # --- 1. Read the file line-by-line ---
    data_rows = []
    with open(file_path, 'r') as f:
        # Skip the exact number of header lines
        for _ in range(NUMBER_OF_HEADER_LINES):
            next(f)
        
        # Process the rest of the file
        for line in f:
            parts = line.split()
            if len(parts) > 5:
                data_rows.append(parts)

    if not data_rows:
        raise ValueError("No valid data rows were found after skipping the header.")
        
    df = pd.DataFrame(data_rows)

    # --- 2. Assign column names and create the datetime index ---
    df.columns = [f'col_{i}' for i in range(len(df.columns))]
    df['datetime'] = pd.to_datetime(df['col_0'].astype(str) + '-' + df['col_1'].astype(str), format='%Y-%j') \
                     + pd.to_timedelta(df['col_2'].astype(str), unit='h')
    df.set_index('datetime', inplace=True)

    # --- 3. Select and rename the correct data columns ---
    required_columns_map = {
        'col_7': 'BZ_GSM',
        'col_6': 'Speed',
        'col_5': 'Proton_Density',
        'col_4': 'Plasma_Temp'
    }
    df_cleaned = df[required_columns_map.keys()].copy()
    df_cleaned.rename(columns=required_columns_map, inplace=True)

    # --- 4. Force data to numeric ---
    for col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # --- 5. Display the final result ---
    print("Successfully parsed and cleaned the data!")
    print("\nDataFrame Info:")
    df_cleaned.info()
    print("\nFirst 5 rows of the cleaned data:")
    print(df_cleaned.head())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")