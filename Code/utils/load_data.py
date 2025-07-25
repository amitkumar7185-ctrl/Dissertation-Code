import pandas as pd
import os

def load_Data():
    folder_path = "Data\\InputData"
    """
    Loads data from multiple CSV files in a folder into a single Pandas DataFrame.    
    """
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")  #Good to print errors to console

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df    

    