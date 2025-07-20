import pandas as pd
def  load_Data():
    file_path = 'Data\Data.csv'  # Update this path if needed
    df = pd.read_csv(file_path)
    return df
    