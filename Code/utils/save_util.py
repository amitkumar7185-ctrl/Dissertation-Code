
def save_pivot_df(pivot_df):
    output_path = 'Data/pivot/pivot_output.xlsx'  
    pivot_df.to_excel(output_path, index=True)
    print("pivot_df saved successfully to Excel.")