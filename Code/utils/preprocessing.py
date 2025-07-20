import pandas as pd
def prepare_pivoted_data(df):
    df['Date'] = pd.to_datetime(df['PerformanceStartDate'])

    pivot_df = df.pivot_table(
        index=['elevatorunitId', 'elevatorunitnumber', 'Date'],
        columns='ItemFieldId',
        values='Readvalue',
        aggfunc='sum'
    ).reset_index()
    # Example: Add lag features or calculate differences to capture anomalies
    pivot_df['door_reversals_diff'] = pivot_df['door_reversals'].diff()
    pivot_df['door_cycles_diff'] = pivot_df['total_door_cycles'].diff()

    # Fill missing values with 0
    pivot_df.fillna(0, inplace=True)
    # Sort by time
    #pivot_df = pivot_df.sort_values(by='Date')
    return pivot_df

def label_faults(pivot_df):
    # Define fault condition (label engineering)
    pivot_df['Fault'] = (
    ((pivot_df['door_reversals'] > 100) |
     (pivot_df['front_door_reversals'] > 100) |
     (pivot_df['rear_door_reversals'] > 100)) |

    ((pivot_df['total_door_cycles'] > 0) & (pivot_df['total_door_cycles'] < 10)) |

    (pivot_df['door_reversals_diff'].abs() > 10) |
    (pivot_df['door_cycles_diff'].abs() > 10) |

    (pivot_df['slow_door_operations'] > 10)
    ).astype(int)   
    return pivot_df

