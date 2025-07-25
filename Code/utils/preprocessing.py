import pandas as pd
def prepare_pivoted_data(df):
    df['Date'] = pd.to_datetime(df['PerformanceStartDate'])

    pivot_df = df.pivot_table(
        index=['elevatorunitId', 'elevatorunitnumber', 'Date'],
        columns='ItemFieldId',
        values='Readvalue',
        aggfunc='sum'
    ).reset_index()

   

    # Fill missing values with 0
    pivot_df.fillna(0, inplace=True)
    # Sort by time
    #pivot_df = pivot_df.sort_values(by='Date')
    return fetureAggregation(pivot_df)

def fetureAggregation(pivot_df):
    # Aggregate features
    pivot_df['total_door_reversals'] = pivot_df['door_reversals'] + pivot_df['front_door_reversals'] + pivot_df['rear_door_reversals']
    pivot_df['total_door_cycles'] = pivot_df['front_door_cycles'] + pivot_df['rear_door_cycles'] + pivot_df['total_door_cycles']


    total_operations = pivot_df['door_operations'] + pivot_df['slow_door_operations']
    pivot_df['slow_door_operation_percentage'] = (pivot_df['slow_door_operations'] / total_operations) * 100
    pivot_df['door_operations'] = total_operations



    pivot_df['door_reversals_percentage'] = (pivot_df['total_door_reversals'] / pivot_df['total_door_cycles']) * 100
    pivot_df.drop(columns=['door_reversals', 'front_door_reversals', 'rear_door_reversals','front_door_cycles','rear_door_cycles'], inplace=True)

     # Example: Add lag features or calculate differences to capture anomalies
    pivot_df['door_reversals_diff'] = pivot_df['total_door_reversals'].diff()
    pivot_df['door_cycles_diff'] = pivot_df['total_door_cycles'].diff()

    # Add more features as needed
    return pivot_df

def label_faults(pivot_df):
    # Define fault condition (label engineering)
    pivot_df['Fault'] = (
    (pivot_df['total_door_reversals'] > 600) |

    ((pivot_df['total_door_cycles'] > 0) & (pivot_df['total_door_cycles'] < 5)) |

    # (pivot_df['door_reversals_diff'].abs() > 10) |
    # (pivot_df['door_cycles_diff'].abs() > 10) |

    (pivot_df['slow_door_operations'] > 2)
    ).astype(int)   
    return pivot_df

