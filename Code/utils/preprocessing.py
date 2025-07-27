import pandas as pd
def prepare_pivoted_data(df):
    df['Date'] = pd.to_datetime(df['PerformanceStartDate'])

    pivot_df = df.pivot_table(
        index=['elevatorunitId', 'elevatorunitnumber', 'Date'],
        columns='ItemFieldId',
        values='Readvalue',
        aggfunc='sum'
    ).reset_index()

    pivot_df.columns = pivot_df.columns.str.strip().str.replace(' ', '_')
    pivot_df=feature_aggregation(pivot_df)

    # Fill missing values with 0
    pivot_df.fillna(0, inplace=True)
    # Sort by time
    #pivot_df = pivot_df.sort_values(by='Date')
    return pivot_df


def feature_aggregation(pivot_df):
    # Combine door operations
    pivot_df['total_door_operations'] = pivot_df[['door_operations', 'front_door_cycles', 'rear_door_cycles']].sum(axis=1)
    # Combine door reversals
    pivot_df['total_door_reversals'] = pivot_df[['door_reversals', 'front_door_reversals', 'rear_door_reversals']].sum(axis=1) 

    pivot_df['door_failure_events'] = pivot_df[['inop4_car_door_open_command_failures', 'inop5_door_open_failures']].sum(axis=1)
    pivot_df['hoistway_faults'] = pivot_df['inop2_hoistway_door_lock_failures']

    # Combine safety chain failures
    pivot_df['safety_chain_issues'] = pivot_df[['inop7_safety_chain_failures_running', 'inop8_safety_chain_failures_idle']].sum(axis=1)
    
    pivot_df['levelling_total_errors'] = pivot_df[['inop11_levelling_errors', 'levelling_errors', 'number_of_non_level_landings']].sum(axis=1)
    pivot_df['startup_delays'] = pivot_df['inop3_elevator_start_delayed']

    # Combine releveling and rescue run time
    pivot_df['average_run_time'] = pivot_df[['one_floor_run_time', 'run_time_releveling', 'run_time_rescue']].mean(axis=1)
    # Combine run starts
    pivot_df['total_run_starts'] = pivot_df[['run_starts_ero', 'run_starts_inspection', 'run_starts_releveling', 'run_starts_rescue']].sum(axis=1)
    pivot_df['total_door_cycles'] = pivot_df['total_door_cycles']

    pivot_df=drop_redundent_columns(pivot_df)
    pivot_df=add_new_feature(pivot_df)  
    return pivot_df

def drop_redundent_columns(pivot_df):
    columns_to_drop = [
    'door_operations','front_door_cycles', 'rear_door_cycles',
    'door_reversals', 'front_door_reversals', 'rear_door_reversals',    
    'inop4_car_door_open_command_failures', 'inop5_door_open_failures',
    'inop2_hoistway_door_lock_failures',
    'inop7_safety_chain_failures_running', 'inop8_safety_chain_failures_idle',
    'inop11_levelling_errors', 'levelling_errors', 'number_of_non_level_landings',
    'inop3_elevator_start_delayed',
    'one_floor_run_time', 'run_time_releveling', 'run_time_rescue',
    'run_starts_ero', 'run_starts_inspection', 'run_starts_releveling', 'run_starts_rescue'
    ]

    pivot_df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return pivot_df

def add_new_feature(pivot_df):
    # Door reversal rate per cycle
    pivot_df['door_reversal_rate'] = pivot_df['total_door_reversals'] / (pivot_df['total_door_operations'] + 1)

    # Safety chain failure ratio
    pivot_df['safety_chain_issues_ratio'] = pivot_df['safety_chain_issues'] / (pivot_df['total_run_starts'] + 1)

    # Ratio of slow operations to total total_operations
    pivot_df['slow_door_operations_ratio'] = pivot_df['slow_door_operations'] / (pivot_df['total_door_operations'] + 1)

    # Slow door operation flag
    pivot_df['is_slow_door'] = (pivot_df['slow_door_operations'] > 5).astype(int)  

    return pivot_df

def label_faults(pivot_df):
    # Define fault condition (label engineering)
    pivot_df['Fault'] = (
    (pivot_df['total_door_reversals'] > 500) |
    (pivot_df['door_failure_events'] > 10) |
    (pivot_df['hoistway_faults'] > 10) |
    (pivot_df['safety_chain_issues'] > 10) |

    (pivot_df['levelling_total_errors'] > 100) |
    (pivot_df['startup_delays'] > 10) |

    (pivot_df['slow_door_operations'] > 5)     
    ).astype(int)   
    return pivot_df

