import numpy as np
import pandas as pd
from pymgrid import Microgrid
from pymgrid.modules import GensetModule, BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt
import seaborn as sns

# Number of time steps for simulation
time_steps = 200

# Initialize modules with sufficient time series length
genset = GensetModule(running_min_production=10, running_max_production=50, genset_cost=0.5)
battery = BatteryModule(min_capacity=0, max_capacity=100, max_charge=50, max_discharge=50, efficiency=1.0, init_soc=50)
renewable = RenewableModule(time_series=50 * np.random.rand(time_steps))
load = LoadModule(time_series=60 * np.random.rand(time_steps))

# Create microgrid
microgrid = Microgrid([genset, battery, ("pv", renewable), load])

def create_fault_actions(genset_max_production, battery_max_discharge):
    # Define fault actions
    fault_actions = {
        'genset_overload': {
            'genset': [np.array([genset_max_production * 1.5, genset_max_production * 1.5])],  # 150% of max production
            'battery': [0.5 * battery_max_discharge]  # Normal operation for battery
        },
        'battery_excessive_discharge': {
            'genset': [np.array([0.5 * genset_max_production, 0.5 * genset_max_production])],  # Normal operation for genset
            'battery': [battery_max_discharge * 1.5]  # 150% of max discharge
        },
        'combined_fault': {
            'genset': [np.array([genset_max_production * 1.5, genset_max_production * 1.5])],  # 150% of max production
            'battery': [battery_max_discharge * 1.5]  # 150% of max discharge
        }
    }
    return fault_actions

# Define max production and discharge for genset and battery
genset_max_production = 50
battery_max_discharge = 50

# Create fault actions
fault_actions = create_fault_actions(genset_max_production, battery_max_discharge)

# Initialize an empty DataFrame to store the results
columns = [
    'load', 'genset_provided_energy', 'genset_co2_production', 'battery_provided_energy',
    'battery_soc', 'balancing_absorbed_energy', 'balancing_overgeneration', 'balancing_reward',
    'fault_label', 'renewable_energy', 'curtailed_energy', 'total_energy_provided', 
    'genset_cost', 'renewable_ratio', 'battery_charge_energy', 'battery_discharge_energy'
]
results_df = pd.DataFrame(columns=columns)

for t in range(time_steps):
    if t < time_steps // 2:
        fault_type = 'no_fault'
        action = None  # Normal operation (no fault)
    else:
        fault_type = np.random.choice(list(fault_actions.keys()))
        action = fault_actions[fault_type]

    if fault_type == 'no_fault':
        # Define normal action (no fault)
        action = {'genset': [np.array([0.8 * genset_max_production, 0.8 * genset_max_production])],
                  'battery': [0.3 * battery_max_discharge]}

    state_data, reward, done, info = microgrid.step(action)

    # Handle missing keys
    balancing_absorbed_energy = info['balancing'][0].get('absorbed_energy', np.nan)
    balancing_overgeneration = info['pv'][0].get('curtailment', np.nan)
    renewable_energy = info['pv'][0].get('provided_energy', np.nan)
    curtailed_energy = info['pv'][0].get('curtailed_energy', np.nan)
    total_energy_provided = info['genset'][0]['provided_energy'] + info['battery'][0]['provided_energy'] + renewable_energy
    battery_charge_energy = info['battery'][0].get('charge_energy', np.nan)
    battery_discharge_energy = info['battery'][0].get('discharge_energy', np.nan)
    
    essential_data_extracted = {
        'load': state_data['load'][0][0],  # Load
        'genset_provided_energy': info['genset'][0]['provided_energy'],  # Genset provided energy
        'genset_co2_production': info['genset'][0]['co2_production'],  # Genset CO2 production
        'battery_provided_energy': info['battery'][0]['provided_energy'],  # Battery provided energy (discharge)
        'battery_soc': state_data['battery'][0][0],  # Battery state of charge (SOC)
        'balancing_absorbed_energy': balancing_absorbed_energy,  # Balancing absorbed energy
        'balancing_overgeneration': balancing_overgeneration,  # Overgeneration (curtailment from PV)
        'balancing_reward': reward,  # Reward
        'fault_label': fault_type,  # Fault label
        'renewable_energy': renewable_energy,  # Renewable energy provided
        'curtailed_energy': curtailed_energy,  # Curtailed energy
        'total_energy_provided': total_energy_provided,  # Total energy provided
        'genset_cost': info['genset'][0].get('cost', np.nan),  # Genset cost
        'renewable_ratio': renewable_energy / (total_energy_provided + 1e-9),  # Renewable energy ratio
        'battery_charge_energy': battery_charge_energy,  # Battery charge energy
        'battery_discharge_energy': battery_discharge_energy  # Battery discharge energy
    }
    
    print(essential_data_extracted)
    
    # Append the data to the DataFrame using pd.concat
    results_df = pd.concat([results_df, pd.DataFrame([essential_data_extracted])], ignore_index=True)

# Feature Engineering: Adding new features
results_df['load_diff'] = results_df['load'].diff().fillna(0)  # Difference between consecutive loads
results_df['genset_load_ratio'] = results_df['genset_provided_energy'] / (results_df['load'] + 1e-9)  # Ratio of genset energy to load
results_df['battery_soc_diff'] = results_df['battery_soc'].diff().fillna(0)  # Difference in battery SOC
results_df['hour'] = results_df.index % 24  # Time of day

# Display the DataFrame
print(results_df)

# Optionally, save the DataFrame to a CSV file
results_df.to_csv('microgrid_fault_data_enhanced.csv', index=False)

# Visualize the results
def visualize_fault_data(df):
    plt.figure(figsize=(18, 14))

    # Plot load over time
    plt.subplot(4, 2, 1)
    sns.lineplot(data=df, x=df.index, y='load', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Load')
    plt.title('Load over Time')

    # Plot genset provided energy over time
    plt.subplot(4, 2, 2)
    sns.lineplot(data=df, x=df.index, y='genset_provided_energy', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Genset Provided Energy')
    plt.title('Genset Provided Energy over Time')

    # Plot battery state of charge over time
    plt.subplot(4, 2, 3)
    sns.lineplot(data=df, x=df.index, y='battery_soc', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Battery State of Charge (SOC)')
    plt.title('Battery State of Charge over Time')

    # Plot balancing absorbed energy over time
    plt.subplot(4, 2, 4)
    sns.lineplot(data=df, x=df.index, y='balancing_absorbed_energy', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Balancing Absorbed Energy')
    plt.title('Balancing Absorbed Energy over Time')

    # Plot new feature: load difference over time
    plt.subplot(4, 2, 5)
    sns.lineplot(data=df, x=df.index, y='load_diff', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Load Difference')
    plt.title('Load Difference over Time')

    # Plot new feature: genset to load ratio over time
    plt.subplot(4, 2, 6)
    sns.lineplot(data=df, x=df.index, y='genset_load_ratio', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Genset to Load Ratio')
    plt.title('Genset to Load Ratio over Time')

    # Plot new feature: renewable ratio over time
    plt.subplot(4, 2, 7)
    sns.lineplot(data=df, x=df.index, y='renewable_ratio', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Renewable Ratio')
    plt.title('Renewable Ratio over Time')

    # Plot new feature: total energy provided over time
    plt.subplot(4, 2, 8)
    sns.lineplot(data=df, x=df.index, y='total_energy_provided', hue='fault_label', palette='tab10')
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy Provided')
    plt.title('Total Energy Provided over Time')

    plt.tight_layout()
    plt.show()

# Visualize the results
visualize_fault_data(results_df)
