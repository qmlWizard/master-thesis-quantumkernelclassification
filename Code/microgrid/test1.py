import numpy as np
from pymgrid import Microgrid
from pymgrid.modules import GensetModule, BatteryModule, LoadModule, RenewableModule
import pandas as pd

# Define the number of components for a complex microgrid
num_gensets = 3
num_batteries = 2
num_renewables = 2
num_loads = 3
num_steps = 100

# Create gensets
gensets = [GensetModule(running_min_production=10 + i*5,
                        running_max_production=50 + i*10,
                        genset_cost=0.5 + i*0.1) for i in range(num_gensets)]

# Create batteries
batteries = [BatteryModule(min_capacity=0,
                           max_capacity=100 + i*50,
                           max_charge=50 + i*25,
                           max_discharge=50 + i*25,
                           efficiency=0.9 + i*0.05,
                           init_soc=0.5) for i in range(num_batteries)]

# Create renewable modules
renewables = [RenewableModule(time_series=50*np.random.rand(num_steps)) for _ in range(num_renewables)]

# Create loads
loads = [LoadModule(time_series=60*np.random.rand(num_steps)) for _ in range(num_loads)]

# Initialize the microgrid
microgrid = Microgrid([*gensets, *batteries, *renewables, *loads])

# Function to flatten the state
def flatten_state(state_tuple):
    flat_state = {}
    for index, state in enumerate(state_tuple):
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_state[f"{index}_{key}_{sub_key}"] = sub_value
                else:
                    flat_state[f"{index}_{key}"] = value
        else:
            flat_state[f"{index}"] = state
    return flat_state

# Simulate the microgrid
states = []
for step in range(num_steps):
    action = microgrid.sample_action()
    state_tuple = microgrid.step(action)
    flat_state = flatten_state(state_tuple)
    states.append(flat_state)

# Convert states to DataFrame
df = pd.DataFrame(states)

# Function to label faults (simple example: label as fault if any state value is becllow a threshold)
def label_faults(row):
    return int((row < 0).any())

# Add fault labels
df['fault'] = df.apply(label_faults, axis=1)

# Save dataset to CSV
df.to_csv('microgrid_data.csv', index=False)

print(df)
