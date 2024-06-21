import numpy as np
import pandas as pd
from pymgrid import Microgrid
from pymgrid.modules import GensetModule, BatteryModule, LoadModule, RenewableModule

# Initialize modules
genset = GensetModule(running_min_production=10, running_max_production=50, genset_cost=0.5)
battery = BatteryModule(min_capacity=0, max_capacity=100, max_charge=50, max_discharge=50, efficiency=1.0, init_soc=0.5)
renewable = RenewableModule(time_series=50*np.random.rand(100))
load = LoadModule(time_series=60*np.random.rand(100))

# Create microgrid
microgrid = Microgrid([genset, battery, ("pv", renewable), load])

# Simulate microgrid operation
time_steps = 100
data = []

for t in range(time_steps):
    microgrid.step(microgrid.sample_action(strict_bound=False))
    
state = microgrid.get_log(drop_singleton_key=True)

state.to_csv('test.csv')
