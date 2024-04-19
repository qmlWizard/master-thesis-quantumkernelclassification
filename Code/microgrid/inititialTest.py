import numpy as np
from pymgrid import Microgrid
from pymgrid.modules import GensetModule, BatteryModule, LoadModule, RenewableModule


genset = GensetModule(running_min_production=10,
                      running_max_production=50,
                      genset_cost=0.5)

battery = BatteryModule(min_capacity=0,
                        max_capacity=100,
                        max_charge=50,
                        max_discharge=50,
                        efficiency=1.0,
                        init_soc=0.5)

# Using random data
renewable = RenewableModule(time_series=50*np.random.rand(50))

load = LoadModule(time_series=60*np.random.rand(50))

microgrid = Microgrid([genset, battery, ("pv", renewable), load])
print("----------------------------------------------------------------------------------------------------------------")
print(microgrid)
print("----------------------------------------------------------------------------------------------------------------")
print(microgrid.sample_action())
print("----------------------------------------------------------------------------------------------------------------")

for j in range(10):
     action = microgrid.sample_action()
     microgrid.step(action)
     
print(microgrid.get_log(drop_singleton_key=True))