#Environment

import numpy as np
from collections import deque
import random

class GridEnvironment:
    def __init__(self, N, demand_data, solar_data, wind_data, day_index, timestep_length):
        self.current_timestep = 0
        self.total_timesteps = int(24 / timestep_length) #Assuming episode is 24 hours
        self.timestep_length = timestep_length  # Length of each timestep in hours
       
        # Need to think about what start / stop time we do. 12am-12am? 4am-4am etc <-- Carla comment: recommend 12am-11:59pm
        self.N = N  # Number of EVs
        self.state_size = 3 # + N + 1 +N # State Size, includes time, and SoC
        self.action_size =  21 #3 **N # State Size

        self.demand_data = demand_data  
        self.solar_data = solar_data
        self.wind_data = wind_data
        self.day_index = day_index

        # Initialize with day 0 data (96 points = 24 hours of 15 min data)
        self.demand = demand_data[day_index,0]  
        self.solar = solar_data[day_index,0]
        self.wind = wind_data[day_index,0]

    
        self.P_EV = [0] * N  # Power status of each EV (non are connected to grid)
        # TODO Answer: If each episode is finite, how does the SoC status roll over to next episode and how does RL agent learn this?
        self.SoC = [50] * N  # SoC status of each EV (non are connected to grid), used by environment ONLY 
    

    def reset(self, day):
        self.current_timestep = 0
        self.demand = 0  
        self.solar = 0  
        self.wind = 0  
        self.P_EV = [0] * self.N  
        #CHANGE WHEN GOING THROUGH MORE THAN ONE DAY OF DATA
        return self.get_state()

    def decode_action(self, action_index):
        """
        Decode a single integer action into actions for each EV.
        
        Args:
        - action: The single integer action to decode that comes from the NN model.
        - N: The number of EVs.
        
        Returns:
        - A list of actions for each EV.
        """
        #action from NN is index,
        #set action_list=[-1,0,1]
        #return action_list[nnoutput_index]
        #actions_list=[-1, 0, 1]
        #action=action_list[action]
        #actions = []
        #for _ in range(self.N):
         #  actions.append(action % 3 - 1)  # Decoding to get -1, 0, 1
          #  action //= 3
        #return actions[::-1]  # Reverse to match the original order
        action = (action_index - 10) / 10.0  # Converts 0 to 200 into -1.0 to 1.0
        return action



    def battery_voltage(self, soc):
        min_voltage = 3.0  # Minimum voltage at 0% SoC
        max_voltage = 4.2  # Maximum voltage at 100% SoC
        soc_array = np.array(soc)
        return 4.2 * (soc_array / 100)

    def get_PEV(self, actions):
        #MAX's CODE
        #ACTION IS A VECTOR OF 0s 1s, -1s
        #return power output of each EV (P_EV) & the SOC for the next state

        #Based on rouh calculations, need roughly 405 EVs
        #10 groups of 41 EVs?
        #Does just multiplying work?
        timestep = (1/60)  # 1 minute
        max_power = (11)/3500  # Maximum power in kW, attempting to scale 100 EVs?
        battery_capacity = (50)/3500 # Battery capacity, scaled by entire system, each represents 100 EVs, testing only having 1 mega EV
        charge_efficiency = 0.90 #changed to .95 from .9
        discharge_efficiency = 0.90
        min_soc = 20
        max_soc = 80
        self.P_EV = [0] * self.N  # Reset power for each EV
        """
        voltage = self.battery_voltage(soc)  # Calculate voltage for each SoC
        power = max_power * voltage / 4.2  # Calculate power for each SoC based on its voltage NEED TO CHECK THIS

        # Ensure new_soc is of a floating point type to accommodate fractional changes
        current_soc=np.array(soc, dtype=float)
        new_soc = np.copy(current_soc) # Cast to float to prevent UFuncTypeError
        
        powerEV = np.zeros_like(soc, dtype=float)  # Initialize powerEV array with zeros

        
        # Charging
        charge_indices = (actions == 1) & (current_soc < max_soc)
        added_energy = np.minimum(power[charge_indices] * timestep, (max_soc - current_soc[charge_indices]) / 100 * battery_capacity) * charge_efficiency
        powerEV[charge_indices] = -(added_energy / timestep)  # Negative with respect to grid
        new_soc[charge_indices] += added_energy / battery_capacity * 100
        new_soc[charge_indices] = np.minimum(new_soc[charge_indices], max_soc)

        # Discharging
        discharge_indices = (actions == -1) & (current_soc > min_soc)
        used_energy = np.minimum(power[discharge_indices] * timestep, (current_soc[discharge_indices] - min_soc) / 100 * battery_capacity) * discharge_efficiency
        powerEV[discharge_indices] = used_energy / timestep  # Positive with respect to grid
        new_soc[discharge_indices] -= used_energy / battery_capacity * 100
        new_soc[discharge_indices] = np.maximum(new_soc[discharge_indices], min_soc)

        # Idle
        idle_indices = (actions == 0)
        powerEV[idle_indices] = 0  # No power exchange for idle

        return new_soc, powerEV
        """
        if actions > 0:  # Charging
            eligible_evs = [i for i in range(self.N) if self.SoC[i] < max_soc]
        elif actions < 0:  # Discharging
            eligible_evs = [i for i in range(self.N) if self.SoC[i] > min_soc]
        else:
            eligible_evs = []
        #print('action test', actions)

        # Determine the number of EVs to affect based on the action percentage
        num_evs_affected = int(abs(actions) * self.N)
        # Ensure we do not sample more EVs than are eligible
        num_evs_affected = min(num_evs_affected, len(eligible_evs))
        selected_evs = random.sample(eligible_evs, num_evs_affected)
        
        voltages= self.battery_voltage(self.SoC)
        SoC_after_action=self.SoC
        PEV_after_action=self.P_EV
        for i in selected_evs:
            if actions > 0:  # Charging
                power = max_power * voltages[i] / 4.2   
                energy_added = power * timestep * charge_efficiency
                SoC_after_action[i] = SoC_after_action[i] + energy_added / battery_capacity * 100
                PEV_after_action[i] = power # Charging ADDs to Demand (should be negative here)
            elif actions < 0:  # Discharging
                power = max_power * voltages[i] /4.2  
                energy_used = power * timestep * discharge_efficiency #energy used will be negative
                SoC_after_action[i] = SoC_after_action[i] - energy_used / battery_capacity * 100
                PEV_after_action[i] = -1*power  ## Discharging subtracts from Demand (should be positive here)
        PEV_after_action=np.array(PEV_after_action)
        return SoC_after_action, PEV_after_action

    def step(self, action):
        #Apply action-> calculate reward -> update state of environment
    
        #Apply action, update P_EV states
        actions=self.decode_action(action)
        actions = np.array(actions)
        #print('action decoded', actions)
        
        next_SoC, next_P_EV = self.get_PEV(actions) #Returns updated SoC & Power levels of each EV AFTER action
        self.SoC = next_SoC.copy()
        #why is this not highlighted green
        self.P_EV = next_P_EV.copy()
        

        #Calculate Reward based upon action within same timestep
        reward = self.calculate_reward(next_P_EV, actions) 
        
        #Move env forward one timestep
        self.current_timestep += 1
        done = self.current_timestep >= self.total_timesteps-1

        if not done:
            next_demand, next_solar, next_wind, next_SoC = self.get_state()
        else:
        # Handle the terminal state here, could be resetting or providing terminal values
        # Make sure to handle the case where you don't have a next state to provide
            next_demand, next_solar, next_wind, next_P_EV, next_SoC= 0, 0, 0, [0] * self.N, [0] * self.N
        
    
        return reward, done, next_demand, next_solar, next_wind , next_P_EV, next_SoC
   
#NEED TO FOCUS ON THE SEQUENCE OF, OBSERVE STATE, CALCULATE ACTION, CALCULATE REWARD etc

    def calculate_reward(self, next_P_EV, action):
        current_demand, current_solar, current_wind, current_SoC = self.get_state()
        
        # Calculate Reward
        reward= -np.abs(current_demand + np.sum(next_P_EV)- (current_solar + current_wind))
    

        return reward

    def get_state(self):
        # Access the current timestep's data correctly
        current_demand = self.demand_data[self.day_index, self.current_timestep]
        current_solar = self.solar_data[self.day_index, self.current_timestep]
        current_wind = self.wind_data[self.day_index, self.current_timestep]
        current_SoC=self.SoC


        # Depending on your needs, return these values directly or along with other state information
        return current_demand, current_solar, current_wind, current_SoC


    def render(self):
        # Is this where we get our animations?!
        pass