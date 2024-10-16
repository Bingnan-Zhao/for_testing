import numpy as np
import matplotlib.pyplot as plt

class Room:
    def __int__(self, room_id, tau_T, tau_H, K_T, K_H, K_V, set_temp, set_hum, set_CO2, area):
        self.room_id = room_id
        self.tau_T = tau_T
        self.tau_H = tau_H
        self.K_T = K_T
        self.K_H = K_H
        self.K_V = K_V
        self.set_temp = set_temp
        self.set_hum = set_hum
        self.set_CO2 = set_CO2
        #initial values
        self.area = area
        self.temp =25
        self.hum = 60
        self.CO2 = 500
        self.occupancy = 1
    
    def compute_internal_heat_gain(self):
        return self.occupancy * 0.1 # 0.1 kW per person
    
    def compute_humidity_gain(self):
        return 0.05 * self.occupancy # 0.05 L/h per person
    
    def compute_CO2_gain(self):
        return 0.02 * self.occupancy
       
class CentralAC:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0
    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return max(0, min(1, output))
    
def simulate_rooms(rooms, central_ac, time_steps=1000, dt=1, T_env=30, C_env=450):
    temp_histories = []
    hum_histories = []
    CO2_histories = []
    ac_powers = []

    for step in range(time_steps):
        total