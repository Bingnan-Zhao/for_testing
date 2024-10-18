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
        #setpoint values
        self.area = area
        self.temp =25
        self.hum = 60
        self.CO2 = 500
        self.occupancy = 1
        self.fan_power = 0 (0-1)
    
    def compute_internal_heat_gain(self):
        return self.occupancy * 0.1 # 0.1 kW per person
    
    def compute_humidity_gain(self):
        return 0.05 * self.occupancy # 0.05 L/h per person
    
    def compute_CO2_gain(self):
        return 0.02 * self.occupancy
       
class PID:
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
    
def simulate_rooms(rooms, central_ac_PID, hum_PID, vent_PID, time_steps=1000, dt=1, T_env=30, C_env=450):
    temp_histories = []
    hum_histories = []
    CO2_histories = []
    ac_powers = []
    vent_powers = []

    for step in range(time_steps):
        total_temp_error = 0
        total_hum_error = 0
        total_CO2_error = 0
        total_area = sum(room.area for room in rooms)

        for room in rooms:
            temp_error = (room.set_temp - room.temp) * room.area
            hum_error = (room.set_hum - room.hum) * room.area
            CO2_error = (room.set_CO2 - room.CO2) * room.area
            total_temp_error += temp_error
            total_hum_error += hum_error
            total_CO2_error += CO2_error
        
        ac_power_temp = central_ac_PID.compute(total_temp_error, dt)
        ac_power_hum = hum_PID.compute(total_hum_error, dt)
        ac_power = max(ac_power_temp, ac_power_hum)
        ac_power = max(0, min(1, ac_power))

        vent_power = vent_PID.compute(C_env - room.CO2, dt)
        vent_power = max(0, min(1, vent_power))

        for room in rooms:
            internal_heat_gain = room.compute_internal_heat_gain()
            humidity_gain = room.compute_humidity_gain()
            CO2_gain = room.compute_CO2_gain()

            room.fan_power = ac_power

            room.temp += (-(room.temp - T_env) / room.tau_T + room.K_T * room.fan_power + internal_heat_gain) * dt 

            room.hum += (-(room.hum - H_env) / room.tau_H + room.K_H * room.fan_power + humidity_gain) * dt

            room.CO2 += (CO2_gain - room.K_V * vent_power) * dt