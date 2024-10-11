#lib checking ...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mt
import threading as td
import math as math
import geopandas as gp
import netCDF4 as nc
import tensorflow as tf
import os as os


#inputdir = 'C:\\Users\\n12130745\\Code\\Python_Code_Files\\for_testing\\'

#data = pd.read_excel(inputdir+"input_condition_details.xlsx")
#print(data)

class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, setpoint=0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
    def compute(self, measurement_value):
        error = self.setpoint - measurement_value
        self.integral += error
        derivative = error - self.last_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return output
def system_stimulation(pid_controller, initial_value, time_steps):
    measurement_value = [initial_value]
    for i in range(time_steps):
        current_value = measurement_value[-1]
        controlle_output = pid_controller.compute(current_value)
        next_value = current_value + controlle_output * 0.1 + np.random.normal(0, 0.5)
        measurement_value.append(next_value)
    return measurement_value


#inoput parameters
Kp = 1.2
Ki = 0.05
Kd = 0.5
setpoint = 40
pid = PID(Kp, Ki, Kd, setpoint) #create PID controller
measurements = system_stimulation(pid, initial_value=25, time_steps=100)


plt.figure(figsize=(12, 6))
plt.plot(measurements, label='System output (tempurature)')
plt.axhline(y=setpoint, color='red', linestyle='--', label='Setpoint (target tempurature)')
plt.title('PID controller stimulation')
plt.xlabel('Time steps')
plt.ylabel('Tempurature')
plt.legend()
plt.show()