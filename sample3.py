import matplotlib.pyplot as plt
import numpy as np

class PID:
    def __init__(self,Kp,Ki,Kd,setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.intergral = 0
    def compute(self,measurement,dt):
        error = self.setpoint - measurement
        self.intergral += error * dt
        derivative =  (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.intergral +self.Kd * derivative
        self.last_error = error
        return output

def room_control_simulation(temp_pid, hum_pid, co2_pid, T_ENV, H_ENV, C_PEOPLE, time_steps, dt):
    tau_T = 300 # indoor thermal time constant
    tau_H = 400 # indoor humidity time constant
    K_T = -0.04  # AC heating or cooling efficiency (up or dowm is + or -)
    K_H = -0.09  # AC humification efficiency (up or dowm is + or -)
    K_V = 0.25   # CO2 ventilation efficiency

    T_room = 25
    H_room = 60
    C_room = 500

    tempuratures = [T_room]
    humidities = [H_room]
    co2_levels = [C_room]

    ac_powers = [0]
    vent_powers = [0]

    for step in range(time_steps):
        current_T = tempuratures[-1]
        current_H = humidities[-1]
        current_C = co2_levels[-1]

        ac_power_temp = -temp_pid.compute(current_T, dt)
        ac_powers_hum = -hum_pid.compute(current_H, dt)
        ac_power = max(ac_power_temp, ac_powers_hum)
        ac_power = max(0, min(1, ac_power))


        vent_power = -co2_pid.compute(current_C, dt)
        vent_power = max(0, min(1, vent_power))

        ac_powers.append(ac_power)
        vent_powers.append(vent_power)

        next_temperature = current_T + ((T_ENV - current_T) / tau_T + K_T * ac_power) * dt
        next_humidity = current_H + ((H_ENV - current_H) / tau_H + K_H * ac_power) * dt
        next_co2 = current_C + C_PEOPLE - K_V * vent_power * dt

        tempuratures.append(next_temperature)
        humidities.append(next_humidity)
        co2_levels.append(next_co2)

        if step > time_steps // 2:
            T_ENV += np.random.normal(0, 0.05)
            H_ENV += np.random.normal(0, 0.05)
            C_PEOPLE += np.random.normal(0, 0.001)
    return tempuratures, humidities, co2_levels, ac_powers, vent_powers



# input parameters
setpoint_temp = 24
setpoint_hum = 50
setpoint_co2 = 550

temp_pid = PID(Kp=1, Ki=0.005, Kd=0.5, setpoint=setpoint_temp)
hum_pid = PID(Kp=1.5, Ki=0.005, Kd=0.3, setpoint=setpoint_hum)
co2_pid = PID(Kp=1.5, Ki=0.001, Kd=1, setpoint=setpoint_co2)

time_steps = 2000
dt = 1
tempuratures, humidities, co2_levels, ac_powers, vent_powers = room_control_simulation(
    temp_pid, hum_pid, co2_pid, T_ENV=30, H_ENV=70, C_PEOPLE=0.15, time_steps=time_steps, dt=dt)

plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)

plt.plot(np.arange(0,(time_steps+1)*dt,dt),tempuratures,label='Room Temperature')
plt.axhline(y=setpoint_temp, color='r', linestyle='--', label='Setpoint(Target Temperature)')
plt.title('Room Temperature, Humidity,and CO2 Control using PID')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(np.arange(0,(time_steps+1)*dt,dt),humidities,label='Room Humidity')
plt.axhline(y=setpoint_hum, color='blue', linestyle='--', label='Setpoint(Target Humidity)')
plt.ylabel('Humidity (%)')
plt.legend()
plt.grid(True)
#print("co2_levels is",co2_levels)
plt.subplot(4, 1, 3)
plt.plot(np.arange(0,(time_steps+1)*dt,dt),co2_levels,label='Room CO2 Level')
plt.axhline(y=setpoint_co2, color='green', linestyle='--', label='Setpoint(Target CO2 Level)')
plt.ylabel('CO2 Level (ppm)')
plt.grid(True)

plt.legend()

plt.subplot(4, 1, 4)
plt.plot(np.arange(0,(time_steps+1)*dt,dt),ac_powers,label='AC Power (%)',color='orange')
plt.plot(np.arange(0,(time_steps+1)*dt,dt),vent_powers,label='Ventilation Power (%)',color='purple')
plt.ylabel('Power (%)')
plt.legend()
plt.xlabel('Time (s)')
plt.show()

