import matplotlib.pyplot as plt
import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.intergral =0
    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.intergral += error * dt
        derivative = (error - self.last_error) / dt

        #self.intergral = max(min(self.intergral, 1), 0)

        output = self.Kp * error + self.Ki * self.intergral + self.Kd * derivative
        self.last_error = error
        return output


def room_tempurature_simulation(pid_controller, ENV_temp, Initial_temp, time_steps, dt):
    tau = 200 #indoor thermal time constant
    Ku = 0.1 #AC heating or cooling constant
    T_room = Initial_temp
    temperatures = [T_room]
    powers = []
    for steps in range(time_steps):
        current_temp = temperatures[-1]
        control_signal = pid_controller.compute(current_temp, dt)
        print("control_signal is",control_signal)

        power_AC = max(0, min(1, control_signal))
        #power_AC = control_signal
        print("power_AC is",power_AC)
        powers.append(power_AC)
        
        next_temperature = current_temp + ((ENV_temp - current_temp)/tau + Ku * power_AC) * dt
        temperatures.append(next_temperature)
        
        if steps > time_steps//2:
            ENV_temp += np.random.normal(0, 0.05)
    return temperatures, powers

#input parameters
Kp = 2.0
Ki = 0.1
Kd = 1.0
setpoint = 27

pid = PID(Kp, Ki, Kd, setpoint)

time_steps = 1000
dt = 1
temperatures, powers = room_tempurature_simulation(pid, ENV_temp=20, Initial_temp=20, time_steps=time_steps, dt=dt)
print(len(temperatures))
print(len(powers))
# 绘制室内温度和空调功率的变化曲线
plt.figure(figsize=(12, 6))

# 室内温度变化
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, time_steps * dt, dt), temperatures[1:], label='Room Temperature')
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint (Target Temperature)')
plt.title('Room Temperature Control using PID')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

# 空调功率变化
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, time_steps * dt, dt), powers, label='AC Power (%)', color='orange')
plt.title('AC Power Output over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Power (0 to 1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()