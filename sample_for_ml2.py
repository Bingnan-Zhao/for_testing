import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
from gym import spaces
import matplotlib.pyplot as plt

# 创建一个简单的HVAC环境
class SimpleHVACEnv(gym.Env):
    """
    一个简单的HVAC环境类，模拟温度、湿度和空气质量的动态变化。
    """
    def __init__(self):
        super(SimpleHVACEnv, self).__init__()
        
        # 定义状态空间，包括 [室内温度, 室内湿度, CO₂浓度]
        #self.observation_space = spaces.Box(low=np.array([18, 30, 400]), 
        #                                    high=np.array([30, 70, 1000]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([400]), 
                                            high=np.array([1000]), dtype=np.float32)
                                                                                
        
        # 定义动作空间[新风阀门开度]
        self.action_space = spaces.Box(low=np.array([0]), 
                                       high=np.array([1]), dtype=np.float32)
        
        # 初始化环境变量
        self.reset()

    def step(self, action):
        # 解包动作值
        #cooling_power, humidifier_power, vent_opening = action
        vent_opening = action[0]

        # 模拟室内状态的变化
        #self.temp -= cooling_power * 0.5  # 模拟降温
        #self.humidity += humidifier_power * 2  # 模拟加湿
        self.co2 -= vent_opening * 30  # 模拟通风
        
        # 限制状态变量范围
        #self.temp = np.clip(self.temp, 18, 30)
        #self.humidity = np.clip(self.humidity, 30, 70)
        self.co2 = np.clip(self.co2, 300, 600)
        self.co2 += np.random.normal(0, 10)  # 模拟CO₂浓度的随机波动

        # 计算奖励
        reward = - ( abs(self.co2 - 400) / 20 + vent_opening * 0.7) #abs(self.temp - 22) + abs(self.humidity - 50) / 10 +cooling_power + humidifier_power +
        
        # 定义结束条件，如果温度或CO₂超出范围
        done = bool(self.co2 < 300 or self.co2 > 600)
        
        return np.array([self.co2]), reward, done, {}

    def reset(self):
        # 环境重置时初始化状态
        #self.temp = 25 + np.random.normal(0, 1)  # 初始温度
        #self.humidity = 45 + np.random.normal(0, 5)  # 初始湿度
        self.co2 = 500 + np.random.normal(0, 50)  # 初始CO₂浓度
        return np.array([self.co2])

# 定义DDPG的Actor网络
def create_actor():
    inputs = layers.Input(shape=(1,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # 使用sigmoid确保输出在[0,1]之间
    model = tf.keras.Model(inputs, outputs)
    return model

# 定义DDPG的Critic网络
def create_critic():
    state_input = layers.Input(shape=(1,))
    action_input = layers.Input(shape=(1,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(64, activation="relu")(concat)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # 输出Q值
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

# 创建DDPG代理类
class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.actor = create_actor()  # 主Actor网络
        self.critic = create_critic()  # 主Critic网络
        self.target_actor = create_actor()  # 目标Actor网络
        self.target_critic = create_critic()  # 目标Critic网络
        
        # 编译Critic网络
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mean_squared_error")

        # 复制网络权重到目标网络
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # 初始化经验缓冲区
        self.buffer = []
        self.buffer_size = 10000
        self.gamma = 0.99  # 折扣因子

    def update_target(self):
        for t, e in zip(self.target_actor.variables, self.actor.variables):
            t.assign(e * 0.995 + t * (1 - 0.995))
        for t, e in zip(self.target_critic.variables, self.critic.variables):
            t.assign(e * 0.995 + t * (1 - 0.995))
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idxs])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def train(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        
        # 计算目标Q值
        target_actions = tf.convert_to_tensor(self.target_actor.predict(next_states))  # 将 next_states 和 target_actions 转换为 TensorFlow 张量
        target_q_values = self.target_critic.predict([tf.convert_to_tensor(next_states), target_actions])
        y = rewards + self.gamma * (1 - dones) * target_q_values[:, 0]
        
        # 更新Critic网络
        self.critic.train_on_batch([states, actions], y)
        
        # 更新Actor网络
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            q_values = self.critic([tf.convert_to_tensor(states), actions_pred])
            actor_loss = -tf.reduce_mean(q_values)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        tf.keras.optimizers.Adam(0.001).apply_gradients(zip(grads, self.actor.trainable_variables))
        
        # 更新目标网络
        self.update_target()

# 初始化环境和代理
env = SimpleHVACEnv()
agent = DDPGAgent(env)

# 训练DDPG代理
episodes = 11  # 设置训练轮次
for ep in range(episodes):
    state = env.reset()
    ep_reward = 0
    done = False
    calculate = 0
    while not done:
        action = agent.actor.predict(state[np.newaxis])[0]
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        print(done)
        print(calculate)
        state = next_state
        ep_reward += reward
        calculate += 1
        if calculate > 2000:
            break
    print(f"Episode {ep+1}, Reward: {ep_reward}")
    print("2")
print("3")    
# 模型评估
def evaluate(agent, env, episodes=5):
    co2_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_co2 = []
        
        while not done:
            # 使用训练后的Actor网络来生成动作
            action = agent.actor.predict(state[np.newaxis])[0]
            state, _, done, _ = env.step(action)
            
            # 记录每一步的状态
            #ep_temp.append(state[0])
            #ep_humidity.append(state[1])
            ep_co2.append(state[0])
        
        #temp_history.append(ep_temp)
        #humidity_history.append(ep_humidity)
        co2_history.append(ep_co2)

    return co2_history

# 运行评估
co2_history = evaluate(agent, env)

# 绘制模型控制效果
plt.figure(figsize=(10, 5))

for co2 in co2_history:
    plt.plot(co2, label="CO₂ Level %d" %co2_history.index(co2))
plt.axhline(y=400, color="r", linestyle="--", label="Target CO₂ (400 ppm)")
plt.xlabel("Time Step")
plt.ylabel("CO₂ Concentration (ppm)")
plt.legend()
plt.title("CO₂ Control")

plt.tight_layout()
plt.show()