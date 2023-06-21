import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import pickle
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
import sys
from matplotlib.lines import fillStyles
from matplotlib.markers import MarkerStyle
from matplotlib.backends.backend_pdf import PdfPages
from HS_utility import record_generate_result_average
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.8               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency 100
MEMORY_CAPACITY = 2000      # memory size
N_ACTIONS = 2               # actions
N_STATES = 2                # Load, Energy, Channel,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 如果使用的是GPU，可以用本行代码得到GPU的编号用来使用，这里的电脑使用的是CPU
print('----------device-------------', device)

class Net(nn.Module):                   # 定义Net类，继承自nn.Module类
    def __init__(self, ):               # __init__是一种特殊的方法，用于定义类的属性及初始化
        super(Net, self).__init__()     # super()在类的继承里，用于子类调用父类的方法，如这里Net类继承了nn.Module里的__init__方法
        self.fc1 = nn.Linear(N_STATES, 64)     # hidden layer 1, 64 neurons
        self.fc1.weight.data.normal_(0, 1.0)   # initialization
        self.fc2 = nn.Linear(64, 64)           # hidden layer 2, 16 neurons
        self.fc2.weight.data.normal_(0, 1.0)   # initialization
        self.out = nn.Linear(64, N_ACTIONS)    # output layer
        self.out.weight.data.normal_(0, 1.0)   # initialization

    def forward(self, x):
        x = self.fc1(x)               # x = W1 * x + b1
        x = F.relu(x)                 # x = F(x)
        x = self.fc2(x)               # x = W2 * x + b2
        x = F.relu(x)                 # x = F(x)
        m = nn.Dropout(p=0.1)
        actions_valuee = m(x)
        actions_value = self.out(actions_valuee)   # actions_value = W3 * x + b3
        # actions_value = self.out(x)  # actions_value = W3 * x + b3
        return actions_value          # 返回数组actions_value


class DQN(object):
    def __init__(self, device):
        self.device = device
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0     # for target updating
        self.memory_counter = 0         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + 1))    # 初始化memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 优化器采用Adam函数
        self.loss_func = nn.MSELoss()   # 均方损失函数
        self.cost_his = []              # 记录误差loss

    def choose_action(self, x, eps=EPSILON):
        x = torch.unsqueeze(torch.FloatTensor(x).to(self.device), 0)   # 括号内先将x数据转换为浮点tensor格式，再将x由列向量转换成横向量
        # input only one sample
        if np.random.uniform() < eps:   # numpy.random.uniform(low,high,size)从一个均匀分布[low,high)中随机采样，默认值low=0， high=1， size=1， greedy
            actions_value = self.eval_net.forward(x)      # x值输入到评估网络中进行计算，输出值为向量actions_value
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()  # torch.max(a,1)，返回a矩阵每一行中最大值的那个元素及索引，troch.max()[1]， 只返回最大值的每个索引
            action = action[0]         # 如果最大值不唯一，取前面的action
        else:   # random
            action = np.random.randint(0, N_ACTIONS)   # 随机选取[0,N_ACTIONS)中的整数为动作
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))        # 在水平方向上平铺拼接数组
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY  # index记录当前memory的序号用于赋值和替换
        self.memory[index, :] = transition             # 赋值/替换memory数组中第index行的数据为当前transition
        self.memory_counter += 1                       # memory计数器 + 1

    def learn(self, double_dqn=False):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(self.device)

        if (double_dqn):                # DDQN
            # q_eval w.r.t the action in experience
            q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

            q_tp1_eval = self.eval_net(b_s_).detach()  # shape (batch, 1)
            a_prime = q_tp1_eval.max(1)[1].view(BATCH_SIZE, 1)

            q_target_tp1_values = self.target_net(b_s_).detach()
            q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.type(torch.int64))
            # q_target_s_a_prime = q_target_s_a_prime.squeeze()
            q_target_d = b_r + GAMMA * q_target_s_a_prime.view(BATCH_SIZE, 1)   # shape (batch, 1)
            loss_d = self.loss_func(q_eval, q_target_d)

            self.optimizer.zero_grad()
            loss_d.backward()
            self.optimizer.step()
        else:                        # DQN
            # q_eval w.r.t the action in experience
            q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
            q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
            q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # return loss

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def transition(state, action):
    import random
    alpha = 0.1
    E_max = 15
    drop_flag = 0
    reward_1 = 0.0
    reward_2 = 0.0
    state_ = [0, 0]
    # ====================================================================
    Utility_offload = record_generate_result_average()
    # ====================================================================
    # No trading, compute by itself
    if action == 0:
        if (state[1] < 400):
            state_[1] = 0
            reward_1 = 3
        elif (state[1] >= 800) and (state[1] < 800):
            state_[1] = state[1] - 400
            reward_1 = 2
        elif (state[1] >= 800) and (state[1] <= 1200):
            state_[1] = state[1] - 400
            reward_1 = 1
        else:
            drop_flag = 1
            state_[1] = 0
            reward_1 = -1
        # energy queue judgement
        if (state[0] >= 1):
            state_[0] = state[0] - 1
            reward_2 = 1
        elif (state[0] >= 2):
            state_[0] = state[0] - 2
            reward_2 = 2
        else:
            drop_flag = 1
            state_[0] = state[0]
            reward_2 = 0
    # ====================================================================
    # Trading, buy UAV
    elif action == 1:
        if (state[1] < 400):
            state_[1] = 0
            reward_1 = Utility_offload + 1

        elif (state[1] >= 400) and (state[1] < 800):
            state_[1] = 0
            reward_1 = Utility_offload + 3

        elif (state[1] >= 800) and (state[1] <= 1200):
            state_[1] = state[1] - 800
            reward_1 = Utility_offload + 1

        else:
            drop_flag = 1
            state_[1] = 0
            reward_1 = Utility_offload - 1
        # energy queue judgement
        if (state[0] >= 3):
            state_[0] = state[0] - 3
            reward_2 = 3
        elif (state[0] >= 3):
            state_[0] = - 3
            reward_2 = 3
        elif (state[0] >= 3):
            state_[0] = state[0] - 3
            reward_2 = 3
        else:
            drop_flag = 1
            state_[0] = state[0]
            reward_2 = 0
    # ====================================================================
    reward = reward_1 - alpha * reward_2
    Task_need = random.randint(0, 850)
    # Task_need = np.random.choice([400, 800])
    E_in = np.random.choice([0, 1, 2, 3])
    state_[0] = min(max(state_[0] + E_in, 0), E_max)
    state_[1] = Task_need + state_[1]
    return drop_flag, state_, reward


def transition_random(state, action):
    import random
    alpha = 0.1
    E_max = 15
    drop_flag = 0
    reward_1 = 0.0
    reward_2 = 0.0
    state_ = [0, 0]
    # ====================================================================
    Utility_offload = record_generate_result_average()
    # ====================================================================
    # No trading, compute by itself
    if action == 0:
        if (state[1] < 400):
            state_[1] = 0
            reward_1 = 3
        elif (state[1] >= 800) and (state[1] < 800):
            state_[1] = state[1] - 400
            reward_1 = 2
        elif (state[1] >= 800) and (state[1] <= 1200):
            state_[1] = state[1] - 400
            reward_1 = 1
        else:
            drop_flag = 1
            state_[1] = 0
            reward_1 = -1
        # energy queue judgement
        if (state[0] >= 1):
            state_[0] = state[0] - 1
            reward_2 = 1
        elif (state[0] >= 2):
            state_[0] = state[0] - 2
            reward_2 = 2
        else:
            drop_flag = 1
            state_[0] = state[0]
            reward_2 = 0
    # ====================================================================
    # Trading, buy UAV
    elif action == 1:
        if (state[1] < 400):
            state_[1] = 0
            reward_1 = Utility_offload + 1

        elif (state[1] >= 400) and (state[1] < 800):
            state_[1] = 0
            reward_1 = Utility_offload + 3

        elif (state[1] >= 800) and (state[1] <= 1200):
            state_[1] = state[1] - 800
            reward_1 = Utility_offload + 1

        else:
            drop_flag = 1
            state_[1] = 0
            reward_1 = Utility_offload - 1
        # energy queue judgement
        if (state[0] >= 3):
            state_[0] = state[0] - 3
            reward_2 = 3
        elif (state[0] >= 3):
            state_[0] = - 3
            reward_2 = 3
        elif (state[0] >= 3):
            state_[0] = state[0] - 3
            reward_2 = 3
        else:
            drop_flag = 1
            state_[0] = state[0]
            reward_2 = 0
    # ====================================================================
    reward = reward_1 - alpha * reward_2
    Task_need = random.randint(0, 850)
    # Task_need = np.random.choice([400, 800])
    E_in = np.random.choice([0, 1, 2, 3])
    state_[0] = min(max(state_[0] + E_in, 0), E_max)
    state_[1] = Task_need + state_[1]
    return drop_flag, state_, reward


if __name__ == '__main__':
    time_line = 100000
    Num_End = 100
    Num_End_d = 2000
    Energy_init = 0

    ep_r = 0.0
    ep_r_random = 0.0
    n_outage = 0
    n_outage_random = 0
    energy_r = 0.0
    energy_random_r = 0
    U_r = 0.0
    U_random_r = 0.0
    # second average
    ep_r_d = 0.0
    ep_r_random_d = 0.0
    n_outage_d = 0
    n_outage_random_d = 0
    energy_r_d = 0.0
    energy_random_r_d = 0
    U_r_d = 0.0
    U_random_r_d = 0.0
    Task_need = 1000

    x_axis = []
    Markov_slot_1 = []
    outage_slot_1 = []
    energy_slot_1 = []
    U_slot_1 = []
    y_axis_random = []
    outage_random = []
    energy_random = []
    U_random = []

    x_axis_d = []
    Markov_slot_1_d = []
    outage_slot_1_d = []
    energy_slot_1_d = []
    U_slot_1_d = []
    y_axis_random_d = []
    outage_random_d = []
    energy_random_d = []
    U_random_d = []

    dqn = DQN(device=device)        # 1. Initialize Q-network of DQN and empty buffer
    s = np.array([Energy_init, Task_need])
    s_random = np.array([Energy_init, Task_need])

    # 2. Set iteration time T and training interval delta
    for i_episode in range(time_line):   # 3. for episode = 1, 2, ..., T do
        # Step 1: choose action according to now state
        a = dqn.choose_action(s)         # 4. Generate trading action \textbf{a}
        # Step 2: obtain the reward and the next state
        drop_flag, s_, r = transition(s, a)   # 5. Observe next state s_{t+1} and immediate reward r
        dqn.store_transition(s, a, r, s_)
        # print(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter >= MEMORY_CAPACITY:
            dqn.learn(double_dqn=True)
            # 6. sample a mini-batch from DQN memory
            # 7. Update DQN network and its target network
        s = s_
        energy_r += s[0]
        n_outage += drop_flag

        # run random
        a_random = np.random.randint(0, 2)
        drop_flag_random, s_random_, r_random = transition_random(s_random, a_random)
        ep_r_random += r_random
        s_random = s_random_
        energy_random_r += s_random[0]
        n_outage_random += drop_flag_random

        if i_episode % Num_End == 99:     # if t mod delta == 0 then
            ave_r = ep_r / Num_End
            ave_r_random = ep_r_random / Num_End
            ave_drop = n_outage / Num_End
            ave_drop_random = n_outage_random / Num_End
            ave_energy = energy_r / Num_End
            ave_energy_random = energy_random_r / Num_End
            ave_U = U_r / Num_End
            ave_U_random = U_random_r / Num_End

            ep_r_d += ave_r
            ep_r_random_d += ave_r_random
            n_outage_d += ave_drop
            n_outage_random_d += ave_drop_random
            energy_r_d += ave_energy
            energy_random_r_d += ave_energy_random
            U_r_d += ave_U
            U_random_r_d += ave_U_random

            x_axis.append(i_episode)
            Markov_slot_1.append(ave_r)
            outage_slot_1.append(ave_drop)
            energy_slot_1.append(ave_energy)
            U_slot_1.append(ave_U)

            y_axis_random.append(ave_r_random)
            outage_random.append(ave_drop_random)
            energy_random.append(ave_energy_random)
            U_random.append(ave_U_random)

            ep_r = ep_r_random = 0.0
            n_outage = n_outage_random = 0.0
            energy_r = energy_random_r = 0.0
            U_r = U_random_r = 0.0

            if i_episode % Num_End_d == 1999:
                ave_r_d = ep_r_d / (Num_End_d / Num_End)
                ave_r_random_d = ep_r_random_d / (Num_End_d / Num_End)
                ave_drop_d = n_outage_d / (Num_End_d / Num_End)
                ave_drop_random_d = n_outage_random_d / (Num_End_d / Num_End)
                ave_energy_d = energy_r_d / (Num_End_d / Num_End)
                ave_energy_random_d = energy_random_r_d / (Num_End_d / Num_End)
                ave_U_d = U_r_d / (Num_End_d / Num_End)
                ave_U_random_d = U_random_r_d / (Num_End_d / Num_End)

                x_axis_d.append(i_episode)
                Markov_slot_1_d.append(ave_r_d)
                outage_slot_1_d.append(ave_drop_d)
                energy_slot_1_d.append(ave_energy_d)
                U_slot_1_d.append(ave_U_d)

                y_axis_random_d.append(ave_r_random_d)
                outage_random_d.append(ave_drop_random_d)
                energy_random_d.append(ave_energy_random_d)
                U_random_d.append(ave_U_random_d)

                ep_r_d = ep_r_random_d = 0.0
                n_outage_d = n_outage_random_d = 0.0
                energy_r_d = energy_random_r_d = 0.0
                U_r_d = U_random_r_d = 0.0
                print('[DQN] EP: ', i_episode, '| Ep_r: ', round(ave_r_d, 2), '| drop_r: ', round(ave_drop_d, 2), '| energy_r: ', round(ave_energy_d, 2))
                print('[RND] Ep: ', i_episode, '| Ep_r: ', round(ave_r_random_d, 2), '| drop_r: ', round(ave_drop_random_d, 2), '| energy_r: ', round(ave_energy_random_d, 2))
                print('--')

    plt.figure(figsize=(7, 5))
    plt.xlim(MEMORY_CAPACITY, time_line)
    plt.plot(x_axis, Markov_slot_1, color='blue', alpha=0.3)
    plt.plot(x_axis, y_axis_random, color='red', alpha=0.3)

    plt.plot(x_axis_d, Markov_slot_1_d, label='C-DQN', color='blue', alpha=0.8)
    plt.plot(x_axis_d, y_axis_random_d, label='Random', color='red', alpha=0.8)
    plt.ylabel('reward')
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig("Contract-guided DQN_1.png", dpi=500,
                bbox_inches='tight')  # Solve the problem of unclear and incomplete pictures
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.xlim(MEMORY_CAPACITY, time_line)
    # plt.ylim(0.0, 0.5)
    plt.plot(x_axis, outage_slot_1, color='blue', alpha=0.3)
    plt.plot(x_axis, outage_random, color='red', alpha=0.3)

    plt.plot(x_axis_d, outage_slot_1_d, label='C-DQN', color='blue', alpha=0.8)
    plt.plot(x_axis_d, outage_random_d, label='Random', color='red', alpha=0.8)
    plt.ylabel('average task drop')
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig("Contract-guided DQN_2.png", dpi=500,
                bbox_inches='tight')  # Solve the problem of unclear and incomplete pictures
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.xlim(MEMORY_CAPACITY, time_line)
    plt.plot(x_axis, energy_slot_1, color='blue', alpha=0.3)
    plt.plot(x_axis, energy_random, color='red', alpha=0.3)

    plt.plot(x_axis_d, energy_slot_1_d, label='C-DQN', color='blue', alpha=0.8)
    plt.plot(x_axis_d, energy_random_d, label='Random', color='red', alpha=0.8)
    plt.ylabel('energy queue')
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig("Contract-guided DQN_3.png", dpi=500,
                bbox_inches='tight')  # Solve the problem of unclear and incomplete pictures
    plt.show()

    pickle.dump(x_axis_d, open("./pickle/x_axis", "wb"))
    pickle.dump(Markov_slot_1_d, open("./pickle/Markov_slot_3", "wb"))
    pickle.dump(outage_slot_1_d, open("./pickle/outage_slot_3", "wb"))
    pickle.dump(energy_slot_1_d, open("./pickle/energy_slot_3", "wb"))


