# 效用表达式
#
# U_social = beta[j] * (pho * (T_max - 1 / (C[j] * mu_p) - delta[j]) - (kappa * a * (mu_p ** 2) * (C[j]) ** 2))
#
# U_relay = beta[j] * (pho * (T_max - 1 / (C[j] * mu_p) - delta[j]) - (theta[j] * pi[j]))
#
# U_UAV = theta[j] * pi[j] - (kappa * a * (mu_p ** 2) * (C[j]) ** 2)

# =================================================================================
# Record
# N = numberCout    # type
# W = 15    # total number
# mu_p = 1  # 1GHz
# pho = 5000
# a =20
# kappa = 10 ** (-13)
# # T_max = 0.101
# T_max = 0.10016
#

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import OrderedDict
import pandas as pd
qtyDf = pd.DataFrame()


# ==========================绘图设置======================================
# 设置线条的颜色
color_list = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
              '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
              '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
              '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C',
              '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9',
              '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
              '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
              '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF']
# 线条标志
line_mark = ['.', ',', 'o', 'v', '^', '<', '>',
             '1', '2', '3', '4', 's', 'p', '*',
             'h', 'H', '+', 'x', 'D', 'd', '|', '_']
# 线条类型
line_style = ['-', '--',
              '-.', ':']

# ==========================================================================
#总和固定是100
unChange = 100
#每次生成随机的个数值，加起来就是 100
numberCout = 8
#想要多来几组
repeattimes = 1
def randomSameSum():
    global qtyDf
    totals = [unChange]
    i = numberCout
    nums = []
    x = np.random.randint(0, i, size=(4,))
    for i in totals:
        print('i')
        while sum(x) != i: x = np.random.randint(2, i, size=(numberCout,))
        print('x')
        nums.append(x)
        # print('nums:', nums)
    df = pd.DataFrame(nums).T
    df.rename(columns = {0:'qty'},inplace = True)
    qtyDf  = qtyDf.append(df)
    return nums
def randomnumSort():
    list_total = []
    for i in range(repeattimes):
        print('repeat', i)
        nums = randomSameSum()
        # print('nums', nums)
        list_total.append(list(nums[0]))
    # print('list_total', list_total)
    list1 = []
    for i in range(0, numberCout):
        # print(float(qtyDf.values[i]))
        list1.append(int(qtyDf.values[i]))
    list1.sort()
    # print(list1)
    list2 = []
    for i in range(len(list_total)):
        list_total[i].sort()
        list2.append([])
        for j in range(numberCout):
            list2[i].append(list_total[i][j] / unChange)
    # print('list2', list2)
    # print('list_total', list_total)
    return list2

# ==========================================================================
N = numberCout    # type
W = 15    # total number
mu_p = 1  # 1GHz
pho = 5000
a =20
kappa = 10 ** (-13)
# T_max = 0.101
T_max = 0.10009

# Incomplete scenario
def generate_result(x, theta_temp):
    # theta = theta_temp[x]
    # beta = theta * W
    beta = theta_temp[x]
    theta = []
    for i in theta_temp[x]:
        theta.append(i * 100)
    delta = [0.1] * N

    w = [0] * N
    C = [0] * N
    w[N - 1] = beta[N - 1]
    C[N - 1] = (1 / mu_p) * (((pho * beta[N - 1]) / (2 * a * kappa * (w[N - 1]))) ** (1 / 3))
    C_flag = C[N - 1]
    for j in range(N - 1, 0, -1):  # 2, 1
        # print(j)
        w_j = ((w[j] * theta[j]) / (theta[j - 1])) + beta[j - 1]
        w[j - 1] = w_j
        C_j = (1 / mu_p) * (((pho * beta[j - 1]) / (2 * a * kappa * (w[j - 1] - w[j]))) ** (1 / 3))
        C[j - 1] = C_j
        C_flag += C_j
    # print('C_list', C)
    pi = [0] * N
    pi[0] = (a * kappa * (mu_p ** 2) * (C[0] ** 2)) / (theta[0])
    for j in range(1, N):
        pi[j] = pi[j - 1] + (a * kappa * (mu_p ** 2) * (C[j] ** 2 - C[j - 1] ** 2)) / (theta[j])

    print('pi', pi)
    print('C', C)

    relay_1_list = []
    for i in range(len(theta)):
        U_UAV = theta[i] * pi[i] - (kappa * a * (mu_p ** 2) * (C[i]) ** 2)
        U_relay_asymmetry = beta[i] * (pho * (T_max - 1 / (C[i] * mu_p) - delta[i]) - (theta[i] * pi[i]))
        relay_1_list.append(U_UAV)
    return relay_1_list

# Complete scenario
def generate_complete(x, theta_temp):
    # theta = theta_temp[x]
    # beta = theta * W
    beta = theta_temp[x]
    theta = []
    for i in theta_temp[x]:
        theta.append(i * 100)
    delta = [0.1] * N

    w = [0] * N
    C = [0] * N

    pi = [0] * N
    for j in range(0, N):
        pi[j] = ((2 / pho) * (mu_p / (kappa * a))**(0.5))**(-2 / 3) / (theta[j])

    for j in range(0, N):
        C[j] = ((theta[j] * pi[j]) / (kappa * a * mu_p) )**(0.5)

    print('pi_complete', pi)
    print('C_complete', C)

    relay_2_list = []
    for i in range(len(theta)):
        U_UAV = theta[i] * pi[i] - (kappa * a * (mu_p ** 2) * (C[i]) ** 2)
        U_complete = beta[i] * (pho * (T_max - 1 / (C[i] * mu_p) - delta[i]) - (theta[i] * pi[i]))
        relay_2_list.append(U_UAV)
    return relay_2_list

# Linear Pricing
def generate_linear(x, theta_temp):
    # theta = theta_temp[x]
    # beta = theta * W
    beta = theta_temp[x]
    theta = []
    for i in theta_temp[x]:
        theta.append(i * 100)
    delta = [0.1] * N

    w = [0] * N
    C = [0] * N
    w[N - 1] = beta[N - 1]
    C[N - 1] = (1 / mu_p) * (((pho * beta[N - 1]) / (2 * a * kappa * (w[N - 1]))) ** (1 / 3))
    C_flag = C[N - 1]
    for j in range(N - 1, 0, -1):  # 2, 1
        # print(j)
        w_j = ((w[j] * theta[j]) / (theta[j - 1])) + beta[j - 1]
        w[j - 1] = w_j
        C_j = (1 / mu_p) * (((pho * beta[j - 1]) / (2 * a * kappa * (w[j - 1] - w[j]))) ** (1 / 3))
        C[j - 1] = C_j
        C_flag += C_j
    # print('C_list', C)
    pi = [0] * N
    for j in range(0, N):
        pi[j] = 0.0018 * (j + 1)    # ((2 / pho) * (mu_p / (kappa * a))**(0.5))**(-2 / 3) / (theta[j])

    print('pi', pi)
    print('C', C)

    relay_1_list = []
    for i in range(len(theta)):
        U_UAV = theta[i] * pi[i] - (kappa * a * (mu_p ** 2) * (C[i]) ** 2)
        U_relay_asymmetry = beta[i] * (pho * (T_max - 1 / (C[i] * mu_p) - delta[i]) - (theta[i] * pi[i]))
        relay_1_list.append(U_UAV)
    return relay_1_list


def generate_result_average():
    v_1_matrix = []
    v_2_matrix = []
    v_3_matrix = []
    # theta_temp = randomnumSort()

    # theta_temp = [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.22]]
    theta_temp = [[0.020, 0.03, 0.06, 0.10, 0.13, 0.16, 0.19, 0.22]]
    print(theta_temp)

    print('generate done')
    for average_time in range(0, repeattimes):
        v_1_sequence = generate_result(average_time, theta_temp)
        v_2_sequence = generate_complete(average_time, theta_temp)
        v_3_sequence = generate_linear(average_time, theta_temp)
        v_1_matrix.append(v_1_sequence)
        v_2_matrix.append(v_2_sequence)
        v_3_matrix.append(v_3_sequence)
        print(average_time)
        print(v_1_matrix)
        print(v_2_matrix)
        print(v_3_matrix)
    return v_1_matrix, v_2_matrix, v_3_matrix


v_1, v_2, v_3 = generate_result_average()
result_mean_1 = list(np.mean(v_1, axis=0))
# print("result_mean_1:", result_mean_1)
value_stage_mean_1 = OrderedDict()
for i in range(3, N):
    value_stage_mean_1[str(i)] = result_mean_1[i - 1] * 100
d_time_1 = np.array([int(x) for x in value_stage_mean_1.keys()])
e_consu_1 = value_stage_mean_1.values()

result_mean_2 = list(np.mean(v_2, axis=0))
# print("result_mean_2:", result_mean_2)
value_stage_mean_2 = OrderedDict()
for i in range(3, N):
    value_stage_mean_2[str(i)] = result_mean_2[i - 1] * 100
d_time_2 = np.array([int(x) for x in value_stage_mean_2.keys()])
e_consu_2 = value_stage_mean_2.values()

result_mean_3 = list(np.mean(v_3, axis=0))
# print("result_mean_3:", result_mean_3)
value_stage_mean_3 = OrderedDict()
for i in range(3, N):
    value_stage_mean_3[str(i)] = result_mean_3[i - 1] * 100
d_time_3 = np.array([int(x) for x in value_stage_mean_3.keys()])
e_consu_3 = value_stage_mean_3.values()


plt.figure(figsize=(8, 5))

# 设置坐标轴刻度
my_x_ticks = np.arange(3, N, 1)
plt.xticks(my_x_ticks)
plt.plot(d_time_1, e_consu_1, color='blue', alpha=0.8, marker=line_mark[17],
         linestyle='-',
         label=r'DRLCI', markersize=10, linewidth=2, clip_on=False)
plt.plot(d_time_2, e_consu_2, color='red', alpha=0.6, marker=line_mark[2],
         linestyle='--',
         label=r'Complete information', markersize=6, linewidth=2, clip_on=False)
plt.plot(d_time_3, e_consu_3, color='green', alpha=0.8, marker=line_mark[3],
         linestyle='--',
         label=r'Linear pricing', markersize=9, linewidth=2, clip_on=False)

plt.xlim(3, 7)
# plt.ylim(430, 660)
plt.legend(fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel("Type of UAVs", fontsize=17)
plt.ylabel("Utility of UAV", fontsize=17)
# plt.grid(linestyle='-.')     # 添加网格
plt.savefig("UAV_utility.pdf", dpi=500, bbox_inches='tight')    # 解决图片不清晰，不完整的问题
plt.show()


