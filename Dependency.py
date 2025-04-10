import numpy as np
import pandas as pd
from scipy.linalg import lstsq
import csv

# 1. 加载数据
data = pd.read_csv('Desktop/data_0247.csv')
t = data['sample_time'].values
x_meas = data['x'].values
y_meas = data['y'].values
theta = data['theta'].values
v_L_encoder = data['v_left'].values
v_R_encoder = data['v_right'].values
# 计算实际速度(采样时间足够小)
def short_sample_calculate(x, y,theta, t):
    dt = np.diff(t)
    v_meas = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
    omega_meas = np.diff(theta) / dt
    return v_meas, omega_meas

#计算实际速度（采样时间长）
def long_sample_calculate(x, y,theta, t):
    dt = np.diff(t)
    #角度差和起始重点位置差distance
    delta_theta = np.diff(theta*np.pi/180)
    #计算角速度
    omega_meas = delta_theta / dt
    delta_distance =np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    #计算半径
    r = delta_distance/2/np.absolute(np.sin(delta_theta/2))
    #计算实际速度
    v_meas = r*omega_meas.T
    print(f"r:\n{r}")
    print(f"omega_meas:\n{omega_meas}")
    print(f"v_meas:\n{v_meas}")
    return v_meas, omega_meas

# 3. 构建观测矩阵和测量向量
v_meas, omega_meas = long_sample_calculate(x_meas, y_meas,theta, t)
H = []
y = []
H.append(v_L_encoder)
H.append(v_R_encoder)
y.append(v_meas)
y.append(omega_meas)


H_old = np.array(H)
y = np.array(y)
H = np.delete(H_old.T, 0, 0)

print(f"H:\n{H}")
print(f"y:\n{y}")

# 4. 最小二乘求解

para_hat=np.linalg.inv(H.T@H)@H.T@y
s=para_hat.reshape(1, -1)
print(f"Estimated slip parameters:  s={s}")

file_path = f"Desktop/parameters.csv"
with open(file_path, 'w' ,newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['theta1', 'theta2', 'theta3', 'theta4'])

with open(file_path, 'a' ,newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for row in s:
        writer.writerow(row)

lists = [H, y,]  # 假设这些列表已经定义

for lst in lists:
    lst.clear()