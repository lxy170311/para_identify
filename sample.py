import numpy as np
import test1
import cv2
import time
import cv2.aruco as aruco
import csv

file_path = f"Desktop/parameters.csv"
#创建初始csv文件用于保存估计的参数
with open(file_path, 'w' ,newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['theta1', 'theta2', 'theta3', 'theta4'])

# 定义全局变量
sample_time=[]
v_r,v_l,y_meas,x_meas,theta_meas=[],[],[],[],[]
V_LS,A_LS=[],[]
num_samples=100


cameraMatrix = []
distCoeffs = []
# 初始化摄像头
cap = cv2.VideoCapture(0)
#aruco marker速度换算
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

#编码器读取左轮速
def get_left_speed():
    return 0.5

#编码器读取右轮速
def get_right_speed():
    return 0.5

#获取机器人实际位姿
def get_marker_info():
    ret, frame = cap.read()
    # 检测Marker
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
        #tvecs矩阵返回的是marker在相机坐标系下的位置
        #rvecs矩阵返回的是marker在相机坐标系下的姿态
        x = tvecs[0][0]
        y = tvecs[0][1]
        theta = rvecs[0][0]
    return x, y, theta

while True:
    sample_time.append(time.time())
    left_speed = get_left_speed()
    right_speed = get_right_speed()
    v_l.append(left_speed)
    v_r.append(right_speed)
    x, y, theta = get_marker_info()
    x_meas.append(x)
    y_meas.append(y)
    theta_meas.append(theta)

    # num_samples个样本后进行线性拟合
    if len(sample_time)>=num_samples:
        v_meas, omega_meas = test1.long_sample_calculate(x_meas, y_meas,theta, sample_time)
        A_LS.append(v_l)
        A_LS.append(v_r)
        V_LS.append(v_meas)
        V_LS.append(omega_meas)
        A_LS_old = np.array(A_LS)
        V_LS = np.array(V_LS)
        #因为原有测量数据为位移和角度差，计算后得到的速度采样V_LS实际上只有num_samples-1个数据，A_LS需要删除第一个数据
        A_LS = np.delete(A_LS_old.T, 0, 0)
        #最小二乘求解
        para_hat=np.linalg.inv(A_LS.T@A_LS)@A_LS.T@V_LS
        #线性拟合后的参数（滑移率）
        s=para_hat.reshape(1, -1)
        print(f"Estimated slip parameters:  s={s}")
        #储存估计的滑移率存储
        with open(file_path, 'w' ,newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in s:
                writer.writerow(row)
        #清空数据
        sample_time.clear()
        v_l.clear()
        v_r.clear()
        x_meas.clear()
        y_meas.clear()
        theta_meas.clear()
        A_LS.fill(0)
        V_LS.fill(0)


