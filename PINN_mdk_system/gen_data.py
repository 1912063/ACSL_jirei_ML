
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyDOE import lhs


np.random.seed(123)


def gen_learning_data(time, tau_max, num_points):
    """
    学習用データを生成する関数
    time: シミュレーション時間
    tau_max: 最大入力トルク
    num_points: データ生成個数
    """
    # lhs: ラテンハイパーキューブサンプリング(乱数生成の凄い版)
    learning_data = lhs(2, num_points)

    learning_data[:, 0] = time*learning_data[:, 0]
    learning_data[:, 1] = 2*tau_max*learning_data[:, 1] - tau_max
    print(learning_data.shape)
    return learning_data

def gen_input_data(time, tau_max, num_data):
    # sin波形入力トルク
    """
    入力トルクを生成する関数
    time: シミュレーション時間
    tau_max: 最大入力トルク
    num_data: データ生成個数
    """
    input_data = np.linspace(0., float(time), num_data).reshape((num_data,1))
    input_array = tau_max*np.sin(input_data)
    input_data = np.concatenate([input_data, input_array],axis=1)
    input_data = torch.from_numpy(input_data)

    return input_data

# tau_max = 2.0
# time = 5.
# #学習データ
# num_data = 1500 ##
# #----------------------------------------------
# # ステップ
# # input_array = tau_max*np.ones((500,1))
# # input_array2 = -1*np.ones((500,1))
# # input_array3 = 0*np.ones((500,1))
# # input_array = np.concatenate([input_array, input_array2, input_array3], axis=0)
# #----------------------------------------------
# # sin
# # input_array = tau_max*np.sin(np.linspace(0, time, num_data)).reshape((num_data,1))
# # #----------------------------------------------
# # learning_data = np.linspace(0., float(time), num_data).reshape((num_data,1)) ##
# # learning_data = np.concatenate([learning_data, input_array],axis=1)

# # lhs：ラテンハイパーキューブサンプリング(乱数生成の凄い版)
# learning_data = lhs(2, 80000)

# learning_data[:, 0] = time*learning_data[:, 0]
# learning_data[:, 1] = 2*tau_max*learning_data[:, 1] - tau_max

# print(learning_data.shape)

# np.save("datas/learning_data.npy", learning_data)