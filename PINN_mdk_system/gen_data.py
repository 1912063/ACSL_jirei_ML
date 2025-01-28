
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyDOE import lhs


np.random.seed(123)


# tau_max = 2.0
time = 10.
#学習データ
num_data = int(40*time*10) ##

# # lhs：ラテンハイパーキューブサンプリング(乱数生成の凄い版)
# learning_data = lhs(2, 80000)

# learning_data[:, 0] = time*learning_data[:, 0]
# learning_data[:, 1] = 2*tau_max*learning_data[:, 1] - tau_max

learning_data = np.linspace(0, time, num_data).reshape(-1,1)

print(learning_data.shape)

np.save("datas/learning_data.npy", learning_data)