
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs


np.random.seed(123)

tau_max = 2.0
time = 5.
#学習データ
num_data = 1500 ##
#----------------------------------------------
# ステップ
# input_array = tau_max*np.ones((500,1))
# input_array2 = -1*np.ones((500,1))
# input_array3 = 0*np.ones((500,1))
# input_array = np.concatenate([input_array, input_array2, input_array3], axis=0)
#----------------------------------------------
# sin
# input_array = tau_max*np.sin(np.linspace(0, time, num_data)).reshape((num_data,1))
# #----------------------------------------------
# learning_data = np.linspace(0., float(time), num_data).reshape((num_data,1)) ##
# learning_data = np.concatenate([learning_data, input_array],axis=1)

# lhs：ラテンハイパーキューブサンプリング(乱数生成の凄い版)
learning_data = lhs(2, 80000)

learning_data[:, 0] = time*learning_data[:, 0]
learning_data[:, 1] = 2*tau_max*learning_data[:, 1] - tau_max

print(learning_data.shape)

np.save("datas/learning_data.npy", learning_data)