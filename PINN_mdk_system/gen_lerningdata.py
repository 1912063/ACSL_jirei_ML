
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(123)


#学習データ
num_data = 1500 ##
#----------------------------------------------
# ステップ
input_array = 5*np.ones((500,1))
input_array2 = -1*np.ones((500,1))
input_array3 = 0*np.ones((500,1))
input_array = np.concatenate([input_array, input_array2, input_array3], axis=0)
#----------------------------------------------
# sin
input_array = 5*np.sin(np.linspace(0, 10, 1500)).reshape((1500,1))
#----------------------------------------------
learning_data = np.linspace(0., 15., num_data).reshape((num_data,1)) ##
learning_data = np.concatenate([learning_data, input_array],axis=1)
print(learning_data.shape)

np.save("datas/learning_data.npy", learning_data)