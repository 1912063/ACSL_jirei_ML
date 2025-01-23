
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from NN_model import my_NNmodel as NN
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
np.random.seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###########################################
# 変更可能箇所
###########################################
# 層構造
input_dim = 2           # 入力層次元数
num_middle_layers = 4  # 中間層 層数
num_middle_neurons = 20 # 中間層 ニューロン数
output_dim = 1          # 出力層 次元数
###########################################
# NN設定条件
# optimizer = "Adam"
optimizer = "L-BFGS"
max_epochs = 10000
###########################################
input_layer = np.full(1, input_dim, dtype=np.int16)
middle_layers = np.full(num_middle_layers, num_middle_neurons, dtype=np.int16)
output_layer = np.full(1, output_dim, dtype=np.int16)
layers = np.concatenate([input_layer, middle_layers, output_layer])

# optimizer = "Adam"
optimizer = "L-BFGS"
max_epochs = 5000

my_Net = NN(layers, optimizer, max_epochs, device).to(device)


my_Net.load_state_dict(torch.load('NN_model.pth', torch.device('cpu')))
loss = np.load("loss.npy", allow_pickle=True).tolist()
plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim(0,len(loss))
plt.grid(linestyle='dotted', linewidth=0.5)
plt.plot(loss)
# plt.show()
output = my_Net.test()
