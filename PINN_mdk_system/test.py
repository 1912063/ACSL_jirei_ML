
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

layers = np.array([2,  20, 20, 20, 20, 1])   #中間層
optimizer = "Adam"
# optimizer = "L-BFGS"
max_epochs = 100000

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
plt.show()
output = my_Net.test()