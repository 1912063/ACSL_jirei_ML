

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
max_epochs = 10

my_Net = NN(layers, optimizer, max_epochs, device).to(device)
if optimizer == "L-BFGS":
    my_Net.optimizer.step(my_Net.closure)
else:
    my_Net.train()

np.save("loss", my_Net.loss_hist)
# output = my_Net.test()
# output = output.reshape(len(output), )
# time_span = 0.025
# x1 = my_Net.L*np.sin(output)
# y1 = -my_Net.L*np.cos(output)

# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.set_aspect('equal')
# ax.grid()

# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text


# def animate(i):
#     thisx = [0, x1[i]]
#     thisy = [0, y1[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*time_span))
#     return line, time_text


# ani = animation.FuncAnimation(fig, animate, range(1, len(output)),
#                             interval=time_span*1000, blit=True, init_func=init)


# ani.save("pendulum.gif",writer=PillowWriter())
# # plt.show()

torch.save(my_Net.state_dict(), "NN_model.pth")
# print("saved")