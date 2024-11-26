import torch
import matplotlib.pyplot as plt

x = torch.arange(-10., 10., 0.01, requires_grad=True)

logsig = 1/(1 + torch.exp(-x))  #ロジスティックシグモイド
tanh = (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))  #ハイパーボリックタンジェント
ReLU = torch.maximum(x, torch.tensor(0.0))  #ReLU
linear = x  #恒等写像

dlogsig_dx = torch.autograd.grad(logsig, x, torch.ones_like(logsig))[0]
dtanh_dx = torch.autograd.grad(tanh, x, torch.ones_like(tanh))[0]
dReLU_dx = torch.autograd.grad(ReLU, x, torch.ones_like(ReLU))[0]
dlinear_dx = torch.autograd.grad(linear, x, torch.ones_like(ReLU))[0]

x = x.detach().numpy()
logsig = logsig.detach().numpy()
dlogsig_dx = dlogsig_dx.detach().numpy()
tanh = tanh.detach().numpy()
dtanh_dx = dtanh_dx.detach().numpy()
ReLU = ReLU.detach().numpy()
dReLU_dx = dReLU_dx.detach().numpy()
linear = linear.detach().numpy()
dlinear_dx = dlinear_dx.detach().numpy()



# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.plot(x, logsig, label="logsig")
# ax1.plot(x, dlogsig_dx, label="dlogsig")

# ax2 = fig.add_subplot(2, 2, 2)
# ax2.plot(x, tanh, label="tanh")
# ax2.plot(x, dtanh_dx, label="dtanh")

# ax3 = fig.add_subplot(2, 2, 3)
# ax3.plot(x, ReLU, label="ReLU")
# ax3.plot(x, dReLU_dx, label="dReLU")

# ax4 = fig.add_subplot(2, 2, 4)
# ax4.plot(x, linear, label="linear")
# ax4.plot(x, dlinear_dx, label="dlinear")
# plt.show()
plt.figure()
plt.plot(x, logsig, label="sigmoid")
plt.plot(x, dlogsig_dx, label="dsigmoid")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, tanh, label="tanh")
plt.plot(x, dtanh_dx, label="dtanh")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, ReLU, label="ReLU")
plt.plot(x, dReLU_dx, label="dReLU")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, linear, label="identity")
plt.plot(x, dlinear_dx, label="didentity")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()