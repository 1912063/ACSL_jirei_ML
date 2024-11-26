import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from sol_ode import solve_ode
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

torch.set_default_dtype(torch.float64)

torch.manual_seed(123)
np.random.seed(123)

class my_NNmodel(torch.nn.Module):
    def __init__(self, layers, optimizer, max_epochs, device):
        super(my_NNmodel, self).__init__()

        self.layers = layers
        self.device = device
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        elif optimizer == "L-BFGS":
            self.optimizer = torch.optim.LBFGS(self.parameters(),lr=1, 
                                        max_iter = max_epochs, 
                                        # max_eval = 100000, 
                                        #   tolerance_grad = 1e-100, 
                                        # tolerance_change = 1e-10, 
                                        history_size = 100, 
                                        line_search_fn = 'strong_wolfe'
                                        )
        
        self.max_epochs = max_epochs
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss()

        #############################################################
        #振り子のパラメータ 自由に変更可能
        self.L = 1.0    #振り子の紐の長さ
        self.d = 0.5
        self.m = 1.0
        self.tau = 5.0
        #初期値
        pi = torch.tensor([np.pi])
        self.x_ini = torch.tensor([[1/2*pi]]).to(self.device) #角度
        self.dx_ini = torch.tensor([[0.0]]).to(self.device) #角速度

        self.x_ini = torch.tensor([[0.0]]).to(self.device) #角度
        self.dx_ini = torch.tensor([[0.0]]).to(self.device) #角速度

        #############################################################

        self.iter = 0
        self.loss_hist = []

        self.learning_data, self.target = self.import_datas()
        
        self.learning_data.requires_grad = True

    def forward(self, x):
        for i in range(len(self.layers)-2):
            u = self.linears[i](x)
            z = self.activation(u)
            x = z

        output = self.linears[-1](x)

        return output
    
    def cal_loss(self, output):

        tau = self.learning_data[:,[1]]
        dxdt = autograd.grad(output, self.learning_data, torch.ones([len(self.learning_data),1]).to(self.device), retain_graph=True, create_graph=True,allow_unused=True)[0]
        # print(dxdt.shape)
        ddxdt = autograd.grad(dxdt[:,[0]], self.learning_data, torch.ones([len(self.learning_data),1]).to(self.device), retain_graph=True, create_graph=True,allow_unused=True)[0]

        #####################################################
        #運動方程式
        f = ddxdt[:,[0]] + self.d/(self.m*self.L)*dxdt[:,[0]] + 9.81/self.L*torch.sin(output) - tau/(self.m*self.L**2) 
        #####################################################

        f_x_ini = output[[0]]
        f_dx_ini = dxdt[0, [0]].reshape((1,1))

        E_x_ini = self.loss_function(f_x_ini, self.x_ini)   #初期角度の誤差関数
        E_dx_ini = self.loss_function(f_dx_ini, self.dx_ini)    #初期角速度の誤差関数

        E = self.loss_function(f, self.target)  #運動方程式の誤差関数

        return E + 5*E_x_ini + 5*E_dx_ini   #重み調整
    
    def train(self):
        
        for i in range(self.max_epochs):
            self.optimizer.zero_grad()
            
            output = self.forward(self.learning_data)
            loss = self.cal_loss(output)
            loss.backward()
            self.loss_hist.append(loss.item())
            self.optimizer.step()

            print("Epochs = ", i+1, "Loss = ", loss)
            self.iter += 1

    def closure(self):
        #PINN.train()
        
        self.optimizer.zero_grad()                   # 勾配情報を0に初期化

        
        output = self.forward(self.learning_data)
        loss = self.cal_loss(output)

        self.loss_hist.append(loss.item())
        
        loss.backward()
                
        self.iter += 1
        
        loss = loss.to('cpu').detach().numpy()

        

        if self.iter % 1 == 0:
            #_ = PINN.test()
            print("------------------------------------------------")
            print("Epochs", self.iter, "loss", loss)

        # if msvcrt.kbhit():
        #     kb = msvcrt.getch()
        #     if kb.decode() == 'a' :
        #         torch.save(self.state_dict(), 'PINN_model.pth')
        #         #t_test = torch.from_numpy(t_test).double().to(device)
        #         #t_learning = torch.from_numpy(t_learning).double().to(device)a

        #         losses  = self.loss_hist
        #         plt.figure()
        #         plt.xscale("log")
        #         plt.yscale("log")
        #         plt.xlabel('Epochs')
        #         plt.ylabel('Loss')
        #         plt.xlim(0,int(self.iter))
        #         plt.grid(linestyle='dotted', linewidth=0.5)
        #         plt.plot(losses)
        #         plt.show()

        return loss

    def import_datas(self):
        learning_data = np.load("datas/learning_data.npy", allow_pickle=True)
        learning_data = torch.from_numpy(learning_data).to(self.device)

        target = torch.zeros((len(learning_data), 1)).to(self.device)

        return learning_data, target
    
    
    def test(self):
        output = self.forward(self.learning_data)
        learning_data = self.learning_data.to("cpu").detach().numpy()#.reshape(len(self.learning_data))
        output = output.to("cpu").detach().numpy()
        self.x_ini = self.x_ini.to("cpu").detach().numpy()
        self.dx_ini = self.dx_ini.to("cpu").detach().numpy()
        state = np.array([self.x_ini, self.dx_ini]).reshape((1,2))
        # G, L, M, D, tau
        # y_larning = integrate.odeint(sol_ode.derivs, state, learning_data[:,0], args=(9.81, self.L, self.m, self.d, self.tau))
        y_larning = solve_ode(state, learning_data[:,1], 9.81, self.L, self.m, self.d, 15/len(learning_data), 15)
        plt.figure()
        plt.plot(learning_data[:,0], y_larning[:,0], label="true")
        plt.plot(learning_data[:,0], output, label="predicted")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\theta$")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim(0,int(self.iter))
        plt.grid(linestyle='dotted', linewidth=0.5)
        plt.plot(self.loss_hist)
        plt.show()

        output = output.reshape(len(output), )
        time_span = 10/1000
        x1 = self.L*np.sin(output)
        y1 = -self.L*np.cos(output)

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text


        def animate(i):
            i = 1*i
            thisx = [0, x1[i]]
            thisy = [0, y1[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*time_span))
            return line, time_text

        print(len(output))
        ani = animation.FuncAnimation(fig, animate, range(1, int(len(output))),
                                    interval=5, blit=True, init_func=init)


        # ani.save("pendulum.gif",writer=PillowWriter())
        ani.save('pendulum.mp4', writer="ffmpeg")
# plt.show()

        return output