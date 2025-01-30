import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import time
plt.rcParams["font.size"] = 16


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
        # self.activation = nn.Softplus()
        self.loss_function = nn.MSELoss()

        # 変更可能箇所
        #############################################################
        # 振り子のパラメータ
        self.L = 1.0    #振り子の紐の長さ
        self.d = 0.5
        self.m = 1.0
        self.g = 9.81
        # self.tau = 2.0 #！！！！値を変更したらgen_learningdata.py内のtauも変更する．！！！
        #初期値
        pi = torch.tensor([np.pi])
        self.x_ini = torch.tensor([[1/2*pi]]).to(self.device) #角度
        self.dx_ini = torch.tensor([[0.0]]).to(self.device) #角速度

        # self.x_ini = torch.tensor([[0.0]]).to(self.device) #角度
        # self.dx_ini = torch.tensor([[6.0]]).to(self.device) #角速度

        #############################################################
        # シミュレーション設定条件
        self.time = 5.
        self.num_data = int(40*self.time*10)    # 2.5 ms刻みでtime分のデータ個数
        #############################################################
        self.iter = 0
        self.loss_hist = []

        self.learning_data, self.target = self.import_datas()
        
        self.learning_data.requires_grad = True

        
        input_data = np.linspace(0., float(self.time), self.num_data).reshape((self.num_data,1))
        input_array = np.zeros(self.num_data).reshape((self.num_data,1))
        input_data = np.concatenate([input_data, input_array],axis=1)
        input_data = torch.from_numpy(input_data).to(self.device)
        state = np.array([self.x_ini, self.dx_ini]).reshape((1,2))
        self.label = self.solve_ode(state, input_data[:,1]) # DDNN用の正解データ
        # self.label[int(self.num_data/2):self.num_data,[0]] = 0  # DDNNの学習区間制限用

    def forward(self, x):
        for i in range(len(self.layers)-2):
            u = self.linears[i](x)
            z = self.activation(u)
            x = z

        output = self.linears[-1](x)

        return output
    
    def cal_loss(self, output):

        ##################################################################################################################################################################################
        # PINNs
        dxdt = autograd.grad(output, self.learning_data, torch.ones([len(self.learning_data),1]).to(self.device), retain_graph=True, create_graph=True,allow_unused=True)[0]
        ddxdt = autograd.grad(dxdt[:,[0]], self.learning_data, torch.ones([len(self.learning_data),1]).to(self.device), retain_graph=True, create_graph=True,allow_unused=True)[0]

        #####################################################
        #運動方程式
        f = ddxdt[:,[0]] + self.d/(self.m*self.L)*dxdt[:,[0]] + self.g/self.L*torch.sin(output)# - tau/(self.m*self.L**2) 
        #####################################################
        f_x_ini = output[[0]]
        f_dx_ini = dxdt[0, [0]].reshape((1,1))

        E_x_ini = self.loss_function(f_x_ini, self.x_ini)   #初期角度の誤差関数
        E_dx_ini = self.loss_function(f_dx_ini, self.dx_ini)    #初期角速度の誤差関数

        E = self.loss_function(f, self.target)  #運動方程式の誤差関数

        #label = torch.from_numpy(self.label)
        #E_label = self.loss_function(output[0:int(self.num_data/2)-1,[0]], label[0:int(self.num_data/2)-1,[0]])

        return E + 5*E_x_ini + 5*E_dx_ini #+ 0.01*E_label   #重み調整
        ##################################################################################################################################################################################
        # # DDNN
        # label = torch.from_numpy(self.label)
        # E = self.loss_function(output, label[:,[0]])
        # # E = self.loss_function(output[0:int(self.num_data/2)-1,[0]], label[0:int(self.num_data/2)-1,[0]])
        # return E
        ##################################################################################################################################################################################
        
    
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
        
        loss = loss.to(self.device).detach().numpy()

        

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
    
    def ode_func(self, x, tau):
        # dxdt = np.concatenate(([x[0,[1]]], [-self.g/self.L*np.sin(x[0,[0]])-self.d/(self.m*self.L)*x[0,[1]]+1/(self.m*self.L**2)*tau]), axis=1)
        dxdt = np.concatenate(([x[0,[1]]], [-self.g/self.L*np.sin(x[0,[0]])-self.d/(self.m*self.L)*x[0,[1]]]), axis=1)
        
        return dxdt
    
    def solve_ode(self, x, tau):
        sol = np.zeros((int(self.num_data), 2))
        for i in range(int(self.num_data)):
            f1 = self.ode_func(x, tau[i])
            f2 = self.ode_func(x + f1*(self.time/self.num_data)/2, tau[i])
            f3 = self.ode_func(x + f2*(self.time/self.num_data)/2, tau[i])
            f4 = self.ode_func(x + f3*(self.time/self.num_data), tau[i])

            x = x + (f1 + 2*f2 + 2*f3 + f4)*self.time/self.num_data/6

            sol[i, :] = x
        return sol
    
    
    def test(self):
        # input_array = self.tau*np.sin(np.linspace(0, time, num_data)).reshape((num_data,1))
        
        ###############################################################
        # １入力の時　シミュレーション時間のみ生成用
        input_data = np.linspace(0., float(self.time), self.num_data).reshape((self.num_data,1))
        input_array = np.zeros(self.num_data).reshape((self.num_data,1))
        input_data = np.concatenate([input_data, input_array],axis=1)
        input_data = torch.from_numpy(input_data).to(self.device)

        # NN計算時間
        time_start = time.time()
        output = self.forward(input_data[:,[0]])
        time_end = time.time()
        time_diff = time_end - time_start
        print("NN計算時間：", time_diff)

        input_data = input_data.to(self.device).detach().numpy()#.reshape(len(self.input_data))
        ###############################################################
        # ２入力の時　入力トルク波形用
        # input_data = np.linspace(0., float(self.time), self.num_data).reshape((self.num_data,1)) ##
        # input_array = self.tau*np.sin(input_data)*0
        # input_data = np.concatenate([input_data, input_array],axis=1)
        # input_data = torch.from_numpy(input_data).to(self.device)

        # output = self.forward(input_data)
        # input_data = input_data.to(self.device).detach().numpy()#.reshape(len(self.input_data))
        ###############################################################
        
        output = output.to(self.device).detach().numpy()
        self.x_ini = self.x_ini.to(self.device).detach().numpy()
        self.dx_ini = self.dx_ini.to(self.device).detach().numpy()
        state = np.array([self.x_ini, self.dx_ini]).reshape((1,2))
        # G, L, M, D, tau
        # y_input = integrate.odeint(sol_ode.derivs, state, input_data[:,0], args=(9.81, self.L, self.m, self.d, self.tau))

        # ４次ルンゲクッタ法計算時間計測
        time_start = time.time()
        y_learning = self.solve_ode(state, input_data[:,1])
        time_end = time.time()
        time_diff = time_end - time_start
        print("数値シミュレーション計算時間：", time_diff)

        # 精度検証
        error_sum = 0.
        for i in range(self.num_data):
            error_sum = abs(y_learning[i,0] - output[i,0])
        error = error_sum/self.num_data
        print("絶対平均誤差：", error)

        plt.figure()
        plt.plot(input_data[:,0], output, label="predicted")

        true_input = np.zeros(int(self.num_data/50)).reshape((int(self.num_data/50),1))
        true_output = np.zeros(int(self.num_data/50)).reshape((int(self.num_data/50),1))
        j = 0
        for i in range(self.num_data):
            if i % 50 ==0:    # 正解点の表示数を減らす用
                true_input[j,0] = input_data[i,0]
                true_output[j,0] = y_learning[i,0]
                j += 1

        plt.plot(true_input[:,0], true_output[:,0], linestyle="None", marker = "o", markersize = 4, label="true")
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
        # plt.show()
        output = output.reshape(len(output), )
        time_span = 10/1000
        x1 = self.L*np.sin(output)
        y1 = -self.L*np.cos(output)
        x2 = self.L*np.sin(y_learning[:,0])
        y2 = -self.L*np.cos(y_learning[:,0])
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        line, = ax.plot([], [], 'o-', lw=2, label="predicted")
        line2, = ax.plot([], [], 'o-', lw=2, label="true")
        ax.legend()
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            line2.set_data([], [])
            time_text.set_text('')
            return line, line2, time_text
        
        def animate(i):
            i = 1*i
            thisx = [0, x1[i]]
            thisy = [0, y1[i]]
            thisx2 = [0, x2[i]]
            thisy2 = [0, y2[i]]
            line.set_data(thisx, thisy)
            line2.set_data(thisx2, thisy2)
            time_text.set_text(time_template % (i*self.time/self.num_data))
            return line, line2, time_text
        print(len(output))
        # ani = animation.FuncAnimation(fig, animate, range(1, int(self.num_data)),
        #                             interval=5, blit=True, init_func=init)
        # # ani.save("pendulum.gif",writer=PillowWriter())
        # ani.save('pendulum.mp4', writer="ffmpeg")
# plt.show()
        return output