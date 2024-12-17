
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from load_data import load_data
import numpy as np

torch.set_default_dtype(torch.float64)

class titanic_classification(nn.Module):
    def __init__(self, layers, epochs, device):
        super().__init__()

        self.epochs = epochs
        self.device = device
        with open("data/origin/train.csv") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
        self.data, self.data_conv, self.survived, self.survived_conv = load_data(device, l, "train")
        

        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        # main文で定義されているNNの構造を配列で全結合層として表している．

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=15, kernel_size=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=15, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=15, kernel_size=5, padding=0)
        # in_channels：入力データの個数（ここでは1人に着目するという意味）
        # out_channels：フィルタの枚数
        # kernel_size：フィルタのサイズ
        # padding：データの空白部分の個数（ここではパティングなしにしている．→データの個数が少なくパティング影響が強く出そうなため）
        
        # Dropout (過学習対策)
        self.dropout = nn.Dropout(0.3)
        
        # 全結合層
        # 出力サイズ計算: input_length / 2 (ストライド2の畳み込み)
        self.fc1 = nn.Linear(15*(5 + 4 + 3 + 2), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # 2値分類

        self.bn1 = nn.BatchNorm1d(num_features=15)  # チャネル数に対応
        self.bn2 = nn.BatchNorm1d(num_features=15)
        self.bn3 = nn.BatchNorm1d(num_features=15)
        self.bn4 = nn.BatchNorm1d(num_features=15)

        self.iter = 1
        self.loss_hist = []
        self.loss_function = nn.MSELoss(reduction ='mean')
        # self.loss_function = nn.BCEWithLogitsLoss()
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        

    def forward(self, x):
        
        for i in range(len(self.layers)-2):         
            z = self.linears[i](x) # linears:全結合層
            x = self.activation(z)

        x = self.linears[-1](x)

        return x
    
    def forward_conv(self, x):

        x1 = self.conv1(x)  # フィルタサイズ2
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        
        x2 = self.conv2(x)  # フィルタサイズ3
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        
        x3 = self.conv3(x)  # フィルタサイズ4
        x3 = self.bn3(x3)
        x3 = self.activation(x3)

        x4 = self.conv4(x)  # フィルタサイズ5
        x4 = self.bn4(x4)
        x4 = self.activation(x4)

        # Flatten
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x3 = torch.flatten(x3, start_dim=1)
        x4 = torch.flatten(x4, start_dim=1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # 全結合層
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # x = self.dropout(x)  # 過学習防止
        x = torch.sigmoid(self.fc3(x))  # 2値分類用
        # x = self.fc3(x) #LOSSがnn.BCEWithLogitsLoss()の時sigmoid不要
        return x
    

    def cal_loss(self, x, y):
        
        E = self.loss_function(x, y)

        return E

    def back_propagation(self):

        for self.iter in range(self.epochs):
            
            self.optimizer.zero_grad()
            # x = self.forward(self.data)
            # loss = self.cal_loss(x, self.survived)

            x = self.forward_conv(self.data_conv)
            loss = self.cal_loss(x, self.survived_conv)

            
            print("Epochs =", self.iter, "loss =", loss)
            loss.backward() # 誤差逆伝播
            self.optimizer.step() # パラメータ更新
            self.loss_hist.append(loss.item()) # 出てきた誤差(loss)のデータ保存
        plt.plot(self.loss_hist, label = 'loss')
        plt.xscale('log')  
        plt.yscale('log')  
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.show()
    
            
    def closure(self):
            
        self.optimizer.zero_grad()
        x = self.forward(self.data)
        loss = self.cal_loss(x, self.survived)
        self.iter += 1
        print("Epochs =", self.iter, "loss =", loss)
        loss.backward()
        self.loss_hist.append(loss.item())
        return loss

    def test(self):
        with open("data/origin/test.csv") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
        data, passengerId = load_data(self.device, l, "test")
        x = data.reshape((len(data),1,len(data.T)))
        # data = torch.transpose(x, 2, 1)
        result = self.forward_conv(x)  # 順伝播

        print(result)
        result = np.array([1 if i >= 0.5 else 0 for i in result]).reshape((len(result), 1))
        
        result = np.concatenate([passengerId, result],axis=1).astype('int64')
        result = np.concatenate([np.array([["PassengerId","Survived"]]), result], axis=0).tolist()
        # print(result)
        with open('data/result.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result)
