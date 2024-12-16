
import csv
import torch
import torch.nn as nn
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

        self.conv1 = nn.Conv1d(in_channels=418, out_channels=418, kernel_size=6, padding=0, groups=19)
        self.conv2 = nn.Conv1d(in_channels=418, out_channels=418, kernel_size=6, padding=0, groups=19)
        
        # Dropout (過学習対策)
        self.dropout = nn.Dropout(0.3)
        
        # 全結合層
        # 出力サイズ計算: input_length / 2 (ストライド2の畳み込み)
        self.fc1 = nn.Linear(418, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 418)  # 2値分類



        self.iter = 1
        self.loss_hist = []
        self.loss_function = nn.MSELoss(reduction ='mean')
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        

    def forward(self, x):
        
        for i in range(len(self.layers)-2):         
            z = self.linears[i](x)
            x = self.activation(z)

        x = self.linears[-1](x)

        return x
    
    def forward_conv(self, x):
        # 畳み込み層 + 活性化関数
        x = self.activation(self.conv1(x))
        # x = self.activation(self.conv2(x))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)  # バッチサイズに対して平坦化
        
        # 全結合層
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # x = self.dropout(x)  # 過学習防止
        x = torch.sigmoid(self.fc3(x))  # 2値分類用
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
            loss.backward()
            self.optimizer.step()
            self.loss_hist.append(loss.item())
            
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
        x = data.reshape((1,len(data),len(data.T)))
        # data = torch.transpose(x, 2, 1)
        result = self.forward_conv(x).reshape((418))
        result = np.array([1 if i >= 0.5 else 0 for i in result]).reshape((len(result), 1))
        

        result = np.concatenate([passengerId, result],axis=1).astype('int64')
        result = np.concatenate([np.array([["PassengerId","Survived"]]), result], axis=0).tolist()
        print(result)
        with open('data/result.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result)
