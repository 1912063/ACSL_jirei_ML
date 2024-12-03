
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
        self.data, self.survived = load_data(device, l, "train")

        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        # main文で定義されているNNの構造を配列で全結合層として表している．

        self.iter = 1
        self.loss_hist = []
        self.loss_function = nn.MSELoss(reduction ='mean')
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        

    def forward(self, x):
        
        for i in range(len(self.layers)-2):         
            z = self.linears[i](x) # linears:全結合層
            x = self.activation(z)

        x = self.linears[-1](x)

        return x
    

    def cal_loss(self, x, y):
        
        E = self.loss_function(x, y)

        return E

    def back_propagation(self):

        for self.iter in range(self.epochs):
            
            self.optimizer.zero_grad()
            x = self.forward(self.data) # 順伝播
            loss = self.cal_loss(x, self.survived) # 誤差の計算
            print("Epochs =", self.iter, "loss =", loss)
            loss.backward() # 誤差逆伝播
            self.optimizer.step() # パラメータ更新
            self.loss_hist.append(loss.item()) # 出てきた誤差(loss)のデータ保存
            
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
        result = self.forward(data)
        result = np.array([1 if i >= 0.5 else 0 for i in result]).reshape((len(result), 1))
        

        result = np.concatenate([passengerId, result],axis=1).astype('int64')
        result = np.concatenate([np.array([["PassengerId","Survived"]]), result], axis=0).tolist()
        print(result)
        with open('data/result.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result)
