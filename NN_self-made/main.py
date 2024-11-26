import torch
import torch.autograd as autograd
import torch.nn as nn
from NN_model import NN221 as NN
import numpy as np

torch.set_default_dtype(torch.float64)

my_net = NN()
target = torch.tensor([[1.], [0.], [0.], [0.]])
target = torch.tensor([[1.]])
learning_data = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [0., 0.]])
learning_data = torch.tensor([[1., 1.]])
lr = 0.01
epochs = 1

for i in range(epochs):
    output = my_net.model(learning_data)
    print("output", output)
    loss = my_net.cal_loss(target, output)
    print("Epochs", i+1, "Loss" , loss.detach().numpy())
    mat_w1, mat_w2, mat_b1, db2_1 = my_net.back_propagetion(loss)
    my_net.update_params(mat_w1, mat_w2, mat_b1, db2_1, lr)

test_data = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [0., 0.]])
output = my_net.test(test_data)
print(output)