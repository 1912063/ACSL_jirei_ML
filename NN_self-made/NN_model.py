import torch
import torch.autograd as autograd
import torch.nn as nn

class NN221():
    def __init__(self):
        self.a = 0
        self.activate_fn_z = nn.Tanh()
        self.activate_fn_z = nn.ReLU()
        self.activate_fn_output = nn.Sigmoid()

        self.loss_function = nn.MSELoss(reduction ='mean')

        # self.w1_11 = torch.tensor([[torch.randn((1, 1))]])
        # self.w1_12 = torch.tensor([[torch.randn((1, 1))]])
        # self.w1_21 = torch.tensor([[torch.randn((1, 1))]])
        # self.w1_22 = torch.tensor([[torch.randn((1, 1))]])
        # self.w2_11 = torch.tensor([[torch.randn((1, 1))]])
        # self.w2_21 = torch.tensor([[torch.randn((1, 1))]])
        # self.b1_1 = torch.tensor([[torch.randn((1, 1))]])
        # self.b1_2 = torch.tensor([[torch.randn((1, 1))]])
        # self.b2_1 = torch.tensor([[torch.randn((1, 1))]])

        self.w1_11 = torch.tensor([1.])
        self.w1_12 = torch.tensor([1.])
        self.w1_21 = torch.tensor([1.])
        self.w1_22 = torch.tensor([1.])
        self.w2_11 = torch.tensor([1.])
        self.w2_21 = torch.tensor([1.])
        self.b1_1 = torch.tensor([1.])
        self.b1_2 = torch.tensor([1.])
        self.b2_1 = torch.tensor([1.])

        # print(self.w1_11)
        # print(self.w2_11)
        # print(self.b1_1)
        # print(self.b2_1)


        self.w1_11.requires_grad = True
        self.w1_12.requires_grad = True
        self.w1_21.requires_grad = True
        self.w1_22.requires_grad = True
        self.w2_11.requires_grad = True
        self.w2_21.requires_grad = True
        self.b1_1.requires_grad = True
        self.b1_2.requires_grad = True
        self.b2_1.requires_grad = True
    
        
    def model(self, input):
        input.requires_grad = True

        z1 = self.activate_fn_z(self.w1_11*input[:, [0]]+self.w1_21*input[:, [1]]+self.b1_1)
        z2 = self.activate_fn_z(self.w1_12*input[:, [0]]+self.w1_22*input[:, [1]]+self.b1_2)
        
        output = self.activate_fn_output(self.w2_11*z1 + self.w2_21*z2+self.b2_1)

        return output
    
    def cal_loss(self, target, output):

        loss = self.loss_function(output, target).reshape((1, 1))
        # print(output-target)
        

        return loss
    
    def cal_grad(self, loss, param):

        dEdp = autograd.grad(loss, param, torch.ones((1,1)), retain_graph=True)

        return dEdp
    
    def back_propagetion(self, loss):

        dw1_11 = self.cal_grad(loss, self.w1_11) 
        # print("dE/dw11", dw1_11)
        dw1_12 = self.cal_grad(loss, self.w1_12)
        dw1_21 = self.cal_grad(loss, self.w1_21)
        dw1_22 = self.cal_grad(loss, self.w1_22)
        dw2_11 = self.cal_grad(loss, self.w2_11)
        dw2_21 = self.cal_grad(loss, self.w2_21)
        db1_1 = self.cal_grad(loss, self.b1_1)
        print("dE/db1_1", db1_1)
        db1_2 = self.cal_grad(loss, self.b1_2)
        db2_1 = self.cal_grad(loss, self.b2_1)

        mat_w1 = torch.tensor([[dw1_11, dw1_12], [dw1_21, dw1_22]])
        mat_w2 = torch.tensor([[dw2_11], [dw2_21]])
        mat_b1 = torch.tensor([[db1_1], [db1_2]])

        return mat_w1, mat_w2, mat_b1, torch.tensor([db2_1])
    
    def update_params(self, mat_w1, mat_w2, mat_b1, db2_1, lr):

        self.w1_11 = self.w1_11 - lr*mat_w1[0, 0]
        self.w1_12 = self.w1_12 - lr*mat_w1[0, 1]
        self.w1_21 = self.w1_21 - lr*mat_w1[1, 0]
        self.w1_22 = self.w1_22 - lr*mat_w1[1, 1]
        self.w2_11 = self.w2_11 - lr*mat_w2[0]
        self.w2_21 = self.w2_21 - lr*mat_w2[1]
        self.b1_1 = self.b1_1 - lr*mat_b1[0]
        self.b1_2 = self.b1_2 - lr*mat_b1[1]
        self.b2_1 = self.b2_1 - lr*db2_1

        # print(self.w1_11)
        # print(self.w2_11)
        # print(self.b1_1)
        # print(self.b2_1)

    def test(self, input):
        # input.requires_grad = True

        z1 = self.activate_fn_z(self.w1_11*input[:, [0]]+self.w1_21*input[:, [1]]+self.b1_1)
        z2 = self.activate_fn_z(self.w1_12*input[:, [0]]+self.w1_22*input[:, [1]]+self.b1_2)

        output = self.activate_fn_output(self.w2_11*z1 + self.w2_21*z2+self.b2_1)

        return output


        





