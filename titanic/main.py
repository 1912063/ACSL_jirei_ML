

import numpy as np
import torch
import torch.nn as nn
from NN_MODEL import titanic_classification

torch.set_default_dtype(torch.float64)

torch.manual_seed(123)
np.random.seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# 選べるoptimizer
"L-BFGS" "Adam" "SGD" "RMSprop"

optim = "L-BFGS"
optim = "Adam"
# optim = "RMSprop"
# optim = "SGD"

epochs = 3000
layers = np.array([6,10,10,1])


NN = titanic_classification(layers, epochs, device).to(device)  # モデルのインスタンス
if optim == "L-BFGS":
    NN.optimizer = torch.optim.LBFGS(NN.parameters(),lr=1, 
                                max_iter = 100, 
                                # max_eval = 100000, 
                                #   tolerance_grad = 1e-100, 
                                # tolerance_change = 1e-10, 
                                history_size = 100, 
                                line_search_fn = 'strong_wolfe'
                                )
    scheduler = None
elif optim == "Adam":
    NN.optimizer = torch.optim.Adam(NN.parameters(), lr=0.0005)
    # scheduler = None

elif optim == "SGD":
    NN.optimizer = torch.optim.SGD(NN.parameters(), lr=0.00001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[500000,10000000], gamma=0.3)
elif optim == "RMSprop":
    NN.optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=0.001)


# print(HLNN.linears[0].bias)
if optim == "L-BFGS":
    NN.optimizer.step(NN.closure)
else:
    NN.back_propagation()

NN.test()

