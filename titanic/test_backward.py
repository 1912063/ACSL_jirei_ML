# import numpy as np
# import pandas as pd
import torch
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt

# backwardは何をしているのか。PytochのAutogradという概念。
# https://zenn.dev/hirayuki/articles/bbc0eec8cd816c183408

x = torch.tensor(3.0, requires_grad=True)
print(x)

y = 2*x
print(y)

y.backward()
# ↓が計算される
print(x.grad)
print('---------------------')



# 伝播のイメージをわかりやすくするために合成関数として考える
x = torch.tensor(2.0, requires_grad=True)
y = 3*x
z = 5*y
z.backward()
print(x.grad)