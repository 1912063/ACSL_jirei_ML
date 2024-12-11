import torch
import csv
import numpy as np
import math

def normalization(data): # データを正規化する関数
    maxdata = max(data)
    mindata = min(data)
    tmp = np.zeros_like(data)
    for i in range(len(data)):
        tmp[i] = (data[i] - mindata) / (maxdata - mindata)
    return tmp

def load_data(device, data, data_type):
    

    data = np.array(data[1:])

    # print(data)
    if data_type == "train":
        passengerId = data[:, 0]
        survived = data[:, 1]
        pclass = data[:, 2]
        name = data[:, 3]
        sex = data[:, 4]
        age = data[:, 5]
        sibSp = data[:, 6]
        parch = data[:, 7]
        ticket = data[:, 8]
        fare = data[:, 9]
        cabin = data[:, 10]
        embarked = data[:, 11]

        pclass = np.array([[float(i) if not isinstance(i, int) else i for i in pclass]]).T
        pclass = normalization(pclass)
        sex = np.array([[1 if i == "male" else 0 for i in sex]]).T
        age = ["0" if i == "" else i for i in age]
        age = np.array([[float(i) if not isinstance(i, int) else i for i in age]]).T
        age = normalization(age)

        sibSp = np.array([[float(i) if not isinstance(i, int) else i for i in sibSp]]).T
        sibSp = normalization(sibSp)
        parch = np.array([[float(i) if not isinstance(i, int) else i for i in parch]]).T
        parch = normalization(parch)
        survived = np.array([[float(i) if not isinstance(i, int) else i for i in survived]]).T

        data = np.concatenate([pclass, sex, age, sibSp, parch], axis=1)

        data = torch.from_numpy(data).to(device)
        survived = torch.from_numpy(survived).to(device)
        return data, survived # data:学習に使用するデータ，survived:教師データ　なので分けて返している


    elif data_type == "test":
        passengerId = data[:, 0]
        survived = 0
        pclass = data[:, 1]
        name = data[:, 2]
        sex = data[:, 3]
        age = data[:, 4]
        sibSp = data[:, 5]
        parch = data[:, 6]
        ticket = data[:, 7]
        fare = data[:, 8]
        cabin = data[:, 9]
        embarked = data[:, 10]

        passengerId = np.array([[float(i) if not isinstance(i, int) else i for i in passengerId]]).reshape((len(data), 1))
        pclass = np.array([[float(i) if not isinstance(i, int) else i for i in pclass]]).T
        pclass = normalization(pclass)
        sex = np.array([[1 if i == "male" else 0 for i in sex]]).T
        age = ["0" if i == "" else i for i in age]
        age = np.array([[float(i) if not isinstance(i, int) else i for i in age]]).T
        age = normalization(age)
        sibSp = np.array([[float(i) if not isinstance(i, int) else i for i in sibSp]]).T
        sibSp = normalization(sibSp)
        parch = np.array([[float(i) if not isinstance(i, int) else i for i in parch]]).T
        parch = normalization(parch)


        
        data = np.concatenate([pclass, sex, age, sibSp, parch], axis=1)

        data = torch.from_numpy(data).to(device)

        return data, passengerId

    # for i in sex:
    #     if i != "male" and i != "female":
    #         print("1")



    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("data/origin/train.csv") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    load_data(device, l, "train")
    