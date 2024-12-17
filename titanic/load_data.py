import torch
import csv
import numpy as np
import math

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

        sex = np.array([[1 if i == "male" else 0 for i in sex]]).T
        
        age = ["0" if i == "" else i for i in age]
        age = np.array([[float(i) if not isinstance(i, int) else i for i in age]]).T

        sibSp = ["0" if i == "" else i for i in sibSp]
        sibSp = np.array([[float(i) if not isinstance(i, int) else i for i in sibSp]]).T

        parch = ["0" if i == "" else i for i in parch]
        parch = np.array([[float(i) if not isinstance(i, int) else i for i in parch]]).T

        fare = ["0" if i == "" else i for i in fare]
        fare = np.array([[float(i) if not isinstance(i, int) else i for i in fare]]).T

        survived = np.array([[float(i) if not isinstance(i, int) else i for i in survived]]).T

        data = np.concatenate([pclass, sex, age, sibSp, parch, fare], axis=1)

        data_conv = data.reshape((891, 1, 6))
        survived_conv = survived
        
        # n = 473
        # data_conv = np.zeros((n,418,6))
        # # data_conv[0,:,:] = data[:418:,:]
        # # data_conv[1,:,:] = data[230:648,:]
        # # data_conv[2,:,:] = data[473:891,:]

        # survived_conv = np.zeros((n,418))
        # for i in range(n):
        #     r = 473*int(np.random.rand(1))
        #     data_conv[i,:,:] = data[i:i+418,:]
        #     survived_conv[[i],:] = survived[i:i+418].T
        # survived_conv[[0],:] = survived[:418].T
        # survived_conv[[1],:] = survived[230:648].T
        # survived_conv[[2],:] = survived[473:891].T

        data = torch.from_numpy(data).to(device)
        data_conv = torch.from_numpy(data_conv).to(device)
        survived = torch.from_numpy(survived).to(device)
        survived_conv = torch.from_numpy(survived_conv).to(device)
        return data, data_conv, survived, survived_conv


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
        sex = np.array([[1 if i == "male" else 0 for i in sex]]).T
        age = ["0" if i == "" else i for i in age]
        age = np.array([[float(i) if not isinstance(i, int) else i for i in age]]).T

        sibSp = ["0" if i == "" else i for i in sibSp]
        sibSp = np.array([[float(i) if not isinstance(i, int) else i for i in sibSp]]).T

        parch = ["0" if i == "" else i for i in parch]
        parch = np.array([[float(i) if not isinstance(i, int) else i for i in parch]]).T

        fare = ["0" if i == "" else i for i in fare]
        fare = np.array([[float(i) if not isinstance(i, int) else i for i in fare]]).T
        
        data = np.concatenate([pclass, sex, age, sibSp, parch, fare], axis=1)

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
