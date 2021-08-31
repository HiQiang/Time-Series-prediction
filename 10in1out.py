# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 16:45
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : final.py
# @Software: PyCharm

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

tr_path = "covid.train.csv"
a = pd.read_csv(tr_path)
tested_positive = a["tested_positive.1"]
tested_positive = tested_positive[0:62]
tested_positive = tested_positive.values

plt.plot(tested_positive)
plt.show()

x = torch.zeros([51, 10])
y = torch.zeros([51, 1])
for i in range(0, 51):
    print(i)
    x[i, :] = torch.from_numpy(tested_positive[i: i+10])
    y[i, :] = tested_positive[i+11]


class Net(torch.nn.Module):
    def __init__(self, feature_in=10, fc1_out=32, fc2_out=64, fc3_out=128,
                 fc4_out=64, fc5_out=32, prediction_out=1):
        super(Net, self).__init__()
        self.FC1 = torch.nn.Sequential(
            torch.nn.Linear(feature_in, fc1_out),
            torch.nn.ReLU(),
        )

        self.FC2 = torch.nn.Sequential(
            torch.nn.Linear(fc1_out, fc2_out),
            torch.nn.ReLU(),
        )
        self.FC3 = torch.nn.Sequential(
            torch.nn.Linear(fc2_out, fc3_out),
            torch.nn.ReLU(),
        )
        self.FC4 = torch.nn.Sequential(
            torch.nn.Linear(fc3_out, fc4_out),
            torch.nn.ReLU(),
        )
        self.FC5 = torch.nn.Sequential(
            torch.nn.Linear(fc4_out, fc5_out),
            torch.nn.ReLU(),
        )
        self.predict = torch.nn.Linear(fc5_out, prediction_out)

    def forward(self, x):
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)
        x = self.FC5(x)
        x = self.predict(x)
        return x


net = Net()
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, )
loss_func = torch.nn.MSELoss()

epochs = 2000
loss_plot = np.zeros(epochs)

for t in range(epochs):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_plot[t] = loss.data.numpy()
    if t+1 % 50 == 0:
        print(loss.data.numpy())

plt.plot(loss_plot)
plt.title("Loss vs Epoch")
plt.show()


pr = net(x)
plt.plot(pr.detach().numpy())
plt.plot(y.numpy())
plt.show()

