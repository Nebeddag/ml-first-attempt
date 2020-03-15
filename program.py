# %%
import numpy as np
import pandas as pd

import torch.autograd
import torch.nn.functional as f
from torch.autograd import Variable

import matplotlib.pyplot as plt

import ml_models.train_model as mdl

# %%
# loading data
np_data = np.load('data\\photos1\\examples_flatten.npy').T

# %%
print(np_data)
np.random.shuffle(np_data)
print(np_data)
print(np_data.shape)

# %%
np_x = np_data[:, : np_data.shape[1] - 1]
print(np_x.shape)
np_y = np_data[:, np_data.shape[1] - 1]
print(np_y.shape)

# %%
all_cnt = np_x.shape[0]
dev_cnt = all_cnt // 100 * 15
test_cnt = all_cnt // 100 * 15
train_cnt = all_cnt - dev_cnt - test_cnt

np_x_train = np_x[: train_cnt, :]
np_y_train = np_y[: train_cnt]

np_x_dev = np_x[train_cnt : train_cnt + dev_cnt, :]
np_y_dev = np_y[train_cnt : train_cnt + dev_cnt]

np_x_test = np_x[train_cnt + dev_cnt :, :]
np_y_test = np_y[train_cnt + dev_cnt :]

# %%
exmpl_img = np_x[0, :].reshape(80, 60, -1)
exmpl_y = np_y[0]
plt.imshow(exmpl_img)

# %%
x_train = torch.from_numpy(np_x_train)
y_train = torch.from_numpy(np_y_train.reshape(-1, 1))

x_dev = torch.from_numpy(np_x_dev)
y_dev = torch.from_numpy(np_y_dev.reshape(-1, 1))

x_test = torch.from_numpy(np_x_test)
y_test = torch.from_numpy(np_y_test.reshape(-1, 1))


#model = mdl.train_model(x_train, y_train)
model = mdl.train_model_ext(x_train, y_train, x_dev, y_dev)

# %%
loss_test = mdl.check_model(x_test, y_test, model)

# %%
