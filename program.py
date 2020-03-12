# %%
import numpy as np
import pandas as pd

import torch.autograd
import torch.nn.functional as f
from torch.autograd import Variable

import matplotlib.pyplot as plt

import ml_models as models

# %%
# loading data
np_data = np.load('data\\photos1\\examples_flatten.npy').T

#%%
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
exmpl_img = np_x[0, :].reshape(80, 60, -1)
exmpl_y = np_y[0]
# %%
plt.imshow(exmpl_img)
print(exmpl_y)

# %%
x_data = Variable(torch.from_numpy(np_x))
y_data = Variable(torch.from_numpy(np_y.reshape(-1 ,1)))

#models.