# %%
import numpy as np
import pandas as pd
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt
import helpers.dataloader as dloader

# %%
# loading data
np_data = np.load('data\\photos1\\examples.npy')
print(np_data)
np.random.shuffle(np_data.T)
print(np_data)
print(np_data.shape)

# %%
np_x = np_data[: np_data.shape[0] - 1, :]
np_y = np_data[np_data.shape[0] - 1, :]

# %%
exmpl = np_x[:, 0].reshape(60, -1)
# %%
plt.imshow(exmpl)

x_data = Variable(torch.from_numpy(np_x.transpose()))
y_data = Variable(torch.from_numpy(np_y.transpose()))

# %%
'''
#%%
p = np.linspace(0,20,100)
plt.plot(p,np.sin(15*p))

#%%
plt.show()

# %%
print(p)
'''

# %%
tmp = dloader.readimgs('data\\photos1\\0\\')
