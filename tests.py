# %%
import torch
from torch import cuda

# %%
print(cuda.is_available())

# %%
device = torch.device("cuda:0" if cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

# %%
print(device)


# %%
