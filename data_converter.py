import numpy as np
import helpers.np_img_loader as img_loader

dir_p = 'data\\photos1\\1\\'
dir_n = 'data\\photos1\\0\\'
output_fname = 'data\\photos1\\examples_flatten'

positive = img_loader.read_and_flat_imgs(dir_p)
#print(np.shape(positive))
#print(positive)

negative = img_loader.read_and_flat_imgs(dir_n)
#print(np.shape(negative))
#print(negative)

all = np.zeros((positive.shape[0] + 1, positive.shape[1] + negative.shape[1]), 'i2')
all[: positive.shape[0], : positive.shape[1]] = positive
all[positive.shape[0], : positive.shape[1]] = np.ones(positive.shape[1])
all[: positive.shape[0], positive.shape[1] :] = negative

#print(all)
np.save(output_fname, all)
