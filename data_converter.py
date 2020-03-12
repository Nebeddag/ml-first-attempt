import numpy as np
import helpers.np_img_loader as img_loader

dir_p = 'data\\photos1\\1\\'
dir_n = 'data\\photos1\\0\\'
output_fname = 'data\\photos1\\examples_flatten'

positive = img_loader.read_and_flat_imgs(dir_p)
print(np.shape(positive))
print(positive)
positive = np.append(positive, np.ones((1, positive.shape[1])), 0)
print(np.shape(positive))
print(positive)
negative = img_loader.read_and_flat_imgs(dir_n)
print(np.shape(negative))
print(negative)
negative = np.append(negative, np.zeros((1, negative.shape[1])), 0)
print(np.shape(negative))
print(negative)
all = np.append(positive, negative, 1)
print(all)
np.save(output_fname, all)