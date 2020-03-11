import numpy as np
import helpers.dataloader as dloader

positive = dloader.readimgs('data\\photos1\\1\\')
print(np.shape(positive))
print(positive)
positive = np.append(positive, np.ones((1, positive.shape[1])), 0)
print(np.shape(positive))
print(positive)
negative = dloader.readimgs('data\\photos1\\0\\')
print(np.shape(negative))
print(negative)
negative = np.append(negative, np.zeros((1, negative.shape[1])), 0)
print(np.shape(negative))
print(negative)
all = np.append(positive, negative, 1)
print(all)
np.save('examples', all)