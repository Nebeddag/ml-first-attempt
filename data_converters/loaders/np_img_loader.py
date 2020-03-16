def read_and_flat_imgs(path: str):
    """reading all pictures from path

    Loads all images from folder, flattens and converts them to array.
    Returns numpy array.
    """
    import numpy as np
    from PIL import Image
    from os import listdir
    from os.path import isfile, join
    
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    im1 = Image.open(onlyfiles[0])
    p1 = np.array(im1).flatten()
    examples_arr = np.zeros((p1.shape[0], len(onlyfiles)), 'i2')

    for i in range(len(onlyfiles)):
        im = Image.open(onlyfiles[i])
        p = np.array(im).flatten()
        examples_arr[:, i] = p
  #
    return examples_arr