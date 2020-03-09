import numpy as np


def readimgs1(path: str):
    """documentation string

    load all images from folder and convert them to matrix
    """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for filename in onlyfiles:
        with open(filename, "rb") as binaryfile:
            bts = binaryfile.read()
            int_values = [x for x in bts]
            print(int_values)
    #
    return onlyfiles


def readimgs(path: str):
    """documentation string

    load all images from folder and convert them to matrix
    """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    export_arr = np.zeros(())
    examples_list = []

    for filename in onlyfiles:
        with open(filename, "rb") as binaryfile:
            bts = binaryfile.read()
            int_values = [x for x in bts]
            examples_list.append(int_values)


    examples_arr = np.asarray(examples_list)
    print(np.shape(examples_arr))
    print(examples_arr)
    #
    return examples_arr
