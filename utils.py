import numpy as np
import pickle
import os


def car_dyn(x, t, ctrl, noise):
    u_0 = ctrl[0] + noise[0]
    u_1 = ctrl[1] + noise[1]
    dxdt = [u_0 * np.cos(x[2]),
            u_0 * np.sin(x[2]),
            u_1]
    return dxdt


def wrapToPi(a):
    if isinstance(a, list):  # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2 * np.pi) - np.pi for x in a]
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_flip(z0):
    flip = 0
    if z0[-1] < 0:
        tf = -z0[-1]
        flip = 1
    else:
        tf = z0[-1]
    return flip, tf


from six.moves import cPickle as pickle  # for performance


def get_folder_name(filename):
    return '/'.join(filename.split('/')[:-1])


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def save_dict(di_, filename_):
    maybe_makedirs(get_folder_name(filename_))
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
