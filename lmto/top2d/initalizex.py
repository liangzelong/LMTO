import numpy as np
import cv2


def init_one(nelx, nely):
    return np.ones((nelx * nely))


def init_volfrac(nelx, nely, volfrac):
    return np.ones((nelx * nely)) * volfrac


def init_load(path, nelx, nely, flip_dim=True):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (nelx, nely))
    if flip_dim:
        img = img.T
    return img / 255.0
