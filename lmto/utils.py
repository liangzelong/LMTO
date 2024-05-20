import cv2
import numpy as np


def generate_sdf_out(
    x,
):
    xout = x > 0.5
    dout = cv2.distanceTransform(xout.astype(np.uint8), cv2.DIST_L2, 3)
    dout=1/(dout+1)
    dout = dout / dout.max()
    return dout.flatten()


def generate_sdf_singlepk(
    x,
):
    xout = x > 0.5
    dout = cv2.distanceTransform(xout.astype(np.uint8), cv2.DIST_L2, 3)
    xin = 1 - x > 0.5
    din = cv2.distanceTransform(xin.astype(np.uint8), cv2.DIST_L2, 3)
    xres = dout - din
    xrmax, xrmin = xres.max(), xres.min()
    xres = (xres - xrmin) / (xrmax - xrmin)
    return xres.flatten()


def generate_sdf_doublepk(
    x,
):
    xout = x > 0.5
    dout = cv2.distanceTransform(xout.astype(np.uint8), cv2.DIST_L2, 3)
    xin = 1 - x > 0.5
    din = cv2.distanceTransform(xin.astype(np.uint8), cv2.DIST_L2, 3)
    xres = dout + din
    xres = xres / xres.max()
    return xres.flatten()


from mmengine import Config
from mmengine.visualization import WandbVisBackend
import os 

def wandb_logger(project_name,task_name):
    save_dir=os.path.dirname(os.path.dirname(__file__))
    init_kwargs={'project':project_name,'name':task_name}
    vis=WandbVisBackend(save_dir=save_dir,init_kwargs=init_kwargs)
    return vis