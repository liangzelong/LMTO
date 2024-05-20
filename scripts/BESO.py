import os
import sys
import cv2
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))

from lmto.top2d import BESO,BridgeForceTop,SensivityFilter,init_one,init_volfrac

def main():
    nelx = 200
    nely = 50
    volfrac = 0.5
    penal = 3.0
    rmin = 1.5
    ft = "conv"
    bc = "BridgeForceTop"
    opt_method = "BESO_OC"
    symmerty = {"dim": 0, "method": "max"}
    # symmerty=None

    topopt = BESO(nelx, nely, volfrac, penal, rmin, ft, bc, opt_method, symmerty)
    x = init_volfrac(nelx, nely,volfrac)
    iter = 0

    while topopt.if_meet_criterion(iter):
        iter += 1
        x = topopt.update_x(iter, x)
        print(f"iter:{iter},c:{topopt.c},v:{topopt.v},change:{topopt.change}")
        cv2.imwrite(f'outputs/TO_outputs/iter_{iter}.png',(255-x.reshape(nelx,nely).T*255).astype(np.uint8))


if __name__ == "__main__":
    main()
