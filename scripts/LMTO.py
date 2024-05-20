import os
import sys
import cv2
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))

from lmto.top2d import LMTO, BridgeForceTop, SensivityFilter, init_one, init_load
from lmto.utils import generate_sdf_out,generate_sdf_singlepk, generate_sdf_doublepk


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
    alpha = 4
    print(nelx, nely, volfrac, penal, rmin, ft, bc, opt_method, symmerty)

    topopt = LMTO(nelx, nely, volfrac, penal, rmin, ft, bc, opt_method, symmerty)
    x = init_one(nelx, nely)
    sdf_ref = init_load("refimage/GothicBridge.png", nelx, nely)
    sdf = generate_sdf_out(sdf_ref)
    iter = 0
    if alpha==10:
        sdf_weight=1.0
    else:
        sdf_weight=1-10**(-alpha)

    while topopt.if_meet_criterion(iter):
        iter += 1
        x = topopt.update_x(iter, x, sdf_weight, sdf)
        print(f"iter:{iter},c:{topopt.c},v:{topopt.v},change:{topopt.change}")
        cv2.imwrite(
            f"outputs/LMTO_outputs/iter_{iter}.png",
            (255 - x.reshape(nelx, nely).T * 255).astype(np.uint8),
        )


if __name__ == "__main__":
    main()
