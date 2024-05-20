import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import argparse
import wandb


from lmto.top2d import LMTO, BridgeForceTop, SensivityFilter, init_one, init_load
from lmto.utils import generate_sdf_out, generate_sdf_singlepk, generate_sdf_doublepk



def main(args):
    pathsplit = args.refpath.split('/')
    wandb.init(project="test", name=f"v{args.volfrac}_a{args.alpha}_{pathsplit[-1]}",config=args)
    wandb.config.update({'refname':pathsplit[-1]})
    topopt = LMTO(
        args.nelx,
        args.nely,
        args.volfrac,
        args.penal,
        args.rmin,
        args.ft,
        args.bc,
        args.opt_method,
        args.symmerty,
    )

    x = init_one(args.nelx, args.nely)
    sdf_ref = init_load(args.refpath, args.nelx, args.nely)
    sdf = generate_sdf_out(sdf_ref)
    iter = 0
    if args.alpha==10:
        alpha=1.0
    else:
        alpha=1-10**(-args.alpha)

    while topopt.if_meet_criterion(iter):
        iter += 1
        x = topopt.update_x(iter, x, alpha, sdf)
        wandb.log(
            {
                "iter": iter,
                "compliance": topopt.c,
                "volume": topopt.v,
                "change": topopt.change,
                "x": wandb.Image(
                    (255 - x.reshape(args.nelx, args.nely).T * 255).astype(np.uint8)
                ),
            }
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nelx", type=int, default=400)
    parser.add_argument("--nely", type=int, default=100)
    parser.add_argument("--volfrac", type=float, default=0.5)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=1.5)
    parser.add_argument("--ft", type=str, default="conv")
    parser.add_argument("--bc", type=str, default="BridgeForceTop")
    parser.add_argument("--opt_method", type=str, default="BESO_OC")
    parser.add_argument("--symmerty", type=dict, default={"dim": 0, "method": "max"})
    parser.add_argument("--refpath", type=str, default="refimage/GothicBridge.png")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
